import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_adaptive_gamma_luma(
    rgb: torch.Tensor,
    num_bins: int = 256,
    gamma_min: float = 0.2,
    gamma_max: float = 1.5,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes Adaptive Gamma Corrected Luminance based on the formulation from:
    T. Trongtirakul, S. S. Agaian, and S. Wu,
    'Adaptive Single Low-Light Image Enhancement by Fractional Stretching in Logarithmic Domain',
    IEEE Access, vol. 11, pp. 143936-143947, 2023 (Eq. 23).

    Args:
        rgb: Input tensor (B, 3, H, W) with float values in [0, 1].
        num_bins: Number of histogram quantization bins (default: 256).
        gamma_min: Minimum allowable adaptive gamma exponent (default: 0.2).
        gamma_max: Maximum allowable adaptive gamma exponent (default: 1.5).
        eps: Small constant for numerical stability.

    Returns:
        Y_agc: Adaptive gamma corrected luma tensor (B, 1, H, W) in [0, 1].
    """
    B, C, H, W = rgb.shape
    assert C == 3, f"Expected 3 channels for RGB input, got {C}"

    # 1. Compute perceptual ITU-R BT.709 luminance
    w_rec709 = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    Y = (rgb * w_rec709).sum(dim=1, keepdim=True).clamp(0.0, 1.0)  # (B, 1, H, W)

    # 2. Batch-wise empirical CDF calculation
    Y_quant = (Y * (num_bins - 1)).round().long().clamp(0, num_bins - 1)  # (B, 1, H, W)
    total_pixels = float(H * W)
    ln2 = 0.6931471805599453

    Y_agc_list = []
    for b in range(B):
        y_b = Y[b, 0]        # (H, W)
        y_q = Y_quant[b, 0]  # (H, W)

        # Empirical PDF & CDF
        hist = torch.histc(y_b, bins=num_bins, min=0.0, max=1.0)
        pdf = hist / (total_pixels + eps)
        cdf = torch.cumsum(pdf, dim=0).clamp(0.0, 1.0)

        # Pixel-wise lookup of CDF and PDF
        c_l = cdf[y_q]  # (H, W)
        p_l = pdf[y_q]  # (H, W)

        # Adaptive exponent from Trongtirakul, Agaian & Wu (Eq. 23)
        # gamma(L) = ln( c(L) / (p(L) + 1.0) + 1.0 + eps ) / ln(2)
        ratio = (c_l / (p_l + 1.0)).clamp_min(0.0)
        gamma_l = torch.log(ratio + 1.0 + eps) / ln2

        # Scale and clamp gamma to reasonable dynamic expansion range [gamma_min, gamma_max]
        gamma_l = gamma_l.clamp(gamma_min, gamma_max)

        # Apply adaptive gamma transformation: Y_agc = Y ^ gamma_l
        y_agc = torch.pow(y_b.clamp_min(eps), gamma_l).clamp(0.0, 1.0)
        Y_agc_list.append(y_agc)

    Y_agc = torch.stack(Y_agc_list, dim=0).unsqueeze(1)  # (B, 1, H, W)
    return Y_agc


class AdaptiveGammaHead(nn.Module):
    """
    Learnable Adaptive Gamma Illumination Head (Strategy 3).
    Formulates illumination adjustment via non-linear adaptive tone mapping:
      gamma(x, y) = clamp(0.5 * exp(tanh(W_gamma * f_L)), 0.1, 1.8)
      Delta_L = ( (Y_base + eps)^gamma - Y_base ) * (1.0 + tanh(head_luma(f_L)))
    """
    def __init__(self, in_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Predicts spatial gamma modulation
        self.gamma_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

        # Zero-initialize the modulation weights so training starts smoothly from base gamma
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.gamma_conv.bias)

    def forward(self, f_L: torch.Tensor, Y_base: torch.Tensor, head_luma: nn.Module) -> torch.Tensor:
        """
        Args:
            f_L: Feature tensor from illumination mid-sequence (B, C, H, W)
            Y_base: Base luminance map (B, 1, H, W) from input image (e.g. Rec.709)
            head_luma: The linear luma head module (B, C, H, W) -> (B, 1, H, W)
        Returns:
            L_hat: Illumination adjustment tensor Delta_L (B, 1, H, W)
        """
        # Spatially adaptive gamma exponent centered around 0.5 (ideal for low-light brightening)
        gamma_mod = torch.tanh(self.gamma_conv(f_L))
        gamma = (0.5 * torch.exp(gamma_mod)).clamp(0.1, 1.8)

        # Non-linear gamma expansion: (Y_base)^gamma - Y_base
        gamma_expanded = torch.pow(Y_base.clamp_min(self.eps), gamma)
        delta_gamma = gamma_expanded - Y_base

        # Neural refinement scale using head_luma (guarantees gradient flow to all parameters)
        res_scale = 1.0 + torch.tanh(head_luma(f_L))

        L_hat = delta_gamma * res_scale
        return L_hat
