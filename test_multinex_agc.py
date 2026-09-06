import os
import sys
import torch
import torch.nn as nn

# Add current directory and basicsr to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'basicsr'))

from basicsr.models.archs.Multinex_arch import Multinex
from basicsr.models.archs.agc_utils import compute_adaptive_gamma_luma, AdaptiveGammaHead


def test_agc_prior():
    print("=== 1. Testing GPU/CPU Adaptive Gamma Prior (Strategy 1) ===")
    x = torch.rand(2, 3, 128, 128)
    y_agc = compute_adaptive_gamma_luma(x)
    assert y_agc.shape == (2, 1, 128, 128), f"Unexpected shape: {y_agc.shape}"
    assert not torch.isnan(y_agc).any(), "NaN detected in Y_agc"
    assert not torch.isinf(y_agc).any(), "Inf detected in Y_agc"
    print(f"  Input: {x.shape} (min={x.min():.4f}, max={x.max():.4f})")
    print(f"  Y_agc: {y_agc.shape} (min={y_agc.min():.4f}, max={y_agc.max():.4f})")
    print("  ✓ Strategy 1 AGC prior test PASSED!")


def test_multinex_agc_forward_and_backward():
    print("\n=== 2. Testing Multinex with AGC (Strategy 1 + Strategy 3) ===")
    illum_flags = dict(
        mean=False,
        rec709=True,
        vmax=True,
        lightness=True,
        ycgco=False,
        l2norm=True,
        agc=True  # Strategy 1
    )
    chroma_flags = dict(
        yuv_uv=False,
        ycbcr_cbcr=True,
        opponent=False,
        chroma_rg=True,
        hsv_s=True
    )

    model = Multinex(
        in_ch=3,
        out_ch=3,
        base_channels=40,
        width_mult=2.0,
        use_depthwise=True,
        use_illum_attn=True,
        use_chroma_attn=True,
        illum_mid=3,
        chroma_mid=3,
        illum_flags=illum_flags,
        chroma_flags=chroma_flags,
        use_adaptive_gamma_head=True,  # Strategy 3
        target_params=45000
    )

    params = model.param_count()
    print(f"  Total Trainable Parameters: {params:,} ({params/1e3:.2f}K)")
    assert params < 50000, f"Parameter budget exceeded: {params}"

    # Forward pass with dummy low-light image batch
    x = torch.rand(2, 3, 128, 128, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 3, 128, 128), f"Unexpected output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected in output"
    print(f"  Forward pass successful: output shape = {out.shape}")

    # Backward pass & gradient flow check
    target = torch.rand(2, 3, 128, 128)
    loss = nn.MSELoss()(out, target)
    loss.backward()

    # Verify gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    print(f"  Backward pass successful: loss = {loss.item():.5f}, all gradients healthy!")
    print("  ✓ Multinex-AGC Strategy 1 + Strategy 3 test PASSED!")


if __name__ == "__main__":
    test_agc_prior()
    test_multinex_agc_forward_and_backward()
    print("\n========================================================")
    print("ALL TESTS PASSED! Multinex-AGC is verified and ready.")
    print("========================================================")
