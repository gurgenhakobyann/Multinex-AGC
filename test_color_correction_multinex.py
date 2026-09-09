import os
import sys
import yaml
import numpy as np
from PIL import Image
from copy import deepcopy
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# Set path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'basicsr'))

from basicsr.models.archs import define_network

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Color Correction test on device: {device}")

    # 1. Load Strategy 1 model
    cfg_path = os.path.join(base_dir, "Options", "Multinex_Strategy1_Prior_LOL-v1.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = define_network(deepcopy(cfg["network_g"]))
    
    ckpt_path = os.path.join(base_dir, "experiments", "Multinex_Strategy1_Prior_LOL-v1", "best_psnr_24.39_104000.pth")
    if not os.path.isfile(ckpt_path):
        fallback = os.path.join(base_dir, "experiments", "Multinex_Strategy1_Prior_LOL-v1", "models", "net_g_latest.pth")
        ckpt_path = fallback

    print(f"Loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("params", ckpt)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(device)
    model.eval()

    input_dir = os.path.join(base_dir, "data", "LOLv1", "Test", "input")
    gt_dir = os.path.join(base_dir, "data", "LOLv1", "Test", "target")

    raw_psnr, raw_ssim = [], []
    guided_psnr, guided_ssim = [], []
    guided_psnr_30, guided_ssim_30 = [], []

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))])
    print(f"Testing on {len(files)} test images...")

    with torch.no_grad():
        for f in files:
            in_np = np.array(Image.open(os.path.join(input_dir, f)).convert("RGB"), dtype=np.float32) / 255.0
            gt_np = np.array(Image.open(os.path.join(gt_dir, f)).convert("RGB"), dtype=np.float32) / 255.0

            x = torch.from_numpy(in_np).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model(x).squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()

            h, w = min(out.shape[0], gt_np.shape[0]), min(out.shape[1], gt_np.shape[1])
            out, gt_np, in_np = out[:h, :w], gt_np[:h, :w], in_np[:h, :w]

            # 1. Raw Multinex
            raw_psnr.append(psnr_fn(gt_np, out, data_range=1.0))
            raw_ssim.append(ssim_fn(gt_np, out, data_range=1.0, channel_axis=2))

            # 2. Guided Chroma (alpha = 0.10)
            sum_low = np.sum(in_np, axis=2, keepdims=True) + 1e-6
            chroma_low = in_np / sum_low
            sum_out = np.sum(out, axis=2, keepdims=True) + 1e-6
            chroma_out = out / sum_out

            chroma_blended_10 = 0.90 * chroma_out + 0.10 * chroma_low
            Y_out = 0.299 * out[..., 0:1] + 0.587 * out[..., 1:2] + 0.114 * out[..., 2:3]
            corr_10 = np.clip(Y_out * (chroma_blended_10 / (np.mean(chroma_blended_10, axis=2, keepdims=True) + 1e-6)), 0.0, 1.0)
            guided_psnr.append(psnr_fn(gt_np, corr_10, data_range=1.0))
            guided_ssim.append(ssim_fn(gt_np, corr_10, data_range=1.0, channel_axis=2))

            # 3. Guided Chroma (alpha = 0.30)
            chroma_blended_30 = 0.70 * chroma_out + 0.30 * chroma_low
            corr_30 = np.clip(Y_out * (chroma_blended_30 / (np.mean(chroma_blended_30, axis=2, keepdims=True) + 1e-6)), 0.0, 1.0)
            guided_psnr_30.append(psnr_fn(gt_np, corr_30, data_range=1.0))
            guided_ssim_30.append(ssim_fn(gt_np, corr_30, data_range=1.0, channel_axis=2))

    print("\n" + "=" * 65)
    print(f"{'COLOR CORRECTION COMPARISON ON MULTINEX STRATEGY 1':^65}")
    print("=" * 65)
    print(f"{'Configuration':<40} | {'PSNR (dB)':>10} | {'SSIM':>8}")
    print("-" * 65)
    print(f"{'Raw Strategy 1 Output (No Post-Correction)':<40} | {np.mean(raw_psnr):>10.3f} | {np.mean(raw_ssim):>8.4f}")
    print(f"{'Strategy 1 + Guided Chroma (alpha=0.10)':<40} | {np.mean(guided_psnr):>10.3f} | {np.mean(guided_ssim):>8.4f}")
    print(f"{'Strategy 1 + Guided Chroma (alpha=0.30)':<40} | {np.mean(guided_psnr_30):>10.3f} | {np.mean(guided_ssim_30):>8.4f}")
    print("=" * 65)

if __name__ == '__main__':
    main()
