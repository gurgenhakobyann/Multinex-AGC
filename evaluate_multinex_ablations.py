import os
import sys
import glob
import yaml
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'basicsr'))

from basicsr.models.archs.Multinex_arch import Multinex

# LOE implementation
def compute_loe(gt, pred):
    # gt, pred: HxWxC float [0, 1]
    L_gt = np.max(gt, axis=2)
    L_pr = np.max(pred, axis=2)
    H, W = L_gt.shape
    step = max(1, min(H, W) // 50)
    L_gt_sub = L_gt[::step, ::step]
    L_pr_sub = L_pr[::step, ::step]
    h_sub, w_sub = L_gt_sub.shape
    n = h_sub * w_sub
    g_flat = L_gt_sub.flatten()
    p_flat = L_pr_sub.flatten()
    diff_g = g_flat[:, None] >= g_flat[None, :]
    diff_p = p_flat[:, None] >= p_flat[None, :]
    return float(np.sum(diff_g ^ diff_p) / (n * (n - 1) / 2))


def evaluate_model(config_path, weights_path, input_dir, gt_dir, device='cuda'):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    net_cfg = cfg['network_g']
    # Instantiate Multinex
    model = Multinex(
        in_ch=net_cfg.get('in_ch', 3),
        out_ch=net_cfg.get('out_ch', 3),
        base_channels=net_cfg.get('base_channels', 40),
        width_mult=net_cfg.get('width_mult', 2.0),
        use_depthwise=net_cfg.get('use_depthwise', True),
        use_illum_attn=net_cfg.get('use_illum_attn', True),
        use_chroma_attn=net_cfg.get('use_chroma_attn', True),
        illum_mid=net_cfg.get('illum_mid', 3),
        chroma_mid=net_cfg.get('chroma_mid', 3),
        illum_flags=net_cfg.get('illum_flags', {}),
        chroma_flags=net_cfg.get('chroma_flags', {}),
        use_adaptive_gamma_head=net_cfg.get('use_adaptive_gamma_head', False),
        target_params=net_cfg.get('target_params', 45000)
    )

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt.get('params', ckpt.get('params_ema', ckpt))
    # clean module prefix if needed
    cleaned_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned_state, strict=True)
    model.to(device)
    model.eval()

    img_names = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))])
    psnr_list, ssim_list, loe_list = [], [], []

    with torch.no_grad():
        for fname in img_names:
            in_p = os.path.join(input_dir, fname)
            gt_p = os.path.join(gt_dir, fname)
            if not os.path.isfile(gt_p):
                continue

            in_img = np.array(Image.open(in_p).convert('RGB'), dtype=np.float32) / 255.0
            gt_img = np.array(Image.open(gt_p).convert('RGB'), dtype=np.float32) / 255.0

            x = torch.from_numpy(in_img).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model(x)
            pr_img = out.squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()

            h = min(pr_img.shape[0], gt_img.shape[0])
            w = min(pr_img.shape[1], gt_img.shape[1])
            pr_img, gt_img = pr_img[:h, :w], gt_img[:h, :w]

            psnr_list.append(psnr_fn(gt_img, pr_img, data_range=1.0))
            ssim_list.append(ssim_fn(gt_img, pr_img, data_range=1.0, channel_axis=2))
            loe_list.append(compute_loe(gt_img, pr_img))

    return {
        'psnr': float(np.mean(psnr_list)),
        'ssim': float(np.mean(ssim_list)),
        'loe': float(np.mean(loe_list)),
        'count': len(psnr_list)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multinex Ablation Strategies on LOL-v1')
    parser.add_argument('--exp_dir', default='experiments', type=str)
    parser.add_argument('--data_root', default='data/LOLv1', type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Multinex Ablations using device: {device}")

    # Fallback search for Test folder
    input_dir = os.path.join(args.data_root, 'Test', 'input')
    gt_dir = os.path.join(args.data_root, 'Test', 'target')
    if not os.path.isdir(input_dir):
        for cand in ['LOLdataset/eval15', 'LOLdataset/our485', '../LOLdataset/eval15']:
            if os.path.isdir(os.path.join(cand, 'low')):
                input_dir = os.path.join(cand, 'low')
                gt_dir = os.path.join(cand, 'high')
                break

    print(f"Using Test LQ: {input_dir}")
    print(f"Using Test GT: {gt_dir}")

    models_to_eval = [
        ('Strategy 1 (AGC Prior Only)',
         'Options/Multinex_Strategy1_Prior_LOL-v1.yaml',
         'Multinex_Strategy1_Prior_LOL_v1'),

        ('Strategy 3 (Learnable Head Only)',
         'Options/Multinex_Strategy3_Head_LOL-v1.yaml',
         'Multinex_Strategy3_Head_LOL_v1'),

        ('Combined (Strategy 1 + Strategy 3)',
         'Options/Multinex_Strategy1_3_Combined_LOL-v1.yaml',
         'Multinex_Strategy1_3_Combined_LOL_v1'),
    ]

    print("\n" + "=" * 70)
    print(f"{'MULTINEX ABLATION STUDY RESULTS ON LOL-v1':^70}")
    print("=" * 70)
    print(f"{'Method / Strategy':<35} | {'PSNR (dB)':>10} | {'SSIM':>8} | {'LOE':>8}")
    print("-" * 70)

    for name, opt_p, exp_name in models_to_eval:
        exp_folder = os.path.join(args.exp_dir, exp_name)
        # Search candidate checkpoint files: best_psnr_*, models/net_g_best, models/net_g_latest, models/net_g_*
        candidates = (
            glob.glob(os.path.join(exp_folder, 'best_psnr_*.pth')) +
            glob.glob(os.path.join(exp_folder, 'best_*.pth')) +
            glob.glob(os.path.join(exp_folder, 'models', 'net_g_best.pth')) +
            glob.glob(os.path.join(exp_folder, 'models', 'net_g_latest.pth')) +
            sorted(glob.glob(os.path.join(exp_folder, 'models', 'net_g_*.pth')), reverse=True)
        )

        if not candidates:
            print(f"{name:<35} | No checkpoints found in {exp_folder}")
            continue

        w_p = candidates[0]
        print(f"[{name}] Loading weights: {w_p}")
        res = evaluate_model(opt_p, w_p, input_dir, gt_dir, device)
        print(f"{name:<35} | {res['psnr']:>10.3f} | {res['ssim']:>8.4f} | {res['loe']:>8.2f}")

    print("=" * 70)


if __name__ == '__main__':
    main()
