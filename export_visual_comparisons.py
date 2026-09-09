import os
import sys
import yaml
import zipfile
import numpy as np
from PIL import Image
from copy import deepcopy
import torch

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'basicsr'))

from basicsr.models.archs import define_network

def export_images():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Exporting comparison images using device: {device}")

    out_dir = os.path.join(base_dir, "visual_results")
    os.makedirs(out_dir, exist_ok=True)

    input_dir = os.path.join(base_dir, "data", "LOLv1", "Test", "input")
    gt_dir = os.path.join(base_dir, "data", "LOLv1", "Test", "target")

    # Pick representative test images: 1.png (classic), 22.png, 778.png
    sample_names = ["1.png", "22.png", "778.png"]
    available_files = [f for f in sample_names if os.path.isfile(os.path.join(input_dir, f))]
    if not available_files:
        available_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])[:3]

    print(f"Selected sample images: {available_files}")

    # Copy input and GT images
    for f in available_files:
        img_in = Image.open(os.path.join(input_dir, f))
        img_in.save(os.path.join(out_dir, f"low_{f}"))
        if os.path.isfile(os.path.join(gt_dir, f)):
            img_gt = Image.open(os.path.join(gt_dir, f))
            img_gt.save(os.path.join(out_dir, f"gt_{f}"))

    # 1. Evaluate Strategy 1 (Best: 24.39 dB / 0.8545 SSIM)
    cfg1_path = os.path.join(base_dir, "Options", "Multinex_Strategy1_Prior_LOL-v1.yaml")
    with open(cfg1_path) as fp:
        cfg1 = yaml.safe_load(fp)
    model1 = define_network(deepcopy(cfg1["network_g"]))
    ckpt1_path = os.path.join(base_dir, "experiments", "Multinex_Strategy1_Prior_LOL-v1", "best_psnr_24.39_104000.pth")
    if not os.path.isfile(ckpt1_path):
        ckpt1_path = os.path.join(base_dir, "experiments", "Multinex_Strategy1_Prior_LOL-v1", "models", "net_g_latest.pth")

    print(f"Loading Strategy 1 weights from: {ckpt1_path}")
    ckpt1 = torch.load(ckpt1_path, map_location=device, weights_only=False)
    model1.load_state_dict({k.replace("module.", ""): v for k, v in ckpt1.get("params", ckpt1).items()})
    model1.to(device)
    model1.eval()

    with torch.no_grad():
        for f in available_files:
            in_np = np.array(Image.open(os.path.join(input_dir, f)).convert("RGB"), dtype=np.float32) / 255.0
            x = torch.from_numpy(in_np).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model1(x).squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
            Image.fromarray((out * 255.0).astype(np.uint8)).save(os.path.join(out_dir, f"strat1_{f}"))

            # Also generate color-corrected Multinex (Guided Chroma alpha=0.30)
            sum_low = np.sum(in_np, axis=2, keepdims=True) + 1e-6
            chroma_low = in_np / sum_low
            sum_out = np.sum(out, axis=2, keepdims=True) + 1e-6
            chroma_out = out / sum_out
            chroma_blended = 0.70 * chroma_out + 0.30 * chroma_low
            Y_out = 0.299 * out[..., 0:1] + 0.587 * out[..., 1:2] + 0.114 * out[..., 2:3]
            corr = np.clip(Y_out * (chroma_blended / (np.mean(chroma_blended, axis=2, keepdims=True) + 1e-6)), 0.0, 1.0)
            Image.fromarray((corr * 255.0).astype(np.uint8)).save(os.path.join(out_dir, f"multinex_guided_{f}"))

    # 2. Evaluate Baseline Multinex (24.10 dB)
    base_ckpt = os.path.join(base_dir, "pretrained_weights", "Multinex_LOL-v1.pth")
    if os.path.isfile(base_ckpt):
        cfg_base_path = os.path.join(base_dir, "Options", "Multinex_LOL-v1.yaml")
        if os.path.isfile(cfg_base_path):
            with open(cfg_base_path) as fp:
                cfg_b = yaml.safe_load(fp)
            # Baseline Multinex does not have agc in illum_flags (4 channels)
            if "illum_flags" in cfg_b["network_g"]:
                cfg_b["network_g"]["illum_flags"]["agc"] = False
            model_b = define_network(deepcopy(cfg_b["network_g"]))
            ckpt_b = torch.load(base_ckpt, map_location=device, weights_only=False)
            model_b.load_state_dict({k.replace("module.", ""): v for k, v in ckpt_b.get("params", ckpt_b).items()})
            model_b.to(device)
            model_b.eval()
            with torch.no_grad():
                for f in available_files:
                    in_np = np.array(Image.open(os.path.join(input_dir, f)).convert("RGB"), dtype=np.float32) / 255.0
                    x = torch.from_numpy(in_np).permute(2, 0, 1).unsqueeze(0).to(device)
                    out = model_b(x).squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
                    Image.fromarray((out * 255.0).astype(np.uint8)).save(os.path.join(out_dir, f"multinex_baseline_{f}"))

    # Zip everything
    zip_path = os.path.join(base_dir, "visual_results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(out_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)

    print(f"\n✓ Successfully exported {len(os.listdir(out_dir))} comparison images to: {out_dir}")
    print(f"✓ Created zip archive: {zip_path}")
    print("Download to local PC using: scp ghakobyan@cluster:~/Multinex-AGC/visual_results.zip .")

if __name__ == "__main__":
    export_images()
