#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-dataset NIQE & BRISQUE testing.

- Evaluates multiple input datasets using a single model weight file.
- Performs standard inference without CLAHE preprocessing.
- Computes NIQE and BRISQUE scores.
- Calculates dataset-level average scores.
- Saves all generated images.
- Outputs per-image and dataset-level NIQE and BRISQUE scores.
"""

import os
import sys
import shutil

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

import argparse
import glob
import csv
import json

from tqdm import tqdm

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models import create_model
from basicsr.utils.options import parse as parse_opt
from niqe_utils import calculate_niqe
from piq import brisque as piq_brisque


# ----------------------------
# Utilities
# ----------------------------

def list_images(input_dir):
    exts = (
        '*.png', '*.PNG',
        '*.jpg', '*.JPG',
        '*.jpeg', '*.JPEG',
        '*.bmp', '*.BMP',
        '*.tif', '*.tiff'
    )

    paths = []

    for ext in exts:
        paths.extend(
            glob.glob(
                os.path.join(input_dir, '**', ext),
                recursive=True
            )
        )

    return sorted(paths)


def imread_rgb(path):
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb_uint8):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, bgr)

    if not ok:
        ext = os.path.splitext(path)[1]
        encoded = cv2.imencode(ext, bgr)[1]

        if encoded is None:
            raise RuntimeError(f"Failed to encode image for: {path}")

        encoded.tofile(path)


def to_tensor(rgb_float01):
    return (
        torch.from_numpy(rgb_float01.transpose(2, 0, 1))
        .unsqueeze(0)
        .contiguous()
    )


def from_tensor_to_uint8_rgb(x):
    x = (
        torch.clamp(x, 0, 1)
        .detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )

    return (x * 255.0 + 0.5).astype(np.uint8)


def compute_niqe_custom(rgb_uint8):
    try:
        score = calculate_niqe(rgb_uint8)
        return float(score)

    except Exception as exc:
        print(f"[WARN] NIQE computation failed: {exc}")
        return float('nan')


def compute_brisque_piq(x_tensor):
    try:
        x_clamped = torch.clamp(x_tensor, 0.0, 1.0)

        score = piq_brisque(
            x_clamped,
            data_range=1.0,
            reduction="none"
        ).item()

        return float(score)

    except Exception as exc:
        print(f"[WARN] BRISQUE computation failed: {exc}")
        return float('nan')


def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))

        if vflip:
            x = torch.flip(x, (-1,))

        if rotate:
            x = torch.rot90(x, dims=(-2, -1))

        y = model(x)

        if rotate:
            y = torch.rot90(y, dims=(-2, -1), k=3)

        if vflip:
            y = torch.flip(y, (-1,))

        if hflip:
            y = torch.flip(y, (-2,))

        return y

    outputs = []

    for hflip in [False, True]:
        for vflip in [False, True]:
            for rotate in [False, True]:
                outputs.append(
                    forward_transformed(
                        x,
                        hflip,
                        vflip,
                        rotate,
                        model
                    )
                )

    return torch.stack(outputs).mean(dim=0)


def make_model(opt, weight_path, device, dp_world=False):
    model = create_model(opt)
    net = model.net_g

    checkpoint = torch.load(
        weight_path,
        map_location=device
    )

    state = None

    for key in ['params_ema', 'params', 'state_dict', 'model', 'net', None]:
        if key is None:
            state = checkpoint
            break

        if isinstance(checkpoint, dict) and key in checkpoint:
            state = checkpoint[key]
            break

    if not isinstance(state, dict):
        raise RuntimeError(
            f"Could not find a valid state dictionary in {weight_path}"
        )

    def try_load(target, state_dict, strict=True):
        try:
            target.load_state_dict(
                state_dict,
                strict=strict
            )
            return True

        except Exception:
            return False

    loaded = try_load(
        net,
        state,
        strict=False
    )

    if not loaded:
        stripped = {
            (
                key.replace('module.', '', 1)
                if key.startswith('module.')
                else key
            ): value
            for key, value in state.items()
        }

        loaded = try_load(
            net,
            stripped,
            strict=False
        )

    if not loaded:
        added = {
            (
                key
                if key.startswith('module.')
                else f'module.{key}'
            ): value
            for key, value in state.items()
        }

        loaded = try_load(
            net,
            added,
            strict=False
        )

    if not loaded:
        raise RuntimeError(
            f"Could not load weights from {weight_path}"
        )

    net = net.to(device)

    if dp_world:
        net = nn.DataParallel(net)

    net.eval()
    return net


def pad_to_factor(x, factor):
    _, _, height, width = x.shape

    padded_height = ((height + factor - 1) // factor) * factor
    padded_width = ((width + factor - 1) // factor) * factor

    pad_height = padded_height - height
    pad_width = padded_width - width

    if pad_height or pad_width:
        x = F.pad(
            x,
            (0, pad_width, 0, pad_height),
            mode='reflect'
        )

    return x, (height, width)


def interleave_infer(x, run, mode='none', stride=4):
    if mode == 'none':
        return run(x)

    if stride not in (2, 4):
        raise ValueError("stride must be 2 or 4")

    parts = [
        x[:, :, :, offset::stride]
        for offset in range(stride)
    ]

    outputs = [
        run(part)
        for part in parts
    ]

    y = torch.zeros_like(x)

    for offset in range(stride):
        y[:, :, :, offset::stride] = outputs[offset]

    return y


class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-dataset NIQE and BRISQUE tester "
            "using a single weight file."
        )
    )

    parser.add_argument(
        '--opt',
        type=str,
        required=True,
        help='Path to BasicSR option YAML.'
    )

    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to the model weight file.'
    )

    parser.add_argument(
        '--input_dirs',
        type=str,
        nargs='+',
        required=True,
        help='Dataset folders.'
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default='./results_multi',
        help='Root results folder.'
    )

    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help='GPU IDs, for example "0" or "0,1".'
    )

    parser.add_argument(
        '--self_ensemble',
        action='store_true',
        help='Enable self-ensemble test-time augmentation.'
    )

    parser.add_argument(
        '--pad_factor',
        type=int,
        default=32,
        help='Padding multiple.'
    )

    parser.add_argument(
        '--thresh_h',
        type=int,
        default=9000,
        help='Automatic interleave threshold for image height.'
    )

    parser.add_argument(
        '--thresh_w',
        type=int,
        default=9000,
        help='Automatic interleave threshold for image width.'
    )

    parser.add_argument(
        '--interleave',
        type=str,
        choices=['auto', 'none', '2', '4'],
        default='auto'
    )

    parser.add_argument(
        '--auto_interleave_stride',
        type=int,
        choices=[2, 4],
        default=4
    )

    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable FP16 autocast on CUDA.'
    )

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('export CUDA_VISIBLE_DEVICES=' + args.gpus)

    multi_gpu = ',' in args.gpus

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    opt = parse_opt(
        args.opt,
        is_train=False
    )
    opt['dist'] = False

    weight_path = os.path.abspath(
        os.path.expanduser(args.weights)
    )

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"Weight file does not exist: {weight_path}"
        )

    weight_name = os.path.splitext(
        os.path.basename(weight_path)
    )[0]

    result_root = os.path.abspath(
        os.path.expanduser(args.result_dir)
    )
    os.makedirs(result_root, exist_ok=True)

    print(f"Loading weights: {weight_path}")

    net = make_model(
        opt,
        weight_path,
        device,
        dp_world=multi_gpu
    )

    run = (
        (lambda x: self_ensemble(x, net))
        if args.self_ensemble
        else (lambda x: net(x))
    )

    datasets_summary = []

    for dataset_path in args.input_dirs:
        dataset_dir = os.path.abspath(
            os.path.expanduser(dataset_path)
        )

        dataset_name = os.path.basename(
            os.path.normpath(dataset_dir)
        )

        print(f"\n=== Dataset: {dataset_name} ===")

        image_paths = list_images(dataset_dir)

        if not image_paths:
            print(
                f"[WARN] No images in dataset folder: "
                f"{dataset_dir}"
            )
            continue

        dataset_root = os.path.join(
            result_root,
            dataset_name
        )

        output_dir = os.path.join(
            dataset_root,
            'images'
        )

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        image_results = []

        for image_path in tqdm(
            image_paths,
            desc=f"Images ({dataset_name})"
        ):
            rgb = (
                imread_rgb(image_path)
                .astype(np.float32)
                / 255.0
            )

            tensor = to_tensor(rgb).to(device)

            image_relative_path = os.path.relpath(
                image_path,
                dataset_dir
            ).replace('\\', '/')

            _, _, original_height, original_width = tensor.shape

            if args.interleave == 'auto':
                if (
                    original_height >= args.thresh_h
                    or original_width >= args.thresh_w
                ):
                    interleave_mode = str(
                        args.auto_interleave_stride
                    )
                    stride = args.auto_interleave_stride

                else:
                    interleave_mode = 'none'
                    stride = 4

            elif args.interleave in ('2', '4'):
                interleave_mode = args.interleave
                stride = int(args.interleave)

            else:
                interleave_mode = 'none'
                stride = 4

            tensor_padded, (height, width) = pad_to_factor(
                tensor,
                args.pad_factor
            )

            amp_context = (
                torch.autocast(
                    device_type='cuda',
                    dtype=torch.float16
                )
                if args.amp and device.type == 'cuda'
                else nullcontext()
            )

            with torch.inference_mode():
                with amp_context:
                    output_padded = interleave_infer(
                        tensor_padded,
                        run,
                        mode=interleave_mode,
                        stride=stride
                    )

            output = output_padded[:, :, :height, :width]
            output_uint8 = from_tensor_to_uint8_rgb(output)

            niqe_value = compute_niqe_custom(output_uint8)
            brisque_value = compute_brisque_piq(output)

            image_results.append(
                (
                    image_relative_path,
                    niqe_value,
                    brisque_value
                )
            )

            output_relative_path = (
                os.path.splitext(image_relative_path)[0]
                + '.png'
            )

            output_path = os.path.join(
                output_dir,
                output_relative_path
            )

            save_rgb(
                output_path,
                output_uint8
            )

        valid_niqe = [
            row[1]
            for row in image_results
            if np.isfinite(row[1])
        ]

        valid_brisque = [
            row[2]
            for row in image_results
            if np.isfinite(row[2])
        ]

        average_niqe = (
            float(np.mean(valid_niqe))
            if valid_niqe
            else float('nan')
        )

        average_brisque = (
            float(np.mean(valid_brisque))
            if valid_brisque
            else float('nan')
        )

        print(f"\n--- {dataset_name} Results ---")
        print(f"Weights: {weight_name}")
        print(f"NIQE:    {average_niqe:.4f}")
        print(f"BRISQUE: {average_brisque:.4f}")

        per_image_csv = os.path.join(
            dataset_root,
            'per_image.csv'
        )

        with open(per_image_csv, 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    'image',
                    'niqe',
                    'brisque'
                ]
            )

            for row in image_results:
                niqe_string = (
                    f"{row[1]:.6f}"
                    if np.isfinite(row[1])
                    else "nan"
                )

                brisque_string = (
                    f"{row[2]:.6f}"
                    if np.isfinite(row[2])
                    else "nan"
                )

                writer.writerow(
                    [
                        row[0],
                        niqe_string,
                        brisque_string
                    ]
                )

        dataset_summary = {
            "dataset": dataset_name,
            "num_images": len(image_paths),
            "weight_name": weight_name,
            "avg_niqe": average_niqe,
            "avg_brisque": average_brisque
        }

        dataset_summary_path = os.path.join(
            dataset_root,
            'dataset_summary.json'
        )

        with open(dataset_summary_path, 'w') as file:
            json.dump(
                dataset_summary,
                file,
                indent=2
            )

        datasets_summary.append(dataset_summary)

    final_json = os.path.join(
        result_root,
        'final_report.json'
    )

    final_csv = os.path.join(
        result_root,
        'final_report.csv'
    )

    with open(final_json, 'w') as file:
        json.dump(
            {
                "weight_name": weight_name,
                "datasets": datasets_summary
            },
            file,
            indent=2
        )

    with open(final_csv, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                'dataset',
                'num_images',
                'weight_name',
                'avg_niqe',
                'avg_brisque'
            ]
        )

        for dataset in datasets_summary:
            writer.writerow(
                [
                    dataset['dataset'],
                    dataset['num_images'],
                    dataset['weight_name'],
                    (
                        f"{dataset['avg_niqe']:.6f}"
                        if np.isfinite(dataset['avg_niqe'])
                        else "nan"
                    ),
                    (
                        f"{dataset['avg_brisque']:.6f}"
                        if np.isfinite(dataset['avg_brisque'])
                        else "nan"
                    )
                ]
            )


if __name__ == '__main__':
    main()