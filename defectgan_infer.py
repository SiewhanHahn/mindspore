#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DefectGAN 推理脚本 — 使用训练好的 StyleGAN2 生成缺陷人脸

支持 Ascend NPU 推理（StyleGAN2 README 确认推理工作正常）。
也可回退到 CPU 模式。

用法:
    # 使用预训练 FFHQ checkpoint
    python defectgan_infer.py --resolution 512 --seeds 42,123,456

    # 使用微调后的 checkpoint
    python defectgan_infer.py --resolution 512 --ckpt output/res_512/training/network-snapshot-000010-G_ema.ckpt

    # 批量生成
    python defectgan_infer.py --resolution 512 --num_images 100 --out_dir output/batch

输出:
    output/res_512/inference/
        seed0042.png
        seed0123.png
        grid.png
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mindspore-course" / "application_example" / "stylegan2" / "src"))

from config import get_config, reset_config
from common import log


def create_parser():
    parser = argparse.ArgumentParser(
        description="DefectGAN Inference - Generate defect face images"
    )
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution (default: 512)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to G_ema checkpoint. "
                             "Default: use pretrained FFHQ checkpoint from config")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                        help="Comma-separated random seeds (default: 42,123,456,789,1024)")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Number of images to generate (random seeds). "
                             "Overrides --seeds if set.")
    parser.add_argument("--truncation_psi", type=float, default=0.5,
                        help="Truncation trick psi value (default: 0.5)")
    parser.add_argument("--device", type=str, default="Ascend",
                        help="Device target: Ascend (default) or CPU")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Default: output/res_{resolution}/inference/")
    parser.add_argument("--no_grid", action="store_true",
                        help="Skip generating a combined grid image")
    return parser


def validate_inference_prerequisites(cfg, ckpt_path):
    """验证推理前提条件"""
    errors = []

    if not ckpt_path.exists():
        errors.append(f"Checkpoint not found: {ckpt_path}")

    if not cfg.stylegan2_src.exists():
        errors.append(f"StyleGAN2 source not found: {cfg.stylegan2_src}")

    if errors:
        log.error("Inference prerequisites check failed:")
        for e in errors:
            log.error("  - %s", e)
        return False

    return True


def load_generator_for_inference(cfg, ckpt_path):
    """加载 Generator 并载入权重（推理模式）"""
    import mindspore as ms
    from mindspore import load_checkpoint
    from model.generator import Generator

    log.info("Creating Generator (inference mode)...")
    log.info("  res=%d, channel_base=%d, num_layers=%d",
             cfg.resolution, cfg.channel_base, cfg.mapping_num_layers)

    generator = Generator(
        z_dim=512,
        w_dim=512,
        c_dim=0,
        img_resolution=cfg.resolution,
        img_channels=3,
        train=False,
        mapping_kwargs={'num_layers': cfg.mapping_num_layers},
        synthesis_kwargs={
            'channel_base': cfg.channel_base,
            'channel_max': cfg.channel_max,
            'num_fp16_res': 4,
            'conv_clamp': 256,
        },
    )

    log.info("Loading checkpoint: %s", ckpt_path.name)
    param_dict = load_checkpoint(str(ckpt_path))

    # 过滤加载：只加载形状匹配的参数（支持 1024→512 跨分辨率迁移）
    gen_params = {p.name: p for p in generator.get_parameters()}
    loaded = 0
    skipped = 0
    for ckpt_key, ckpt_val in param_dict.items():
        if ckpt_key in gen_params:
            gen_param = gen_params[ckpt_key]
            if ckpt_val.shape == gen_param.shape:
                gen_param.set_data(ckpt_val)
                loaded += 1
            else:
                skipped += 1
    log.info("  %d loaded, %d skipped (shape mismatch)", loaded, skipped)

    if loaded == 0:
        raise RuntimeError(
            f"No matching parameters found. Checkpoint resolution may not match {cfg.resolution}.\n"
            f"Try: python run_defectgan.py test --resolution {cfg.resolution}"
        )

    return generator


def generate_images(generator, seeds, truncation_psi, out_dir, cfg, generate_grid=True):
    """使用 Generator 生成图片并保存"""
    import mindspore as ms
    from mindspore import Tensor, ops
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating %d images (psi=%.2f)...", len(seeds), truncation_psi)

    generated_paths = []

    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        z = np.random.randn(1, 512).astype(np.float32)
        z_tensor = Tensor(z, ms.float32)

        # 生成
        img_tensor = generator(z_tensor, None, truncation_psi=truncation_psi)

        # 转换: [-1, 1] -> [0, 255]
        img_np = img_tensor.asnumpy()
        img_np = np.clip((img_np + 1) * 127.5, 0, 255).astype(np.uint8)

        # CHW -> HWC
        img_np = img_np[0].transpose(1, 2, 0)

        # 保存
        out_path = out_dir / f"seed{seed:04d}.png"
        img = Image.fromarray(img_np, 'RGB')
        img.save(out_path)
        generated_paths.append(out_path)

        if (i + 1) % 10 == 0:
            log.info("  %d/%d done", i + 1, len(seeds))

    log.info("Saved %d images to %s", len(generated_paths), out_dir)

    # 生成网格图
    if generate_grid and len(seeds) >= 4:
        grid_path = make_image_grid(generated_paths, out_dir)
        log.info("Grid saved to %s", grid_path)

    return generated_paths


def make_image_grid(image_paths, out_dir, cols=4):
    """将多张图片拼接成网格图"""
    from PIL import Image

    images = [Image.open(p) for p in image_paths[:16]]  # 最多 16 张
    if not images:
        return None

    img_w, img_h = images[0].size
    n_images = len(images)
    rows = (n_images + cols - 1) // cols

    grid_w = cols * img_w
    grid_h = rows * img_h
    grid = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * img_w, row * img_h))

    grid_path = out_dir / "grid.png"
    grid.save(grid_path)
    return grid_path


def run_inference(cfg, args):
    """执行推理主流程"""
    import mindspore as ms
    from mindspore import context

    # ---- 设备设置 ----
    device = args.device
    log.info("Setting device target: %s", device)

    try:
        context.set_context(device_target=device, device_id=0)
        # Ascend 推理建议使用 PYNATIVE_MODE
        if device == "Ascend":
            context.set_context(mode=context.PYNATIVE_MODE)
        log.info("  Device context set successfully")
    except Exception as e:
        log.warning("Failed to set %s: %s", device, e)
        log.info("Falling back to CPU")
        context.set_context(device_target="CPU", device_id=0)

    # ---- 确定 checkpoint 路径 ----
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
    else:
        ckpt_path = cfg.pretrained_ckpt

    if not validate_inference_prerequisites(cfg, ckpt_path):
        sys.exit(1)

    # ---- 输出目录 ----
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = cfg.output_inference_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 解析 seeds ----
    if args.num_images:
        seeds = list(range(args.num_images))
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # ---- 加载模型并生成 ----
    log.info("=" * 60)
    log.info("DefectGAN Inference")
    log.info("  Resolution: %dx%d", cfg.resolution, cfg.resolution)
    log.info("  Checkpoint: %s", ckpt_path)
    log.info("  Seeds: %d images", len(seeds))
    log.info("  Truncation psi: %.2f", args.truncation_psi)
    log.info("  Output: %s", out_dir)
    log.info("=" * 60)

    generator = load_generator_for_inference(cfg, ckpt_path)
    generate_images(
        generator, seeds, args.truncation_psi, out_dir, cfg,
        generate_grid=not args.no_grid,
    )

    log.info("=" * 60)
    log.info("Inference complete!")
    log.info("Output: %s", out_dir)
    log.info("=" * 60)


def main():
    parser = create_parser()
    args = parser.parse_args()

    reset_config()
    cfg = get_config(
        resolution=args.resolution,
        infer_device=args.device,
    )

    try:
        run_inference(cfg, args)
    except ImportError as e:
        log.error("Import error: %s", e)
        log.error("Make sure MindSpore is installed and the conda environment is active.")
        log.error("  conda activate mindspore_env")
        sys.exit(1)
    except Exception as e:
        log.error("Inference failed: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
