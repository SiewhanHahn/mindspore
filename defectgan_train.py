#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DefectGAN StyleGAN2 微调训练脚本

在预训练 FFHQ 生成器上微调，使模型偏向生成带缺陷特征的人脸。
遵循 StyleGAN2 原始训练流程，使用 CPU 训练（Ascend 不支持 StyleGAN2 训练）。

用法:
    python defectgan_train.py --resolution 512 --total_kimg 100 --batch_size 2
    python defectgan_train.py --resolution 512 --total_kimg 10 --batch_size 1 --snap 5

训练输出:
    output/res_512/training/
        network-snapshot-000010-G.ckpt
        network-snapshot-000010-D.ckpt
        network-snapshot-000010-G_ema.ckpt
        fakes000010.png
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mindspore-course" / "application_example" / "stylegan2" / "src"))

from config import get_config, reset_config
from common import log, format_size


def create_parser():
    parser = argparse.ArgumentParser(
        description="DefectGAN StyleGAN2 Fine-tuning (CPU)"
    )
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution (default: 512)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (default: 2)")
    parser.add_argument("--total_kimg", type=float, default=100,
                        help="Total training images in thousands (default: 100)")
    parser.add_argument("--snap", type=int, default=10,
                        help="Snapshot interval in ticks (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--xflips", action="store_true", default=False,
                        help="Enable horizontal flip augmentation")
    parser.add_argument("--start_over", action="store_true", default=False,
                        help="Start training from scratch (no pretrained weights)")
    return parser


def validate_training_prerequisites(cfg):
    """验证训练前提条件，返回是否就绪"""
    errors = []

    if not cfg.dataset_zip.exists():
        errors.append(f"Dataset not found: {cfg.dataset_zip}")

    if not cfg.pretrained_ckpt.exists():
        errors.append(f"Pretrained checkpoint not found: {cfg.pretrained_ckpt}")

    if not cfg.stylegan2_src.exists():
        errors.append(f"StyleGAN2 source not found: {cfg.stylegan2_src}")

    if errors:
        log.error("Training prerequisites check failed:")
        for e in errors:
            log.error("  - %s", e)
        return False

    log.info("Training prerequisites: OK")
    return True


def create_generator(cfg, train_mode=True):
    """创建 Generator，使用分辨率正确的参数"""
    from model.generator import Generator

    generator = Generator(
        z_dim=512,
        w_dim=512,
        c_dim=0,
        img_resolution=cfg.resolution,
        img_channels=3,
        batch_size=cfg.batch_size,
        train=train_mode,
        mapping_kwargs={'num_layers': cfg.mapping_num_layers},
        synthesis_kwargs={
            'channel_base': cfg.channel_base,
            'channel_max': cfg.channel_max,
            'num_fp16_res': 4,
            'conv_clamp': 256,
        },
    )
    return generator


def create_discriminator(cfg):
    """创建 Discriminator，使用分辨率正确的参数"""
    from model.discriminator import Discriminator

    discriminator = Discriminator(
        c_dim=0,
        img_resolution=cfg.resolution,
        img_channels=3,
        architecture='resnet',
        channel_base=cfg.channel_base,
        channel_max=cfg.channel_max,
        batch_size=cfg.batch_size,
    )
    return discriminator


def _filtered_load(net, param_dict, name="model"):
    """过滤加载：只加载形状匹配的参数（支持跨分辨率迁移）"""
    net_params = {p.name: p for p in net.get_parameters()}
    loaded = 0
    skipped = 0
    for ckpt_key, ckpt_val in param_dict.items():
        if ckpt_key in net_params:
            net_param = net_params[ckpt_key]
            if ckpt_val.shape == net_param.shape:
                net_param.set_data(ckpt_val)
                loaded += 1
            else:
                skipped += 1
    log.info("  %s: %d loaded, %d skipped (shape mismatch)", name, loaded, skipped)
    return loaded, skipped


def load_pretrained_weights(generator, generator_ema, discriminator, cfg):
    """加载预训练 FFHQ 权重到 Generator 和 G_ema（过滤加载，支持跨分辨率迁移）"""
    from mindspore import load_checkpoint

    ckpt_path = str(cfg.pretrained_ckpt)
    log.info("Loading pretrained weights from %s", cfg.pretrained_ckpt.name)

    param_dict = load_checkpoint(ckpt_path)

    # 过滤加载到 G_ema（推理副本）
    _filtered_load(generator_ema, param_dict, "G_ema")

    # 过滤加载到训练 Generator
    _filtered_load(generator, param_dict, "Generator")

    # Discriminator 从头训练（新域）
    log.info("  Discriminator: training from scratch")

    return param_dict


def setup_optimizers(generator, discriminator, cfg):
    """创建 G 和 D 的 Adam 优化器"""
    import mindspore.nn as nn

    # 学习率调整（与 StyleGAN2 原始代码一致）
    g_reg_interval = 4
    d_reg_interval = 16
    mb_ratio_g = g_reg_interval / (g_reg_interval + 1)
    mb_ratio_d = d_reg_interval / (d_reg_interval + 1)

    g_lr = cfg.learning_rate * mb_ratio_g
    d_lr = cfg.learning_rate * mb_ratio_d

    g_optimizer = nn.Adam(
        generator.trainable_params(),
        learning_rate=g_lr,
        beta1=0.0,
        beta2=0.99,
        eps=1e-8,
    )

    d_optimizer = nn.Adam(
        discriminator.trainable_params(),
        learning_rate=d_lr,
        beta1=0.0,
        beta2=0.99,
        eps=1e-8,
    )

    log.info("Optimizers created:")
    log.info("  G lr: %.6f (base=%.4f, mb_ratio=%.4f)", g_lr, cfg.learning_rate, mb_ratio_g)
    log.info("  D lr: %.6f (base=%.4f, mb_ratio=%.4f)", d_lr, cfg.learning_rate, mb_ratio_d)

    return g_optimizer, d_optimizer


def setup_training_cells(generator, discriminator, g_optimizer, d_optimizer):
    """构建 GAN 训练的 loss cell 和 TrainOneStepCell"""
    from loss.stylegan2_loss import StyleGANLoss, CustomWithLossCell
    import mindspore.nn as nn
    import mindspore as ms

    # Loss 计算单元
    gan_loss = StyleGANLoss(
        generator.mapping,
        generator.synthesis,
        discriminator,
    )

    # 带梯度的 loss cell
    loss_cell = CustomWithLossCell(
        generator.mapping,
        generator.synthesis,
        discriminator,
        gan_loss,
    )

    # G 和 D 各自的 TrainOneStepCell
    # 注意：TrainOneStepCell 通过 requires_grad 切换训练目标
    g_train_cell = nn.TrainOneStepCell(loss_cell, g_optimizer)
    d_train_cell = nn.TrainOneStepCell(loss_cell, d_optimizer)

    return g_train_cell, d_train_cell


def run_training(cfg, args):
    """
    执行训练主循环。

    遵循 StyleGAN2 train.py 的流程：
    1. 加载数据集
    2. 初始化模型
    3. 交替训练 G 和 D
    4. EMA 更新 G_ema
    5. 定期保存 snapshot
    """
    import mindspore as ms
    from mindspore import Tensor, context

    from training_dataset.dataset import Ffhq
    from common import progress_bar

    # ---- 设置设备 ----
    context.set_context(device_target=cfg.train_device, device_id=0)
    log.info("Device target: %s", cfg.train_device)

    # ---- 加载数据集 ----
    log.info("Loading dataset: %s", cfg.dataset_zip)
    dataset = Ffhq(
        path=str(cfg.dataset_zip),
        batch_size=cfg.batch_size,
        resolution=cfg.resolution,
        use_labels=False,
        xflip=args.xflips,
        random_seed=cfg.random_seed,
    )
    dataset_size = len(dataset)
    batch_num = dataset_size // cfg.batch_size
    log.info("  Dataset size: %d, batches: %d", dataset_size, batch_num)

    # ---- 初始化模型 ----
    log.info("Initializing models (res=%d, channel_base=%d, num_layers=%d)",
             cfg.resolution, cfg.channel_base, cfg.mapping_num_layers)

    generator = create_generator(cfg, train_mode=True)
    generator_ema = create_generator(cfg, train_mode=False)  # 推理副本
    discriminator = create_discriminator(cfg)

    # ---- 加载预训练权重 ----
    if not args.start_over:
        load_pretrained_weights(generator, generator_ema, discriminator, cfg)
    else:
        log.warning("Starting from scratch — no pretrained weights loaded")

    # ---- 优化器 ----
    g_optimizer, d_optimizer = setup_optimizers(generator, discriminator, cfg)

    # ---- 创建输出目录 ----
    out_dir = cfg.output_training_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # ---- 训练状态变量 ----
    cur_nimg = 0
    cur_tick = 0
    total_nimg = int(cfg.total_kimg * 1000)
    tick_start_nimg = 0
    batch_idx = 0
    kimg_per_tick = 1  # 每个 tick 对应 1000 张图片
    snap_interval = args.snap
    g_reg_interval = 4
    d_reg_interval = 16

    log.info("Training targets:")
    log.info("  Total kimg: %.1f (%d images)", cfg.total_kimg, total_nimg)
    log.info("  Snap interval: %d ticks", snap_interval)
    log.info("  G reg interval: %d, D reg interval: %d", g_reg_interval, d_reg_interval)
    log.info("=" * 60)
    log.info("Starting training loop...")
    log.info("=" * 60)

    start_time = time.time()

    # ---- 训练循环 ----
    try:
        while cur_nimg < total_nimg:
            for batch_num_idx in range(batch_num):
                # (a) 加载一个 batch 的真实图片
                real_img_np, real_c_np = dataset.get_all(batch_num_idx)
                real_img = Tensor(real_img_np / 127.5 - 1.0, ms.float32)  # 归一化到 [-1, 1]

                # (b) 生成随机隐变量 z
                gen_z = Tensor(
                    np.random.randn(cfg.batch_size, 512).astype(np.float32),
                    ms.float32,
                )

                # (c) 交替训练 D 和 G
                # D 训练
                if batch_idx % d_reg_interval == 0:
                    # 只更新 D 的参数
                    for p in discriminator.get_parameters():
                        p.requires_grad = True
                    for p in generator.get_parameters():
                        p.requires_grad = False

                    d_loss = d_optimizer(
                        generator.mapping,
                        generator.synthesis,
                        discriminator,
                        real_img, real_c_np, gen_z, None
                    )

                # G 训练
                if batch_idx % g_reg_interval == 0:
                    for p in discriminator.get_parameters():
                        p.requires_grad = False
                    for p in generator.get_parameters():
                        p.requires_grad = True

                    g_loss = g_optimizer(
                        generator.mapping,
                        generator.synthesis,
                        discriminator,
                        real_img, real_c_np, gen_z, None
                    )

                # (d) EMA 更新 G_ema
                ema_beta = 0.5 ** (cfg.batch_size / (10 * 1000))
                for p_ema, p in zip(generator_ema.get_parameters(),
                                    generator.get_parameters()):
                    p_ema.set_data(p_ema * ema_beta + p * (1 - ema_beta))

                # (e) 更新计数器
                cur_nimg += cfg.batch_size
                batch_idx += 1

                # (f) Tick 维护
                if cur_nimg < tick_start_nimg + kimg_per_tick * 1000:
                    continue

                # 到达 tick 边界
                tick_duration = time.time() - start_time
                cur_tick += 1
                kimg_so_far = cur_nimg / 1000.0

                # 打印进度
                log.info(
                    "tick %d / kimg %.1f / time %.1fs",
                    cur_tick, kimg_so_far, tick_duration
                )

                # 保存模型快照
                if cur_tick % snap_interval == 0:
                    import mindspore as ms
                    kimg_str = f"{int(kimg_so_far):06d}"

                    # 保存 G_ema（推理用）
                    ms.save_checkpoint(generator_ema,
                                       str(out_dir / f"network-snapshot-{kimg_str}-G_ema.ckpt"))
                    # 保存 Generator
                    ms.save_checkpoint(generator,
                                       str(out_dir / f"network-snapshot-{kimg_str}-G.ckpt"))
                    # 保存 Discriminator
                    ms.save_checkpoint(discriminator,
                                       str(out_dir / f"network-snapshot-{kimg_str}-D.ckpt"))

                    log.info("  Snapshot saved at kimg %s", kimg_str)

                tick_start_nimg = cur_nimg
                start_time = time.time()

                if cur_nimg >= total_nimg:
                    break

            if cur_nimg >= total_nimg:
                break

    except KeyboardInterrupt:
        log.warning("Training interrupted by user")
        log.info("Saving final snapshot...")
        ms.save_checkpoint(generator_ema,
                           str(out_dir / "network-snapshot-interrupt-G_ema.ckpt"))
        ms.save_checkpoint(generator,
                           str(out_dir / "network-snapshot-interrupt-G.ckpt"))
        ms.save_checkpoint(discriminator,
                           str(out_dir / "network-snapshot-interrupt-D.ckpt"))

    log.info("=" * 60)
    log.info("Training completed!")
    log.info("Total kimg: %.1f", cur_nimg / 1000.0)
    log.info("Output: %s", out_dir)
    log.info("=" * 60)


def main():
    parser = create_parser()
    args = parser.parse_args()

    reset_config()
    cfg = get_config(
        resolution=args.resolution,
        batch_size=args.batch_size,
        total_kimg=args.total_kimg,
        snap_interval=args.snap,
        random_seed=args.seed,
    )

    print("=" * 60)
    print("  DefectGAN StyleGAN2 Fine-tuning")
    print(f"  Resolution:  {cfg.resolution}x{cfg.resolution}")
    print(f"  Batch size:  {cfg.batch_size}")
    print(f"  Total kimg:  {cfg.total_kimg}")
    print(f"  Device:      {cfg.train_device}")
    print(f"  channel_base:{cfg.channel_base}")
    print(f"  num_layers:  {cfg.mapping_num_layers}")
    print("=" * 60)

    if not validate_training_prerequisites(cfg):
        log.error("Cannot start training. Fix the issues above and try again.")
        sys.exit(1)

    cfg.ensure_dirs()
    run_training(cfg, args)


if __name__ == "__main__":
    main()
