#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DefectGAN 主入口 — 统计引导的真实人脸缺陷生成

子命令:
    build   从 CelebA 源数据构建 FFHQ 格式数据集
    test    验证数据集、模型和 checkpoint 是否就绪
    train   StyleGAN2 微调训练（CPU）
    infer   生成缺陷人脸图片（Ascend NPU）
    all     依次执行 build -> test -> train -> infer

用法:
    python run_defectgan.py test --resolution 512
    python run_defectgan.py train --resolution 512 --total_kimg 100 --batch_size 2
    python run_defectgan.py infer --resolution 512 --seeds 42,123,456
    python run_defectgan.py build --resolution 512
    python run_defectgan.py all --resolution 512 --total_kimg 50
"""

import sys
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config, reset_config
from common import log, print_environment_info


def create_main_parser():
    """创建主命令行解析器"""
    parser = argparse.ArgumentParser(
        description="DefectGAN - Statistical Guided Real Face Defect Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_defectgan.py test
  python run_defectgan.py train --resolution 512 --total_kimg 100
  python run_defectgan.py infer --resolution 512 --seeds 42,123,456,789
  python run_defectgan.py build --resolution 512
  python run_defectgan.py all --resolution 512 --total_kimg 50 --batch_size 2
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- build 子命令 ----
    build_parser = subparsers.add_parser("build", help="Build FFHQ zip dataset from CelebA source")
    build_parser.add_argument("--resolution", type=int, default=512)
    build_parser.add_argument("--num-defect", type=int, default=700)
    build_parser.add_argument("--num-normal", type=int, default=300)
    build_parser.add_argument("--seed", type=int, default=42)

    # ---- test 子命令 ----
    test_parser = subparsers.add_parser("test", help="Validate dataset, model, checkpoint")
    test_parser.add_argument("--resolution", type=int, default=512)
    test_parser.add_argument("--skip-checkpoint", action="store_true")

    # ---- train 子命令 ----
    train_parser = subparsers.add_parser("train", help="Fine-tune StyleGAN2 (CPU)")
    train_parser.add_argument("--resolution", type=int, default=512)
    train_parser.add_argument("--batch_size", type=int, default=2)
    train_parser.add_argument("--total_kimg", type=float, default=100)
    train_parser.add_argument("--snap", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--xflips", action="store_true", default=False)
    train_parser.add_argument("--start_over", action="store_true", default=False)

    # ---- infer 子命令 ----
    infer_parser = subparsers.add_parser("infer", help="Generate defect face images")
    infer_parser.add_argument("--resolution", type=int, default=512)
    infer_parser.add_argument("--ckpt", type=str, default=None)
    infer_parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    infer_parser.add_argument("--num_images", type=int, default=None)
    infer_parser.add_argument("--truncation_psi", type=float, default=0.5)
    infer_parser.add_argument("--device", type=str, default="Ascend")
    infer_parser.add_argument("--out_dir", type=str, default=None)
    infer_parser.add_argument("--no_grid", action="store_true")

    # ---- all 子命令 ----
    all_parser = subparsers.add_parser("all", help="Run full pipeline: build -> test -> train -> infer")
    all_parser.add_argument("--resolution", type=int, default=512)
    all_parser.add_argument("--batch_size", type=int, default=2)
    all_parser.add_argument("--total_kimg", type=float, default=100)
    all_parser.add_argument("--snap", type=int, default=10)
    all_parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    all_parser.add_argument("--skip-build", action="store_true",
                            help="Skip dataset building (use existing zip)")
    all_parser.add_argument("--skip-test", action="store_true",
                            help="Skip validation step")

    return parser


def cmd_build(args):
    """执行 build 子命令"""
    from build_dataset import CelebADefectDatasetBuilder

    reset_config()
    cfg = get_config(
        resolution=args.resolution,
        num_defect_max=args.num_defect,
        num_normal_max=args.num_normal,
        random_seed=args.seed,
    )

    issues = cfg.validate()
    if issues:
        log.warning("Some paths missing (may be OK for build):")
        for i in issues:
            log.warning("  - %s", i)

    builder = CelebADefectDatasetBuilder(cfg)
    builder.run()


def cmd_test(args):
    """执行 test 子命令"""
    from defectgan_test import DefectGANValidator

    reset_config()
    cfg = get_config(resolution=args.resolution)

    validator = DefectGANValidator(cfg, skip_checkpoint=args.skip_checkpoint)
    success = validator.run_all()
    sys.exit(0 if success else 1)


def cmd_train(args):
    """执行 train 子命令"""
    import subprocess

    cmd = [sys.executable, str(PROJECT_ROOT / "defectgan_train.py")]
    cmd.extend(["--resolution", str(args.resolution)])
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--total_kimg", str(args.total_kimg)])
    cmd.extend(["--snap", str(args.snap)])
    cmd.extend(["--seed", str(args.seed)])
    if args.xflips:
        cmd.append("--xflips")
    if args.start_over:
        cmd.append("--start_over")

    log.info("Launching: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def cmd_infer(args):
    """执行 infer 子命令"""
    import subprocess

    cmd = [sys.executable, str(PROJECT_ROOT / "defectgan_infer.py")]
    cmd.extend(["--resolution", str(args.resolution)])
    cmd.extend(["--truncation_psi", str(args.truncation_psi)])
    cmd.extend(["--device", args.device])
    cmd.extend(["--seeds", args.seeds])

    if args.ckpt:
        cmd.extend(["--ckpt", args.ckpt])
    if args.num_images:
        cmd.extend(["--num_images", str(args.num_images)])
    if args.out_dir:
        cmd.extend(["--out_dir", args.out_dir])
    if args.no_grid:
        cmd.append("--no_grid")

    log.info("Launching: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def cmd_all(args):
    """执行 all 子命令：依次运行 build -> test -> train -> infer"""
    log.info("=" * 60)
    log.info("DefectGAN Full Pipeline")
    log.info("=" * 60)

    pipeline_steps = []

    # Step 1: Build
    if not args.skip_build:
        pipeline_steps.append(("BUILD", lambda: cmd_build(args)))
    else:
        log.info("[SKIP] Build (--skip-build)")

    # Step 2: Test
    if not args.skip_test:
        pipeline_steps.append(("TEST", lambda: cmd_test(args)))
    else:
        log.info("[SKIP] Test (--skip-test)")

    # Step 3: Train
    pipeline_steps.append(("TRAIN", lambda: cmd_train(args)))

    # Step 4: Infer
    pipeline_steps.append(("INFER", lambda: cmd_infer(args)))

    for step_name, step_fn in pipeline_steps:
        log.info("=" * 40)
        log.info("  STEP: %s", step_name)
        log.info("=" * 40)
        try:
            step_fn()
        except SystemExit as e:
            if e.code != 0:
                log.error("[FAIL] %s step failed with code %s", step_name, e.code)
                log.error("Pipeline stopped.")
                sys.exit(e.code)
        except Exception as e:
            log.error("[FAIL] %s step failed: %s", step_name, e)
            sys.exit(1)

    log.info("=" * 60)
    log.info("Full pipeline completed successfully!")
    log.info("=" * 60)


def main():
    parser = create_main_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\nRun 'python run_defectgan.py <command> --help' for command details.")
        sys.exit(1)

    print_environment_info()

    command_map = {
        "build": cmd_build,
        "test": cmd_test,
        "train": cmd_train,
        "infer": cmd_infer,
        "all": cmd_all,
    }

    fn = command_map[args.command]
    fn(args)


if __name__ == "__main__":
    main()
