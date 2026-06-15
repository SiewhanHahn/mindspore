#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DefectGAN 验证脚本 — 检查数据集、模型初始化、checkpoint 加载

每个测试项目显式输出 PASS/FAIL，不存在静默失败。
任何项目 FAIL 时以非零退出码退出。

用法:
    python defectgan_test.py                          # 默认 512
    python defectgan_test.py --resolution 512         # 显式指定
    python defectgan_test.py --resolution 128         # 128 模式
    python defectgan_test.py --skip-checkpoint        # 跳过 checkpoint 检查
"""

import os
import sys
import argparse
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mindspore-course" / "application_example" / "stylegan2" / "src"))

from config import get_config, reset_config
from common import log, count_images_in_zip, detect_mindspore, format_size


class DefectGANValidator:
    """逐项验证 DefectGAN 环境是否就绪"""

    def __init__(self, cfg, skip_checkpoint: bool = False):
        self.cfg = cfg
        self.skip_checkpoint = skip_checkpoint
        self.results = []  # [(test_name, passed: bool, message: str)]
        self.all_passed = True

    def check(self, name: str, passed: bool, message: str = ""):
        """记录一个检查项的结果"""
        status = "[PASS]" if passed else "[FAIL]"
        self.results.append((name, passed, message))
        if not passed:
            self.all_passed = False
        print(f"  {status} {name}")
        if message:
            print(f"         {message}")

    # ---------------------------------------------------------------
    # 各检查项
    # ---------------------------------------------------------------

    def check_environment(self):
        """检查 Python 和 MindSpore 环境"""
        print("\n--- Environment ---")

        available, version, device = detect_mindspore()
        if available:
            self.check("MindSpore", True, f"Version: {version}, Device: {device}")
        else:
            self.check("MindSpore", False,
                       f"MindSpore not installed ({version}). "
                       f"Activate conda env: conda activate mindspore_env")

        self.check("Python", True, f"Version: {sys.version.split()[0]}")

    def check_dataset_zip(self):
        """检查数据集 zip 文件"""
        print("\n--- Dataset Zip ---")
        zip_path = self.cfg.dataset_zip

        if not zip_path.exists():
            self.check("Zip exists", False,
                       "File not found: {zip_path}")
            return

        zip_size = zip_path.stat().st_size
        self.check("Zip exists", True,
                   f"{zip_path.name} ({format_size(zip_size)})")

        # 检查 zip 可读性
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                all_names = zf.namelist()
                png_files = [n for n in all_names if n.endswith('.png')]
                json_files = [n for n in all_names if n.endswith('.json')]

                self.check("Zip readable", True,
                           f"{len(png_files)} PNGs, {len(json_files)} JSON files")

                # 验证 FFHQ 格式
                has_dataset_json = 'dataset.json' in all_names
                self.check("FFHQ format", has_dataset_json,
                           "dataset.json found" if has_dataset_json else
                           "dataset.json NOT found — may not be FFHQ format")

                if png_files:
                    # 检查第一张图片的格式
                    info = zf.getinfo(png_files[0])
                    is_stored = info.compress_type == 0
                    self.check("ZIP_STORED", is_stored,
                               f"Compression: {'STORED (correct)' if is_stored else f'type={info.compress_type} (expected 0=STORED)'}")

                    # 检查图片尺寸
                    try:
                        import io
                        from PIL import Image
                        img_data = zf.read(png_files[0])
                        img = Image.open(io.BytesIO(img_data))
                        expected = (self.cfg.resolution, self.cfg.resolution)
                        size_ok = img.size == expected
                        self.check("Image resolution", size_ok,
                                   f"{img.size} (expected {expected})" if size_ok else
                                   f"Got {img.size}, expected {expected}")
                    except Exception as e:
                        self.check("Image resolution", False, str(e))

        except Exception as e:
            self.check("Zip readable", False, str(e))

    def check_dataset_loading(self):
        """检查通过 Ffhq 类加载数据集"""
        print("\n--- Dataset Loading (Ffhq) ---")
        try:
            from training_dataset.dataset import Ffhq

            dataset = Ffhq(
                path=str(self.cfg.dataset_zip),
                use_labels=False,
                resolution=self.cfg.resolution,
                xflip=False,
                batch_size=1,
            )
            self.check("Ffhq load", True,
                       f"Dataset loaded, {len(dataset)} samples available")
        except Exception as e:
            self.check("Ffhq load", False, str(e))

    def check_checkpoint(self):
        """检查预训练 checkpoint"""
        if self.skip_checkpoint:
            print("\n--- Checkpoint (SKIPPED) ---")
            self.check("Checkpoint", True, "Skipped by user request")
            return

        print("\n--- Checkpoint ---")
        ckpt_path = self.cfg.pretrained_ckpt

        if not ckpt_path.exists():
            self.check("Checkpoint exists", False,
                       "File not found: {ckpt_path}")
            return

        ckpt_size = ckpt_path.stat().st_size
        self.check("Checkpoint exists", True,
                   f"{ckpt_path.name} ({format_size(ckpt_size)})")

        try:
            import mindspore as ms
from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE)
import mindspore as ms
            from mindspore import load_checkpoint

            param_dict = load_checkpoint(str(ckpt_path))
            self.check("Checkpoint load", True,
                       f"{len(param_dict)} parameters loaded")
        except Exception as e:
            self.check("Checkpoint load", False, str(e))

    def check_generator_init(self):
        """检查 Generator 初始化"""
        print("\n--- Generator Init ---")
        try:
            from model.generator import Generator

            gen = Generator(
                z_dim=512,
                w_dim=512,
                c_dim=0,
                img_resolution=self.cfg.resolution,
                img_channels=3,
                mapping_kwargs={'num_layers': self.cfg.mapping_num_layers},
                synthesis_kwargs={
                    'channel_base': self.cfg.channel_base,
                    'channel_max': self.cfg.channel_max,
                    'num_fp16_res': 4,
                    'conv_clamp': 256,
                },
            )
            self.check("Generator init", True,
                       f"Created with res={self.cfg.resolution}, "
                       f"channel_base={self.cfg.channel_base}, "
                       f"num_layers={self.cfg.mapping_num_layers}")
        except Exception as e:
            self.check("Generator init", False, str(e))

    def check_checkpoint_loading_into_generator(self):
        """检查将 checkpoint 加载到 Generator"""
        if self.skip_checkpoint:
            print("\n--- Checkpoint -> Generator (SKIPPED) ---")
            return

        print("\n--- Checkpoint -> Generator ---")
        try:
            import mindspore as ms
            from mindspore import load_checkpoint, load_param_into_net
            from model.generator import Generator

            gen = Generator(
                z_dim=512, w_dim=512, c_dim=0,
                img_resolution=self.cfg.resolution,
                img_channels=3,
                mapping_kwargs={'num_layers': self.cfg.mapping_num_layers},
                synthesis_kwargs={
                    'channel_base': self.cfg.channel_base,
                    'channel_max': self.cfg.channel_max,
                    'num_fp16_res': 4,
                    'conv_clamp': 256,
                },
            )

            # 过滤加载：只加载形状匹配的参数
            # 预训练 checkpoint 是 1024 的，Gen 是 512，会多出一个 1024x1024 的块
            param_dict = load_checkpoint(str(self.cfg.pretrained_ckpt))
            gen_params = {p.name: p for p in gen.get_parameters()}
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
            if loaded > 0:
                self.check("Ckpt -> Generator", True,
                           f"{loaded} params loaded, {skipped} skipped (1024->512 transfer)")
            else:
                self.check("Ckpt -> Generator", False,
                           f"No matching params (checkpoint resolution != {self.cfg.resolution})")
        except Exception as e:
            self.check("Ckpt -> Generator", False, str(e))

    # ---------------------------------------------------------------
    # 运行所有检查
    # ---------------------------------------------------------------

    def run_all(self):
        print("=" * 60)
        print("  DefectGAN Validation")
        print(f"  Resolution: {self.cfg.resolution}x{self.cfg.resolution}")
        print(f"  Project:    {self.cfg.project_root}")
        print("=" * 60)

        # 依次运行各检查
        self.check_environment()
        self.check_dataset_zip()
        self.check_dataset_loading()
        self.check_checkpoint()
        self.check_generator_init()
        self.check_checkpoint_loading_into_generator()

        # 汇总
        print("\n" + "=" * 60)
        passed = sum(1 for _, p, _ in self.results if p)
        failed = sum(1 for _, p, _ in self.results if not p)
        print(f"  Results: {passed} passed, {failed} failed")
        print("=" * 60)

        if not self.all_passed:
            print("\n[FAIL] Some checks failed! See details above.")
            print("Common fixes:")
            print("  1. conda activate mindspore_env")
            print("  2. Check dataset exists at:", self.cfg.dataset_zip)
            print("  3. Check checkpoint exists at:", self.cfg.pretrained_ckpt)
            return False
        else:
            print("\n[OK] All checks passed!")
            return True


# ================================================================
# CLI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DefectGAN Test - Validate dataset, model, checkpoint"
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Target resolution (default: 512)"
    )
    parser.add_argument(
        "--skip-checkpoint", action="store_true",
        help="Skip checkpoint-related checks"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reset_config()
    cfg = get_config(resolution=args.resolution)

    validator = DefectGANValidator(cfg, skip_checkpoint=args.skip_checkpoint)
    success = validator.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
