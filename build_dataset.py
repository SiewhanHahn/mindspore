#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DefectGAN 数据集构建器

合并了 build_celeba_defect_dataset.py 和 rebuild_dataset.py 的功能：

1. 读取 CelebA 标注，按缺陷属性筛选样本
2. 构建 70% 缺陷 + 30% 正常 的平衡数据集
3. 输出 FFHQ 格式 ZIP (ZIP_STORED + PNG, compress_level=0)

用法:
    python build_dataset.py                          # 默认 512x512
    python build_dataset.py --resolution 128         # 128x128
    python build_dataset.py --num-defect 500 --num-normal 200
"""

import os
import json
import zipfile
import random
import io
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from PIL import Image

# 将项目根目录加入 sys.path（支持直接运行和导入）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config, reset_config
from common import (
    log, progress_bar, resolve_image_path,
    normalize_image_filename, format_size
)


class CelebADefectDatasetBuilder:
    """从 CelebA 源数据构建缺陷/正常平衡数据集（FFHQ 格式 ZIP）"""

    def __init__(self, cfg):
        """
        Args:
            cfg: DefectGANConfig 实例
        """
        self.cfg = cfg
        self.attr_data: Dict[str, Dict[str, int]] = {}
        self.attr_names: List[str] = []
        self.defect_files: List[str] = []
        self.normal_files: List[str] = []
        self.skipped_count: int = 0
        self.missing_count: int = 0

    # ---------------------------------------------------------------
    # Step 1: 读取标注文件
    # ---------------------------------------------------------------

    def read_annotations(self) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
        """读取 list_attr_celeba.txt，返回 {img_name: {attr: 1/-1}} 和属性名列表"""
        attr_file = self.cfg.annotations_file

        if not attr_file.exists():
            raise FileNotFoundError(
                f"标注文件不存在: {attr_file}\n"
                f"请将 CelebA 的 list_attr_celeba.txt 放到 {self.cfg.anno_dir}"
            )

        log.info("Reading annotations from %s", attr_file.name)
        attr_data = {}
        attr_names = []

        with open(attr_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_imgs = int(lines[0].strip())
        attr_names = lines[1].strip().split()
        log.info("  Total images in annotation: %d", total_imgs)
        log.info("  Attributes: %d", len(attr_names))

        for line in progress_bar(lines[2:], desc="Reading annotations"):
            parts = line.strip().split()
            if len(parts) < len(attr_names) + 1:
                self.skipped_count += 1
                continue

            img_name = parts[0]
            attr_values = [int(x) for x in parts[1:len(attr_names) + 1]]
            attr_data[img_name] = {
                attr_names[i]: attr_values[i] for i in range(len(attr_names))
            }

        log.info("  Parsed %d image annotations", len(attr_data))
        self.attr_data = attr_data
        self.attr_names = attr_names
        return attr_data, attr_names

    # ---------------------------------------------------------------
    # Step 2: 扫描磁盘上的实际文件并匹配标注
    # ---------------------------------------------------------------

    def scan_and_match_files(self) -> Tuple[List[str], List[str]]:
        """扫描源图片目录，逐个匹配标注，分类为 defect / normal"""
        img_dir = self.cfg.source_images_dir

        if not img_dir.exists():
            raise FileNotFoundError(
                f"源图片目录不存在: {img_dir}\n"
                f"请将 CelebA 128x128 图片放到 {img_dir}"
            )

        # 收集所有图片文件
        img_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        all_files = [
            f.name for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in img_exts
            # 也匹配 .jpg.jpg (双扩展名)
            or (f.is_file() and os.path.splitext(f.name.lower())[1] in img_exts)
        ]

        # 更健壮的收集：直接遍历所有文件，尝试匹配
        all_files = []
        for f in img_dir.iterdir():
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if name_lower.endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(f.name)

        log.info("Scanning %d files in %s", len(all_files), img_dir.name)

        defect_files = []
        normal_files = []
        matched = 0
        unmatched = 0

        for filename in progress_bar(all_files, desc="Matching files"):
            # 标准化文件名用于标注匹配
            norm_name = normalize_image_filename(filename)

            if norm_name not in self.attr_data:
                unmatched += 1
                continue

            attrs = self.attr_data[norm_name]
            matched += 1

            if self._has_defect(attrs):
                defect_files.append(filename)
            else:
                normal_files.append(filename)

        log.info("  Matched: %d, Unmatched: %d", matched, unmatched)
        log.info("  Defect:  %d files", len(defect_files))
        log.info("  Normal:  %d files", len(normal_files))

        self.defect_files = defect_files
        self.normal_files = normal_files
        return defect_files, normal_files

    def _has_defect(self, attrs: Dict[str, int]) -> bool:
        """检查属性字典中是否包含任一缺陷属性"""
        for attr_name in self.cfg.defect_attrs:
            if attrs.get(attr_name, -1) == 1:
                return True
        return False

    # ---------------------------------------------------------------
    # Step 3: 采样平衡数据集
    # ---------------------------------------------------------------

    def sample_balanced(self) -> List[str]:
        """
        70% 缺陷 + 30% 正常 平衡采样。
        返回最终选中的文件名列表。
        """
        random.seed(self.cfg.random_seed)

        num_defect = min(len(self.defect_files), self.cfg.num_defect_max)
        num_normal = min(len(self.normal_files), self.cfg.num_normal_max)

        if num_defect == 0 and num_normal == 0:
            raise RuntimeError(
                "没有可用的样本！请检查源图片目录和标注文件是否匹配。"
            )

        selected_defect = random.sample(self.defect_files, num_defect) if self.defect_files else []
        selected_normal = random.sample(self.normal_files, num_normal) if self.normal_files else []

        final_list = selected_defect + selected_normal
        random.shuffle(final_list)

        log.info("Sampled dataset:")
        log.info("  Defect: %d / %d available (%.0f%%)",
                 num_defect, len(self.defect_files),
                 100 * num_defect / max(len(final_list), 1))
        log.info("  Normal: %d / %d available (%.0f%%)",
                 num_normal, len(self.normal_files),
                 100 * num_normal / max(len(final_list), 1))
        log.info("  Total:  %d images", len(final_list))

        return final_list

    # ---------------------------------------------------------------
    # Step 4: 构建 FFHQ 格式 ZIP
    # ---------------------------------------------------------------

    def build_zip(self, selected_files: List[str]) -> Path:
        """
        将选中的图片打包为 FFHQ 格式 ZIP 文件。
        FFHQ 格式规范:
          - 目录结构: 00000/img00000000.png
          - 压缩方式: ZIP_STORED (不压缩)
          - 图片格式: PNG, compress_level=0
          - 元数据: dataset.json
        """
        zip_path = self.cfg.dataset_zip
        self.cfg.ensure_dirs()

        target_size = (self.cfg.resolution, self.cfg.resolution)
        log.info("Building FFHQ zip: %s", zip_path.name)
        log.info("  Target resolution: %dx%d", *target_size)
        log.info("  Compression: ZIP_STORED, PNG compress_level=0")

        built_count = 0
        error_count = 0
        labels = []

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for idx, filename in enumerate(progress_bar(selected_files, desc="Building zip")):
                img_path = resolve_image_path(self.cfg.source_images_dir, filename)

                if img_path is None:
                    error_count += 1
                    log.debug("  Missing file: %s", filename)
                    continue

                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')

                    # 按需 resize
                    if img.size != target_size:
                        img = img.resize(target_size, Image.LANCZOS)

                    # FFHQ 目录结构
                    idx_str = f'{idx:08d}'
                    archive_name = f'{idx_str[:5]}/img{idx_str}.png'

                    # 编码为 PNG (compress_level=0, 无压缩)
                    buf = io.BytesIO()
                    img.save(buf, format='png', compress_level=0, optimize=False)

                    zf.writestr(archive_name, buf.getvalue())
                    labels.append(None)
                    built_count += 1

                except Exception as e:
                    error_count += 1
                    log.warning("  Error processing %s: %s", filename, e)

            # 写入 metadata
            metadata = {'labels': None}  # 无条件 GAN，不需要标签
            zf.writestr('dataset.json', json.dumps(metadata))

        # 报告
        zip_size = os.path.getsize(zip_path) if zip_path.exists() else 0
        log.info("Zip build complete:")
        log.info("  Path:    %s", zip_path)
        log.info("  Size:    %s", format_size(zip_size))
        log.info("  Images:  %d", built_count)
        if error_count > 0:
            log.warning("  Errors:  %d files skipped", error_count)

        if built_count == 0:
            raise RuntimeError(
                f"没有成功写入任何图片！请检查:\n"
                f"  1. 源图片目录是否包含可读图片: {self.cfg.source_images_dir}\n"
                f"  2. 文件名是否与标注文件匹配"
            )

        return zip_path

    # ---------------------------------------------------------------
    # 主流程
    # ---------------------------------------------------------------

    def run(self) -> Path:
        """执行完整的数据集构建流程，返回 zip 路径"""
        print("=" * 60)
        print("  DefectGAN Dataset Builder")
        print(f"  Resolution: {self.cfg.resolution}x{self.cfg.resolution}")
        print("=" * 60)

        if not self.attr_data:
            self.read_annotations()

        if not self.defect_files:
            self.scan_and_match_files()

        selected = self.sample_balanced()
        zip_path = self.build_zip(selected)

        print("\n" + "=" * 60)
        print("  Dataset build complete!")
        print(f"  Output: {zip_path}")
        print("=" * 60)

        return zip_path


# ================================================================
# CLI 入口
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DefectGAN Dataset Builder - Build FFHQ-format zip from CelebA source"
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Target image resolution (default: 512)"
    )
    parser.add_argument(
        "--num-defect", type=int, default=700,
        help="Maximum number of defect samples (default: 700)"
    )
    parser.add_argument(
        "--num-normal", type=int, default=300,
        help="Maximum number of normal samples (default: 300)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 重置并创建新配置
    reset_config()
    cfg = get_config(
        resolution=args.resolution,
        num_defect_max=args.num_defect,
        num_normal_max=args.num_normal,
        random_seed=args.seed,
    )

    # 验证环境
    issues = cfg.validate()
    if issues:
        log.error("Environment check failed:")
        for i in issues:
            log.error("  - %s", i)
        sys.exit(1)

    try:
        builder = CelebADefectDatasetBuilder(cfg)
        builder.run()
    except Exception as e:
        log.error("Dataset build failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
