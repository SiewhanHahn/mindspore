#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DefectGAN 共享工具函数：日志、路径解析、图片验证、进度显示
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime


# ================================================================
# 日志配置
# ================================================================

def setup_logging(name: str = "defectgan", level: int = logging.INFO) -> logging.Logger:
    """创建带时间戳的 logger，同时输出到 stdout"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


log = setup_logging()


# ================================================================
# 路径工具
# ================================================================

def resolve_image_path(base_dir: Path, img_name: str) -> Optional[Path]:
    """
    解析图片路径，自动处理 CelebA 常见的扩展名变体。

    CelebA 图片文件名可能是:
      - 000001.jpg       (原始)
      - 000001.jpg.jpg   (重复扩展名)
      - 000001.png       (PNG 格式)
      - 000001.jpeg      (JPEG 变体)

    按优先级依次尝试: 原名 → .jpg.jpg → .jpg → .png → .jpeg
    返回第一个存在的路径，都不存在则返回 None。
    """
    # 1. 直接尝试原名
    direct = base_dir / img_name
    if direct.exists():
        return direct

    # 2. 标准化扩展名：用 os.path.splitext 逐步剥离
    stem, ext = os.path.splitext(img_name)

    # 候选扩展名列表，按可能性排序
    candidates = [
        stem + ".jpg.jpg",   # 双扩展名
        stem + ".jpg",       # 标准 JPEG
        stem + ".png",       # PNG
        stem + ".jpeg",      # JPEG 长写法
        stem + ".JPG",
        stem + ".PNG",
    ]

    for cand in candidates:
        cand_path = base_dir / cand
        if cand_path.exists():
            return cand_path

    return None


def normalize_image_filename(filename: str) -> str:
    """
    将任意图片扩展名标准化为 .jpg（用于与标注文件中的文件名匹配）。

    例:
        '000001.jpg.jpg' -> '000001.jpg'
        '000001.png'     -> '000001.jpg'
        '000001.jpg'     -> '000001.jpg'
    """
    # 剥离所有扩展名后缀直到不再是已知图片扩展名
    known_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    stem = filename
    while True:
        base, ext = os.path.splitext(stem)
        if ext.lower() in known_exts:
            stem = base
        else:
            break
    return stem + ".jpg"


# ================================================================
# 图片工具
# ================================================================

def get_image_size(img_path: Path) -> Optional[Tuple[int, int]]:
    """快速获取图片尺寸（不依赖 PIL），返回 (width, height) 或 None"""
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            return img.size
    except Exception:
        return None


def check_image_valid(img_path: Path, expected_size: Optional[Tuple[int, int]] = None) -> bool:
    """
    验证图片是否可以正常打开，可选检查尺寸。
    返回 True/False，不抛异常。
    """
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            img.verify()
        # verify() 后需要重新打开以检查尺寸
        if expected_size:
            with Image.open(img_path) as img:
                if img.size != expected_size:
                    return False
        return True
    except Exception:
        return False


# ================================================================
# 进度显示
# ================================================================

def progress_bar(iterable, desc: str = "", total: int = None, unit: str = "it"):
    """统一的进度条包装，自动选择 tqdm 或简单打印"""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, unit=unit)
    except ImportError:
        # 无 tqdm 时的简单回退
        if total and total > 0:
            log.info("%s (total: %d)", desc, total)
        return iterable


# ================================================================
# 环境检测
# ================================================================

def detect_mindspore() -> Tuple[bool, str, str]:
    """
    检测 MindSpore 环境。
    返回 (可用, 版本, 设备目标)
    """
    try:
        import mindspore as ms
        version = ms.__version__
        device = ms.get_context("device_target") if hasattr(ms, 'get_context') else "unknown"
        return True, version, device
    except ImportError:
        return False, "not installed", "N/A"
    except Exception as e:
        return False, str(e), "N/A"


def detect_device_capability() -> dict:
    """
    检测硬件能力，返回设备信息字典。
    """
    info = {
        "python_version": sys.version,
        "mindspore_available": False,
        "mindspore_version": "N/A",
        "default_device": "N/A",
    }

    available, version, device = detect_mindspore()
    info["mindspore_available"] = available
    info["mindspore_version"] = version
    info["default_device"] = device

    return info


def print_environment_info():
    """打印完整环境信息（用于诊断）"""
    print("=" * 60)
    print("  DefectGAN Environment")
    print("=" * 60)
    print(f"  Python:      {sys.version}")
    print(f"  Working dir: {os.getcwd()}")

    available, version, device = detect_mindspore()
    status = "OK" if available else "NOT FOUND"
    print(f"  MindSpore:   {status} | {version} (device: {device})")

    print("=" * 60)


# ================================================================
# 文件系统工具
# ================================================================

def format_size(size_bytes: int) -> str:
    """字节数转为人类可读的大小字符串"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def count_images_in_zip(zip_path: Path) -> int:
    """统计 zip 文件中的图片数量"""
    import zipfile
    if not zip_path.exists():
        return 0
    with zipfile.ZipFile(zip_path, 'r') as zf:
        return sum(1 for n in zf.namelist() if n.lower().endswith(('.png', '.jpg', '.jpeg')))
