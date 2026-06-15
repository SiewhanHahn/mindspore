#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DefectGAN 集中配置模块

所有路径基于 config.py 自身所在目录自动解析，无硬编码绝对路径。
分辨率驱动参数（channel_base、num_layers、lr）自动计算。

用法:
    from config import get_config
    cfg = get_config(resolution=512)
    print(cfg.dataset_zip)  # -> 自动解析的完整路径
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DefectGANConfig:
    """DefectGAN 全局配置，所有路径相对于 project_root 自动计算"""

    # --- 用户可覆盖项 ---
    resolution: int = 512
    batch_size: int = 2
    total_kimg: float = 25000
    snap_interval: int = 10
    random_seed: int = 42
    train_device: str = "CPU"          # StyleGAN2 训练不支持 Ascend
    infer_device: str = "Ascend"       # 推理支持 Ascend
    num_defect_max: int = 700
    num_normal_max: int = 300
    truncation_psi: float = 0.5

    # --- 内部字段（由 __post_init__ 计算）---
    project_root: Path = field(init=False)

    def __post_init__(self):
        # config.py 所在目录即为项目根目录
        self.project_root = Path(__file__).resolve().parent

    # ================================================================
    # 路径属性 — 全部相对于 project_root 自动计算
    # ================================================================

    @property
    def anno_dir(self) -> Path:
        return self.project_root / "Anno"

    @property
    def annotations_file(self) -> Path:
        return self.anno_dir / "list_attr_celeba.txt"

    @property
    def bbox_file(self) -> Path:
        return self.anno_dir / "list_bbox_celeba.txt"

    @property
    def landmarks_file(self) -> Path:
        return self.anno_dir / "list_landmarks_celeba.txt"

    @property
    def landmarks_align_file(self) -> Path:
        return self.anno_dir / "list_landmarks_align_celeba.txt"

    @property
    def source_images_dir(self) -> Path:
        """CelebA 原始图片目录（128x128 缩略图）"""
        return self.project_root / "celeba-128"

    @property
    def ckpt_dir(self) -> Path:
        return self.project_root / "ckpt"

    @property
    def pretrained_ckpt_dir(self) -> Path:
        """预训练 FFHQ checkpoint 目录"""
        return self.ckpt_dir / "ffhq"

    @property
    def pretrained_ckpt(self) -> Path:
        """G_ema.ckpt 预训练生成器权重"""
        return self.pretrained_ckpt_dir / "G_ema.ckpt"

    @property
    def dataset_dir(self) -> Path:
        return self.project_root / "dataset"

    @property
    def dataset_zip(self) -> Path:
        """数据集 zip 路径。优先查找 dataset/ 子目录，其次项目根目录。"""
        # 首选 dataset/ 子目录
        path = self.dataset_dir / f"celeba_defect_ffhq_{self.resolution}.zip"
        if path.exists():
            return path
        # 回退：项目根目录（兼容旧布局）
        root_path = self.project_root / f"celeba_defect_ffhq_{self.resolution}.zip"
        if root_path.exists():
            return root_path
        # 都不存在时返回 dataset/ 路径（用于构建新数据集）
        return path

    @property
    def stylegan2_src(self) -> Path:
        """MindSpore StyleGAN2 源码目录"""
        return self.project_root / "mindspore-course" / "application_example" / "stylegan2" / "src"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output" / f"res_{self.resolution}"

    @property
    def output_training_dir(self) -> Path:
        return self.output_dir / "training"

    @property
    def output_inference_dir(self) -> Path:
        return self.output_dir / "inference"

    # ================================================================
    # 分辨率驱动的模型参数（与 StyleGAN2 train.py 保持一致）
    # ================================================================

    @property
    def channel_base(self) -> int:
        """合成网络通道基数。512+ 分辨率需要更大容量"""
        return 32768 if self.resolution >= 512 else 16384

    @property
    def channel_max(self) -> int:
        return 512

    @property
    def mapping_num_layers(self) -> int:
        """mapping 网络层数"""
        return 8 if self.resolution >= 512 else 2

    @property
    def learning_rate(self) -> float:
        """分辨率相关的学习率"""
        if self.resolution >= 1024:
            return 0.002
        return 0.0025  # 512 和 128 都用 0.0025

    # ================================================================
    # 缺陷属性定义
    # ================================================================

    @property
    def defect_attrs(self) -> list:
        """用于判定"缺陷"的 CelebA 属性名列表"""
        return [
            "Eyeglasses",
            "Double_Chin",
            "Bags_Under_Eyes",
            "Wearing_Hat",
            "Chubby",
            "Goatee",
            "Mustache",
        ]

    # ================================================================
    # 工具方法
    # ================================================================

    def ensure_dirs(self):
        """创建所有需要的输出目录"""
        dirs = [
            self.dataset_dir,
            self.output_training_dir,
            self.output_inference_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list:
        """验证关键文件和目录是否存在，返回问题列表"""
        issues = []
        checks = [
            (self.annotations_file, "CelebA 标注文件"),
            (self.pretrained_ckpt, "预训练 checkpoint G_ema.ckpt"),
            (self.source_images_dir, "源图片目录 celeba-128/"),
            (self.stylegan2_src, "StyleGAN2 源码目录"),
        ]
        for path, desc in checks:
            if not path.exists():
                issues.append(f"缺失: {desc} ({path})")
        return issues


# ================================================================
# 单例工厂
# ================================================================

_config_instance: Optional[DefectGANConfig] = None


def get_config(**overrides) -> DefectGANConfig:
    """
    获取配置单例。首次调用时可用关键字参数覆盖默认值。

    用法:
        cfg = get_config()                      # 默认 512
        cfg = get_config(resolution=128)        # 开发/测试模式
        cfg = get_config(batch_size=4, total_kimg=100)
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = DefectGANConfig(**overrides)
    return _config_instance


def reset_config():
    """重置配置单例（测试用）"""
    global _config_instance
    _config_instance = None


# ================================================================
# 自检
# ================================================================

if __name__ == "__main__":
    cfg = get_config()
    print(f"Project Root:     {cfg.project_root}")
    print(f"Resolution:       {cfg.resolution}")
    print(f"channel_base:     {cfg.channel_base}")
    print(f"mapping_layers:   {cfg.mapping_num_layers}")
    print(f"learning_rate:    {cfg.learning_rate}")
    print(f"Train Device:     {cfg.train_device}")
    print(f"Infer Device:     {cfg.infer_device}")
    print(f"Dataset Zip:      {cfg.dataset_zip}")
    print(f"Pretrained Ckpt:  {cfg.pretrained_ckpt}")
    print(f"Annotations:      {cfg.annotations_file}")
    print(f"Defect Attrs:     {cfg.defect_attrs}")
    print(f"Source Images:    {cfg.source_images_dir}")
    print()
    issues = cfg.validate()
    if issues:
        print("[WARN] Config issues:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("[OK] All critical paths verified")
