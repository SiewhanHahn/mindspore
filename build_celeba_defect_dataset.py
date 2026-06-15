#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: 此脚本已被 build_dataset.py 取代。

旧功能:
1. Read CelebA annotations, filter defect samples
2. Build 70% defect + 30% normal dataset
3. Build FFHQ format Zip
4. Test training on MindSpore CPU

请使用新入口:
    python run_defectgan.py build --resolution 512
    python build_dataset.py --resolution 512
"""

import os
import json
import zipfile
import shutil
import random
import io
from PIL import Image
from tqdm import tqdm


class CelebADefectBuilder:
    def __init__(self):
        self.anno_dir = r'e:\kaiyuan\Anno'
        self.img_dir = r'e:\kaiyuan\celeba-128'
        self.output_dir = r'e:\kaiyuan\dataset'
        self.zip_path = os.path.join(self.output_dir, 'celeba_defect_ffhq_128.zip')
        
        os.makedirs(self.output_dir, exist_ok=True)

    def read_attr_file(self):
        attr_file = os.path.join(self.anno_dir, 'list_attr_celeba.txt')
        attr_data = {}
        attr_names = []
        
        with open(attr_file, 'r') as f:
            lines = f.readlines()
            total_imgs = int(lines[0].strip())
            attr_names = lines[1].strip().split()
            
            for line in tqdm(lines[2:], desc="Reading annotations"):
                parts = line.strip().split()
                img_name = parts[0]
                attr_values = [int(x) for x in parts[1:]]
                attr_data[img_name] = {
                    attr_names[i]: attr_values[i] for i in range(len(attr_names))
                }
        
        print(f"OK: Read {len(attr_data)} images")
        return attr_data, attr_names

    def filter_defect_samples(self, attr_data):
        defect_list = []
        normal_list = []
        
        for img_name, attrs in tqdm(attr_data.items(), desc="Filtering samples"):
            has_defect = False
            
            if attrs.get('Eyeglasses', -1) == 1:
                has_defect = True
            if attrs.get('Double_Chin', -1) == 1:
                has_defect = True
            if attrs.get('Bags_Under_Eyes', -1) == 1:
                has_defect = True
            if attrs.get('Wearing_Hat', -1) == 1:
                has_defect = True
            if attrs.get('Chubby', -1) == 1:
                has_defect = True
            if attrs.get('Goatee', -1) == 1:
                has_defect = True
            if attrs.get('Mustache', -1) == 1:
                has_defect = True
            
            if has_defect:
                defect_list.append(img_name)
            else:
                normal_list.append(img_name)
        
        print(f"OK: Defect samples: {len(defect_list)}")
        print(f"OK: Normal samples: {len(normal_list)}")
        return defect_list, normal_list

    def build_balanced_dataset(self, defect_list, normal_list):
        random.seed(42)
        
        num_defect = min(len(defect_list), 700)
        num_normal = min(len(normal_list), 300)
        
        selected_defect = random.sample(defect_list, num_defect)
        selected_normal = random.sample(normal_list, num_normal)
        
        final_dataset = selected_defect + selected_normal
        random.shuffle(final_dataset)
        
        print(f"OK: Final dataset: {len(final_dataset)} images")
        print(f"    - Defect: {len(selected_defect)} (70%)")
        print(f"    - Normal: {len(selected_normal)} (30%)")
        
        return final_dataset

    def build_ffhq_zip(self, selected_files):
        print("\nBuilding FFHQ format Zip...")
        
        labels = []
        
        with zipfile.ZipFile(self.zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for idx, img_name in enumerate(tqdm(selected_files, desc="Building Zip")):
                img_path = os.path.join(self.img_dir, img_name)
                
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.img_dir, img_name.replace('.jpg', '.jpg.jpg'))
                
                if not os.path.exists(img_path):
                    continue
                
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    
                    if img.size != (128, 128):
                        img = img.resize((128, 128), Image.LANCZOS)
                    
                    idx_str = f'{idx:08d}'
                    archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
                    
                    img_bits = io.BytesIO()
                    img.save(img_bits, format='png', compress_level=0, optimize=False)
                    
                    zf.writestr(archive_fname, img_bits.getbuffer())
                    labels.append(None)
                    
                except Exception as e:
                    continue
            
            metadata = {'labels': labels if all(x is not None for x in labels) else None}
            zf.writestr('dataset.json', json.dumps(metadata))
        
        print(f"OK: Zip created: {self.zip_path}")
        print(f"    Size: {os.path.getsize(self.zip_path) / 1024 / 1024:.2f} MB")
        return self.zip_path


def main():
    print("=" * 60)
    print("CelebA Defect Dataset Builder")
    print("=" * 60)
    
    builder = CelebADefectBuilder()
    
    attr_data, attr_names = builder.read_attr_file()
    defect_list, normal_list = builder.filter_defect_samples(attr_data)
    final_dataset = builder.build_balanced_dataset(defect_list, normal_list)
    zip_path = builder.build_ffhq_zip(final_dataset)
    
    print("\n" + "=" * 60)
    print("Dataset building complete!")
    print("=" * 60)
    print("\nNow please run these commands to test training on MindSpore CPU:")
    print("-" * 60)
    print("conda activate mindspore_env")
    print("cd E:\\kaiyuan\\mindspore-course\\application_example\\stylegan2\\src")
    print("python train.py --data_dir=E:\\kaiyuan\\dataset\\celeba_defect_ffhq_128.zip --resume_paper=E:\\kaiyuan\\ckpt\\ffhq\\ --img_res=128 --out_dir=E:\\kaiyuan\\defect_output\\training\\ --device_target=CPU --total_kimg=0.1 --batch_size=1")
    print("-" * 60)


if __name__ == '__main__':
    main()
