#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 项目文件结构清理和优化脚本
"""

import os
import shutil
from pathlib import Path

def create_directories():
    """创建标准的项目目录结构"""
    directories = [
        'src',           # 源代码
        'tests',         # 测试文件
        'docs',          # 文档
        'notebooks',     # Jupyter notebooks
        'data',          # 数据文件
        'outputs',       # 输出结果
        'scripts'        # 工具脚本
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 创建目录: {dir_name}/")

def organize_files():
    """重新组织项目文件"""
    file_moves = {
        # 源代码到src/
        'ai_enhanced_housing_model.py': 'src/',
        'ai_enhanced_housing_model_fixed.py': 'src/',
        
        # 测试文件到tests/
        'test_ai_enhanced_system.py': 'tests/',
        'test_simulation.py': 'tests/',
        'test_simulation_fixed.py': 'tests/',
        
        # 文档到docs/
        'AI_Enhanced_Housing_ABM_Guide.md': 'docs/',
        
        # notebook到notebooks/
        'ai_enhanced_housing_market_simulation_fixed.ipynb': 'notebooks/',
        
        # 脚本到scripts/
        'run_simulation.py': 'scripts/',
    }
    
    print("\n🔄 重新组织文件结构...")
    for src_file, dest_dir in file_moves.items():
        if os.path.exists(src_file):
            dest_path = os.path.join(dest_dir, src_file)
            shutil.move(src_file, dest_path)
            print(f"📦 移动: {src_file} → {dest_path}")
        else:
            print(f"⚠️ 文件不存在: {src_file}")

def update_imports():
    """更新移动后文件的导入路径"""
    import_updates = {
        'notebooks/ai_enhanced_housing_market_simulation_fixed.ipynb': [
            ('from ai_enhanced_housing_model_fixed import', 'import sys; sys.path.append("../src"); from ai_enhanced_housing_model_fixed import')
        ],
        'scripts/run_simulation.py': [
            ('from ai_enhanced_housing_model_fixed import', 'import sys; sys.path.append("../src"); from ai_enhanced_housing_model_fixed import')
        ],
        'tests/test_simulation_fixed.py': [
            ('from ai_enhanced_housing_model_fixed import', 'import sys; sys.path.append("../src"); from ai_enhanced_housing_model_fixed import')
        ]
    }
    
    print("\n🔧 更新导入路径...")
    for file_path, replacements in import_updates.items():
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for old_import, new_import in replacements:
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    print(f"✅ 更新导入: {file_path}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

def create_init_files():
    """创建__init__.py文件"""
    init_dirs = ['src', 'tests']
    
    print("\n📄 创建__init__.py文件...")
    for dir_name in init_dirs:
        init_path = os.path.join(dir_name, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write(f'"""AI增强住房市场仿真系统 - {dir_name}模块"""\n')
            print(f"✅ 创建: {init_path}")

def clean_cache():
    """清理缓存文件"""
    cache_patterns = [
        '__pycache__',
        '.ipynb_checkpoints',
        '*.pyc',
        '*.pyo',
        '*.pyd'
    ]
    
    print("\n🧹 清理缓存文件...")
    for root, dirs, files in os.walk('.'):
        # 删除缓存目录
        for cache_dir in cache_patterns[:2]:  # __pycache__ 和 .ipynb_checkpoints
            if cache_dir in dirs:
                cache_path = os.path.join(root, cache_dir)
                shutil.rmtree(cache_path)
                print(f"🗑️ 删除缓存目录: {cache_path}")

def main():
    """主函数"""
    print("🧹 开始项目清理和优化...")
    print("=" * 50)
    
    # 1. 创建目录结构
    create_directories()
    
    # 2. 重新组织文件
    organize_files()
    
    # 3. 更新导入路径
    update_imports()
    
    # 4. 创建__init__.py
    create_init_files()
    
    # 5. 清理缓存
    clean_cache()
    
    print("\n🎉 项目清理完成！")
    print("\n📋 新的文件结构:")
    print("├── src/                    # 源代码")
    print("├── tests/                  # 测试文件") 
    print("├── docs/                   # 文档")
    print("├── notebooks/              # Jupyter notebooks")
    print("├── scripts/                # 工具脚本")
    print("├── data/                   # 数据文件")
    print("├── outputs/                # 输出结果")
    print("├── requirements_ai_enhanced.txt")
    print("├── README.md")
    print("└── .gitignore")

if __name__ == "__main__":
    main() 