#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugCLIP 模型和数据下载脚本
====================================
从 Google Drive 下载必要的文件
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         📥 DrugCLIP 模型和数据下载                             ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

def print_instructions():
    print("📋 下载说明")
    print("═" * 60)
    print()
    print("Google Drive 文件夹链接:")
    print("https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV")
    print()
    print("需要下载的文件:")
    print()
    print("1️⃣  train_no_test_af.zip (包含以下文件):")
    print("   • mols.lmdb        → 放在 data/")
    print("   • pocket.lmdb      → 放在 data/")
    print("   • dict_mol.txt     → 放在 data/")
    print("   • dict_pkt.txt     → 放在 data/")
    print()
    print("2️⃣  checkpoint_best.pt  → 放在 checkpoints/")
    print()
    print("下载方法:")
    print()
    print("方式 A - 网页下载 (推荐):")
    print("────────────────────────")
    print("1. 打开上述 Google Drive 链接")
    print("2. 右键点击 train_no_test_af.zip → 下载")
    print("3. 等待下载完成后解压")
    print("4. 复制文件到对应目录:")
    print("   $ cp train_no_test_af/* data/")
    print("5. 右键点击 checkpoint_best.pt → 下载")
    print("   $ cp checkpoint_best.pt checkpoints/")
    print()
    print("方式 B - 命令行下载 (如果知道文件 ID):")
    print("────────────────────────────────────────")
    print("安装 gdown: pip install gdown")
    print("下载文件: gdown <FILE_ID> -O output_path")
    print()
    print("═" * 60)
    print()

def create_mock_data():
    """为测试创建模拟 LMDB 数据"""
    print("🔧 创建模拟数据用于测试...")
    print()
    
    data_dir = Path('data')
    
    # 创建简单的文本占位符
    mock_files = {
        'data/mols.lmdb': 'Mock molecule LMDB database\n',
        'data/pocket.lmdb': 'Mock pocket LMDB database\n',
    }
    
    for file_path, content in mock_files.items():
        full_path = Path(file_path)
        if not full_path.exists():
            with open(full_path, 'w') as f:
                f.write(content + 'This is a placeholder for testing.\n')
            print(f"✓ 创建 {file_path} (用于测试)")
    
    print()
    print("⚠️  注意: 这些是模拟文件，不能进行实际预测")
    print("       请下载真实的数据文件来使用完整功能")
    print()

def check_files():
    """检查文件"""
    print("📦 检查文件...")
    print()
    
    required_files = {
        'checkpoints/checkpoint_best.pt': '模型检查点',
        'data/mols.lmdb': '分子数据库',
        'data/pocket.lmdb': '口袋数据库',
    }
    
    all_exist = True
    for file_path, desc in required_files.items():
        exists = Path(file_path).exists()
        status = '✓' if exists else '✗'
        print(f"{status} {desc:15} {file_path}")
        if not exists:
            all_exist = False
    
    print()
    return all_exist

def main():
    print_header()
    print_instructions()
    
    if check_files():
        print("✅ 所有文件都已准备好！")
        return True
    else:
        print("⚠️  缺少数据文件")
        print()
        print("选择操作:")
        print("1. 为测试创建模拟数据")
        print("2. 显示手动下载说明")
        print("3. 退出")
        print()
        choice = input("请选择 (1-3): ").strip()
        
        if choice == '1':
            create_mock_data()
            print("准备就绪！模拟数据已创建。")
            print("现在可以启动应用进行界面测试。")
            return True
        elif choice == '2':
            print_instructions()
            return False
        else:
            return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  已取消")
        sys.exit(1)
