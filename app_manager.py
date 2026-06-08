#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugCLIP Web 应用启动管理脚本
====================================
管理模型和数据下载，启动应用
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path

class DrugCLIPManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / 'data'
        self.checkpoint_dir = self.base_dir / 'checkpoints'
        self.backend_dir = self.base_dir / 'web_app' / 'backend'
        
    def check_environment(self):
        """检查环境"""
        print("🔍 检查环境...")
        print()
        
        # 检查 Python 版本
        if sys.version_info < (3, 8):
            print("❌ Python 版本过低，需要 >= 3.8")
            return False
        print(f"✓ Python {sys.version.split()[0]}")
        
        # 检查目录
        for d in [self.data_dir, self.checkpoint_dir]:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
        print(f"✓ 数据目录已创建")
        
        return True
    
    def check_model_and_data(self):
        """检查模型和数据文件"""
        print()
        print("📦 检查模型和数据文件...")
        print()
        
        required_files = {
            'checkpoints/checkpoint_best.pt': '模型检查点',
            'data/mols.lmdb': '分子 LMDB 数据库',
            'data/pocket.lmdb': '口袋 LMDB 数据库',
            'data/dict_mol.txt': '分子词典',
            'data/dict_pkt.txt': '口袋词典',
        }
        
        missing_files = []
        existing_files = []
        
        for file_path, description in required_files.items():
            full_path = self.base_dir / file_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"✓ {description:20} {file_path:30} ({size_mb:.1f} MB)")
                existing_files.append(file_path)
            else:
                print(f"✗ {description:20} {file_path:30} (缺失)")
                missing_files.append(file_path)
        
        return existing_files, missing_files
    
    def create_mock_data(self):
        """创建模拟数据用于测试"""
        print()
        print("🔧 创建模拟数据文件用于测试...")
        
        # 创建字典文件
        dict_mol = self.data_dir / 'dict_mol.txt'
        if not dict_mol.exists():
            with open(dict_mol, 'w') as f:
                f.write("C\nN\nO\nF\nS\nCl\nBr\nI\nP\n")
            print(f"✓ 创建 {dict_mol.name}")
        
        dict_pkt = self.data_dir / 'dict_pkt.txt'
        if not dict_pkt.exists():
            with open(dict_pkt, 'w') as f:
                f.write("C\nN\nO\nS\n")
            print(f"✓ 创建 {dict_pkt.name}")
        
        # 注意：LMDB 数据库和模型检查点是二进制文件，需要实际下载
        print()
        print("⚠️  模型和 LMDB 数据库需要从 Google Drive 下载")
        return False
    
    def print_download_guide(self):
        """打印下载指南"""
        print()
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║         📥 模型和数据下载指南                                   ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        print()
        print("数据来源: Google Drive")
        print("链接: https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV")
        print()
        print("需要下载的文件:")
        print()
        print("1️⃣  从 train_no_test_af.zip 中提取:")
        print("   ✓ mols.lmdb          → 放在 data/ 目录")
        print("   ✓ pocket.lmdb        → 放在 data/ 目录")
        print("   ✓ dict_mol.txt       → 放在 data/ 目录")
        print("   ✓ dict_pkt.txt       → 放在 data/ 目录")
        print()
        print("2️⃣  模型检查点:")
        print("   ✓ checkpoint_best.pt → 放在 checkpoints/ 目录")
        print()
        print("下载步骤:")
        print("  1. 打开上述 Google Drive 链接")
        print("  2. 下载文件到本地")
        print("  3. 解压 train_no_test_af.zip")
        print("  4. 复制文件到对应目录:")
        print(f"     - 数据文件 → {self.data_dir}/")
        print(f"     - 模型文件 → {self.checkpoint_dir}/")
        print()
        print("💡 或使用 gdown 下载 (如果知道文件 ID):")
        print("   $ gdown <FILE_ID> -O checkpoints/checkpoint_best.pt")
        print()
    
    def try_auto_download(self):
        """尝试自动下载"""
        print()
        print("🔄 尝试自动下载模型和数据...")
        print()
        
        # 这里可以添加具体的下载逻辑
        # 由于 Google Drive 链接通常需要特殊处理，这里只是提示
        print("⚠️  自动下载需要手动获取 Google Drive 文件链接")
        print("   请手动下载或使用 gdown 工具")
        
        return False
    
    def install_dependencies(self):
        """安装 Python 依赖"""
        print()
        print("📚 安装 Python 依赖...")
        print()
        
        requirements_file = self.backend_dir / 'requirements.txt'
        
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', '-r', str(requirements_file)],
                check=True
            )
            print("✓ 依赖安装完成")
            return True
        except Exception as e:
            print(f"❌ 依赖安装失败: {e}")
            return False
    
    def start_app(self):
        """启动应用"""
        print()
        print("🚀 启动 DrugCLIP Web 应用...")
        print()
        
        app_file = self.backend_dir / 'app.py'
        
        if not app_file.exists():
            print(f"❌ 应用文件不存在: {app_file}")
            return False
        
        try:
            print("━" * 60)
            print("Flask 应用已启动！")
            print()
            print("📱 Web 界面: http://localhost:5000")
            print()
            print("按 Ctrl+C 停止应用")
            print("━" * 60)
            print()
            
            # 启动 Flask 应用
            os.chdir(str(self.backend_dir))
            subprocess.run([sys.executable, 'app.py'])
            
        except KeyboardInterrupt:
            print()
            print("⏹️  应用已停止")
            return True
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return False
    
    def run(self, skip_download_check=False, install_deps=True, start_app=True):
        """运行管理器"""
        print()
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║         🧬 DrugCLIP Web 应用启动管理器                         ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        print()
        
        # 1. 检查环境
        if not self.check_environment():
            return False
        
        # 2. 检查模型和数据
        existing, missing = self.check_model_and_data()
        
        if missing and not skip_download_check:
            print()
            print("⚠️  缺少必需的文件:")
            for f in missing:
                print(f"   - {f}")
            
            self.print_download_guide()
            
            # 创建模拟数据用于测试
            print("🔧 创建模拟数据文件用于测试...")
            self.create_mock_data()
        
        # 3. 安装依赖
        if install_deps:
            if not self.install_dependencies():
                print()
                print("⚠️  依赖安装可能失败，但继续尝试启动应用...")
        
        # 4. 启动应用
        if start_app:
            return self.start_app()
        
        return True


def main():
    parser = argparse.ArgumentParser(description='DrugCLIP Web 应用管理器')
    parser.add_argument('--skip-download-check', action='store_true', help='跳过下载检查')
    parser.add_argument('--no-install-deps', action='store_true', help='不安装依赖')
    parser.add_argument('--no-start', action='store_true', help='不启动应用')
    parser.add_argument('--check-only', action='store_true', help='仅检查文件，不启动')
    
    args = parser.parse_args()
    
    manager = DrugCLIPManager()
    
    if args.check_only:
        manager.check_environment()
        manager.check_model_and_data()
    else:
        manager.run(
            skip_download_check=args.skip_download_check,
            install_deps=not args.no_install_deps,
            start_app=not args.no_start
        )


if __name__ == '__main__':
    main()
