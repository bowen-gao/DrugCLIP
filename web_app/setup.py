#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script to initialize the web application
"""

import os
import sys
import argparse
from pathlib import Path

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/emb_cache',
        'checkpoints',
        'logs',
        'uploads'
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_env_file():
    """Create .env file with default settings"""
    env_file = '.env'
    
    env_content = """# DrugCLIP Web Application Configuration
# Flask Settings
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Model Settings
DEVICE=cuda
MODEL_PATH=checkpoints/checkpoint_best.pt

# Data Paths
MOL_LMDB_PATH=data/mols.lmdb
POCKET_LMDB_PATH=data/pocket.lmdb
EMB_CACHE_DIR=data/emb_cache

# Upload Settings
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=uploads
"""
    
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✓ Created configuration file: {env_file}")
    else:
        print(f"⚠ Configuration file already exists: {env_file}")

def print_instructions():
    """Print setup instructions"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           DrugCLIP Web Application - Setup Complete          ║
╚══════════════════════════════════════════════════════════════╝

✓ Project structure created successfully!

Next Steps:
───────────

1. Install dependencies:
   cd web_app/backend
   pip install -r requirements.txt

2. Download model checkpoint and data:
   - Download from: https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV?usp=sharing
   - Place checkpoint in: ./checkpoints/checkpoint_best.pt
   - Place data in: ./data/

3. Run the application:
   
   Option A - Local development:
   ───────────────────────────────
   cd web_app/backend
   python app.py
   
   Then open: http://localhost:5000

   Option B - Docker:
   ──────────────────
   docker-compose up --build
   
   Then open: http://localhost:5000

4. Access the web interface:
   - Single Prediction: Upload one molecule for prediction
   - Batch Prediction: Upload multiple molecules from SMILES file
   - Results Analysis: View charts and statistics
   - Help: View documentation

📝 Configuration:
──────────────────
- Edit .env file to customize settings
- Change DEVICE to 'cpu' if GPU is not available

📚 API Documentation:
───────────────────────
- See API_DOCS.md for detailed endpoint documentation

🔗 Resources:
──────────────
- Paper: https://arxiv.org/pdf/2310.06367.pdf
- GitHub: https://github.com/bowen-gao/DrugCLIP
- PDB: https://www.rcsb.org/

💡 Tips:
────────
- Use SMILES format for molecules
- Use PDB ID for protein pockets
- Download results as CSV or JSON for analysis

For more information, see the help section in the web interface.
""")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup DrugCLIP Web Application')
    args = parser.parse_args()
    
    print("Setting up DrugCLIP Web Application...")
    print()
    
    setup_directories()
    create_env_file()
    
    print()
    print_instructions()
