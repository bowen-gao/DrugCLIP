#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for DrugCLIP Web Application
"""

import os

class Config:
    """Base configuration"""
    
    # Flask settings
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', '../checkpoints/checkpoint_best.pt')
    DEVICE = os.getenv('DEVICE', 'cuda')  # 'cuda' or 'cpu'
    
    # Data paths
    MOL_LMDB_PATH = os.getenv('MOL_LMDB_PATH', '../data/mols.lmdb')
    POCKET_LMDB_PATH = os.getenv('POCKET_LMDB_PATH', '../data/pocket.lmdb')
    EMB_CACHE_DIR = os.getenv('EMB_CACHE_DIR', '../data/emb_cache')
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'smi', 'txt', 'smiles', 'pdb', 'mol2'}
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/drugclip_uploads')
    
    # Cache settings
    CACHE_TIMEOUT = 3600  # 1 hour
    
    # Batch settings
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', 1000))

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    DEVICE = 'cpu'  # Use CPU for development

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    DEVICE = 'cuda'  # Use GPU for production

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEVICE = 'cpu'
    MODEL_PATH = 'mock'  # Use mock model for testing
