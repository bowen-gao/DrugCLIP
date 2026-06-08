#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugCLIP Web Application Backend
Flask application for molecular virtual screening
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
from model_service import ModelService
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')
CORS(app)

# Load configuration
app.config.from_object(Config)

# Initialize model service
model_service = None

def init_model():
    """Initialize model service"""
    global model_service
    try:
        model_service = ModelService(
            model_path=app.config['MODEL_PATH'],
            device=app.config['DEVICE']
        )
        logger.info("Model service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        model_service = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_service is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict drug-protein binding affinity
    
    Expected JSON body:
    {
        "mol_smi": "SMILES string",
        "pocket_id": "PDB pocket ID",
        "pocket_pdb": "PDB file content (optional)"
    }
    
    Returns:
    {
        "success": bool,
        "score": float,
        "rank": int,
        "message": str
    }
    """
    try:
        if model_service is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded',
                'message': 'Please restart the server'
            }), 503
        
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({
                'success': False,
                'error': 'Invalid request',
                'message': 'No JSON data provided'
            }), 400
        
        mol_smi = data.get('mol_smi', '').strip()
        pocket_id = data.get('pocket_id', '').strip()
        
        if not mol_smi or not pocket_id:
            return jsonify({
                'success': False,
                'error': 'Missing required fields',
                'message': 'mol_smi and pocket_id are required'
            }), 400
        
        # Run prediction
        logger.info(f"Processing prediction: SMILES={mol_smi}, Pocket={pocket_id}")
        result = model_service.predict(mol_smi, pocket_id)
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple molecules
    
    Expected JSON body:
    {
        "molecules": [
            {"smi": "SMILES1", "name": "mol1"},
            {"smi": "SMILES2", "name": "mol2"}
        ],
        "pocket_id": "PDB pocket ID"
    }
    
    Returns:
    {
        "success": bool,
        "results": [
            {"name": str, "smi": str, "score": float, "rank": int},
            ...
        ],
        "timestamp": str
    }
    """
    try:
        if model_service is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        data = request.get_json()
        molecules = data.get('molecules', [])
        pocket_id = data.get('pocket_id', '').strip()
        
        if not molecules or not pocket_id:
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400
        
        logger.info(f"Batch prediction: {len(molecules)} molecules, Pocket={pocket_id}")
        results = model_service.batch_predict(molecules, pocket_id)
        
        # Sort by score descending
        results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
        # Add rank
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload_molecules', methods=['POST'])
def upload_molecules():
    """
    Upload molecules from file (SMILES format)
    
    Expected: File upload with .smi or .smilesfile
    
    Returns:
    {
        "success": bool,
        "molecules": [{"smi": str, "name": str}, ...],
        "count": int
    }
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        # Check file extension
        allowed_extensions = {'smi', 'txt', 'smiles'}
        if not any(file.filename.endswith('.' + ext) for ext in allowed_extensions):
            return jsonify({
                'success': False,
                'error': 'Unsupported file format. Use .smi, .txt, or .smiles'
            }), 400
        
        # Parse molecules
        molecules = []
        content = file.read().decode('utf-8')
        
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 1:
                smi = parts[0]
                name = parts[1] if len(parts) > 1 else f"mol_{len(molecules)}"
                molecules.append({
                    'smi': smi,
                    'name': name
                })
        
        logger.info(f"Uploaded {len(molecules)} molecules from file")
        
        return jsonify({
            'success': True,
            'molecules': molecules,
            'count': len(molecules)
        })
    
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example molecules and pockets for testing"""
    examples = {
        'molecules': [
            {'smi': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'name': 'Ibuprofen'},
            {'smi': 'CC(=O)Oc1ccccc1C(=O)O', 'name': 'Aspirin'},
            {'smi': 'c1ccccc1', 'name': 'Benzene'},
        ],
        'pockets': [
            {'id': '1a30', 'name': 'HIV Protease'},
            {'id': '1bma', 'name': 'Estrogen Receptor'},
            {'id': '3pbl', 'name': 'Kinase'},
        ]
    }
    return jsonify(examples)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Initialize model on startup
    init_model()
    
    # Run Flask app
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
