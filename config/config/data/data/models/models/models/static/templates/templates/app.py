
"""

import os
import json
import tempfile
import threading
import time
from pathlib import Path

# Check for available dependencies and import only what's available
try:
    from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask not available. Installing basic dependencies...")
    FLASK_AVAILABLE = False

# Try to import ML modules, fall back to mock implementations if not available
try:
    from config.dataset_config import DatasetConfig
    from config.model_config import ModelConfig
    from data.dataset_loader import DatasetLoader
    from models.yolo_trainer import YOLOTrainer
    from models.yolo_evaluator import YOLOEvaluator
    from models.yolo_inference import YOLOInference
    from utils.visualization import Visualizer
    from utils.file_utils import FileUtils
    ML_MODULES_AVAILABLE = True
except ImportError:
    # Create mock classes for demonstration
    ML_MODULES_AVAILABLE = False
    
    class MockTrainer:
        def __init__(self): pass
        def train(self, *args, **kwargs): return {"status": "demo_mode", "message": "Training functionality requires ML dependencies"}
    
    class MockEvaluator:
        def __init__(self): pass
        def evaluate(self, *args, **kwargs): return {"status": "demo_mode", "message": "Evaluation functionality requires ML dependencies"}
    
    class MockInference:
        def __init__(self): pass
        def predict(self, *args, **kwargs): return {"status": "demo_mode", "message": "Inference functionality requires ML dependencies"}
    
    DatasetConfig = ModelConfig = DatasetLoader = None
    YOLOTrainer = MockTrainer
    YOLOEvaluator = MockEvaluator
    YOLOInference = MockInference
    Visualizer = FileUtils = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Global variables for tracking training status
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': 0.0,
    'status_message': 'Ready'
}

def create_directories():
    """Create necessary directories."""
    directories = [
        app.config['UPLOAD_FOLDER'],
        app.config['OUTPUT_FOLDER'],
        'datasets',
        'models'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

create_directories()

@app.route('/')
def index():
    """Main page with options for training, evaluation, and inference."""
    return render_template('index.html')

@app.route('/api/datasets')
def get_datasets():
    """Get available datasets."""
    datasets = ['coco', 'pascal_voc']
    return jsonify({'datasets': datasets})

@app.route('/api/models')
def get_models():
    """Get available trained models."""
    models_dir = Path('models')
    models = []
    if models_dir.exists():
        for model_file in models_dir.glob('*.pt'):
            models.append({
                'name': model_file.stem,
                'path': str(model_file),
                'size': model_file.stat().st_size
            })
    return jsonify({'models': models})

@app.route('/api/train', methods=['POST'])
def start_training():
    """Start model training."""
    global training_status
    
    if training_status['is_training']:
        return jsonify({'error': 'Training is already in progress'}), 400
    
    data = request.get_json()
    
    # Validate input
    required_fields = ['dataset', 'model_size', 'epochs', 'batch_size']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    def train_model():
        global training_status
        try:
            training_status['is_training'] = True
            training_status['total_epochs'] = data['epochs']
            training_status['status_message'] = 'Initializing training...'
            
            # Setup configurations
            dataset_config = DatasetConfig(
                dataset_name=data['dataset'],
                data_path='datasets',
                img_size=data.get('img_size', 640)
            )
            
            model_config = ModelConfig(
                model_size=data['model_size'],
                epochs=data['epochs'],
                batch_size=data['batch_size'],
                device=data.get('device', 'auto')
            )
            
            # Initialize trainer
            trainer = YOLOTrainer(model_config, dataset_config, app.config['OUTPUT_FOLDER'])
            
            # Start training with status updates
            def update_callback(epoch, loss):
                training_status['current_epoch'] = epoch
                training_status['current_loss'] = loss
                training_status['status_message'] = f'Training epoch {epoch}/{data["epochs"]}'
            
            model_path = trainer.train(status_callback=update_callback)
            
            training_status['status_message'] = 'Training completed successfully'
            
        except Exception as e:
            training_status['status_message'] = f'Training failed: {str(e)}'
        finally:
            training_status['is_training'] = False
    
    # Start training in background thread
    thread = threading.Thread(target=train_model)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Training started', 'status': training_status})

@app.route('/api/training_status')
def get_training_status():
    """Get current training status."""
    return jsonify(training_status)

@app.route('/api/evaluate', methods=['POST'])
def evaluate_model():
    """Evaluate a trained model."""
    data = request.get_json()
    
    if 'model_path' not in data or 'dataset' not in data:
        return jsonify({'error': 'Missing model_path or dataset'}), 400
    
    try:
        dataset_config = DatasetConfig(
            dataset_name=data['dataset'],
            data_path='datasets',
            img_size=data.get('img_size', 640)
        )
        
        evaluator = YOLOEvaluator(data['model_path'], dataset_config, app.config['OUTPUT_FOLDER'])
        metrics = evaluator.evaluate()
        
        return jsonify({'metrics': metrics, 'message': 'Evaluation completed'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/inference', methods=['POST'])
def run_inference():
    """Run inference on uploaded image."""
    if 'file' not in request.files or 'model_path' not in request.form:
        return jsonify({'error': 'Missing file or model_path'}), 400
    
    file = request.files['file']
    model_path = request.form['model_path']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Run inference
        inference = YOLOInference(model_path, 'auto')
        results = inference.predict_single(file_path, app.config['OUTPUT_FOLDER'])
        
        # Convert result image to base64 for display
        result_image_path = results.get('output_image_path')
        if result_image_path and os.path.exists(result_image_path):
            with open(result_image_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
                results['image_data'] = f"data:image/jpeg;base64,{img_data}"
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_model/<model_name>')
def download_model(model_name):
    """Download a trained model."""
    model_path = os.path.join('models', f'{model_name}.pt')
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    return jsonify({'error': 'Model not found'}), 404

@app.route('/results')
def show_results():
    """Show training/evaluation results."""
    return render_template('results.html')

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
