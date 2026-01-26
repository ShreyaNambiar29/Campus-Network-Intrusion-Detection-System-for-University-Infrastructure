#!/usr/bin/env python3
"""
Campus Network Intrusion Detection System - Production API Server
Final Year Academic Project

A production-ready REST API server for real-time intrusion detection
with comprehensive logging, monitoring, and security features.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import jwt
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configuration
class Config:
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'model_outputs/final_ids_model.pkl')
    PREDICTION_THRESHOLD = float(os.getenv('PREDICTION_THRESHOLD', '0.5'))
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'campus-ids-secret-key-change-in-production')
    API_KEY_REQUIRED = os.getenv('API_KEY_REQUIRED', 'true').lower() == 'true'
    
    # Performance
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '1000'))
    CACHE_TTL = int(os.getenv('CACHE_TTL', '300'))  # 5 minutes
    
    # Monitoring
    METRICS_ENABLED = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
    
    # Alerts
    ALERT_WEBHOOK = os.getenv('ALERT_WEBHOOK')
    SMTP_SERVER = os.getenv('SMTP_SERVER')
    ALERT_EMAIL = os.getenv('ALERT_EMAIL')

app.config.from_object(Config)

# Initialize Redis for caching (optional)
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    redis_client = None

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ids_api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
if app.config['METRICS_ENABLED']:
    prediction_counter = Counter('ids_predictions_total', 'Total predictions made', ['result'])
    prediction_latency = Histogram('ids_prediction_duration_seconds', 'Prediction latency')
    attack_counter = Counter('ids_attacks_detected_total', 'Total attacks detected')
    api_requests = Counter('ids_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
    system_health = Gauge('ids_system_health', 'System health status (1=healthy, 0=unhealthy)')

# Global model variable
model = None
model_metadata = None

def load_model():
    """Load the trained IDS model"""
    global model, model_metadata
    
    try:
        model_path = app.config['MODEL_PATH']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        if os.path.exists(metadata_path):
            model_metadata = joblib.load(metadata_path)
        else:
            model_metadata = {'version': '1.0', 'timestamp': datetime.now().isoformat()}
            
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        logger.info(f"Model metadata: {model_metadata}")
        
        if app.config['METRICS_ENABLED']:
            system_health.set(1)
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        if app.config['METRICS_ENABLED']:
            system_health.set(0)
        return False

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not app.config['API_KEY_REQUIRED']:
            return f(*args, **kwargs)
            
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authorization header required'}), 401
            
        try:
            # Extract token from "Bearer <token>"
            token = auth_header.split(' ')[1] if len(auth_header.split(' ')) > 1 else auth_header
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            g.user_id = payload.get('user_id', 'anonymous')
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid API key'}), 401
            
        return f(*args, **kwargs)
    return decorated_function

def validate_features(features: Union[Dict, List[Dict]]) -> bool:
    """Validate input features"""
    if isinstance(features, dict):
        features = [features]
        
    for feature_set in features:
        if not isinstance(feature_set, dict):
            return False
        if len(feature_set) != 122:  # Expected number of features after preprocessing
            return False
        if not all(isinstance(v, (int, float)) for v in feature_set.values()):
            return False
            
    return True

def preprocess_features(features: Union[Dict, List[Dict]]) -> np.ndarray:
    """Preprocess features for model prediction"""
    if isinstance(features, dict):
        features = [features]
        
    # Convert to DataFrame for consistent processing
    df = pd.DataFrame(features)
    
    # Ensure correct column order (assuming model expects specific order)
    if model_metadata and 'feature_columns' in model_metadata:
        df = df[model_metadata['feature_columns']]
    
    return df.values

@app.before_request
def before_request():
    """Log request and start timing"""
    g.start_time = time.time()
    
    # Skip logging for health and metrics endpoints
    if request.endpoint not in ['health', 'metrics']:
        logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    """Log response and update metrics"""
    duration = time.time() - getattr(g, 'start_time', 0)
    
    if app.config['METRICS_ENABLED'] and request.endpoint not in ['metrics']:
        api_requests.labels(
            method=request.method,
            endpoint=request.endpoint or 'unknown',
            status=response.status_code
        ).inc()
    
    # Skip logging for health and metrics endpoints
    if request.endpoint not in ['health', 'metrics']:
        logger.info(f"Response: {response.status_code} ({duration:.3f}s)")
    
    return response

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'redis_available': REDIS_AVAILABLE,
        'version': model_metadata.get('version', 'unknown') if model_metadata else 'unknown'
    }
    
    status_code = 200 if model is not None else 503
    return jsonify(health_status), status_code

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    if not app.config['METRICS_ENABLED']:
        return jsonify({'error': 'Metrics disabled'}), 404
        
    return generate_latest(), 200, {'Content-Type': 'text/plain'}

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    """Single prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Features required in request body'}), 400
        
        features = data['features']
        
        # Validate features
        if not validate_features(features):
            return jsonify({'error': 'Invalid features format'}), 400
        
        # Start timing for metrics
        start_time = time.time()
        
        # Preprocess and predict
        X = preprocess_features(features)
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Calculate metrics
        prediction_time = time.time() - start_time
        confidence = float(max(prediction_proba))
        
        # Prepare response
        result = {
            'prediction': 'Attack' if prediction == 1 else 'Normal',
            'confidence': round(confidence, 4),
            'prediction_time_ms': round(prediction_time * 1000, 2),
            'timestamp': datetime.now().isoformat(),
            'model_version': model_metadata.get('version', '1.0') if model_metadata else '1.0'
        }
        
        # Update metrics
        if app.config['METRICS_ENABLED']:
            prediction_counter.labels(result=result['prediction']).inc()
            prediction_latency.observe(prediction_time)
            
            if result['prediction'] == 'Attack':
                attack_counter.inc()
        
        # Log attack detection
        if result['prediction'] == 'Attack':
            logger.warning(f"ATTACK DETECTED - Confidence: {confidence:.4f}, Source: {request.remote_addr}")
            
            # Send alert if configured
            if app.config['ALERT_WEBHOOK']:
                send_alert(result, request.remote_addr)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
@require_api_key
def predict_batch():
    """Batch prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Features array required in request body'}), 400
        
        features_list = data['features']
        
        # Validate batch size
        if len(features_list) > app.config['MAX_BATCH_SIZE']:
            return jsonify({'error': f'Batch size exceeds limit of {app.config["MAX_BATCH_SIZE"]}'}), 400
        
        # Validate features
        if not validate_features(features_list):
            return jsonify({'error': 'Invalid features format'}), 400
        
        # Start timing
        start_time = time.time()
        
        # Preprocess and predict
        X = preprocess_features(features_list)
        predictions = model.predict(X)
        prediction_probas = model.predict_proba(X)
        
        # Calculate metrics
        prediction_time = time.time() - start_time
        
        # Prepare results
        results = []
        attack_count = 0
        
        for i, (pred, proba) in enumerate(zip(predictions, prediction_probas)):
            confidence = float(max(proba))
            prediction_label = 'Attack' if pred == 1 else 'Normal'
            
            if prediction_label == 'Attack':
                attack_count += 1
            
            results.append({
                'id': i,
                'prediction': prediction_label,
                'confidence': round(confidence, 4)
            })
        
        # Prepare response
        response = {
            'results': results,
            'summary': {
                'total_samples': len(features_list),
                'attacks_detected': attack_count,
                'normal_traffic': len(features_list) - attack_count,
                'attack_rate': round(attack_count / len(features_list), 4)
            },
            'prediction_time_ms': round(prediction_time * 1000, 2),
            'timestamp': datetime.now().isoformat(),
            'model_version': model_metadata.get('version', '1.0') if model_metadata else '1.0'
        }
        
        # Update metrics
        if app.config['METRICS_ENABLED']:
            prediction_counter.labels(result='Attack').inc(attack_count)
            prediction_counter.labels(result='Normal').inc(len(features_list) - attack_count)
            prediction_latency.observe(prediction_time)
            attack_counter.inc(attack_count)
        
        # Log batch results
        logger.info(f"Batch prediction: {len(features_list)} samples, {attack_count} attacks detected")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': 'Batch prediction failed', 'details': str(e)}), 500

@app.route('/model/info', methods=['GET'])
@require_api_key
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    info = {
        'model_type': type(model).__name__,
        'loaded_at': model_metadata.get('loaded_at', 'unknown') if model_metadata else 'unknown',
        'version': model_metadata.get('version', 'unknown') if model_metadata else 'unknown',
        'features_expected': 122,
        'classes': ['Normal', 'Attack'],
        'performance': model_metadata.get('performance', {}) if model_metadata else {}
    }
    
    return jsonify(info), 200

@app.route('/stats', methods=['GET'])
@require_api_key  
def stats():
    """Get API usage statistics"""
    if not app.config['METRICS_ENABLED']:
        return jsonify({'error': 'Metrics disabled'}), 404
    
    # This would typically come from your metrics system
    # For now, return basic info
    stats_data = {
        'uptime_seconds': time.time() - getattr(app, 'start_time', time.time()),
        'model_loaded': model is not None,
        'redis_available': REDIS_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(stats_data), 200

def send_alert(prediction_result: Dict, source_ip: str):
    """Send alert for detected attack"""
    try:
        alert_data = {
            'alert_type': 'intrusion_detected',
            'severity': 'high',
            'timestamp': prediction_result['timestamp'],
            'source_ip': source_ip,
            'confidence': prediction_result['confidence'],
            'model_version': prediction_result['model_version']
        }
        
        # Implementation depends on your alerting system
        # Could be webhook, email, Slack, etc.
        logger.info(f"Alert would be sent: {alert_data}")
        
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded'}), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

def create_directories():
    """Create necessary directories"""
    os.makedirs('logs', exist_ok=True)

if __name__ == '__main__':
    # Create necessary directories
    create_directories()
    
    # Record start time
    app.start_time = time.time()
    
    # Load model
    if not load_model():
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Enable CORS for development
    if os.getenv('ENVIRONMENT') == 'development':
        CORS(app)
    
    logger.info("Campus Network IDS API Server starting...")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Redis available: {REDIS_AVAILABLE}")
    logger.info(f"Metrics enabled: {app.config['METRICS_ENABLED']}")
    
    # Run server
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('ENVIRONMENT') == 'development'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
