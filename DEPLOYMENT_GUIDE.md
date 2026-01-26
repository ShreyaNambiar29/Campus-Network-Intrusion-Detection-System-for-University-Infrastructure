# Production Deployment Guide
# Campus Network Intrusion Detection System

## 🎯 Deployment Overview

This guide provides step-by-step instructions for deploying the Campus Network IDS in a production university environment.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Ubuntu 20.04 LTS or CentOS 8+ (recommended)
- **Python Version**: 3.8 or higher
- **RAM**: Minimum 4GB, Recommended 8GB+
- **CPU**: Minimum 2 cores, Recommended 4+ cores
- **Storage**: Minimum 10GB available space
- **Network**: Gigabit Ethernet connection

### Network Access Requirements
- Access to university network traffic (mirror port or flow data)
- Outbound internet access for updates and alerts
- SMTP server access for email notifications
- Optional: API access to existing SIEM systems

## 🚀 Deployment Methods

### Method 1: Docker Deployment (Recommended)

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 idsuser && chown -R idsuser:idsuser /app
USER idsuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start application
CMD ["python", "production_server.py"]
```

#### Step 2: Create Docker Compose Configuration
```yaml
version: '3.8'

services:
  ids-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ids-api
    restart: unless-stopped

volumes:
  redis_data:
```

#### Step 3: Deploy with Docker
```bash
# Clone repository
git clone <repository-url>
cd Campus-Network-Intrusion-Detection-System-for-University-Infrastructure

# Build and start services
docker-compose up -d

# Verify deployment
docker-compose ps
docker-compose logs ids-api
```

### Method 2: Native Linux Deployment

#### Step 1: Environment Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.9 python3.9-venv python3-pip -y

# Create application user
sudo useradd -m -s /bin/bash idsadmin
sudo usermod -aG sudo idsadmin

# Create application directory
sudo mkdir -p /opt/campus-ids
sudo chown idsadmin:idsadmin /opt/campus-ids
```

#### Step 2: Application Installation
```bash
# Switch to application user
sudo su - idsadmin

# Clone repository
cd /opt/campus-ids
git clone <repository-url> .

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_ids_model.py
```

#### Step 3: System Service Configuration
```bash
# Create systemd service file
sudo tee /etc/systemd/system/campus-ids.service << EOF
[Unit]
Description=Campus Network IDS Service
After=network.target

[Service]
Type=simple
User=idsadmin
WorkingDirectory=/opt/campus-ids
Environment=PATH=/opt/campus-ids/venv/bin
ExecStart=/opt/campus-ids/venv/bin/python production_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable campus-ids
sudo systemctl start campus-ids

# Check service status
sudo systemctl status campus-ids
```

## 🔧 Configuration

### Production Configuration File
```python
# config/production.py
import os

class ProductionConfig:
    # Model Configuration
    MODEL_PATH = "/opt/campus-ids/model_outputs/final_ids_model.pkl"
    PREDICTION_THRESHOLD = 0.5
    BATCH_SIZE = 1000
    
    # Performance Settings
    MAX_WORKERS = 4
    TIMEOUT = 30
    MAX_REQUESTS = 10000
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FILE = "/var/log/campus-ids/ids.log"
    LOG_ROTATION = "daily"
    LOG_RETENTION_DAYS = 30
    
    # Alert Configuration
    SMTP_SERVER = "smtp.university.edu"
    SMTP_PORT = 587
    ALERT_EMAIL = "security@university.edu"
    SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")
    
    # Security Settings
    API_KEY_REQUIRED = True
    RATE_LIMIT = "1000/hour"
    CORS_ENABLED = False
    
    # Monitoring
    METRICS_ENABLED = True
    HEALTH_CHECK_ENDPOINT = "/health"
    METRICS_ENDPOINT = "/metrics"
```

### Network Traffic Integration
```python
# traffic_collector.py
import asyncio
import pcapy
from scapy.all import *

class TrafficCollector:
    def __init__(self, interface="eth0"):
        self.interface = interface
        self.ids_service = IDSService()
        
    async def start_collection(self):
        """Start collecting network traffic"""
        cap = pcapy.open_live(self.interface, 65536, True, 0)
        
        while True:
            header, packet = cap.next()
            features = self.extract_features(packet)
            
            # Send to IDS for analysis
            result = await self.ids_service.predict(features)
            
            if result['prediction'] == 'Attack':
                await self.send_alert(result, packet)
                
    def extract_features(self, packet):
        """Extract NSL-KDD compatible features from packet"""
        # Implementation depends on your specific feature extraction logic
        pass
```

## 🔒 Security Hardening

### SSL/TLS Configuration
```bash
# Generate SSL certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/ids.key \
    -out /etc/ssl/certs/ids.crt

# Update nginx configuration
sudo tee /etc/nginx/sites-available/campus-ids << EOF
server {
    listen 443 ssl;
    server_name ids.university.edu;
    
    ssl_certificate /etc/ssl/certs/ids.crt;
    ssl_certificate_key /etc/ssl/private/ids.key;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF
```

### Firewall Configuration
```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8000/tcp  # Block direct access to app port
```

### API Authentication
```python
# auth.py
import jwt
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'API key required'}), 401
            
        try:
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid API key'}), 401
            
        return f(*args, **kwargs)
    return decorated_function
```

## 📊 Monitoring & Alerting

### Prometheus Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
prediction_counter = Counter('ids_predictions_total', 'Total predictions made')
prediction_latency = Histogram('ids_prediction_duration_seconds', 'Prediction latency')
attack_counter = Counter('ids_attacks_detected_total', 'Total attacks detected')
system_health = Gauge('ids_system_health', 'System health status')
```

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Campus IDS Monitoring",
    "panels": [
      {
        "title": "Predictions per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ids_predictions_total[1m])"
          }
        ]
      },
      {
        "title": "Attack Detection Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(ids_attacks_detected_total[1h])"
          }
        ]
      }
    ]
  }
}
```

### Log Monitoring with ELK Stack
```yaml
# logstash.conf
input {
  file {
    path => "/var/log/campus-ids/ids.log"
    type => "ids-logs"
  }
}

filter {
  if [type] == "ids-logs" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "campus-ids-%{+YYYY.MM.dd}"
  }
}
```

## 🔧 Maintenance & Updates

### Automated Backup Script
```bash
#!/bin/bash
# backup_ids.sh

BACKUP_DIR="/backup/campus-ids"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup model files
tar -czf $BACKUP_DIR/models_$DATE.tar.gz /opt/campus-ids/model_outputs/

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/campus-ids/config/

# Backup logs (last 7 days)
find /var/log/campus-ids/ -name "*.log" -mtime -7 | \
    tar -czf $BACKUP_DIR/logs_$DATE.tar.gz -T -

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -type f -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR"
```

### Update Procedure
```bash
#!/bin/bash
# update_ids.sh

echo "Starting IDS update procedure..."

# Stop service
sudo systemctl stop campus-ids

# Backup current version
./backup_ids.sh

# Pull latest code
cd /opt/campus-ids
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Run tests
python -m pytest tests/

# Restart service
sudo systemctl start campus-ids

# Verify deployment
sleep 10
curl -f http://localhost:8000/health || echo "Health check failed!"

echo "Update completed successfully"
```

## 📋 Deployment Verification

### Health Check Script
```bash
#!/bin/bash
# health_check.sh

echo "Campus IDS Health Check"
echo "======================"

# Check service status
echo "1. Service Status:"
sudo systemctl is-active campus-ids

# Check API endpoint
echo "2. API Health:"
curl -s http://localhost:8000/health | jq .

# Check log for errors
echo "3. Recent Errors:"
tail -n 50 /var/log/campus-ids/ids.log | grep ERROR | tail -5

# Check system resources
echo "4. System Resources:"
echo "   CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "   Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%\n", $3/$2*100)}')"
echo "   Disk Usage: $(df -h /opt/campus-ids | tail -1 | awk '{print $5}')"

# Check model performance
echo "5. Model Performance:"
python /opt/campus-ids/test_ids_model.py --quick-test
```

### Performance Benchmark
```python
# benchmark.py
import time
import requests
import statistics
from concurrent.futures import ThreadPoolExecutor

def benchmark_api():
    """Benchmark API performance"""
    endpoint = "http://localhost:8000/predict"
    test_data = {...}  # Sample test data
    
    def single_request():
        start = time.time()
        response = requests.post(endpoint, json=test_data)
        return time.time() - start
    
    # Run concurrent tests
    with ThreadPoolExecutor(max_workers=10) as executor:
        times = list(executor.map(lambda x: single_request(), range(100)))
    
    print(f"Average response time: {statistics.mean(times):.3f}s")
    print(f"95th percentile: {statistics.quantiles(times, n=20)[18]:.3f}s")
    print(f"Max response time: {max(times):.3f}s")
    print(f"Throughput: {100/sum(times):.1f} requests/second")

if __name__ == "__main__":
    benchmark_api()
```

## 📞 Support & Troubleshooting

### Common Issues & Solutions

#### Issue: High Memory Usage
```bash
# Monitor memory usage
ps aux | grep campus-ids
free -h

# Solution: Adjust batch size in config
# Restart service with new configuration
```

#### Issue: Slow Predictions
```bash
# Check CPU usage
top -p $(pgrep -f campus-ids)

# Solution: Scale horizontally or upgrade hardware
# Consider GPU acceleration for future versions
```

#### Issue: False Positive Alerts
```bash
# Review recent predictions
tail -f /var/log/campus-ids/ids.log | grep "ATTACK_DETECTED"

# Solution: Adjust prediction threshold
# Retrain model with more recent data
```

### Emergency Contacts
- **Primary**: [Student Name] - [Email] - [Phone]
- **Academic Supervisor**: [Professor] - [Email]
- **University IT**: [IT Contact] - [Email] - [Emergency Phone]

---

**Document Version**: 1.0  
**Deployment Date**: January 2026  
**Next Review**: April 2026
