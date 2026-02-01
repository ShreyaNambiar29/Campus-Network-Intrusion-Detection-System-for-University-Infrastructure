# Real-time Packet Monitoring Documentation

## Overview

The Campus Network IDS now includes **real-time network packet monitoring** using Scapy to detect TCP SYN-based port scanning attacks. This system integrates seamlessly with the existing alert infrastructure.

## Features

### ✅ What's New
- **Real-time packet capture** using Scapy
- **Port scan detection** based on SYN packet analysis  
- **Automatic alert generation** via existing alert system
- **Configurable thresholds** via environment variables
- **Memory-efficient** sliding window implementation
- **Duplicate prevention** with cooldown periods
- **Production-ready** with proper error handling and logging

### 🎯 Detection Capabilities
- **Attack Type**: TCP SYN-based port scanning
- **Detection Method**: Sliding time window analysis
- **Default Threshold**: 15+ SYN packets in 5 seconds
- **Alert Severity**: HIGH
- **Anomaly Score**: 0.95

## Configuration

### Environment Variables (backend/.env)
```bash
# Real-time Packet Monitoring Configuration
PORT_SCAN_THRESHOLD=15      # Number of SYN packets to trigger alert
PORT_SCAN_TIME_WINDOW=5     # Time window in seconds
PORT_SCAN_COOLDOWN=30       # Seconds before same IP can trigger new alert
```

## Architecture

### System Integration
```
Internet Traffic → Scapy Sniffer → SYN Analysis → Alert Creation → MongoDB → Dashboard
```

### Components
1. **PacketMonitor Class** (`services/packet_monitor.py`)
   - Handles packet capture and analysis
   - Manages sliding time windows
   - Creates alerts via existing system

2. **Background Thread** (started in `main.py`)
   - Runs packet sniffing without blocking FastAPI
   - Automatic cleanup of old data

3. **Alert Integration**
   - Uses existing `AlertCreate` model
   - Inserts directly into MongoDB
   - Follows same alert format as other detections

## Usage

### Starting with Packet Monitoring

#### Option 1: Standard Mode (Limited Functionality)
```bash
# Start server normally
cd backend
source ../venv/bin/activate
export $(grep -v '^#' .env | xargs)
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*Note: Will log permission errors but server works normally*

#### Option 2: Full Packet Monitoring (Requires Root)
```bash
# Start server with root privileges for packet capture
cd backend  
source ../venv/bin/activate
export $(grep -v '^#' .env | xargs)
sudo $(which python) -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Testing Port Scan Detection

1. **Start server with root privileges**
2. **Generate port scan traffic:**
   ```bash
   # From another terminal
   sudo nmap -sS -p 1-100 127.0.0.1
   ```
3. **Check dashboard** for new HIGH severity alerts
4. **Monitor logs** for detection messages

### Monitoring Status API

```bash
# Get packet monitoring status (requires authentication)
curl -X GET "http://localhost:8000/api/monitoring/status" \
  -H "Authorization: Bearer YOUR_FIREBASE_TOKEN"
```

**Response:**
```json
{
  "message": "Packet monitoring status retrieved successfully",
  "monitoring": {
    "monitoring": true,
    "tracked_ips": 5,
    "recent_alerts": 2,
    "config": {
      "threshold": 15,
      "time_window": 5,
      "cooldown": 30
    }
  }
}
```

## Security Considerations

### Permissions
- **Packet capture requires root privileges** on most systems
- **Production deployment** should run with minimal required permissions
- **Consider using capabilities** instead of full root access:
  ```bash
  sudo setcap cap_net_raw=eip /path/to/python
  ```

### Network Interface
- **Monitors all available network interfaces** by default
- **Uses BPF filter** `"tcp"` for efficiency
- **Memory usage** kept minimal with `store=False`

## Performance

### Optimizations
- **BPF filtering**: Only TCP packets processed
- **Sliding windows**: Efficient time-based cleanup
- **Memory management**: Automatic cleanup of old data
- **Background threading**: Non-blocking operation

### Resource Usage
- **Memory**: ~1-5MB additional for tracking data
- **CPU**: Minimal impact with BPF filtering
- **Network**: No impact on traffic flow

## Alert Generation

### Alert Details
```python
AlertCreate(
    source_ip="192.168.1.100",        # Scanning source IP
    destination_ip="*",               # Multiple targets
    attack_type="Port Scan",          # Attack classification
    severity=Severity.HIGH,           # High severity
    anomaly_score=0.95               # High confidence
)
```

### Duplicate Prevention
- **Cooldown period**: 30 seconds (configurable)
- **Per-source tracking**: Prevents spam from same IP
- **Automatic cleanup**: Old entries removed automatically

## Dashboard Integration

### Automatic Updates
- **New alerts** appear in dashboard automatically
- **HIGH severity** alerts prominently displayed  
- **Charts update** to reflect new attack types
- **Real-time statistics** include port scan data

### Visual Indicators
- 🔴 **High Severity Badge**: Port scan alerts marked as HIGH
- 📊 **Attack Type Chart**: Shows "Port Scan" category
- ⏰ **Real-time Timestamps**: Shows exact detection time
- 🎯 **Source IP Tracking**: Identifies attacking sources

## Troubleshooting

### Common Issues

#### Permission Denied
```
ERROR: Permission denied: could not open /dev/bpf0
```
**Solution**: Start server with `sudo` or set capabilities

#### Scapy Not Available
```
ERROR: Scapy not available. Install with: pip install scapy
```
**Solution**: 
```bash
pip install scapy==2.5.0
```

#### No Network Interface
```
WARNING: No IPv4 address found on interface
```
**Solution**: Normal on some systems, monitoring still works on available interfaces

### Logs and Debugging

#### Enable Debug Logging
```python
logging.basicConfig(level=logging.DEBUG)
```

#### Key Log Messages
- `PacketMonitor initialized with threshold=15, window=5s, cooldown=30s`
- `Starting real-time packet monitoring...`
- `Port scan detected from X.X.X.X: N SYN packets in 5 seconds`
- `Port scan alert created: <alert_id> for source IP: X.X.X.X`

## Development

### Testing New Detection Rules
1. **Modify threshold** in `.env` file
2. **Restart server** to load new config
3. **Generate test traffic** with nmap
4. **Check alert generation** in logs and dashboard

### Adding New Detection Types
1. **Extend PacketMonitor class** in `packet_monitor.py`
2. **Add new detection methods** following existing pattern
3. **Create appropriate alert types** in models
4. **Update documentation** with new capabilities

## Production Deployment

### Security Best Practices
- **Use service accounts** with minimal required permissions
- **Network segmentation** for monitoring interfaces
- **Log rotation** for packet monitoring logs
- **Rate limiting** for API endpoints

### Performance Monitoring
- **Monitor memory usage** of packet monitoring process
- **Track alert generation rates** to prevent false positives
- **Set up automated restarts** if packet monitoring fails
- **Configure log rotation** for high-volume environments

### High Availability
- **Multiple monitoring nodes** for redundancy  
- **Load balancing** for API requests
- **Database replication** for alert storage
- **Monitoring health checks** via `/health` endpoint

---

## Quick Start Summary

1. **Install Scapy**: `pip install scapy==2.5.0`
2. **Configure thresholds** in `.env` file  
3. **Start with sudo**: `sudo python -m uvicorn main:app --host 0.0.0.0 --port 8000`
4. **Test detection**: `sudo nmap -sS -p 1-100 127.0.0.1`
5. **Check dashboard** for HIGH severity "Port Scan" alerts

Your Campus Network IDS now provides **real-time network intrusion detection** with automatic alert generation! 🚀
