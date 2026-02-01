# Campus Network Intrusion Detection System (IDS)

A comprehensive, production-ready Campus Network Intrusion Detection System built with modern web technologies for university infrastructure security monitoring.

![Campus IDS Dashboard](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)

## 🚀 Features

### Backend (FastAPI)
- **Firebase Authentication**: Secure authentication using Firebase Auth with role-based access control
- **Real-time Packet Monitoring**: Live network packet capture and analysis using Scapy
- **Port Scan Detection**: Automatic detection of TCP SYN-based port scanning attacks
- **Real-time Alert Management**: Create, view, and resolve security alerts with user tracking
- **Anomaly Detection**: Built-in scoring system with automatic severity adjustment
- **Role-Based Security**: Admin and viewer roles with appropriate permissions
- **MongoDB Integration**: Scalable document-based storage with MongoDB Atlas
- **RESTful API**: Comprehensive API with automatic documentation
- **Health Monitoring**: Built-in health checks and monitoring endpoints
- **CORS Support**: Cross-origin resource sharing for frontend integration
- **Production Ready**: Docker containerization and cloud deployment ready

### Frontend (Vanilla JS)
- **Firebase Authentication**: Secure login/logout with email verification
- **Role-Based UI**: Different features available based on user role (Admin/Viewer)
- **Professional Dashboard**: Dark cybersecurity-themed professional interface
- **Real-time Updates**: Auto-refresh every 10 seconds with manual controls
- **Interactive Charts**: Severity distribution and timeline visualization using Chart.js
- **Responsive Design**: Mobile-friendly responsive layout
- **Toast Notifications**: User-friendly notification system
- **Attack Simulation**: Built-in attack simulation for testing (verified users only)
- **Keyboard Shortcuts**: Enhanced user experience with shortcuts

### Security Features
- **Firebase Authentication**: Enterprise-grade authentication with email verification
- **Real-time Network Monitoring**: Live packet capture and TCP SYN flood detection
- **Port Scan Detection**: Automatic detection with configurable thresholds
- **Role-Based Access Control**: Admin and Viewer roles with appropriate permissions
- **Session Management**: Secure token-based authentication with automatic refresh
- **Alert Severity Levels**: LOW, MEDIUM, HIGH, CRITICAL with color coding
- **Anomaly Scoring**: 0.0 to 1.0 scoring with automatic severity escalation
- **Status Tracking**: OPEN/RESOLVED status management with user attribution
- **IP Address Tracking**: Source and destination IP monitoring
- **Attack Type Classification**: Categorized threat detection including real-time port scans

## 🏗️ Architecture

```
Campus Network IDS
├── Backend (FastAPI + MongoDB)
│   ├── Real-time Packet Monitoring (Scapy)
│   ├── API Endpoints
│   ├── Database Models
│   ├── Detection Service
│   └── Health Monitoring
└── Frontend (HTML + CSS + JS)
    ├── Dashboard Interface
    ├── Real-time Updates
    ├── Charts & Visualizations
    └── Alert Management
```

## 📦 Installation & Setup

### Prerequisites

- Python 3.11+
- MongoDB Atlas account
- Node.js (for development tools, optional)

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Campus-Network-Intrusion-Detection-System-for-University-Infrastructure
```

2. **Setup Python environment**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Configuration**
```bash
cp .env.example .env
```

Edit `.env` with your MongoDB Atlas connection string:
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/campus_ids?retryWrites=true&w=majority
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
```

5. **Run the backend**

**Standard Mode** (without packet monitoring):
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Full Security Mode** (with real-time packet monitoring):
```bash
sudo $(which python) -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*Note: Root privileges required for packet capture*

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Monitoring Status: `http://localhost:8000/api/monitoring/status` (auth required)

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Update API Configuration**
Edit `js/app.js` and update the API_BASE_URL:
```javascript
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000', // For local development
    // API_BASE_URL: 'https://your-backend-url.onrender.com', // For production
    REFRESH_INTERVAL: 10000,
};
```

3. **Serve the frontend**

For development, you can use Python's built-in server:
```bash
python -m http.server 3000
```

Or use any static file server like `live-server`:
```bash
npx live-server --port=3000
```

Access the dashboard at `http://localhost:3000`

## 🚀 Deployment

### Backend Deployment

#### Option 1: Deploy to Render

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**: Add `MONGODB_URI`

#### Option 2: Deploy to Railway

1. Connect your GitHub repository to Railway
2. Deploy from the `/backend` directory
3. Add environment variable: `MONGODB_URI`
4. Railway will automatically detect the Dockerfile

#### Option 3: Docker Deployment

```bash
cd backend
docker build -t campus-ids-backend .
docker run -p 8000:8000 -e MONGODB_URI="your-mongodb-uri" campus-ids-backend
```

### Frontend Deployment

#### Deploy to Vercel

1. Connect your GitHub repository to Vercel
2. Set the root directory to `/frontend`
3. Update the API_BASE_URL in `js/app.js` to your deployed backend URL
4. Deploy automatically

#### Deploy to Netlify

1. Connect your GitHub repository to Netlify
2. Set the publish directory to `/frontend`
3. Update the API_BASE_URL in `js/app.js`
4. Deploy

## 📊 API Documentation

### Endpoints

#### Alerts Management
- `GET /api/alerts` - Get all alerts
- `POST /api/alerts` - Create new alert
- `GET /api/alerts/{id}` - Get specific alert
- `PUT /api/alerts/{id}/resolve` - Resolve alert
- `GET /api/alerts/stats/summary` - Get alert statistics

#### System
- `GET /health` - Health check
- `GET /` - API information
- `POST /api/simulate-attack` - Simulate network attack

### Request/Response Examples

#### Create Alert
```bash
curl -X POST "http://localhost:8000/api/alerts" \
-H "Content-Type: application/json" \
-d '{
  "source_ip": "192.168.1.100",
  "destination_ip": "10.0.0.50",
  "attack_type": "SQL Injection",
  "severity": "HIGH"
}'
```

#### Response
```json
{
  "id": "65f1234567890abcdef12345",
  "source_ip": "192.168.1.100",
  "destination_ip": "10.0.0.50",
  "attack_type": "SQL Injection",
  "severity": "HIGH",
  "anomaly_score": 0.856,
  "timestamp": "2024-02-01T10:30:00Z",
  "status": "OPEN"
}
```

## 🛡️ Security Features

### Alert Severity Levels
- **LOW**: Basic security events, low priority
- **MEDIUM**: Moderate security concerns requiring attention
- **HIGH**: Serious security threats needing immediate action
- **CRITICAL**: Critical security breaches requiring emergency response

### Anomaly Detection
- Automatic anomaly score generation (0.0 - 1.0)
- Score > 0.8 automatically escalates to HIGH severity
- Score > 0.9 automatically escalates to CRITICAL severity

### Attack Types Supported
- SQL Injection
- DDoS Attack
- Port Scanning
- Brute Force Login
- Malware Detection
- Unauthorized Access Attempt
- Data Exfiltration
- Cross-Site Scripting (XSS)
- Man-in-the-Middle Attack
- Phishing Attempt

## 🎨 UI Features

### Dashboard Components
- **Summary Cards**: Total, High, Critical, and Open alerts
- **Severity Chart**: Doughnut chart showing alert distribution
- **Timeline Chart**: Line chart showing alerts over 24 hours
- **Alerts Table**: Comprehensive alert listing with actions
- **Real-time Updates**: Auto-refresh with status indicators

### Color Coding
- 🟢 **LOW**: Green
- 🟡 **MEDIUM**: Yellow  
- 🟠 **HIGH**: Orange
- 🔴 **CRITICAL**: Red (with pulse animation)

## 🔧 Configuration

### Environment Variables (Backend)
```env
MONGODB_URI=mongodb+srv://...           # Required: MongoDB connection string
HOST=0.0.0.0                          # Optional: Server host (default: 0.0.0.0)
PORT=8000                             # Optional: Server port (default: 8000)
ENVIRONMENT=development               # Optional: Environment type
SECRET_KEY=your-secret-key           # Optional: For future authentication
```

### Frontend Configuration
```javascript
// js/app.js
const CONFIG = {
    API_BASE_URL: 'https://your-backend-url.com',
    REFRESH_INTERVAL: 10000, // 10 seconds
    CHART_COLORS: {
        LOW: '#26de81',
        MEDIUM: '#ffa726', 
        HIGH: '#ff9800',
        CRITICAL: '#ff4757'
    }
};
```

## 🧪 Testing

### Manual Testing
1. Start the backend server
2. Open the frontend dashboard
3. Click "Simulate Attack" to generate test alerts
4. Verify alerts appear in the table and charts
5. Test resolving alerts
6. Verify auto-refresh functionality

### API Testing
Use the built-in FastAPI docs at `/docs` to test all endpoints interactively.

## 📈 Monitoring & Logging

### Health Checks
- `/health` endpoint provides system status
- Database connectivity verification
- Automatic status indicators in frontend

### Logging
- Structured logging throughout the application
- Error tracking and reporting
- Performance monitoring capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the API documentation at `/docs`
- Review the deployment logs for troubleshooting

## 🚀 Future Enhancements

- [ ] User authentication and authorization
- [ ] Real network integration (SNMP, NetFlow)
- [ ] Machine learning-based anomaly detection
- [ ] Email/SMS alert notifications
- [ ] Advanced filtering and search
- [ ] Historical data analytics
- [ ] Integration with SIEM systems
- [ ] Multi-tenant support for multiple campuses

---

**Built with ❤️ for Campus Network Security**
