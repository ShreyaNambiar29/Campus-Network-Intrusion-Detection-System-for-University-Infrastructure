#!/bin/bash

# Campus Network IDS - Development Startup Script

echo "🚀 Starting Campus Network Intrusion Detection System"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Check Firebase service account file
if [ ! -f "campus-network-ids-firebase-adminsdk-fbsvc-d3b08bfd26.json" ]; then
    echo "⚠️  Firebase service account file not found"
    echo "   Expected: campus-network-ids-firebase-adminsdk-fbsvc-d3b08bfd26.json"
    echo "   Please make sure it's in the project root directory"
fi

# Function to start backend
start_backend() {
    echo "📡 Starting Backend Server..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "🔧 Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "📦 Installing Python dependencies..."
    pip install -q -r requirements.txt
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        echo "⚠️  .env file not found. Creating from example..."
        cp .env.example .env
        echo "⚠️  Please update .env with your MongoDB URI before proceeding"
        echo "⚠️  You can get a free MongoDB Atlas cluster at: https://www.mongodb.com/atlas"
        echo "✅ Firebase configuration is already set up for campus-network-ids project"
        read -p "Press Enter after updating .env file..."
    fi
    
    # Start the backend server
    echo "🚀 Starting FastAPI server on http://localhost:8000"
    echo "📚 API Documentation will be available at http://localhost:8000/docs"
    uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd ..
}

# Function to start frontend
start_frontend() {
    echo "🌐 Starting Frontend Server..."
    cd frontend
    
    # Start a simple HTTP server
    echo "🚀 Starting frontend server on http://localhost:3000"
    python3 -m http.server 3000 &
    FRONTEND_PID=$!
    cd ..
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    echo "✅ Servers stopped"
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup SIGINT

# Start both servers
start_backend
sleep 3
start_frontend

echo ""
echo "✅ Campus Network IDS is now running!"
echo "=================================================="
echo "🌐 Frontend Dashboard: http://localhost:3000"
echo "📡 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo "=================================================="
echo ""
echo "💡 Tips:"
echo "   - Make sure your MongoDB Atlas URI is configured in backend/.env"
echo "   - Update the API_BASE_URL in frontend/js/app.js if needed"
echo "   - Press Ctrl+C to stop both servers"
echo ""

# Wait for user input to keep script running
read -p "Press Enter to stop the servers..."
cleanup
