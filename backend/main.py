from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from database import connect_to_mongo, close_mongo_connection, check_database_health, get_database
from routers import alerts, auth
from models import HealthResponse
from services.detection_service import DetectionService
from services.packet_monitor import start_background_monitoring, get_packet_monitor
from core.firebase_auth import get_verified_user, UserInfo

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 STARTING UP CAMPUS NETWORK IDS API")
    try:
        logger.info("🔌 CONNECTING TO MONGODB...")
        await connect_to_mongo()
        logger.info("✅ CONNECTED TO MONGODB SUCCESSFULLY")
    except Exception as e:
        logger.error(f"❌ FAILED TO CONNECT TO MONGODB: {e}")
        # You might want to exit here in production
    
    # Start real-time packet monitoring
    try:
        logger.info("🔍 STARTING REAL-TIME PACKET MONITORING...")
        start_background_monitoring()
        logger.info("✅ REAL-TIME PACKET MONITORING STARTED")
    except Exception as e:
        logger.error(f"❌ FAILED TO START PACKET MONITORING: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 SHUTTING DOWN CAMPUS NETWORK IDS API")
    try:
        # Stop packet monitoring
        monitor = get_packet_monitor()
        monitor.stop_monitoring()
        logger.info("✅ PACKET MONITORING STOPPED")
    except Exception as e:
        logger.error(f"❌ ERROR STOPPING PACKET MONITORING: {e}")
    
    await close_mongo_connection()


app = FastAPI(
    title="Campus Network Intrusion Detection System",
    description="A comprehensive IDS for university campus network security monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "https://vercel.app",
        "https://*.vercel.app",
        "*"  # For development only - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(alerts.router)
app.include_router(auth.router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        db_connected = await check_database_health()
        
        return HealthResponse(
            status="healthy" if db_connected else "degraded",
            timestamp=datetime.utcnow(),
            database_connected=db_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@app.get("/api/monitoring/status")
async def get_monitoring_status(current_user: UserInfo = Depends(get_verified_user)):
    """Get real-time packet monitoring status - Requires authentication"""
    try:
        monitor = get_packet_monitor()
        stats = monitor.get_monitoring_stats()
        return {
            "message": "Packet monitoring status retrieved successfully",
            "monitoring": stats
        }
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring status: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Campus Network Intrusion Detection System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/api/simulate-attack")
async def simulate_attack(current_user: UserInfo = Depends(get_verified_user)):
    """Simulate a network attack for testing purposes - Requires authentication"""
    try:
        from routers.alerts import create_alert
        from database import get_database
        
        # Generate simulated attack
        simulated_alert = DetectionService.simulate_network_attack()
        
        # Get database instance
        db = await get_database()
        
        # Create the alert using the router function
        created_alert = await create_alert(simulated_alert, db)
        
        return {
            "message": "Simulated attack created successfully",
            "alert": created_alert
        }
        
    except Exception as e:
        logger.error(f"Failed to simulate attack: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to simulate attack: {str(e)}"
        )


@app.get("/debug/test-alert")
async def create_test_alert():
    """Create a test alert directly in database for debugging"""
    try:
        from bson import ObjectId
        
        logger.info("🧪 DEBUG: Creating test alert...")
        
        # Get database connection
        db = await get_database()
        alerts_collection = db.alerts
        
        # Create test alert document
        test_alert = {
            "_id": ObjectId(),
            "source_ip": "192.168.1.100",
            "destination_ip": "192.168.1.1",
            "attack_type": "Debug Test Alert",
            "severity": "HIGH",
            "anomaly_score": 0.99,
            "timestamp": datetime.utcnow(),
            "status": "OPEN"
        }
        
        logger.info(f"🧪 DEBUG: Inserting test alert: {test_alert}")
        
        # Insert test alert
        result = await alerts_collection.insert_one(test_alert)
        
        logger.info(f"✅ DEBUG: Test alert created with ID: {result.inserted_id}")
        
        # Return the created alert
        return {
            "message": "Test alert created successfully",
            "alert_id": str(result.inserted_id),
            "alert": {
                "id": str(test_alert["_id"]),
                "source_ip": test_alert["source_ip"],
                "destination_ip": test_alert["destination_ip"],
                "attack_type": test_alert["attack_type"],
                "severity": test_alert["severity"],
                "anomaly_score": test_alert["anomaly_score"],
                "timestamp": test_alert["timestamp"],
                "status": test_alert["status"]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ DEBUG: Failed to create test alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create test alert: {str(e)}"
        )


@app.get("/debug/db-status")
async def get_database_status():
    """Get database connection status and alert count"""
    try:
        logger.info("🔍 DEBUG: Checking database status...")
        
        # Get database connection
        db = await get_database()
        alerts_collection = db.alerts
        
        # Count total alerts
        total_alerts = await alerts_collection.count_documents({})
        
        # Count alerts by status
        open_alerts = await alerts_collection.count_documents({"status": "OPEN"})
        resolved_alerts = await alerts_collection.count_documents({"status": "RESOLVED"})
        
        # Get latest alert
        latest_alert_cursor = alerts_collection.find().sort("timestamp", -1).limit(1)
        latest_alert = await latest_alert_cursor.to_list(length=1)
        latest_alert_info = None
        
        if latest_alert:
            alert = latest_alert[0]
            latest_alert_info = {
                "id": str(alert["_id"]),
                "attack_type": alert["attack_type"],
                "severity": alert["severity"],
                "timestamp": alert["timestamp"],
                "source_ip": alert["source_ip"]
            }
        
        result = {
            "database_connected": True,
            "collection_name": "alerts",
            "total_alerts": total_alerts,
            "open_alerts": open_alerts,
            "resolved_alerts": resolved_alerts,
            "latest_alert": latest_alert_info,
            "timestamp": datetime.utcnow()
        }
        
        logger.info(f"✅ DEBUG: Database status: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ DEBUG: Database status check failed: {e}")
        return {
            "database_connected": False,
            "error": str(e),
            "timestamp": datetime.utcnow()
        }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True if os.getenv("ENVIRONMENT") == "development" else False
    )
