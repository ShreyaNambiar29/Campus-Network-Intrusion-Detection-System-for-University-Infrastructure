from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from database import connect_to_mongo, close_mongo_connection, check_database_health
from routers import alerts, auth
from models import HealthResponse
from services.detection_service import DetectionService
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
    logger.info("Starting up Campus Network IDS API")
    try:
        await connect_to_mongo()
        logger.info("Connected to MongoDB successfully")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        # You might want to exit here in production
    
    yield
    
    # Shutdown
    logger.info("Shutting down Campus Network IDS API")
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
