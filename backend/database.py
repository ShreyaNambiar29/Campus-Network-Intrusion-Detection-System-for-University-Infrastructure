import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import logging

# Configure detailed logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Database:
    client: AsyncIOMotorClient = None
    database = None


db = Database()


async def get_database():
    logger.debug("📊 DATABASE: get_database() called")
    if db.database is None:
        logger.error("❌ DATABASE: Database not initialized!")
        raise ConnectionError("Database not connected")
    return db.database


async def connect_to_mongo():
    """Create database connection"""
    try:
        logger.info("🔌 DATABASE: Attempting to connect to MongoDB...")
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise ValueError("MONGODB_URI environment variable is not set")
        
        logger.debug(f"📡 DATABASE: Using MongoDB URI: {mongodb_uri[:50]}...")
        
        db.client = AsyncIOMotorClient(mongodb_uri)
        db.database = db.client.campus_ids
        
        logger.info("🔍 DATABASE: Testing connection with ping...")
        # Test the connection
        await db.client.admin.command('ping')
        logger.info("✅ DATABASE: Connected to MongoDB successfully")
        
    except ConnectionFailure as e:
        logger.error(f"❌ DATABASE: Failed to connect to MongoDB: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ DATABASE: Unexpected error connecting to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close database connection"""
    if db.client:
        db.client.close()
        logger.info("Disconnected from MongoDB")


async def check_database_health():
    """Check if database is accessible"""
    try:
        await db.client.admin.command('ping')
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
