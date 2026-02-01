from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from datetime import datetime
from bson import ObjectId
from pymongo.errors import DuplicateKeyError
import logging

from database import get_database
from models import Alert, AlertCreate, AlertResponse, AlertUpdate, AlertsStats, Status, Severity
from services.detection_service import DetectionService
from core.firebase_auth import get_current_user, get_admin_user, get_verified_user, UserInfo

# Configure detailed logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=List[AlertResponse])
async def get_all_alerts(
    current_user: UserInfo = Depends(get_verified_user),
    db=Depends(get_database)
):
    """Get all alerts sorted by timestamp (latest first) - Requires authentication"""
    logger.info(f"🔍 GET /api/alerts called by user: {current_user.email}")
    
    try:
        logger.debug("📊 Querying MongoDB for alerts...")
        alerts_cursor = db.alerts.find().sort("timestamp", -1)
        alerts = []
        
        alert_count = 0
        async for alert in alerts_cursor:
            alert_count += 1
            logger.debug(f"📄 Processing alert {alert_count}: {alert['_id']}")
            
            alert_response = AlertResponse(
                id=str(alert["_id"]),
                source_ip=alert["source_ip"],
                destination_ip=alert["destination_ip"],
                attack_type=alert["attack_type"],
                severity=alert["severity"],
                anomaly_score=alert.get("anomaly_score"),
                timestamp=alert["timestamp"],
                status=alert["status"]
            )
            alerts.append(alert_response)
        
        logger.info(f"✅ Successfully fetched {len(alerts)} alerts for {current_user.email}")
        return alerts
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch alerts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch alerts: {str(e)}"
        )


@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    alert_data: AlertCreate, 
    current_user: UserInfo = Depends(get_verified_user),
    db=Depends(get_database)
):
    """Create a new alert - Requires authentication"""
    try:
        # Process alert through detection service
        processed_alert = DetectionService.process_alert(alert_data)
        
        # Create alert document with user information
        alert_dict = {
            "source_ip": processed_alert.source_ip,
            "destination_ip": processed_alert.destination_ip,
            "attack_type": processed_alert.attack_type,
            "severity": processed_alert.severity.value,
            "anomaly_score": processed_alert.anomaly_score,
            "timestamp": datetime.utcnow(),
            "status": Status.OPEN.value,
            "created_by": current_user.uid,
            "created_by_email": current_user.email
        }
        
        result = await db.alerts.insert_one(alert_dict)
        
        # Fetch the created alert
        created_alert = await db.alerts.find_one({"_id": result.inserted_id})
        
        return AlertResponse(
            id=str(created_alert["_id"]),
            source_ip=created_alert["source_ip"],
            destination_ip=created_alert["destination_ip"],
            attack_type=created_alert["attack_type"],
            severity=created_alert["severity"],
            anomaly_score=created_alert.get("anomaly_score"),
            timestamp=created_alert["timestamp"],
            status=created_alert["status"]
        )
        
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Alert already exists"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create alert: {str(e)}"
        )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: str, 
    current_user: UserInfo = Depends(get_verified_user),
    db=Depends(get_database)
):
    """Get a specific alert by ID - Requires authentication"""
    try:
        if not ObjectId.is_valid(alert_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid alert ID format"
            )
        
        alert = await db.alerts.find_one({"_id": ObjectId(alert_id)})
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        return AlertResponse(
            id=str(alert["_id"]),
            source_ip=alert["source_ip"],
            destination_ip=alert["destination_ip"],
            attack_type=alert["attack_type"],
            severity=alert["severity"],
            anomaly_score=alert.get("anomaly_score"),
            timestamp=alert["timestamp"],
            status=alert["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch alert: {str(e)}"
        )


@router.put("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: str, 
    current_user: UserInfo = Depends(get_admin_user),
    db=Depends(get_database)
):
    """Mark an alert as resolved - Requires admin role"""
    try:
        if not ObjectId.is_valid(alert_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid alert ID format"
            )
        
        result = await db.alerts.update_one(
            {"_id": ObjectId(alert_id)},
            {"$set": {
                "status": Status.RESOLVED.value,
                "resolved_by": current_user.uid,
                "resolved_by_email": current_user.email,
                "resolved_at": datetime.utcnow()
            }}
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        # Fetch updated alert
        updated_alert = await db.alerts.find_one({"_id": ObjectId(alert_id)})
        
        return AlertResponse(
            id=str(updated_alert["_id"]),
            source_ip=updated_alert["source_ip"],
            destination_ip=updated_alert["destination_ip"],
            attack_type=updated_alert["attack_type"],
            severity=updated_alert["severity"],
            anomaly_score=updated_alert.get("anomaly_score"),
            timestamp=updated_alert["timestamp"],
            status=updated_alert["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resolve alert: {str(e)}"
        )


@router.get("/stats/summary", response_model=AlertsStats)
async def get_alerts_stats(
    current_user: UserInfo = Depends(get_verified_user),
    db=Depends(get_database)
):
    """Get alerts statistics - Requires authentication"""
    try:
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_alerts": {"$sum": 1},
                    "high_severity_alerts": {
                        "$sum": {"$cond": [{"$eq": ["$severity", "HIGH"]}, 1, 0]}
                    },
                    "critical_alerts": {
                        "$sum": {"$cond": [{"$eq": ["$severity", "CRITICAL"]}, 1, 0]}
                    },
                    "open_alerts": {
                        "$sum": {"$cond": [{"$eq": ["$status", "OPEN"]}, 1, 0]}
                    }
                }
            }
        ]
        
        result = await db.alerts.aggregate(pipeline).to_list(1)
        
        if not result:
            return AlertsStats(
                total_alerts=0,
                high_severity_alerts=0,
                critical_alerts=0,
                open_alerts=0
            )
        
        stats = result[0]
        return AlertsStats(
            total_alerts=stats.get("total_alerts", 0),
            high_severity_alerts=stats.get("high_severity_alerts", 0),
            critical_alerts=stats.get("critical_alerts", 0),
            open_alerts=stats.get("open_alerts", 0)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch alert statistics: {str(e)}"
        )
