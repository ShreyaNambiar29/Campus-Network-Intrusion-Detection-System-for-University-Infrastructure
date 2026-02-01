from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from datetime import datetime
from typing import Optional
from bson import ObjectId


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Status(str, Enum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"


class AlertBase(BaseModel):
    source_ip: str = Field(..., description="Source IP address")
    destination_ip: str = Field(..., description="Destination IP address")
    attack_type: str = Field(..., description="Type of attack detected")
    severity: Severity = Field(..., description="Severity level")
    anomaly_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Anomaly detection score")


class AlertCreate(AlertBase):
    pass


class AlertUpdate(BaseModel):
    status: Status


class Alert(AlertBase):
    id: Optional[str] = Field(default=None, alias="_id")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: Status = Field(default=Status.OPEN)

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={ObjectId: str}
    )


class AlertResponse(BaseModel):
    id: str
    source_ip: str
    destination_ip: str
    attack_type: str
    severity: Severity
    anomaly_score: Optional[float]
    timestamp: datetime
    status: Status

    model_config = ConfigDict(from_attributes=True)


class AlertsStats(BaseModel):
    total_alerts: int
    high_severity_alerts: int
    critical_alerts: int
    open_alerts: int


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database_connected: bool
