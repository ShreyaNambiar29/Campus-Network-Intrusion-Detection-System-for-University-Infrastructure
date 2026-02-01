import random
from models import AlertCreate, Severity
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DetectionService:
    
    @staticmethod
    def generate_anomaly_score() -> float:
        """Generate a random anomaly score between 0 and 1"""
        return round(random.uniform(0.0, 1.0), 3)
    
    @staticmethod
    def determine_severity_from_score(anomaly_score: Optional[float]) -> Severity:
        """Determine severity based on anomaly score"""
        if anomaly_score is None:
            return Severity.LOW
        
        if anomaly_score > 0.9:
            return Severity.CRITICAL
        elif anomaly_score > 0.8:
            return Severity.HIGH
        elif anomaly_score > 0.6:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    @staticmethod
    def process_alert(alert_data: AlertCreate) -> AlertCreate:
        """Process alert data and enhance with detection logic"""
        # Generate anomaly score if not provided
        if alert_data.anomaly_score is None:
            alert_data.anomaly_score = DetectionService.generate_anomaly_score()
        
        # Auto-adjust severity based on anomaly score if score is high
        calculated_severity = DetectionService.determine_severity_from_score(alert_data.anomaly_score)
        
        # Use the higher severity between provided and calculated
        severity_levels = {
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4
        }
        
        current_severity_level = severity_levels.get(alert_data.severity, 1)
        calculated_severity_level = severity_levels.get(calculated_severity, 1)
        
        if calculated_severity_level > current_severity_level:
            alert_data.severity = calculated_severity
            logger.info(f"Severity upgraded to {calculated_severity} based on anomaly score {alert_data.anomaly_score}")
        
        return alert_data
    
    @staticmethod
    def simulate_network_attack() -> AlertCreate:
        """Simulate a network attack for testing purposes"""
        attack_types = [
            "SQL Injection",
            "DDoS Attack",
            "Port Scanning",
            "Brute Force Login",
            "Malware Detection",
            "Unauthorized Access Attempt",
            "Data Exfiltration",
            "Cross-Site Scripting (XSS)",
            "Man-in-the-Middle Attack",
            "Phishing Attempt"
        ]
        
        # Generate random IPs
        source_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        dest_ip = f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}"  # Campus network
        
        attack_type = random.choice(attack_types)
        severity = random.choice(list(Severity))
        
        alert = AlertCreate(
            source_ip=source_ip,
            destination_ip=dest_ip,
            attack_type=attack_type,
            severity=severity
        )
        
        return DetectionService.process_alert(alert)
