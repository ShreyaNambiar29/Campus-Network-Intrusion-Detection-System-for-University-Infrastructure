"""
Real-time Network Packet Monitor for Campus Network IDS
Detects TCP SYN-based port scanning using Scapy
"""

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Set

try:
    from scapy.all import sniff, TCP, IP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

from models import AlertCreate, Severity, Status
from database import get_database
from bson import ObjectId

# Configure detailed logging for debugging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add console handler if not present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class PacketMonitor:
    """
    Real-time packet monitoring system for detecting network intrusions
    """
    
    def __init__(self):
        # Configuration from environment variables
        self.port_scan_threshold = int(os.getenv('PORT_SCAN_THRESHOLD', '15'))
        self.time_window = int(os.getenv('PORT_SCAN_TIME_WINDOW', '5'))
        self.cooldown_period = int(os.getenv('PORT_SCAN_COOLDOWN', '30'))
        
        # Tracking data structures
        self.syn_packets: Dict[str, deque] = defaultdict(deque)
        self.recent_alerts: Dict[str, datetime] = {}
        self.monitoring = False
        
        logger.info(f"PacketMonitor initialized with threshold={self.port_scan_threshold}, "
                   f"window={self.time_window}s, cooldown={self.cooldown_period}s")

    def start_monitoring(self):
        """Start packet monitoring in a background thread"""
        if not SCAPY_AVAILABLE:
            logger.error("🚫 SCAPY NOT AVAILABLE: Install with: pip install scapy")
            return
            
        if self.monitoring:
            logger.warning("⚠️ PACKET MONITORING ALREADY RUNNING")
            return
            
        self.monitoring = True
        logger.info("🚀 STARTING REAL-TIME PACKET MONITORING...")
        logger.info(f"📊 MONITORING CONFIG: threshold={self.port_scan_threshold}, window={self.time_window}s, cooldown={self.cooldown_period}s")
        
        try:
            logger.info("🔍 STARTING SCAPY PACKET SNIFFING with filter='tcp'...")
            # Start packet sniffing with BPF filter for TCP packets only
            sniff(
                filter="tcp",
                prn=self._process_packet,
                store=False,  # Don't store packets in memory
                stop_filter=lambda x: not self.monitoring
            )
            logger.info("✅ PACKET SNIFFING STARTED SUCCESSFULLY")
        except PermissionError:
            logger.error(
                "🚫 PERMISSION DENIED: Packet monitoring requires root privileges. "
                "Run with: sudo python -m uvicorn main:app --host 0.0.0.0 --port 8000"
            )
        except Exception as e:
            logger.error(f"❌ PACKET MONITORING ERROR: {e}")
        finally:
            self.monitoring = False
            logger.info("🛑 PACKET MONITORING STOPPED")

    def stop_monitoring(self):
        """Stop packet monitoring"""
        self.monitoring = False
        logger.info("Stopping packet monitoring...")

    def _process_packet(self, packet):
        """Process captured TCP packets"""
        try:
            if not packet.haslayer(TCP) or not packet.haslayer(IP):
                return
                
            tcp_layer = packet[TCP]
            ip_layer = packet[IP]
            
            # Check for SYN packets (flags == 2 or "S")
            if tcp_layer.flags == 2:  # SYN flag
                src_ip = ip_layer.src
                dst_port = tcp_layer.dport
                current_time = datetime.utcnow()
                
                logger.debug(f"📦 SYN PACKET DETECTED: {src_ip} -> port {dst_port}")
                
                # Track SYN packets from this source IP
                self._track_syn_packet(src_ip, dst_port, current_time)
                
        except Exception as e:
            logger.error(f"❌ ERROR PROCESSING PACKET: {e}")

    def _track_syn_packet(self, src_ip: str, dst_port: int, timestamp: datetime):
        """Track SYN packets and detect port scanning"""
        # Add timestamp to the deque for this source IP
        self.syn_packets[src_ip].append(timestamp)
        
        # Clean up old timestamps outside the time window
        cutoff_time = timestamp - timedelta(seconds=self.time_window)
        while (self.syn_packets[src_ip] and 
               self.syn_packets[src_ip][0] < cutoff_time):
            self.syn_packets[src_ip].popleft()
        
        # Check if threshold is exceeded
        packet_count = len(self.syn_packets[src_ip])
        
        logger.debug(f"📈 SYN TRACKING: {src_ip} has {packet_count} SYN packets in {self.time_window}s window")
        
        if packet_count >= self.port_scan_threshold:
            logger.warning(f"🚨 THRESHOLD EXCEEDED: {src_ip} sent {packet_count} SYN packets (threshold: {self.port_scan_threshold})")
            self._handle_port_scan_detection(src_ip, packet_count, timestamp)
        else:
            logger.debug(f"📊 Below threshold: {packet_count}/{self.port_scan_threshold} for {src_ip}")

    def _handle_port_scan_detection(self, src_ip: str, packet_count: int, timestamp: datetime):
        """Handle port scan detection and create alert"""
        # Check cooldown period to prevent duplicate alerts
        if src_ip in self.recent_alerts:
            time_since_last_alert = timestamp - self.recent_alerts[src_ip]
            if time_since_last_alert.total_seconds() < self.cooldown_period:
                logger.info(f"🔄 COOLDOWN ACTIVE: Skipping duplicate alert for {src_ip} (last alert {time_since_last_alert.total_seconds():.1f}s ago)")
                return  # Skip duplicate alert
        
        # Update recent alerts tracker
        self.recent_alerts[src_ip] = timestamp
        
        logger.critical(f"🚨 PORT SCAN DETECTED: {src_ip} sent {packet_count} SYN packets in {self.time_window} seconds")
        logger.info(f"🎯 CREATING ALERT for source IP: {src_ip}")
        
        # Create port scan alert
        asyncio.create_task(self._create_port_scan_alert(src_ip, packet_count))
        
        logger.warning(f"⚠️ Port scan detected from {src_ip}: {packet_count} SYN packets "
                      f"in {self.time_window} seconds")

    async def _create_port_scan_alert(self, src_ip: str, packet_count: int):
        """Create alert in database using existing alert system"""
        try:
            logger.info(f"🔧 CREATING AlertCreate object for {src_ip}")
            
            # Create alert data
            alert_data = AlertCreate(
                source_ip=src_ip,
                destination_ip="*",  # Multiple targets in port scan
                attack_type="Port Scan",
                severity=Severity.HIGH,
                anomaly_score=0.95
            )
            
            logger.info(f"✅ ALERT OBJECT CREATED: {alert_data.dict()}")
            
            # Get database connection
            logger.info("🗄️ GETTING DATABASE CONNECTION...")
            db = await get_database()
            alerts_collection = db.alerts
            logger.info("✅ DATABASE CONNECTION OBTAINED")
            
            # Prepare alert document
            alert_doc = {
                "_id": ObjectId(),
                "source_ip": alert_data.source_ip,
                "destination_ip": alert_data.destination_ip,
                "attack_type": alert_data.attack_type,
                "severity": alert_data.severity.value,
                "anomaly_score": alert_data.anomaly_score,
                "timestamp": datetime.utcnow(),
                "status": Status.OPEN.value
            }
            
            logger.info(f"📄 PREPARED ALERT DOCUMENT: {alert_doc}")
            
            # Insert alert into database
            logger.info("💾 INSERTING ALERT INTO MONGODB...")
            result = await alerts_collection.insert_one(alert_doc)
            
            logger.critical(f"✅ ALERT SUCCESSFULLY INSERTED: ID={result.inserted_id} for source IP: {src_ip}")
            logger.info(f"🎉 PORT SCAN ALERT CREATION COMPLETED: {result.inserted_id}")
            
        except Exception as e:
            logger.error(f"❌ FAILED TO CREATE PORT SCAN ALERT: {e}")
            logger.error(f"❌ EXCEPTION TYPE: {type(e).__name__}")
            import traceback
            logger.error(f"❌ TRACEBACK: {traceback.format_exc()}")

    def cleanup_old_data(self):
        """Clean up old tracking data to prevent memory leaks"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(seconds=self.time_window * 2)
        
        # Clean up SYN packet tracking
        for src_ip in list(self.syn_packets.keys()):
            while (self.syn_packets[src_ip] and 
                   self.syn_packets[src_ip][0] < cutoff_time):
                self.syn_packets[src_ip].popleft()
            
            # Remove empty entries
            if not self.syn_packets[src_ip]:
                del self.syn_packets[src_ip]
        
        # Clean up recent alerts (older than cooldown period)
        alert_cutoff = current_time - timedelta(seconds=self.cooldown_period * 2)
        self.recent_alerts = {
            ip: alert_time for ip, alert_time in self.recent_alerts.items()
            if alert_time > alert_cutoff
        }

    def get_monitoring_stats(self) -> Dict:
        """Get current monitoring statistics"""
        return {
            "monitoring": self.monitoring,
            "tracked_ips": len(self.syn_packets),
            "recent_alerts": len(self.recent_alerts),
            "config": {
                "threshold": self.port_scan_threshold,
                "time_window": self.time_window,
                "cooldown": self.cooldown_period
            }
        }


# Global packet monitor instance
packet_monitor = PacketMonitor()


def start_packet_monitor():
    """Start packet monitoring in a separate thread"""
    logger.info("🚀 STARTING PACKET MONITORING THREAD...")
    packet_monitor.start_monitoring()


def start_background_monitoring():
    """Start packet monitoring as background daemon thread"""
    if not SCAPY_AVAILABLE:
        logger.warning("⚠️ SCAPY NOT AVAILABLE: Packet monitoring disabled.")
        return
        
    logger.info("🎬 INITIALIZING BACKGROUND PACKET MONITORING...")
    monitoring_thread = threading.Thread(
        target=start_packet_monitor,
        daemon=True,
        name="PacketMonitor"
    )
    monitoring_thread.start()
    logger.info("✅ PACKET MONITORING BACKGROUND THREAD STARTED")
    
    # Start periodic cleanup task
    cleanup_thread = threading.Thread(
        target=_periodic_cleanup,
        daemon=True,
        name="PacketMonitorCleanup"
    )
    cleanup_thread.start()
    logger.info("✅ PACKET MONITOR CLEANUP THREAD STARTED")


def _periodic_cleanup():
    """Periodic cleanup of old tracking data"""
    while True:
        try:
            time.sleep(60)  # Run cleanup every minute
            packet_monitor.cleanup_old_data()
            logger.debug("Packet monitor cleanup completed")
        except Exception as e:
            logger.error(f"Error in packet monitor cleanup: {e}")


def get_packet_monitor():
    """Get the global packet monitor instance"""
    return packet_monitor
