#!/bin/bash

# Campus Network IDS - Real-time Packet Monitoring Test Script
# This script demonstrates how to test the port scan detection functionality

echo "🔐 Campus Network IDS - Packet Monitoring Test"
echo "=============================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  WARNING: Packet monitoring requires root privileges."
    echo "   For full functionality, restart the server with:"
    echo ""
    echo "   sudo $(pwd)/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000"
    echo ""
    echo "   Or start with sudo and then test with:"
    echo "   sudo nmap -sS 127.0.0.1"
    echo ""
else
    echo "✅ Running with root privileges - packet monitoring will work!"
    echo ""
fi

echo "📊 Current Configuration:"
echo "   PORT_SCAN_THRESHOLD: ${PORT_SCAN_THRESHOLD:-15} SYN packets"
echo "   PORT_SCAN_TIME_WINDOW: ${PORT_SCAN_TIME_WINDOW:-5} seconds"
echo "   PORT_SCAN_COOLDOWN: ${PORT_SCAN_COOLDOWN:-30} seconds"
echo ""

echo "🎯 To test port scan detection:"
echo "   1. Start server with sudo privileges"
echo "   2. Run: sudo nmap -sS -p 1-100 127.0.0.1"
echo "   3. Check dashboard for new HIGH severity alerts"
echo ""

echo "🔍 Monitor logs for detection messages like:"
echo "   'Port scan detected from X.X.X.X: N SYN packets in 5 seconds'"
echo ""

echo "📈 Check monitoring status at:"
echo "   GET http://localhost:8000/api/monitoring/status"
echo "   (Requires authentication)"
echo ""

echo "✨ Dashboard will automatically update with detected alerts!"
