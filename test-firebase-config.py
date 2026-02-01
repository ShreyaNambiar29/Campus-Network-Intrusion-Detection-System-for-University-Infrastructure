#!/usr/bin/env python3
"""
Firebase Configuration Test Script
Tests Firebase Admin SDK setup for Campus Network IDS
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add backend to path
sys.path.append('backend')

try:
    from core.firebase_auth import firebase_auth
    print("✅ Firebase auth module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Firebase auth module: {e}")
    print("   Make sure you're in the project root and dependencies are installed")
    sys.exit(1)

def test_firebase_config():
    """Test Firebase configuration"""
    print("\n🔥 Testing Firebase Admin SDK Configuration")
    print("=" * 50)
    
    # Check if service account file exists
    service_account_path = "campus-network-ids-firebase-adminsdk-fbsvc-d3b08bfd26.json"
    if os.path.exists(service_account_path):
        print("✅ Firebase service account file found")
    else:
        print("❌ Firebase service account file not found")
        return False
    
    # Check environment variables
    env_file = Path("backend/.env")
    if env_file.exists():
        print("✅ Backend .env file found")
        
        # Read and check key variables
        with open(env_file) as f:
            env_content = f.read()
            if "FIREBASE_SERVICE_ACCOUNT_PATH" in env_content:
                print("✅ FIREBASE_SERVICE_ACCOUNT_PATH configured")
            if "FIREBASE_ADMIN_EMAILS" in env_content:
                print("✅ FIREBASE_ADMIN_EMAILS configured")
    else:
        print("❌ Backend .env file not found")
        return False
    
    # Test Firebase initialization
    try:
        if firebase_auth.app:
            print("✅ Firebase Admin SDK initialized successfully")
            print(f"   Project ID: {firebase_auth.app.project_id}")
            return True
    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        return False
    
    return True

def test_frontend_config():
    """Test frontend Firebase configuration"""
    print("\n🌐 Testing Frontend Firebase Configuration")
    print("=" * 50)
    
    config_file = Path("frontend/js/firebase-config.js")
    if config_file.exists():
        print("✅ Frontend Firebase config file found")
        
        with open(config_file) as f:
            content = f.read()
            if "campus-network-ids" in content:
                print("✅ Project ID matches in frontend config")
            if "firebaseConfig" in content:
                print("✅ Firebase config object found")
        return True
    else:
        print("❌ Frontend Firebase config file not found")
        return False

def main():
    print("🔐 Campus Network IDS - Firebase Configuration Test")
    print("=" * 55)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test backend configuration
    backend_ok = test_firebase_config()
    
    # Test frontend configuration  
    frontend_ok = test_frontend_config()
    
    print("\n📋 Test Results Summary")
    print("=" * 30)
    
    if backend_ok and frontend_ok:
        print("✅ All Firebase configurations are ready!")
        print("\n🚀 You can now start the development servers:")
        print("   ./start-dev.sh")
        print("\n🔧 Don't forget to:")
        print("   1. Update MONGODB_URI in backend/.env")
        print("   2. Enable Email/Password auth in Firebase Console")
        print("   3. Add your domain to Firebase authorized domains")
        return 0
    else:
        print("❌ Some configurations need attention")
        print("   Please check the errors above and fix them")
        return 1

if __name__ == "__main__":
    sys.exit(main())
