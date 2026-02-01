#!/usr/bin/env python3
"""
Simple Firebase Configuration Test Script
Tests Firebase Admin SDK setup without FastAPI dependencies
"""

import os
import sys
import json
import logging
from pathlib import Path

def test_service_account_file():
    """Test if service account file exists and is valid"""
    print("\n🔐 Testing Service Account File")
    print("=" * 40)
    
    service_account_path = "campus-network-ids-firebase-adminsdk-fbsvc-d3b08bfd26.json"
    if os.path.exists(service_account_path):
        print("✅ Firebase service account file found")
        
        try:
            with open(service_account_path) as f:
                config = json.load(f)
                
            required_fields = [
                'type', 'project_id', 'private_key_id', 'private_key',
                'client_email', 'client_id', 'auth_uri', 'token_uri'
            ]
            
            for field in required_fields:
                if field in config:
                    print(f"✅ {field}: Present")
                else:
                    print(f"❌ {field}: Missing")
                    return False
            
            print(f"✅ Project ID: {config.get('project_id')}")
            print(f"✅ Client Email: {config.get('client_email')}")
            return True
            
        except json.JSONDecodeError:
            print("❌ Service account file is not valid JSON")
            return False
        except Exception as e:
            print(f"❌ Error reading service account file: {e}")
            return False
    else:
        print("❌ Firebase service account file not found")
        return False

def test_env_configuration():
    """Test environment configuration"""
    print("\n⚙️ Testing Environment Configuration")
    print("=" * 40)
    
    env_file = Path("backend/.env")
    if env_file.exists():
        print("✅ Backend .env file found")
        
        try:
            with open(env_file) as f:
                env_content = f.read()
            
            checks = [
                ('FIREBASE_SERVICE_ACCOUNT_PATH', 'Service account path'),
                ('FIREBASE_ADMIN_EMAILS', 'Admin emails'),
                ('FIREBASE_ADMIN_DOMAINS', 'Admin domains')
            ]
            
            all_present = True
            for var_name, description in checks:
                if var_name in env_content:
                    print(f"✅ {description}: Configured")
                else:
                    print(f"❌ {description}: Not configured")
                    all_present = False
            
            return all_present
            
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
            return False
    else:
        print("❌ Backend .env file not found")
        return False

def test_firebase_admin_import():
    """Test if Firebase Admin SDK can be imported"""
    print("\n🔥 Testing Firebase Admin SDK")
    print("=" * 40)
    
    try:
        import firebase_admin
        from firebase_admin import credentials, auth
        print("✅ Firebase Admin SDK imported successfully")
        print(f"   Version: {firebase_admin.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Firebase Admin SDK: {e}")
        return False

def test_firebase_initialization():
    """Test Firebase Admin SDK initialization"""
    print("\n🚀 Testing Firebase Initialization")
    print("=" * 40)
    
    try:
        import firebase_admin
        from firebase_admin import credentials
        
        # Check if already initialized
        if firebase_admin._apps:
            print("✅ Firebase Admin SDK already initialized")
            app = firebase_admin.get_app()
            print(f"   Project ID: {app.project_id}")
            return True
        
        # Try to initialize
        service_account_path = "campus-network-ids-firebase-adminsdk-fbsvc-d3b08bfd26.json"
        if os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            app = firebase_admin.initialize_app(cred)
            print("✅ Firebase Admin SDK initialized successfully")
            print(f"   Project ID: {app.project_id}")
            return True
        else:
            print("❌ Service account file not found for initialization")
            return False
            
    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        return False

def test_frontend_config():
    """Test frontend Firebase configuration"""
    print("\n🌐 Testing Frontend Configuration")
    print("=" * 40)
    
    config_file = Path("frontend/js/firebase-config.js")
    if config_file.exists():
        print("✅ Frontend Firebase config file found")
        
        try:
            with open(config_file) as f:
                content = f.read()
            
            checks = [
                ('campus-network-ids', 'Project ID'),
                ('firebaseConfig', 'Config object'),
                ('firebase.initializeApp', 'Initialization'),
                ('firebase.auth()', 'Auth service')
            ]
            
            all_present = True
            for check, description in checks:
                if check in content:
                    print(f"✅ {description}: Found")
                else:
                    print(f"❌ {description}: Missing")
                    all_present = False
            
            return all_present
            
        except Exception as e:
            print(f"❌ Error reading frontend config: {e}")
            return False
    else:
        print("❌ Frontend Firebase config file not found")
        return False

def main():
    print("🔐 Campus Network IDS - Firebase Configuration Test")
    print("=" * 55)
    
    # Run all tests
    tests = [
        ("Service Account File", test_service_account_file),
        ("Environment Config", test_env_configuration),
        ("Firebase SDK Import", test_firebase_admin_import),
        ("Firebase Initialization", test_firebase_initialization),
        ("Frontend Config", test_frontend_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))
    
    print("\n📋 Test Results Summary")
    print("=" * 30)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 30)
    if all_passed:
        print("🎉 All Firebase configuration tests PASSED!")
        print("   Your Firebase setup is ready to use.")
    else:
        print("⚠️ Some tests FAILED!")
        print("   Please check the errors above and fix the configuration.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
