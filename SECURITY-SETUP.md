# Security Setup Guide

## 🔒 Protected Files

The following files contain sensitive credentials and are protected by `.gitignore`:

### Backend Environment File
- `backend/.env` - Contains actual MongoDB URI, Firebase credentials, and other secrets
- `backend/.env.example` - Template with placeholder values (safe to commit)

### Firebase Service Account Key
- `campus-network-ids-firebase-adminsdk-*.json` - Actual Firebase credentials (NEVER commit)
- `campus-network-ids-firebase-adminsdk-TEMPLATE.json` - Template file (safe to commit)

## 🚀 Setup Instructions

### 1. Backend Environment Setup
```bash
# Copy the template
cp backend/.env.example backend/.env

# Edit backend/.env and replace placeholder values with actual credentials:
# - MONGODB_URI: Your actual MongoDB connection string
# - FIREBASE_SERVICE_ACCOUNT: Your actual Firebase service account JSON
# - SECRET_KEY: Generate a secure secret key
```

### 2. Firebase Service Account Setup

**Option A: Using JSON String (Recommended)**
1. Copy your Firebase service account JSON content
2. Paste it as a single line in `backend/.env` under `FIREBASE_SERVICE_ACCOUNT`

**Option B: Using File Path**
1. Copy the template: `cp campus-network-ids-firebase-adminsdk-TEMPLATE.json campus-network-ids-firebase-adminsdk-YOUR-KEY-ID.json`
2. Replace the content with your actual Firebase credentials
3. Update `FIREBASE_SERVICE_ACCOUNT_PATH` in `backend/.env`

### 3. Packet Monitoring Configuration
Adjust these values in `backend/.env` as needed:
- `PORT_SCAN_THRESHOLD`: Number of SYN packets to trigger alert (default: 15)
- `PORT_SCAN_TIME_WINDOW`: Time window in seconds (default: 5)
- `PORT_SCAN_COOLDOWN`: Cooldown between alerts in seconds (default: 30)

## ⚠️ Security Notes

1. **Never commit sensitive files to Git**
2. **Keep actual credentials in local `.env` files only**
3. **Use environment variables in production**
4. **Rotate keys regularly**
5. **Use different credentials for development and production**

## 🔍 Verification

Check that sensitive files are ignored:
```bash
git check-ignore backend/.env
git check-ignore campus-network-ids-firebase-adminsdk-*.json
```

Both commands should return the file paths, confirming they are ignored.
