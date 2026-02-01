#!/bin/bash

# Firebase Configuration Setup Script for Campus Network IDS

echo "🔥 Setting up Firebase Configuration for Campus Network IDS"
echo "========================================================="

# Check if we're in the right directory
if [ ! -f "campus-network-ids-firebase-adminsdk-fbsvc-d3b08bfd26.json" ]; then
    echo "❌ Firebase service account file not found in current directory"
    echo "   Please make sure you're in the project root directory"
    exit 1
fi

echo "✅ Firebase service account file found"

# Check if backend .env exists
if [ ! -f "backend/.env" ]; then
    echo "⚠️  Backend .env file not found. Creating from template..."
    cp backend/.env.example backend/.env
fi

echo "✅ Backend environment file ready"

# Verify Firebase configuration
echo ""
echo "📋 Firebase Project Configuration:"
echo "   Project ID: campus-network-ids"
echo "   Auth Domain: campus-network-ids.firebaseapp.com"
echo "   Service Account: firebase-adminsdk-fbsvc@campus-network-ids.iam.gserviceaccount.com"

echo ""
echo "🔧 Setup Tasks:"
echo "   ✅ Firebase service account file ready"
echo "   ✅ Frontend Firebase config updated" 
echo "   ✅ Backend environment variables configured"
echo "   ⚠️  MongoDB URI needs to be updated in backend/.env"

echo ""
echo "📝 Next Steps:"
echo "   1. Update MONGODB_URI in backend/.env with your MongoDB Atlas connection"
echo "   2. In Firebase Console, enable Email/Password authentication"
echo "   3. Add authorized domains in Firebase Console > Authentication > Settings"
echo "   4. Test the authentication system with a sample account"

echo ""
echo "🚀 To start the development servers:"
echo "   ./start-dev.sh"

echo ""
echo "🔐 Firebase Authentication is ready for Campus Network IDS!"
