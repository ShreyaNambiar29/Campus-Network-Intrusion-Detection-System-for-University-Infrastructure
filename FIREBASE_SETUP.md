# Firebase Setup Guide for Campus Network IDS

## Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project" or "Add project"
3. Enter project name: `campus-network-ids`
4. Enable/disable Google Analytics as needed
5. Click "Create project"

## Step 2: Enable Authentication

1. In Firebase Console, go to "Authentication"
2. Click "Get started"
3. Go to "Sign-in method" tab
4. Enable "Email/Password" provider
5. Optionally enable "Email link (passwordless sign-in)"

## Step 3: Get Web App Configuration

1. In Firebase Console, click on Project settings (gear icon)
2. Scroll down to "Your apps" section
3. Click "Web" icon to add web app
4. Enter app nickname: "Campus IDS Dashboard"
5. Check "Also set up Firebase Hosting" (optional)
6. Click "Register app"
7. Copy the Firebase configuration object

## Step 4: Configure Frontend

1. Open `frontend/js/firebase-config.js`
2. Replace the `firebaseConfig` object with your configuration:

```javascript
const firebaseConfig = {
    apiKey: "your-api-key",
    authDomain: "your-project.firebaseapp.com",
    projectId: "your-project-id",
    storageBucket: "your-project.appspot.com",
    messagingSenderId: "123456789",
    appId: "1:123456789:web:abcdef"
};
```

## Step 5: Setup Backend Firebase Admin

### Option A: Service Account JSON (Recommended)

1. In Firebase Console, go to Project settings > Service accounts
2. Click "Generate new private key"
3. Download the JSON file
4. **For production**: Convert to single line JSON string and set as environment variable:
   ```bash
   FIREBASE_SERVICE_ACCOUNT='{"type":"service_account","project_id":"..."}'
   ```
5. **For development**: Save file and set path:
   ```bash
   FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/serviceAccount.json
   ```

### Option B: Individual Environment Variables

Set each field as environment variable:
```bash
FIREBASE_TYPE=service_account
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-...@your-project.iam.gserviceaccount.com
# ... and so on
```

## Step 6: Configure Admin Users

Set admin users in environment variables:

```bash
# Admin emails (comma-separated)
FIREBASE_ADMIN_EMAILS=admin@university.edu,security@university.edu

# Admin domains (comma-separated)
FIREBASE_ADMIN_DOMAINS=university.edu
```

## Step 7: Test Authentication

1. Start your backend server
2. Open the frontend login page
3. Create a test account with university email
4. Verify email address
5. Sign in and test dashboard access
6. Test admin functions if you're an admin user

## Firestore Rules (Optional)

If you plan to use Firestore for additional data:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Allow authenticated users to read/write their own data
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Allow admins to read all data
    match /{document=**} {
      allow read: if request.auth != null && 
        request.auth.token.email in ['admin@university.edu'];
    }
  }
}
```

## Security Notes

1. **Never commit service account keys to version control**
2. **Use environment variables for all sensitive configuration**
3. **Restrict Firebase project access to authorized personnel**
4. **Set up proper email domains for admin access**
5. **Enable MFA for Firebase Console access**
6. **Regularly rotate service account keys**

## Troubleshooting

### Authentication Issues
- Check Firebase configuration in `firebase-config.js`
- Verify service account setup in backend
- Check browser console for errors
- Test with simple email/password first

### Permission Issues
- Verify admin email configuration
- Check user role assignment logic
- Test with different user accounts
- Check backend logs for auth errors

### API Connection Issues
- Verify backend URL in configuration
- Check CORS settings
- Test API endpoints with Postman
- Verify Firebase Admin SDK initialization
