import os
import json
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, auth
from pydantic import BaseModel
from enum import Enum

logger = logging.getLogger(__name__)

# Security scheme for Bearer token
security = HTTPBearer()

# User roles enum
class UserRole(str, Enum):
    ADMIN = "admin"
    VIEWER = "viewer"

# User info model
class UserInfo(BaseModel):
    uid: str
    email: str
    name: Optional[str] = None
    role: UserRole = UserRole.VIEWER
    email_verified: bool = False

class FirebaseAuth:
    def __init__(self):
        self.app = None
        self._initialize_firebase()

    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase is already initialized
            if firebase_admin._apps:
                self.app = firebase_admin.get_app()
                logger.info("Firebase Admin SDK already initialized")
                return

            # Get Firebase service account configuration
            firebase_config = self._get_firebase_config()
            
            if firebase_config:
                # Initialize with service account
                cred = credentials.Certificate(firebase_config)
                self.app = firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized successfully with service account")
            else:
                # Initialize with default credentials (for Google Cloud environments)
                self.app = firebase_admin.initialize_app()
                logger.info("Firebase Admin SDK initialized with default credentials")
                
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
            raise Exception(f"Firebase initialization failed: {e}")

    def _get_firebase_config(self) -> Optional[Dict]:
        """Get Firebase service account configuration from environment variables"""
        try:
            # Method 1: Direct JSON string from environment variable
            firebase_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            if firebase_json:
                return json.loads(firebase_json)
            
            # Method 2: Path to service account file
            service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
            if service_account_path and os.path.exists(service_account_path):
                with open(service_account_path, 'r') as f:
                    return json.load(f)
            
            # Method 3: Individual environment variables (for security)
            firebase_config = {}
            required_fields = [
                "type", "project_id", "private_key_id", "private_key",
                "client_email", "client_id", "auth_uri", "token_uri",
                "auth_provider_x509_cert_url", "client_x509_cert_url"
            ]
            
            for field in required_fields:
                env_var = f"FIREBASE_{field.upper()}"
                value = os.getenv(env_var)
                if value:
                    firebase_config[field] = value
                    
            if len(firebase_config) == len(required_fields):
                # Fix private key formatting
                if 'private_key' in firebase_config:
                    firebase_config['private_key'] = firebase_config['private_key'].replace('\\n', '\n')
                return firebase_config
                
            logger.warning("Firebase service account configuration not found in environment variables")
            return None
            
        except Exception as e:
            logger.error(f"Error loading Firebase configuration: {e}")
            return None

    async def verify_token(self, token: str) -> UserInfo:
        """Verify Firebase ID token and return user information"""
        try:
            # Verify the ID token
            decoded_token = auth.verify_id_token(token)
            
            # Extract user information
            uid = decoded_token['uid']
            email = decoded_token.get('email', '')
            name = decoded_token.get('name', '')
            email_verified = decoded_token.get('email_verified', False)
            
            # Determine user role based on email domain or custom claims
            role = self._determine_user_role(email, decoded_token)
            
            return UserInfo(
                uid=uid,
                email=email,
                name=name,
                role=role,
                email_verified=email_verified
            )
            
        except auth.InvalidIdTokenError:
            logger.warning(f"Invalid Firebase ID token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        except auth.ExpiredIdTokenError:
            logger.warning(f"Expired Firebase ID token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token has expired"
            )
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )

    def _determine_user_role(self, email: str, decoded_token: Dict[str, Any]) -> UserRole:
        """Determine user role based on email domain or custom claims"""
        try:
            # Check custom claims first
            if 'role' in decoded_token:
                role_claim = decoded_token['role'].lower()
                if role_claim == 'admin':
                    return UserRole.ADMIN
                elif role_claim == 'viewer':
                    return UserRole.VIEWER
            
            # Check admin emails from environment
            admin_emails = os.getenv("FIREBASE_ADMIN_EMAILS", "").split(",")
            admin_emails = [email.strip() for email in admin_emails if email.strip()]
            
            if email in admin_emails:
                return UserRole.ADMIN
            
            # Check admin domains
            admin_domains = os.getenv("FIREBASE_ADMIN_DOMAINS", "").split(",")
            admin_domains = [domain.strip() for domain in admin_domains if domain.strip()]
            
            email_domain = email.split("@")[-1] if "@" in email else ""
            if email_domain in admin_domains:
                return UserRole.ADMIN
            
            # Default to viewer role
            return UserRole.VIEWER
            
        except Exception as e:
            logger.warning(f"Error determining user role: {e}")
            return UserRole.VIEWER

# Global Firebase Auth instance
firebase_auth = FirebaseAuth()

# Dependency functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserInfo:
    """Dependency to get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    token = credentials.credentials
    return await firebase_auth.verify_token(token)

async def get_admin_user(
    current_user: UserInfo = Depends(get_current_user)
) -> UserInfo:
    """Dependency to ensure user has admin role"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

async def get_verified_user(
    current_user: UserInfo = Depends(get_current_user)
) -> UserInfo:
    """Dependency to ensure user has verified email"""
    if not current_user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )
    return current_user

# Optional dependency - allows both authenticated and anonymous access
async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[UserInfo]:
    """Optional dependency to get current user if authenticated"""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        return await firebase_auth.verify_token(token)
    except HTTPException:
        return None
