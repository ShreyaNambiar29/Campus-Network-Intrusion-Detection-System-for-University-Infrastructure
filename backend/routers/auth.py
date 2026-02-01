from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from core.firebase_auth import get_current_user, get_optional_user, UserInfo, UserRole

router = APIRouter(prefix="/api/auth", tags=["authentication"])


class UserProfileResponse(BaseModel):
    uid: str
    email: str
    name: Optional[str]
    role: UserRole
    email_verified: bool


class AuthStatusResponse(BaseModel):
    authenticated: bool
    user: Optional[UserProfileResponse] = None


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(current_user: UserInfo = Depends(get_current_user)):
    """Get current user profile information"""
    return UserProfileResponse(
        uid=current_user.uid,
        email=current_user.email,
        name=current_user.name,
        role=current_user.role,
        email_verified=current_user.email_verified
    )


@router.get("/status", response_model=AuthStatusResponse)
async def get_auth_status(user: Optional[UserInfo] = Depends(get_optional_user)):
    """Get authentication status - allows anonymous access"""
    if user:
        return AuthStatusResponse(
            authenticated=True,
            user=UserProfileResponse(
                uid=user.uid,
                email=user.email,
                name=user.name,
                role=user.role,
                email_verified=user.email_verified
            )
        )
    else:
        return AuthStatusResponse(authenticated=False)


@router.post("/verify-role")
async def verify_user_role(current_user: UserInfo = Depends(get_current_user)):
    """Verify user role and permissions"""
    return {
        "uid": current_user.uid,
        "email": current_user.email,
        "role": current_user.role,
        "permissions": {
            "can_view_alerts": True,
            "can_create_alerts": current_user.email_verified,
            "can_resolve_alerts": current_user.role == UserRole.ADMIN,
            "can_manage_users": current_user.role == UserRole.ADMIN
        }
    }
