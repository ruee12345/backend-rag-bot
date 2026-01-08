from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# Simple user database for demo
fake_users_db = {
    "employee@test.com": {
        "email": "employee@test.com",
        "password": "employee123",
        "role": "employee",
        "name": "Test Employee"
    },
    "admin@test.com": {
        "email": "admin@test.com",
        "password": "admin123",
        "role": "admin",
        "name": "Test Admin"
    }
}

class LoginRequest(BaseModel):
    email: str
    password: str
    role: Optional[str] = "employee"

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

@router.post("/login")
async def login(request: LoginRequest):
    """Simple login for demo (will replace with JWT/SSO later)"""
    user = fake_users_db.get(request.email)
    
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check role if specified
    if request.role and user["role"] != request.role:
        raise HTTPException(
            status_code=403, 
            detail=f"User is not a {request.role}. Role: {user['role']}"
        )
    
    # Create simple token (demo only - will use JWT later)
    token_data = {
        "email": user["email"],
        "role": user["role"],
        "name": user["name"]
    }
    
    # In production, use JWT tokens
    fake_token = f"demo-token-{user['email']}"
    
    return LoginResponse(
        access_token=fake_token,
        user={
            "email": user["email"],
            "role": user["role"],
            "name": user["name"]
        }
    )

@router.get("/me")
async def get_current_user(token: str):
    """Get current user info (demo)"""
    if not token.startswith("demo-token-"):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Extract email from token
    email = token.replace("demo-token-", "")
    user = fake_users_db.get(email)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "email": user["email"],
        "role": user["role"],
        "name": user["name"]
    }