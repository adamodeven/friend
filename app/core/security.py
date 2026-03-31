from fastapi import Header, HTTPException, status

from app.core.config import get_settings


def require_admin_token(x_admin_token: str = Header(default="")) -> None:
    expected = get_settings().admin_token
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid admin token")

