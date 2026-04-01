from fastapi import Header, HTTPException, Query, status

from app.core.config import get_settings


def require_admin_token(
    x_admin_token: str = Header(default=""),
    token: str | None = Query(default=None),
) -> None:
    expected = get_settings().admin_token
    provided = x_admin_token or token or ""
    if not expected or provided != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid admin token")
