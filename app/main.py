from fastapi import FastAPI

from app.api.routes import admin_router, health_router, messages_router, twilio_router
from app.core.config import get_settings
from app.core.logging import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(title=settings.app_name)
app.include_router(health_router)
app.include_router(twilio_router)
app.include_router(messages_router, prefix="/api")
app.include_router(admin_router, prefix="/api")

