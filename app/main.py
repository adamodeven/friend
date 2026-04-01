import asyncio
import logging

from fastapi import FastAPI

from app.api.routes import admin_router, health_router, messages_router, twilio_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.llm.warmup import warmup_ollama_text_model

configure_logging()
settings = get_settings()

app = FastAPI(title=settings.app_name)
app.include_router(health_router)
app.include_router(twilio_router)
app.include_router(messages_router, prefix="/api")
app.include_router(admin_router, prefix="/api")

logger = logging.getLogger(__name__)


@app.on_event("startup")
async def _startup_warmup() -> None:
    if settings.llm_provider.lower() != "ollama":
        return
    asyncio.create_task(asyncio.to_thread(warmup_ollama_text_model, logger=logger))
