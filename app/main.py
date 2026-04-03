import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import admin_router, health_router, messages_router, twilio_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.llm.warmup import warmup_ollama_text_model

configure_logging()
settings = get_settings()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if settings.llm_provider.lower() != "ollama":
        yield
        return

    asyncio.create_task(asyncio.to_thread(warmup_ollama_text_model, logger=logger))
    yield


app = FastAPI(title=settings.app_name, lifespan=_lifespan)
app.include_router(health_router)
app.include_router(twilio_router)
app.include_router(messages_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
