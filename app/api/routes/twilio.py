from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from starlette.requests import ClientDisconnect

from app.core.config import get_settings
from app.schemas.transport import InboundSmsPayload
from app.transport.twilio_adapter import TwilioTransport
from app.worker.tasks import process_inbound_sms_task

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)


@router.post("/twilio", response_class=PlainTextResponse)
async def twilio_webhook(
    request: Request,
    x_twilio_signature: str = Header(default=""),
) -> str:
    try:
        form = await request.form()
    except ClientDisconnect:
        logger.warning("twilio webhook client disconnected before form parse")
        return "ok"

    data = {k: str(v) for k, v in form.multi_items()}
    transport = TwilioTransport()
    settings = get_settings()
    if transport.enabled and settings.twilio_validate_signature:
        is_valid = transport.validate_webhook(str(request.url), data, x_twilio_signature)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid twilio signature")

    payload = InboundSmsPayload.from_twilio_form(data)
    process_inbound_sms_task.delay(payload.model_dump(mode="json", by_alias=True))
    return "ok"
