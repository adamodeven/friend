from __future__ import annotations

import logging
from urllib.parse import parse_qsl

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from starlette.requests import ClientDisconnect

from app.core.config import get_settings
from app.schemas.transport import InboundSmsPayload
from app.transport.twilio_adapter import TwilioTransport
from app.worker.tasks import process_inbound_sms_task

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)


def _parse_twilio_form_data(raw_body: bytes, form_items: list[tuple[str, str]] | None) -> dict[str, str]:
    if form_items:
        return {k: str(v) for k, v in form_items}
    decoded = raw_body.decode("utf-8", errors="ignore")
    parsed = parse_qsl(decoded, keep_blank_values=True)
    return {k: str(v) for k, v in parsed}


@router.post("/twilio", response_class=PlainTextResponse)
async def twilio_webhook(
    request: Request,
    x_twilio_signature: str = Header(default=""),
) -> str:
    try:
        raw_body = await request.body()
    except ClientDisconnect:
        logger.warning("twilio webhook client disconnected before body read")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="webhook body unavailable")

    try:
        form = await request.form()
        form_items = [(k, str(v)) for k, v in form.multi_items()]
    except Exception:
        form_items = None

    data = _parse_twilio_form_data(raw_body, form_items)
    if not data:
        logger.warning("twilio webhook empty payload")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty twilio payload")

    transport = TwilioTransport()
    settings = get_settings()
    if transport.enabled and settings.twilio_validate_signature:
        is_valid = transport.validate_webhook(str(request.url), data, x_twilio_signature)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid twilio signature")

    payload = InboundSmsPayload.from_twilio_form(data)
    sid = payload.message_sid or "unknown"
    body_preview = (payload.body or "").replace("\n", " ").strip()
    if len(body_preview) > 160:
        body_preview = f"{body_preview[:157]}..."
    logger.info(
        "twilio inbound accepted sid=%s from=%s to=%s num_media=%s body=%s",
        sid,
        payload.from_number,
        payload.to_number,
        payload.num_media,
        body_preview,
    )
    payload_json = payload.model_dump(mode="json", by_alias=True)
    try:
        process_inbound_sms_task.delay(payload_json)
        logger.info("twilio inbound queued sid=%s", sid)
    except Exception:
        logger.exception("failed to enqueue inbound sid=%s, processing inline", sid)
        process_inbound_sms_task(payload_json)
    return "ok"
