from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_session
from app.domain.conversation_manager import ConversationManager
from app.schemas.transport import InboundSmsPayload
from app.transport.twilio_adapter import TwilioTransport

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)


@router.post("/twilio", response_class=PlainTextResponse)
async def twilio_webhook(
    request: Request,
    x_twilio_signature: str = Header(default=""),
    session: Session = Depends(get_session),
) -> str:
    form = await request.form()
    data = {k: str(v) for k, v in form.multi_items()}
    transport = TwilioTransport()
    settings = get_settings()
    if transport.enabled and settings.twilio_validate_signature:
        is_valid = transport.validate_webhook(str(request.url), data, x_twilio_signature)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid twilio signature")

    payload = InboundSmsPayload.from_twilio_form(data)
    manager = ConversationManager()
    try:
        result = manager.process_inbound(session, payload)
        if result.skipped_duplicate:
            return "ok"

        for chunk in result.outgoing_messages:
            transport.send_sms(to_number=payload.from_number, body=chunk)

        session.commit()
    except Exception as exc:
        session.rollback()
        logger.exception("twilio webhook processing failed: %s", exc)
        transport.send_sms(
            to_number=payload.from_number,
            body="my bad, i hit a processing hiccup. resend that and i got you.",
        )
    return "ok"
