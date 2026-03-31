from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import require_admin_token
from app.db.session import get_session
from app.domain.conversation_manager import ConversationManager
from app.schemas.transport import InboundSmsPayload
from app.transport.twilio_adapter import TwilioTransport

router = APIRouter(prefix="/messages", tags=["messages"])


class SimulateMessageRequest(BaseModel):
    from_number: str
    to_number: str
    body: str
    message_sid: str


@router.post("/simulate", dependencies=[Depends(require_admin_token)])
def simulate_message(payload: SimulateMessageRequest, session: Session = Depends(get_session)) -> dict:
    manager = ConversationManager()
    inbound = InboundSmsPayload(
        From=payload.from_number,
        To=payload.to_number,
        Body=payload.body,
        MessageSid=payload.message_sid,
        NumMedia=0,
        media=[],
    )
    result = manager.process_inbound(session, inbound)
    transport = TwilioTransport()
    for msg in result.outgoing_messages:
        transport.send_sms(to_number=payload.from_number, body=msg)
    session.commit()
    return {"skipped_duplicate": result.skipped_duplicate, "outgoing_messages": result.outgoing_messages}

