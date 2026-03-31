from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.db.models import ConversationMessage, MessageDirection


def inbound_message_exists(session: Session, external_id: str | None) -> bool:
    if not external_id:
        return False
    stmt: Select[tuple[uuid.UUID]] = select(ConversationMessage.id).where(
        ConversationMessage.direction == MessageDirection.inbound,
        ConversationMessage.external_id == external_id,
    )
    return session.execute(stmt).first() is not None


def create_message(
    session: Session,
    *,
    user_id: uuid.UUID,
    direction: MessageDirection,
    body: str,
    channel: str = "sms",
    external_id: str | None = None,
    message_type: str = "text",
    metadata_json: dict | None = None,
    created_at: datetime | None = None,
) -> ConversationMessage:
    msg = ConversationMessage(
        user_id=user_id,
        direction=direction,
        body=body,
        channel=channel,
        external_id=external_id,
        message_type=message_type,
        metadata_json=metadata_json or {},
    )
    if created_at:
        msg.created_at = created_at
    session.add(msg)
    session.flush()
    return msg


def list_recent_messages(session: Session, user_id: uuid.UUID, limit: int = 30) -> list[ConversationMessage]:
    stmt = (
        select(ConversationMessage)
        .where(ConversationMessage.user_id == user_id)
        .order_by(ConversationMessage.created_at.desc())
        .limit(limit)
    )
    return list(reversed(session.execute(stmt).scalars().all()))

