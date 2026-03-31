from __future__ import annotations

import logging

from twilio.base.exceptions import TwilioRestException
from twilio.request_validator import RequestValidator
from twilio.rest import Client

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class TwilioTransport:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._enabled = bool(settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_from_number)
        self._client = (
            Client(username=settings.twilio_account_sid, password=settings.twilio_auth_token)
            if self._enabled
            else None
        )
        self._validator = RequestValidator(settings.twilio_auth_token) if settings.twilio_auth_token else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def validate_webhook(self, url: str, params: dict[str, str], signature: str) -> bool:
        if not self._validator:
            return True
        return self._validator.validate(url, params, signature)

    def send_sms(self, *, to_number: str, body: str) -> str | None:
        if not self._client:
            logger.info("twilio disabled, would send to %s: %s", to_number, body)
            return None
        try:
            msg = self._client.messages.create(
                from_=self._settings.twilio_from_number,
                to=to_number,
                body=body,
            )
            return msg.sid
        except TwilioRestException as exc:  # pragma: no cover
            logger.exception("failed to send twilio sms: %s", exc)
            return None

