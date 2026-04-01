from __future__ import annotations

import asyncio

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from starlette.requests import ClientDisconnect

from app.api.routes import twilio as twilio_route
from app.main import app


def test_twilio_webhook_enqueues_background_task(monkeypatch) -> None:
    captured: dict = {}

    def fake_delay(payload: dict) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(twilio_route.process_inbound_sms_task, "delay", fake_delay)

    client = TestClient(app)
    response = client.post(
        "/webhooks/twilio",
        data={
            "From": "+12488290272",
            "To": "+17622516270",
            "Body": "is this thing on??",
            "MessageSid": "SMTEST123",
            "NumMedia": "0",
        },
    )

    assert response.status_code == 200
    assert response.text == "ok"
    assert captured["payload"]["MessageSid"] == "SMTEST123"
    assert captured["payload"]["Body"] == "is this thing on??"


def test_twilio_webhook_accepts_smssid_alias(monkeypatch) -> None:
    captured: dict = {}

    def fake_delay(payload: dict) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(twilio_route.process_inbound_sms_task, "delay", fake_delay)

    client = TestClient(app)
    response = client.post(
        "/webhooks/twilio",
        data={
            "From": "+12488290272",
            "To": "+17622516270",
            "SmsSid": "SMALIASED123",
            "NumMedia": "0",
        },
    )

    assert response.status_code == 200
    assert response.text == "ok"
    assert captured["payload"]["MessageSid"] == "SMALIASED123"
    assert captured["payload"]["Body"] == ""


def test_parse_twilio_form_data_falls_back_to_raw_body_when_form_items_missing() -> None:
    raw = b"From=%2B12488290272&To=%2B17622516270&Body=hey+there&SmsSid=SMRAW123&NumMedia=0"
    parsed = twilio_route._parse_twilio_form_data(raw, None)
    assert parsed["From"] == "+12488290272"
    assert parsed["To"] == "+17622516270"
    assert parsed["Body"] == "hey there"
    assert parsed["SmsSid"] == "SMRAW123"


def test_twilio_webhook_returns_503_on_client_disconnect() -> None:
    class _DisconnectingRequest:
        async def body(self) -> bytes:
            raise ClientDisconnect()

        async def form(self):  # pragma: no cover
            return {}

        url = "https://friend.example/webhooks/twilio"

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(twilio_route.twilio_webhook(_DisconnectingRequest(), x_twilio_signature=""))  # type: ignore[arg-type]

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "webhook body unavailable"
