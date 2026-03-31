from __future__ import annotations

from fastapi.testclient import TestClient

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
