from __future__ import annotations

from types import SimpleNamespace

from app.transport import twilio_adapter
from app.transport.twilio_adapter import TwilioTransport


def test_send_sms_skips_when_outbound_disabled(monkeypatch) -> None:
    fake_settings = SimpleNamespace(
        twilio_account_sid="AC123",
        twilio_auth_token="token",
        twilio_from_number="+17622516270",
        twilio_outbound_enabled=False,
    )

    monkeypatch.setattr(twilio_adapter, "get_settings", lambda: fake_settings)
    transport = TwilioTransport()
    sid = transport.send_sms(to_number="+12488290272", body="test message")

    assert sid is not None
    assert sid.startswith("SM_DRYRUN_")


def test_send_sms_uses_client_when_outbound_enabled(monkeypatch) -> None:
    fake_settings = SimpleNamespace(
        twilio_account_sid="AC123",
        twilio_auth_token="token",
        twilio_from_number="+17622516270",
        twilio_outbound_enabled=True,
    )

    class _Messages:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def create(self, *, from_: str, to: str, body: str):  # noqa: ANN001,ANN202
            self.calls.append({"from": from_, "to": to, "body": body})
            return SimpleNamespace(sid="SM_REAL_123")

    class _Client:
        def __init__(self) -> None:
            self.messages = _Messages()

    monkeypatch.setattr(twilio_adapter, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(twilio_adapter, "Client", lambda username, password: _Client())

    transport = TwilioTransport()
    sid = transport.send_sms(to_number="+12488290272", body="hello")

    assert sid == "SM_REAL_123"
