from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field


class InboundMedia(BaseModel):
    media_url: str
    content_type: str | None = None


class InboundSmsPayload(BaseModel):
    from_number: str = Field(alias="From")
    to_number: str = Field(alias="To")
    body: str = Field(default="", alias="Body")
    message_sid: str = Field(alias="MessageSid")
    num_media: int = Field(default=0, alias="NumMedia")
    media: list[InboundMedia] = Field(default_factory=list)

    @classmethod
    def from_twilio_form(cls, form: dict[str, str]) -> "InboundSmsPayload":
        num_media = int(form.get("NumMedia", "0"))
        media = []
        for i in range(num_media):
            media.append(
                InboundMedia(
                    media_url=form.get(f"MediaUrl{i}", ""),
                    content_type=form.get(f"MediaContentType{i}"),
                )
            )
        data = dict(form)
        if not data.get("MessageSid") and data.get("SmsSid"):
            data["MessageSid"] = data["SmsSid"]
        if not data.get("MessageSid") and data.get("SmsMessageSid"):
            data["MessageSid"] = data["SmsMessageSid"]
        if not data.get("MessageSid"):
            from_num = data.get("From", "")
            to_num = data.get("To", "")
            body = data.get("Body", "")
            digest = hashlib.sha1(f"{from_num}|{to_num}|{body}".encode("utf-8")).hexdigest()[:12]
            data["MessageSid"] = f"missing-{digest}"
        if "Body" not in data or data.get("Body") is None:
            data["Body"] = ""
        data["media"] = media
        return cls.model_validate(data)
