from __future__ import annotations

from pydantic import BaseModel, Field


class InboundMedia(BaseModel):
    media_url: str
    content_type: str | None = None


class InboundSmsPayload(BaseModel):
    from_number: str = Field(alias="From")
    to_number: str = Field(alias="To")
    body: str = Field(alias="Body")
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
        data["media"] = media
        return cls.model_validate(data)

