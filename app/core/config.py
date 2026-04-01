from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_name: str = "friend"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    timezone: str = "America/New_York"

    database_url: str = "postgresql+psycopg://friend:friend@localhost:5432/friend"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    llm_provider: str = "openai"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_text_model: str = "gpt-5.4-mini"
    openai_intent_model: str = "gpt-5.4-nano"
    openai_composer_model: str = "gpt-5.4-mini"
    openai_fallback_text_model: str = "gpt-5.4-nano"
    openai_vision_model: str = "gpt-5.4-mini"
    openai_timeout_seconds: int = 45
    openai_fallback_to_ollama: bool = True

    ollama_base_url: str = "http://localhost:11434"
    ollama_text_model: str = "llama3.2:1b"
    ollama_intent_model: str = ""
    ollama_composer_model: str = ""
    ollama_fallback_text_model: str = "llama3.2:1b"
    ollama_vision_model: str = "llava:13b"
    ollama_timeout_seconds: int = 45
    ollama_keep_alive: str = "30m"
    ollama_auto_pull_missing_models: bool = True
    ollama_warmup_on_startup: bool = True
    ollama_intent_num_ctx: int = 512
    ollama_intent_num_predict: int = 96
    ollama_option_num_gpu: int | None = None
    ollama_option_main_gpu: int | None = None
    ollama_option_num_thread: int | None = None
    ollama_option_num_batch: int | None = None
    ollama_option_low_vram: bool = False

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_to_number: str = ""
    twilio_webhook_auth_token: str = ""
    twilio_validate_signature: bool = False

    admin_token: str = "change-me"
    default_style: str = "casual_cool"
    user_phone_number: str = ""
    user_name: str = "User"
    attachments_dir: str = "/tmp/friend_attachments"
    inbound_dedup_window_minutes: int = 10
    max_sms_chars: int = 320

    reminder_min_spacing_minutes: int = 30
    reminder_max_per_day: int = 10
    checkin_default_minutes: int = 90
    sleepy_hours_start: int = 1
    sleepy_hours_end: int = 8

    @property
    def attachments_path(self) -> Path:
        path = Path(self.attachments_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
