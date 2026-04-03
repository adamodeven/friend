# Friend: Personal AI Accountability via SMS

Production-minded MVP for a one-user, always-on execution manager that works entirely through texting.

Core stack:
- FastAPI API
- PostgreSQL source of truth
- Redis + Celery worker/beat for background reminders
- Twilio SMS/MMS transport
- OpenAI or Ollama for intent extraction, conversational style, and image understanding
- Deterministic scheduling/state transitions in domain services

## 1) What This Repo Includes

- Natural language inbound parsing (`need to`, `due`, `in class rn`, `what's due this week`, etc.)
- Multiple-task extraction from a single text and lightweight next-step generation
- Dependency/blocker capture (`need to fix portfolio website first`) with prerequisite task linking
- Brief follow-ups only when timing or blocker ambiguity would materially change planning
- Durable task/project/reminder state in Postgres
- MMS ingestion pipeline for assignment screenshots with extracted artifacts plus task creation/update when possible
- Deterministic reminder scheduler (outside LLM)
- Adaptive context handling (class/driving/dinner/all-nighter)
- Message style layer (`casual_cool`, `direct`, `more_serious`)
- SMS chunking for natural text-bubble output
- Admin/debug routes + CLI
- Alembic migrations
- Docker + Docker Compose suitable for Portainer
- Tests for core state and reminder behavior
- Conversation stress tests covering mixed turns, slips, blocker replans, timeline queries, and screenshot-derived tasks

## 2) Repository Structure

```text
app/
  api/routes/             # FastAPI routes (twilio, admin, health, message simulation)
  core/                   # config, logging, security, time parsing helpers
  db/                     # SQLAlchemy models, session, repositories
  domain/                 # deterministic state engine, reminders, timeline, memory helpers
  llm/                    # Ollama adapter, extraction, reply composer, style/chunking
  transport/              # Twilio transport adapter
  ingestion/              # attachment download + image extraction
  worker/                 # Celery app + scheduled tasks
  cli/                    # admin/debug CLI
alembic/                  # migrations
tests/                    # pytest coverage for key logic
docker/                   # API/worker Dockerfiles
scripts/                  # bootstrap + webhook simulation helpers
prompts/                  # sample prompt templates
config/                   # style profile config example
```

## 3) Quick Start (Local Dev)

### Prereqs
- Python 3.12+
- Docker + Docker Compose

### Steps
1. Copy env file:
```bash
cp .env.example .env
```
2. Fill required values in `.env` (see env checklist below).
3. Install dependencies:
```bash
python3 -m pip install -e '.[dev]'
```
4. Start core services for local app/worker run:
```bash
docker compose up -d postgres redis
```
If `LLM_PROVIDER=ollama`, also start:
```bash
docker compose up -d ollama ollama-init
```
5. Run migrations:
```bash
alembic upgrade head
```
6. Bootstrap user/profile from `USER_PROFILE.md`:
```bash
python3 scripts/bootstrap_user.py
```
7. Run API:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port "${APP_PORT:-8000}"
```
8. Run worker:
```bash
celery -A app.worker.celery_app.celery_app worker -l INFO
```
9. Run beat scheduler:
```bash
celery -A app.worker.celery_app.celery_app beat -l INFO
```
10. First run note:
- `ollama-init` auto-pulls `OLLAMA_TEXT_MODEL` for you.
- set `OLLAMA_PULL_VISION_ON_STARTUP=true` if you also want pre-pull of `OLLAMA_VISION_MODEL`.
- first boot can take a while while models download.

## 4) Docker Deploy (Portainer-Friendly)

1. Copy `docker-compose.yml` into Portainer stack.
2. Add matching `.env` values in Portainer environment.
3. Deploy stack.
4. `ollama-init` automatically pulls your configured models before API/worker/beat start.
5. API starts with `alembic upgrade head` automatically.

For local Docker deploy:
```bash
docker compose up --build
```

Note:
- API host binding uses `APP_PORT` from `.env` (default `8000`).
- Preflight your stack file before deploying with:
```bash
docker compose config
```

If you change model names later and want to pull again:
```bash
docker compose run --rm ollama-init
```

## 5) Twilio Setup

1. Buy/configure Twilio number with SMS + MMS enabled.
2. Set webhook URL to:
```text
https://<your-domain>/webhooks/twilio
```
3. Set method to `POST`.
4. Fill these `.env` fields:
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM_NUMBER`
- `TWILIO_TO_NUMBER` (your personal number for one-user mode)
- `TWILIO_VALIDATE_SIGNATURE=false` (recommended behind reverse proxy for MVP)
- `TWILIO_OUTBOUND_ENABLED=true` (set `false` to process inbound + AI logic without paying for outbound sends during testing)

## 6) Provider Setup (Default: OpenAI)

This repo now defaults to `LLM_PROVIDER=openai`.

### OpenAI setup (recommended default)

- `LLM_PROVIDER=openai`
- Set `OPENAI_API_KEY`
- Keep `OPENAI_BASE_URL=https://api.openai.com/v1` unless you use a compatible gateway
- Suggested model split:
  - `OPENAI_INTENT_MODEL=gpt-5.4-nano`
  - `OPENAI_COMPOSER_MODEL=gpt-5.4-mini`
  - `OPENAI_TEXT_MODEL=gpt-5.4-mini`
  - `OPENAI_FALLBACK_TEXT_MODEL=gpt-5.4-nano`
  - `OPENAI_VISION_MODEL=gpt-5.4-mini`
- `OPENAI_TIMEOUT_SECONDS=45`
- `OPENAI_FALLBACK_TO_OLLAMA=true` keeps replies alive if OpenAI is rate-limited or temporarily unavailable
- `OPENAI_RATE_LIMIT_COOLDOWN_SECONDS=300` avoids repeated OpenAI 429 retries by temporarily routing directly to Ollama fallback
- `OPENAI_INSUFFICIENT_QUOTA_COOLDOWN_SECONDS=3600` pauses OpenAI retries longer when billing/quota is exhausted, and skips low-quality Ollama text fallback in that mode

### Ollama setup (optional, still fully supported)

- `LLM_PROVIDER=ollama`
- For Docker stack services, `OLLAMA_BASE_URL=http://ollama:11434`
- For host-local API process, use `OLLAMA_BASE_URL=http://localhost:11434`
- Default models in this repo:
  - `OLLAMA_TEXT_MODEL=llama3.2:1b`
  - `OLLAMA_FALLBACK_TEXT_MODEL=llama3.2:1b`
  - `OLLAMA_VISION_MODEL=llava:13b`
- `OLLAMA_PULL_VISION_ON_STARTUP=false` by default so chat starts faster.
- `OLLAMA_TIMEOUT_SECONDS=45` default favors fail-fast behavior on CPU-only hosts.
- `OLLAMA_AUTO_PULL_MISSING_MODELS=true` auto-recovers if a configured model is missing.
- `OLLAMA_WARMUP_ON_STARTUP=true` pre-warms the text model on API/worker startup to reduce first-message latency.
- `OLLAMA_INTENT_NUM_CTX=512` and `OLLAMA_INTENT_NUM_PREDICT=96` keep intent extraction fast for short SMS.
- Model pull is automated by the `ollama-init` compose service.
- For low-RAM/CPU hosts, keep `OLLAMA_TEXT_MODEL=llama3.2:1b` for better SMS reliability.
- `WORKER_CONCURRENCY=1` is recommended for one-user CPU boxes to reduce inference contention.

### GPU profile (GTX 1050 friendly)

If you have a GTX 1050, these settings are tuned for stable latency and VRAM limits:

- `OLLAMA_TEXT_MODEL=llama3.2:1b`
- `OLLAMA_CONTEXT_LENGTH=1024`
- `OLLAMA_NUM_PARALLEL=1`
- `OLLAMA_MAX_LOADED_MODELS=1`
- `OLLAMA_MAX_QUEUE=64`
- `OLLAMA_FLASH_ATTENTION=1`
- `OLLAMA_KV_CACHE_TYPE=q4_0`
- `OLLAMA_DOCKER_RUNTIME=nvidia`
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- `OLLAMA_OPTION_NUM_GPU=999`
- `OLLAMA_OPTION_MAIN_GPU=0`
- `OLLAMA_OPTION_NUM_THREAD=4`
- `OLLAMA_OPTION_NUM_BATCH=64`
- `OLLAMA_OPTION_LOW_VRAM=true`

### Host prerequisites for Docker GPU acceleration

On Ubuntu host:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-container-toolkit
echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
docker run --rm --gpus=all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker exec friend-ollama ollama ps
```

`ollama ps` should show processor usage on GPU (not CPU-only).

## 7) Simulate Inbound Messages

### Via Twilio-form webhook simulation
```bash
bash scripts/simulate_twilio_webhook.sh "yo I need to finish the CAD for the enclosure by tomorrow night"
```

### Stress simulation (many mixed inbound texts)
```bash
APP_PORT=8045 ROUNDS=120 bash scripts/stress_simulate_twilio.sh
```

Screenshot notes:
- MMS attachments are stored under `ATTACHMENTS_DIR` / `attachments_dir` after download.
- Screenshot ingestion creates an extracted artifact even when task confidence is low.
- When the screenshot clearly maps to an existing active task, the task is updated instead of blindly duplicated.
- Attachment failures are isolated so the text part of the turn still processes and replies.

### Via admin simulation endpoint
```bash
curl -X POST http://localhost:8000/api/messages/simulate \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: change-me" \
  -d '{
    "from_number":"+15555550111",
    "to_number":"+15555550222",
    "body":"what do i have due this week",
    "message_sid":"SM_LOCAL_001"
  }'
```

If you changed `APP_PORT`, replace `8000` in the URL above with your configured port.

## 8) Test The Conversation Flow

Relevant end-to-end conversation hardening tests:

```bash
pytest -q tests/test_conversation_manager.py tests/test_stress_conversation_behaviors.py
```

## 9) Admin/Debug Surface

Protected by `X-Admin-Token`.

- `GET /api/admin/tasks/active`
- `GET /api/admin/deadlines/upcoming?days=7`
- `GET /api/admin/messages/recent?limit=20`
- `GET /api/admin/profile`
- `GET /api/admin/notes/recent`
- `POST /api/admin/reminders/run` (forces reminder schedule + send pass)

## 10) CLI Commands

```bash
friend-admin active-tasks
friend-admin upcoming --days 7
friend-admin messages --limit 20
friend-admin run-reminders
```

## 11) Architecture Notes

### Deterministic source of truth
- Task/project/reminder/deadline state lives in Postgres.
- Reminder scheduling and state transitions live in `app/domain`, not LLM prompts.

### LLM role (non-authoritative)
- Intent classification
- Task/deadline extraction
- Conversation phrasing/style (LLM-first via reply brief composer)
- Screenshot understanding for assignment ingestion

### Response architecture (refactored)
- Layer 1 deterministic state/action engine:
  - updates tasks, deadlines, reminders, context, notes
  - outputs structured `StateOutcome` goals/facts (not end-user wording)
- Layer 2 reply-brief builder:
  - packages latest message, thread window, active tasks, deadlines, profile notes, context flags
  - emits `ReplyBrief` for generation
- Layer 3 LLM conversation composer:
  - generates fresh human texting output from the reply brief
  - semantic chunking for 1-3 natural bubbles
  - repetition guard retries once if wording is too similar to recent assistant output
- Layer 4 failure fallback:
  - minimal safety-only outputs for model failure/timeouts
  - separated from normal flow

### Reliability controls
- Inbound dedup via Twilio `MessageSid` + unique constraint
- Processing audit rows (`processing_jobs`)
- Background reminder pipeline with status tracking (`pending/sent/skipped/failed`)

### Modularity
- Transport adapter isolated in `app/transport`
- LLM adapter isolated in `app/llm`
- Domain logic isolated in `app/domain`
- Storage isolated in `app/db`

## 12) Environment Variables You Must Fill

Required for production:
- `DATABASE_URL`
- `REDIS_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM_NUMBER`
- `TWILIO_TO_NUMBER`
- `LLM_PROVIDER`
- `WORKER_CONCURRENCY`
- `ADMIN_TOKEN`
- `USER_PHONE_NUMBER`

If `LLM_PROVIDER=openai`:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_TEXT_MODEL`
- `OPENAI_INTENT_MODEL`
- `OPENAI_COMPOSER_MODEL`
- `OPENAI_FALLBACK_TEXT_MODEL`
- `OPENAI_VISION_MODEL`
- `OPENAI_TIMEOUT_SECONDS`
- `OPENAI_FALLBACK_TO_OLLAMA`
- `OPENAI_RATE_LIMIT_COOLDOWN_SECONDS`
- `OPENAI_INSUFFICIENT_QUOTA_COOLDOWN_SECONDS`

If `LLM_PROVIDER=ollama`:
- `OLLAMA_BASE_URL`
- `OLLAMA_TEXT_MODEL`
- `OLLAMA_FALLBACK_TEXT_MODEL`
- `OLLAMA_VISION_MODEL`
- `OLLAMA_PULL_VISION_ON_STARTUP`
- `OLLAMA_TIMEOUT_SECONDS`
- `OLLAMA_AUTO_PULL_MISSING_MODELS`
- `OLLAMA_WARMUP_ON_STARTUP`
- `OLLAMA_CONTEXT_LENGTH`
- `OLLAMA_NUM_PARALLEL`
- `OLLAMA_MAX_LOADED_MODELS`
- `OLLAMA_MAX_QUEUE`
- `OLLAMA_FLASH_ATTENTION`
- `OLLAMA_KV_CACHE_TYPE`
- `OLLAMA_DOCKER_RUNTIME`
- `NVIDIA_VISIBLE_DEVICES`
- `NVIDIA_DRIVER_CAPABILITIES`
- `OLLAMA_INTENT_NUM_CTX`
- `OLLAMA_INTENT_NUM_PREDICT`

Recommended:
- `TIMEZONE`
- `DEFAULT_STYLE`
- `ATTACHMENTS_DIR`

## 13) Tests

```bash
pytest -q
```

Targeted stress/behavior suite:
```bash
pytest -q tests/test_stress_conversation_behaviors.py tests/test_reply_composer.py tests/test_intent_extraction.py tests/test_state_engine.py tests/test_time_utils.py
```

Current suite validates:
- natural-language task add/update across single-task and multi-task turns
- deadline inference, ambiguity handling, and timeline-specific time phrases
- deterministic task creation, dependencies, subtasks, blocker-aware replanning, and profile memory capture
- reminder scheduling, escalation, dedup spacing, and context-aware reminder deferral
- inbound webhook idempotency plus end-to-end conversation/state progression
- attachment ingestion and screenshot-derived artifact/task capture
- Alembic upgrade paths and model-to-schema parity for the contract-critical tables

## 14) Quick Troubleshooting

- Incoming shows in Twilio but no outbound:
  - confirm `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` are set in `.env`
  - for reverse-proxy deployments, keep `TWILIO_VALIDATE_SIGNATURE=false`
  - if you intentionally disabled sends for test mode, check `TWILIO_OUTBOUND_ENABLED` (set to `true` to re-enable live SMS)
  - check API logs: `docker logs -f friend-api`

## 15) Known MVP Limitations / Next Upgrades

- Single-user mode only (by design for now)
- Attachment OCR uses model vision parsing directly; no dedicated OCR fallback yet
- No web dashboard yet (admin routes + CLI only)
- Reminder escalation is rule-based and simple; can be expanded with richer behavior scoring
- No full push-based analytics/metrics stack yet (structured logs and DB audit are in place)

Suggested next upgrades:
- Add stronger temporal parser with explicit ambiguity resolution memory
- Add per-task effort estimation calibration based on past misses
- Add richer dependency auto-linking and critical-path views
- Add backup automation (pg_dump + encrypted object storage)
- Add optional secondary transport (iMessage bridge or WhatsApp) via adapter swap
