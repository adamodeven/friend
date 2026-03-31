# Friend: Personal AI Accountability via SMS

Production-minded MVP for a one-user, always-on execution manager that works entirely through texting.

Core stack:
- FastAPI API
- PostgreSQL source of truth
- Redis + Celery worker/beat for background reminders
- Twilio SMS/MMS transport
- Ollama (local) for intent extraction, conversational style, and image understanding
- Deterministic scheduling/state transitions in domain services

## 1) What This Repo Includes

- Natural language inbound parsing (`need to`, `due`, `in class rn`, `what's due this week`, etc.)
- Durable task/project/reminder state in Postgres
- MMS ingestion pipeline for assignment screenshots
- Deterministic reminder scheduler (outside LLM)
- Adaptive context handling (class/driving/dinner/all-nighter)
- Message style layer (`casual_cool`, `direct`, `more_serious`)
- SMS chunking for natural text-bubble output
- Admin/debug routes + CLI
- Alembic migrations
- Docker + Docker Compose suitable for Portainer
- Tests for core state and reminder behavior

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
docker compose up -d postgres redis ollama ollama-init
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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
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
- `ollama-init` auto-pulls `OLLAMA_TEXT_MODEL` and `OLLAMA_VISION_MODEL` for you.
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

## 6) Ollama Setup

- `LLM_PROVIDER=ollama`
- For Docker stack services, `OLLAMA_BASE_URL=http://ollama:11434`
- For host-local API process, use `OLLAMA_BASE_URL=http://localhost:11434`
- Default models in this repo:
  - `OLLAMA_TEXT_MODEL=llama3.1:8b`
  - `OLLAMA_VISION_MODEL=llava:13b`
- Model pull is automated by the `ollama-init` compose service.

## 7) Simulate Inbound Messages

### Via Twilio-form webhook simulation
```bash
bash scripts/simulate_twilio_webhook.sh "yo I need to finish the CAD for the enclosure by tomorrow night"
```

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

## 8) Admin/Debug Surface

Protected by `X-Admin-Token`.

- `GET /api/admin/tasks/active`
- `GET /api/admin/deadlines/upcoming?days=7`
- `GET /api/admin/messages/recent?limit=20`
- `GET /api/admin/profile`
- `GET /api/admin/notes/recent`
- `POST /api/admin/reminders/run` (forces reminder schedule + send pass)

## 9) CLI Commands

```bash
friend-admin active-tasks
friend-admin upcoming --days 7
friend-admin messages --limit 20
friend-admin run-reminders
```

## 10) Architecture Notes

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

## 11) Environment Variables You Must Fill

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
- `OLLAMA_BASE_URL`
- `OLLAMA_TEXT_MODEL`
- `OLLAMA_VISION_MODEL`
- `ADMIN_TOKEN`
- `USER_PHONE_NUMBER`

Recommended:
- `TIMEZONE`
- `DEFAULT_STYLE`
- `ATTACHMENTS_DIR`

## 12) Tests

```bash
pytest -q
```

Current suite validates:
- fallback intent parsing
- deterministic task creation + context block updates
- reminder scheduling + dedup spacing behavior
- inbound webhook idempotency behavior

## 13) Quick Troubleshooting

- Incoming shows in Twilio but no outbound:
  - confirm `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` are set in `.env`
  - for reverse-proxy deployments, keep `TWILIO_VALIDATE_SIGNATURE=false`
  - check API logs: `docker logs -f friend-api`

## 14) Known MVP Limitations / Next Upgrades

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
