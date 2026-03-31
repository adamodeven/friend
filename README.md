# Friend: Personal AI Accountability via SMS

Production-minded MVP for a one-user, always-on execution manager that works entirely through texting.

Core stack:
- FastAPI API
- PostgreSQL source of truth
- Redis + Celery worker/beat for background reminders
- Twilio SMS/MMS transport
- OpenAI for intent extraction, conversational style, and image understanding
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
  llm/                    # OpenAI adapters, extraction, reply composer, style/chunking
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
4. Start Postgres + Redis:
```bash
docker compose up -d postgres redis
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

## 4) Docker Deploy (Portainer-Friendly)

1. Copy `docker-compose.yml` into Portainer stack.
2. Add matching `.env` values in Portainer environment.
3. Deploy stack.
4. API starts with `alembic upgrade head` automatically.

For local Docker deploy:
```bash
docker compose up --build
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

## 6) Simulate Inbound Messages

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

## 7) Admin/Debug Surface

Protected by `X-Admin-Token`.

- `GET /api/admin/tasks/active`
- `GET /api/admin/deadlines/upcoming?days=7`
- `GET /api/admin/messages/recent?limit=20`
- `GET /api/admin/profile`
- `GET /api/admin/notes/recent`
- `POST /api/admin/reminders/run` (forces reminder schedule + send pass)

## 8) CLI Commands

```bash
friend-admin active-tasks
friend-admin upcoming --days 7
friend-admin messages --limit 20
friend-admin run-reminders
```

## 9) Architecture Notes

### Deterministic source of truth
- Task/project/reminder/deadline state lives in Postgres.
- Reminder scheduling and state transitions live in `app/domain`, not LLM prompts.

### LLM role (non-authoritative)
- Intent classification
- Task/deadline extraction
- Conversation phrasing/style
- Screenshot understanding for assignment ingestion

### Reliability controls
- Inbound dedup via Twilio `MessageSid` + unique constraint
- Processing audit rows (`processing_jobs`)
- Background reminder pipeline with status tracking (`pending/sent/skipped/failed`)

### Modularity
- Transport adapter isolated in `app/transport`
- LLM adapter isolated in `app/llm`
- Domain logic isolated in `app/domain`
- Storage isolated in `app/db`

## 10) Environment Variables You Must Fill

Required for production:
- `DATABASE_URL`
- `REDIS_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM_NUMBER`
- `TWILIO_TO_NUMBER`
- `OPENAI_API_KEY`
- `ADMIN_TOKEN`
- `USER_PHONE_NUMBER`

Recommended:
- `TIMEZONE`
- `DEFAULT_STYLE`
- `ATTACHMENTS_DIR`

## 11) Tests

```bash
pytest -q
```

Current suite validates:
- fallback intent parsing
- deterministic task creation + context block updates
- reminder scheduling + dedup spacing behavior
- inbound webhook idempotency behavior

## 12) Known MVP Limitations / Next Upgrades

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

