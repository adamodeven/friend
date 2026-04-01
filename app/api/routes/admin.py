from __future__ import annotations

import asyncio
import json
from html import escape
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import require_admin_token
from app.db.models import ConversationMessage, PlanningNote, UserProfile
from app.db.repositories.message_repo import list_recent_messages
from app.db.repositories.task_repo import list_active_tasks, list_upcoming_deadlines
from app.db.repositories.user_repo import get_or_create_primary_user
from app.db.session import get_session
from app.schemas.admin import MessageView, ProfileView, TaskView
from app.worker.tasks import schedule_reminders_task, send_due_reminders_task

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_token)])


@router.get("/tasks/active", response_model=list[TaskView])
def active_tasks(session: Session = Depends(get_session)) -> list[TaskView]:
    user = get_or_create_primary_user(session)
    tasks = list_active_tasks(session, user.id)
    return [
        TaskView(
            id=t.id,
            title=t.title,
            status=t.status.value,
            priority=t.priority,
            deadline_at=t.deadline_at,
            project_id=t.project_id,
        )
        for t in tasks
    ]


@router.get("/deadlines/upcoming", response_model=list[TaskView])
def upcoming_deadlines(days: int = 7, session: Session = Depends(get_session)) -> list[TaskView]:
    user = get_or_create_primary_user(session)
    tasks = list_upcoming_deadlines(session, user.id, within_days=days)
    return [
        TaskView(
            id=t.id,
            title=t.title,
            status=t.status.value,
            priority=t.priority,
            deadline_at=t.deadline_at,
            project_id=t.project_id,
        )
        for t in tasks
    ]


@router.get("/messages/recent", response_model=list[MessageView])
def recent_messages(limit: int = 20, session: Session = Depends(get_session)) -> list[MessageView]:
    user = get_or_create_primary_user(session)
    messages = list_recent_messages(session, user.id, limit=limit)
    return [
        MessageView(
            id=m.id,
            direction=m.direction.value,
            body=m.body,
            created_at=m.created_at,
        )
        for m in messages
    ]


@router.get("/messages/stream")
async def stream_messages(
    limit: int = 120,
    poll_seconds: float = 1.0,
    heartbeat_seconds: float = 15.0,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    user = get_or_create_primary_user(session)
    safe_limit = min(max(limit, 10), 300)
    safe_poll_seconds = min(max(poll_seconds, 0.25), 5.0)
    safe_heartbeat_seconds = min(max(heartbeat_seconds, 3.0), 60.0)

    async def event_generator():
        last_payload = ""
        last_emit = datetime.now(tz=timezone.utc)
        try:
            while True:
                session.expire_all()
                messages = list_recent_messages(session, user.id, limit=safe_limit)
                payload = [
                    {
                        "id": str(message.id),
                        "direction": message.direction.value,
                        "body": message.body,
                        "created_at": message.created_at.isoformat(),
                    }
                    for message in messages
                ]
                serialized = json.dumps(payload, ensure_ascii=True)
                now = datetime.now(tz=timezone.utc)
                if serialized != last_payload:
                    yield f"event: messages\ndata: {serialized}\n\n"
                    last_payload = serialized
                    last_emit = now
                elif (now - last_emit).total_seconds() >= safe_heartbeat_seconds:
                    yield "event: ping\ndata: {}\n\n"
                    last_emit = now
                await asyncio.sleep(safe_poll_seconds)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/live", response_class=HTMLResponse)
def live_monitor() -> str:
    title = escape("Friend Live SMS Monitor")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0d1117;
      --panel: #111827;
      --muted: #9ca3af;
      --line: #1f2937;
      --inbound: #a7f3d0;
      --outbound: #93c5fd;
      --text: #e5e7eb;
    }}
    body {{
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      background: radial-gradient(circle at top, #101926, var(--bg) 55%);
      color: var(--text);
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1024px;
      margin: 24px auto;
      padding: 0 16px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }}
    h1 {{
      font-size: 18px;
      margin: 0;
      font-weight: 600;
    }}
    #status {{
      font-size: 12px;
      color: var(--muted);
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: rgba(17, 24, 39, 0.9);
      backdrop-filter: blur(2px);
    }}
    .row {{
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      display: grid;
      grid-template-columns: 90px 90px 1fr;
      gap: 10px;
      align-items: baseline;
    }}
    .row:first-child {{
      border-top: 0;
    }}
    .ts {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .dir {{
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .dir.inbound {{ color: var(--inbound); }}
    .dir.outbound {{ color: var(--outbound); }}
    .body {{
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.35;
    }}
    .controls {{
      margin: 10px 0 16px;
      display: flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }}
    button {{
      border: 1px solid #374151;
      border-radius: 8px;
      background: #111827;
      color: #d1d5db;
      padding: 6px 10px;
      cursor: pointer;
    }}
    button:hover {{
      background: #1f2937;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>{title}</h1>
      <div id="status">connecting…</div>
    </div>
    <div class="controls">
      <span>auto-updates every second</span>
      <button id="clear">clear screen</button>
    </div>
    <div id="feed" class="panel"></div>
  </div>
  <script>
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    const statusEl = document.getElementById("status");
    const feed = document.getElementById("feed");
    const seen = new Map();

    document.getElementById("clear").addEventListener("click", () => {{
      feed.innerHTML = "";
      seen.clear();
    }});

    function fmtTime(iso) {{
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return "--:--";
      return d.toLocaleTimeString([], {{ hour: "2-digit", minute: "2-digit", second: "2-digit" }});
    }}

    function render(messages) {{
      const sorted = [...messages].sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
      for (const msg of sorted) {{
        const prior = seen.get(msg.id);
        const sig = `${{msg.direction}}|${{msg.body}}|${{msg.created_at}}`;
        if (prior === sig) continue;
        seen.set(msg.id, sig);
      }}
      const html = sorted.map((msg) => {{
        const safeBody = String(msg.body ?? "")
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;");
        const dir = msg.direction || "unknown";
        return `<div class="row"><div class="ts">${{fmtTime(msg.created_at)}}</div><div class="dir ${{dir}}">${{dir}}</div><div class="body">${{safeBody}}</div></div>`;
      }}).join("");
      feed.innerHTML = html;
      window.scrollTo(0, document.body.scrollHeight);
    }}

    if (!token) {{
      statusEl.textContent = "missing token in url. open /api/admin/live?token=YOUR_ADMIN_TOKEN";
    }} else {{
      const streamUrl = `/api/admin/messages/stream?token=${{encodeURIComponent(token)}}&limit=180&poll_seconds=1`;
      const source = new EventSource(streamUrl);
      source.addEventListener("open", () => {{
        statusEl.textContent = "live";
      }});
      source.addEventListener("messages", (event) => {{
        try {{
          const data = JSON.parse(event.data);
          render(Array.isArray(data) ? data : []);
        }} catch {{
          statusEl.textContent = "stream parse hiccup";
        }}
      }});
      source.addEventListener("ping", () => {{
        statusEl.textContent = "live";
      }});
      source.addEventListener("error", () => {{
        statusEl.textContent = "reconnecting…";
      }});
    }}
  </script>
</body>
</html>"""


@router.post("/reminders/run")
def force_reminder_run() -> dict:
    scheduled = schedule_reminders_task()
    sent = send_due_reminders_task()
    return {"schedule_result": scheduled, "send_result": sent}


@router.get("/profile", response_model=ProfileView)
def profile(session: Session = Depends(get_session)) -> ProfileView:
    user = get_or_create_primary_user(session)
    profile = session.execute(select(UserProfile).where(UserProfile.user_id == user.id)).scalars().first()
    if not profile:
        return ProfileView(style="casual_cool", planning_preferences={}, bio=None)
    return ProfileView(
        style=profile.style.value,
        planning_preferences=profile.planning_preferences or {},
        bio=profile.bio,
    )


@router.get("/notes/recent")
def recent_notes(limit: int = 15, session: Session = Depends(get_session)) -> dict:
    user = get_or_create_primary_user(session)
    now = datetime.now(tz=ZoneInfo(user.timezone))
    week_ago = now - timedelta(days=7)
    notes = (
        session.execute(
            select(PlanningNote)
            .where(PlanningNote.user_id == user.id, PlanningNote.created_at >= week_ago)
            .order_by(PlanningNote.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return {
        "count": len(notes),
        "items": [
            {
                "type": note.note_type,
                "content": note.content,
                "weight": note.weight,
                "created_at": note.created_at.isoformat(),
            }
            for note in notes
        ],
    }
