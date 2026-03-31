from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
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

