from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import typer
from rich.console import Console
from rich.table import Table

from app.db.repositories.message_repo import list_recent_messages
from app.db.repositories.task_repo import list_active_tasks, list_upcoming_deadlines
from app.db.repositories.user_repo import get_or_create_primary_user
from app.db.session import SessionLocal
from app.worker.tasks import schedule_reminders_task, send_due_reminders_task

app = typer.Typer(help="Friend admin/debug CLI")
console = Console()


@app.command("active-tasks")
def active_tasks() -> None:
    session = SessionLocal()
    try:
        user = get_or_create_primary_user(session)
        tasks = list_active_tasks(session, user.id)
        table = Table(title="Active Tasks")
        table.add_column("Title")
        table.add_column("Status")
        table.add_column("Priority")
        table.add_column("Deadline")
        for task in tasks:
            deadline = task.deadline_at.astimezone(ZoneInfo(user.timezone)).strftime("%Y-%m-%d %H:%M") if task.deadline_at else "-"
            table.add_row(task.title, task.status.value, str(task.priority), deadline)
        console.print(table)
    finally:
        session.close()


@app.command("upcoming")
def upcoming(days: int = 7) -> None:
    session = SessionLocal()
    try:
        user = get_or_create_primary_user(session)
        tasks = list_upcoming_deadlines(session, user.id, within_days=days)
        table = Table(title=f"Upcoming Deadlines ({days}d)")
        table.add_column("Title")
        table.add_column("Due")
        for task in tasks:
            due = task.deadline_at.astimezone(ZoneInfo(user.timezone)).strftime("%a %m/%d %I:%M%p") if task.deadline_at else "-"
            table.add_row(task.title, due)
        console.print(table)
    finally:
        session.close()


@app.command("messages")
def messages(limit: int = 20) -> None:
    session = SessionLocal()
    try:
        user = get_or_create_primary_user(session)
        msgs = list_recent_messages(session, user.id, limit=limit)
        for msg in msgs:
            timestamp = msg.created_at.astimezone(ZoneInfo(user.timezone)).strftime("%m-%d %H:%M")
            console.print(f"[{timestamp}] {msg.direction.value}: {msg.body}")
    finally:
        session.close()


@app.command("run-reminders")
def run_reminders() -> None:
    scheduled = schedule_reminders_task()
    sent = send_due_reminders_task()
    console.print({"scheduled": scheduled, "sent": sent})


if __name__ == "__main__":
    app()

