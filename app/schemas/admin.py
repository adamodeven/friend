from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class TaskView(BaseModel):
    id: UUID
    title: str
    status: str
    priority: int
    deadline_at: datetime | None = None
    project_id: UUID | None = None


class MessageView(BaseModel):
    id: UUID
    direction: str
    body: str
    created_at: datetime


class ProfileView(BaseModel):
    style: str
    planning_preferences: dict[str, Any]
    bio: str | None

