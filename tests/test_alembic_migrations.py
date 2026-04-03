from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, inspect

from app.core.config import get_settings
from app.db.models import DeadlineEvent, Reminder, Task, TaskDependency, UserProfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def _alembic_config(database_url: str) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _upgrade(monkeypatch: pytest.MonkeyPatch, database_url: str, revision: str) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    get_settings.cache_clear()
    command.upgrade(_alembic_config(database_url), revision)
    get_settings.cache_clear()


def _column_names(inspector, table_name: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table_name)}


def test_existing_0001_database_upgrades_cleanly_to_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    database_url = f"sqlite+pysqlite:///{tmp_path / 'existing.sqlite'}"

    _upgrade(monkeypatch, database_url, "0001_initial_schema")

    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert "baseline_profile_json" not in _column_names(inspector, "user_profiles")
    assert "source_message_id" not in _column_names(inspector, "tasks")
    engine.dispose()

    _upgrade(monkeypatch, database_url, "head")

    engine = create_engine(database_url)
    inspector = inspect(engine)

    task_columns = _column_names(inspector, "tasks")
    assert "source_message_id" in task_columns
    assert "next_step" in task_columns
    assert "deadline_confidence" in task_columns
    assert "slip_count" in task_columns
    assert "reminder_pause_until" in task_columns

    profile_columns = _column_names(inspector, "user_profiles")
    assert "baseline_profile_json" in profile_columns

    dependency_columns = _column_names(inspector, "task_dependencies")
    assert "metadata_json" in dependency_columns
    unique_constraints = {constraint["name"] for constraint in inspector.get_unique_constraints("task_dependencies")}
    assert "uq_task_dependency_edge" in unique_constraints

    reminder_columns = _column_names(inspector, "reminders")
    assert {"attempt_count", "cooldown_until"}.issubset(reminder_columns)

    deadline_columns = _column_names(inspector, "deadline_events")
    assert {"source_phrase", "is_ambiguous", "granularity", "metadata_json"}.issubset(deadline_columns)

    task_indexes = {index["name"] for index in inspector.get_indexes("tasks")}
    assert "ix_tasks_source_message_id" in task_indexes
    engine.dispose()


def test_fresh_upgrade_head_matches_current_models_for_scoped_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    database_url = f"sqlite+pysqlite:///{tmp_path / 'fresh.sqlite'}"
    _upgrade(monkeypatch, database_url, "head")

    engine = create_engine(database_url)
    inspector = inspect(engine)

    scoped_models = {
        "user_profiles": UserProfile,
        "tasks": Task,
        "task_dependencies": TaskDependency,
        "reminders": Reminder,
        "deadline_events": DeadlineEvent,
    }

    for table_name, model in scoped_models.items():
        assert _column_names(inspector, table_name) == set(model.__table__.columns.keys())

    engine.dispose()
