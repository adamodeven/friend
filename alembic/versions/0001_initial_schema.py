"""initial schema

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-03-31 11:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


message_direction_enum = sa.Enum("inbound", "outbound", name="message_direction")
task_status_enum = sa.Enum("active", "blocked", "completed", "archived", name="task_status")
reminder_status_enum = sa.Enum("pending", "sent", "skipped", "failed", name="reminder_status")
job_status_enum = sa.Enum("queued", "running", "done", "failed", name="job_status")
profile_style_enum = sa.Enum("casual_cool", "direct", "more_serious", name="profile_style")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("phone_number", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_users_phone_number", "users", ["phone_number"], unique=True)

    op.create_table(
        "user_profiles",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("style", profile_style_enum, nullable=False),
        sa.Column("bedtime", sa.Time(), nullable=True),
        sa.Column("wake_time", sa.Time(), nullable=True),
        sa.Column("planning_preferences", sa.JSON(), nullable=False),
        sa.Column("bio", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_user_profiles_user_id", "user_profiles", ["user_id"], unique=True)

    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("direction", message_direction_enum, nullable=False),
        sa.Column("channel", sa.String(length=30), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("message_type", sa.String(length=50), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("channel", "direction", "external_id", name="uq_message_external_direction"),
    )
    op.create_index("ix_conversation_messages_user_id", "conversation_messages", ["user_id"], unique=False)
    op.create_index("ix_conversation_messages_external_id", "conversation_messages", ["external_id"], unique=False)
    op.create_index("ix_conversation_messages_direction", "conversation_messages", ["direction"], unique=False)
    op.create_index("ix_conversation_messages_created_at", "conversation_messages", ["created_at"], unique=False)

    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_projects_user_id", "projects", ["user_id"], unique=False)
    op.create_index("ix_projects_title", "projects", ["title"], unique=False)

    op.create_table(
        "milestones",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_milestones_user_id", "milestones", ["user_id"], unique=False)
    op.create_index("ix_milestones_project_id", "milestones", ["project_id"], unique=False)
    op.create_index("ix_milestones_due_at", "milestones", ["due_at"], unique=False)

    op.create_table(
        "tasks",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("parent_task_id", sa.Uuid(), sa.ForeignKey("tasks.id"), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", task_status_enum, nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("estimated_minutes", sa.Integer(), nullable=True),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("soft_deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("start_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("blocked_reason", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.Column("extraction_confidence", sa.Float(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_tasks_user_id", "tasks", ["user_id"], unique=False)
    op.create_index("ix_tasks_project_id", "tasks", ["project_id"], unique=False)
    op.create_index("ix_tasks_parent_task_id", "tasks", ["parent_task_id"], unique=False)
    op.create_index("ix_tasks_title", "tasks", ["title"], unique=False)
    op.create_index("ix_tasks_status", "tasks", ["status"], unique=False)
    op.create_index("ix_tasks_deadline_at", "tasks", ["deadline_at"], unique=False)
    op.create_index("ix_tasks_soft_deadline_at", "tasks", ["soft_deadline_at"], unique=False)
    op.create_index("ix_tasks_created_at", "tasks", ["created_at"], unique=False)

    op.create_table(
        "task_dependencies",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("predecessor_task_id", sa.Uuid(), sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("successor_task_id", sa.Uuid(), sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("dependency_type", sa.String(length=30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_task_dependencies_user_id", "task_dependencies", ["user_id"], unique=False)
    op.create_index("ix_task_dependencies_predecessor_task_id", "task_dependencies", ["predecessor_task_id"], unique=False)
    op.create_index("ix_task_dependencies_successor_task_id", "task_dependencies", ["successor_task_id"], unique=False)

    op.create_table(
        "reminders",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("task_id", sa.Uuid(), sa.ForeignKey("tasks.id"), nullable=True),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", reminder_status_enum, nullable=False),
        sa.Column("escalation_level", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(length=255), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_reminders_user_id", "reminders", ["user_id"], unique=False)
    op.create_index("ix_reminders_task_id", "reminders", ["task_id"], unique=False)
    op.create_index("ix_reminders_scheduled_for", "reminders", ["scheduled_for"], unique=False)
    op.create_index("ix_reminders_status", "reminders", ["status"], unique=False)

    op.create_table(
        "schedule_blocks",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("block_type", sa.String(length=50), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source_message_id", sa.Uuid(), sa.ForeignKey("conversation_messages.id"), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_schedule_blocks_user_id", "schedule_blocks", ["user_id"], unique=False)
    op.create_index("ix_schedule_blocks_block_type", "schedule_blocks", ["block_type"], unique=False)
    op.create_index("ix_schedule_blocks_starts_at", "schedule_blocks", ["starts_at"], unique=False)
    op.create_index("ix_schedule_blocks_ends_at", "schedule_blocks", ["ends_at"], unique=False)

    op.create_table(
        "attachments",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("message_id", sa.Uuid(), sa.ForeignKey("conversation_messages.id"), nullable=True),
        sa.Column("external_media_id", sa.String(length=128), nullable=True),
        sa.Column("media_url", sa.Text(), nullable=False),
        sa.Column("media_content_type", sa.String(length=128), nullable=True),
        sa.Column("local_path", sa.Text(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("analysis", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_attachments_user_id", "attachments", ["user_id"], unique=False)
    op.create_index("ix_attachments_message_id", "attachments", ["message_id"], unique=False)
    op.create_index("ix_attachments_external_media_id", "attachments", ["external_media_id"], unique=False)
    op.create_index("ix_attachments_sha256", "attachments", ["sha256"], unique=False)

    op.create_table(
        "extracted_artifacts",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("source_attachment_id", sa.Uuid(), sa.ForeignKey("attachments.id"), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("context", sa.String(length=255), nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("structured_data", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_task_id", sa.Uuid(), sa.ForeignKey("tasks.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_extracted_artifacts_user_id", "extracted_artifacts", ["user_id"], unique=False)
    op.create_index("ix_extracted_artifacts_source_attachment_id", "extracted_artifacts", ["source_attachment_id"], unique=False)
    op.create_index("ix_extracted_artifacts_due_at", "extracted_artifacts", ["due_at"], unique=False)
    op.create_index("ix_extracted_artifacts_created_task_id", "extracted_artifacts", ["created_task_id"], unique=False)

    op.create_table(
        "planning_notes",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("note_type", sa.String(length=60), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("related_task_id", sa.Uuid(), sa.ForeignKey("tasks.id"), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_planning_notes_user_id", "planning_notes", ["user_id"], unique=False)
    op.create_index("ix_planning_notes_note_type", "planning_notes", ["note_type"], unique=False)
    op.create_index("ix_planning_notes_related_task_id", "planning_notes", ["related_task_id"], unique=False)
    op.create_index("ix_planning_notes_created_at", "planning_notes", ["created_at"], unique=False)

    op.create_table(
        "daily_summary_snapshots",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("open_task_count", sa.Integer(), nullable=False),
        sa.Column("due_soon_count", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_daily_summary_snapshots_user_id", "daily_summary_snapshots", ["user_id"], unique=False)
    op.create_index("ix_daily_summary_snapshots_snapshot_date", "daily_summary_snapshots", ["snapshot_date"], unique=False)

    op.create_table(
        "deadline_events",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("task_id", sa.Uuid(), sa.ForeignKey("tasks.id"), nullable=True),
        sa.Column("milestone_id", sa.Uuid(), sa.ForeignKey("milestones.id"), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_hard", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(length=60), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_deadline_events_user_id", "deadline_events", ["user_id"], unique=False)
    op.create_index("ix_deadline_events_task_id", "deadline_events", ["task_id"], unique=False)
    op.create_index("ix_deadline_events_milestone_id", "deadline_events", ["milestone_id"], unique=False)
    op.create_index("ix_deadline_events_due_at", "deadline_events", ["due_at"], unique=False)

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("job_type", sa.String(length=80), nullable=False),
        sa.Column("status", job_status_enum, nullable=False),
        sa.Column("input_payload", sa.JSON(), nullable=False),
        sa.Column("output_payload", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_processing_jobs_user_id", "processing_jobs", ["user_id"], unique=False)
    op.create_index("ix_processing_jobs_job_type", "processing_jobs", ["job_type"], unique=False)
    op.create_index("ix_processing_jobs_status", "processing_jobs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_processing_jobs_status", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_job_type", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_user_id", table_name="processing_jobs")
    op.drop_table("processing_jobs")

    op.drop_index("ix_deadline_events_due_at", table_name="deadline_events")
    op.drop_index("ix_deadline_events_milestone_id", table_name="deadline_events")
    op.drop_index("ix_deadline_events_task_id", table_name="deadline_events")
    op.drop_index("ix_deadline_events_user_id", table_name="deadline_events")
    op.drop_table("deadline_events")

    op.drop_index("ix_daily_summary_snapshots_snapshot_date", table_name="daily_summary_snapshots")
    op.drop_index("ix_daily_summary_snapshots_user_id", table_name="daily_summary_snapshots")
    op.drop_table("daily_summary_snapshots")

    op.drop_index("ix_planning_notes_created_at", table_name="planning_notes")
    op.drop_index("ix_planning_notes_related_task_id", table_name="planning_notes")
    op.drop_index("ix_planning_notes_note_type", table_name="planning_notes")
    op.drop_index("ix_planning_notes_user_id", table_name="planning_notes")
    op.drop_table("planning_notes")

    op.drop_index("ix_extracted_artifacts_created_task_id", table_name="extracted_artifacts")
    op.drop_index("ix_extracted_artifacts_due_at", table_name="extracted_artifacts")
    op.drop_index("ix_extracted_artifacts_source_attachment_id", table_name="extracted_artifacts")
    op.drop_index("ix_extracted_artifacts_user_id", table_name="extracted_artifacts")
    op.drop_table("extracted_artifacts")

    op.drop_index("ix_attachments_sha256", table_name="attachments")
    op.drop_index("ix_attachments_external_media_id", table_name="attachments")
    op.drop_index("ix_attachments_message_id", table_name="attachments")
    op.drop_index("ix_attachments_user_id", table_name="attachments")
    op.drop_table("attachments")

    op.drop_index("ix_schedule_blocks_ends_at", table_name="schedule_blocks")
    op.drop_index("ix_schedule_blocks_starts_at", table_name="schedule_blocks")
    op.drop_index("ix_schedule_blocks_block_type", table_name="schedule_blocks")
    op.drop_index("ix_schedule_blocks_user_id", table_name="schedule_blocks")
    op.drop_table("schedule_blocks")

    op.drop_index("ix_reminders_status", table_name="reminders")
    op.drop_index("ix_reminders_scheduled_for", table_name="reminders")
    op.drop_index("ix_reminders_task_id", table_name="reminders")
    op.drop_index("ix_reminders_user_id", table_name="reminders")
    op.drop_table("reminders")

    op.drop_index("ix_task_dependencies_successor_task_id", table_name="task_dependencies")
    op.drop_index("ix_task_dependencies_predecessor_task_id", table_name="task_dependencies")
    op.drop_index("ix_task_dependencies_user_id", table_name="task_dependencies")
    op.drop_table("task_dependencies")

    op.drop_index("ix_tasks_created_at", table_name="tasks")
    op.drop_index("ix_tasks_soft_deadline_at", table_name="tasks")
    op.drop_index("ix_tasks_deadline_at", table_name="tasks")
    op.drop_index("ix_tasks_status", table_name="tasks")
    op.drop_index("ix_tasks_title", table_name="tasks")
    op.drop_index("ix_tasks_parent_task_id", table_name="tasks")
    op.drop_index("ix_tasks_project_id", table_name="tasks")
    op.drop_index("ix_tasks_user_id", table_name="tasks")
    op.drop_table("tasks")

    op.drop_index("ix_milestones_due_at", table_name="milestones")
    op.drop_index("ix_milestones_project_id", table_name="milestones")
    op.drop_index("ix_milestones_user_id", table_name="milestones")
    op.drop_table("milestones")

    op.drop_index("ix_projects_title", table_name="projects")
    op.drop_index("ix_projects_user_id", table_name="projects")
    op.drop_table("projects")

    op.drop_index("ix_conversation_messages_created_at", table_name="conversation_messages")
    op.drop_index("ix_conversation_messages_direction", table_name="conversation_messages")
    op.drop_index("ix_conversation_messages_external_id", table_name="conversation_messages")
    op.drop_index("ix_conversation_messages_user_id", table_name="conversation_messages")
    op.drop_table("conversation_messages")

    op.drop_index("ix_user_profiles_user_id", table_name="user_profiles")
    op.drop_table("user_profiles")

    op.drop_index("ix_users_phone_number", table_name="users")
    op.drop_table("users")

    profile_style_enum.drop(op.get_bind(), checkfirst=True)
    job_status_enum.drop(op.get_bind(), checkfirst=True)
    reminder_status_enum.drop(op.get_bind(), checkfirst=True)
    task_status_enum.drop(op.get_bind(), checkfirst=True)
    message_direction_enum.drop(op.get_bind(), checkfirst=True)
