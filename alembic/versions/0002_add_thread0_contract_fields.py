"""add thread 0 contract fields

Revision ID: 0002_add_thread0_contract_fields
Revises: 0001_initial_schema
Create Date: 2026-04-03 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_add_thread0_contract_fields"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("user_profiles") as batch_op:
        batch_op.add_column(sa.Column("baseline_profile_json", sa.JSON(), nullable=True))

    op.execute(sa.text("UPDATE user_profiles SET baseline_profile_json = '{}' WHERE baseline_profile_json IS NULL"))

    with op.batch_alter_table("user_profiles") as batch_op:
        batch_op.alter_column("baseline_profile_json", existing_type=sa.JSON(), nullable=False)

    with op.batch_alter_table("tasks") as batch_op:
        batch_op.add_column(sa.Column("source_message_id", sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column("next_step", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("deadline_source_phrase", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("deadline_confidence", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("deadline_is_ambiguous", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("deadline_granularity", sa.String(length=30), nullable=True))
        batch_op.add_column(sa.Column("deadline_timezone", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("last_progress_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("blocked_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("blocker_details_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("slip_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("last_slipped_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("last_slip_reason", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("reminder_escalation_level", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("last_reminder_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("reminder_pause_until", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_foreign_key(
            "fk_tasks_source_message_id_conversation_messages",
            "conversation_messages",
            ["source_message_id"],
            ["id"],
        )
        batch_op.create_index("ix_tasks_source_message_id", ["source_message_id"], unique=False)

    op.execute(
        sa.text(
            """
            UPDATE tasks
            SET deadline_confidence = CASE
                    WHEN deadline_at IS NOT NULL OR soft_deadline_at IS NOT NULL THEN COALESCE(extraction_confidence, 0.0)
                    ELSE 0.0
                END,
                deadline_is_ambiguous = false,
                deadline_granularity = 'unknown',
                blocker_details_json = '{}',
                slip_count = 0,
                reminder_escalation_level = 0
            WHERE deadline_confidence IS NULL
               OR deadline_is_ambiguous IS NULL
               OR deadline_granularity IS NULL
               OR blocker_details_json IS NULL
               OR slip_count IS NULL
               OR reminder_escalation_level IS NULL
            """
        )
    )

    with op.batch_alter_table("tasks") as batch_op:
        batch_op.alter_column("deadline_confidence", existing_type=sa.Float(), nullable=False)
        batch_op.alter_column("deadline_is_ambiguous", existing_type=sa.Boolean(), nullable=False)
        batch_op.alter_column("deadline_granularity", existing_type=sa.String(length=30), nullable=False)
        batch_op.alter_column("blocker_details_json", existing_type=sa.JSON(), nullable=False)
        batch_op.alter_column("slip_count", existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column("reminder_escalation_level", existing_type=sa.Integer(), nullable=False)

    with op.batch_alter_table("task_dependencies") as batch_op:
        batch_op.add_column(sa.Column("metadata_json", sa.JSON(), nullable=True))

    op.execute(sa.text("UPDATE task_dependencies SET metadata_json = '{}' WHERE metadata_json IS NULL"))
    op.execute(
        sa.text(
            """
            DELETE FROM task_dependencies
            WHERE id IN (
                SELECT id
                FROM (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY user_id, predecessor_task_id, successor_task_id, dependency_type
                               ORDER BY created_at, id
                           ) AS row_num
                    FROM task_dependencies
                ) ranked
                WHERE row_num > 1
            )
            """
        )
    )

    with op.batch_alter_table("task_dependencies") as batch_op:
        batch_op.alter_column("metadata_json", existing_type=sa.JSON(), nullable=False)
        batch_op.create_unique_constraint(
            "uq_task_dependency_edge",
            ["user_id", "predecessor_task_id", "successor_task_id", "dependency_type"],
        )

    with op.batch_alter_table("reminders") as batch_op:
        batch_op.add_column(sa.Column("attempt_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("cooldown_until", sa.DateTime(timezone=True), nullable=True))

    op.execute(sa.text("UPDATE reminders SET attempt_count = 0 WHERE attempt_count IS NULL"))

    with op.batch_alter_table("reminders") as batch_op:
        batch_op.alter_column("attempt_count", existing_type=sa.Integer(), nullable=False)

    with op.batch_alter_table("deadline_events") as batch_op:
        batch_op.add_column(sa.Column("source_phrase", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("is_ambiguous", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("granularity", sa.String(length=30), nullable=True))
        batch_op.add_column(sa.Column("metadata_json", sa.JSON(), nullable=True))

    op.execute(
        sa.text(
            """
            UPDATE deadline_events
            SET is_ambiguous = false,
                granularity = 'unknown',
                metadata_json = '{}'
            WHERE is_ambiguous IS NULL
               OR granularity IS NULL
               OR metadata_json IS NULL
            """
        )
    )

    with op.batch_alter_table("deadline_events") as batch_op:
        batch_op.alter_column("is_ambiguous", existing_type=sa.Boolean(), nullable=False)
        batch_op.alter_column("granularity", existing_type=sa.String(length=30), nullable=False)
        batch_op.alter_column("metadata_json", existing_type=sa.JSON(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("deadline_events") as batch_op:
        batch_op.drop_column("metadata_json")
        batch_op.drop_column("granularity")
        batch_op.drop_column("is_ambiguous")
        batch_op.drop_column("source_phrase")

    with op.batch_alter_table("reminders") as batch_op:
        batch_op.drop_column("cooldown_until")
        batch_op.drop_column("attempt_count")

    with op.batch_alter_table("task_dependencies") as batch_op:
        batch_op.drop_constraint("uq_task_dependency_edge", type_="unique")
        batch_op.drop_column("metadata_json")

    with op.batch_alter_table("tasks") as batch_op:
        batch_op.drop_index("ix_tasks_source_message_id")
        batch_op.drop_constraint("fk_tasks_source_message_id_conversation_messages", type_="foreignkey")
        batch_op.drop_column("reminder_pause_until")
        batch_op.drop_column("last_reminder_at")
        batch_op.drop_column("reminder_escalation_level")
        batch_op.drop_column("last_slip_reason")
        batch_op.drop_column("last_slipped_at")
        batch_op.drop_column("slip_count")
        batch_op.drop_column("blocker_details_json")
        batch_op.drop_column("blocked_at")
        batch_op.drop_column("last_progress_at")
        batch_op.drop_column("started_at")
        batch_op.drop_column("deadline_timezone")
        batch_op.drop_column("deadline_granularity")
        batch_op.drop_column("deadline_is_ambiguous")
        batch_op.drop_column("deadline_confidence")
        batch_op.drop_column("deadline_source_phrase")
        batch_op.drop_column("next_step")
        batch_op.drop_column("source_message_id")

    with op.batch_alter_table("user_profiles") as batch_op:
        batch_op.drop_column("baseline_profile_json")
