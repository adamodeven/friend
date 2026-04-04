from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.db.models import PlanningNote, Reminder, ReminderStatus, ScheduleBlock, Task, TaskDependency, TaskStatus, User
from app.db.repositories.message_repo import create_message
from app.db.repositories.task_repo import create_task, create_task_dependency
from app.domain.state_engine import StateEngine
from app.db.models import MessageDirection
from app.schemas.intent import ExtractedTask, IntentResult


def test_add_task_creates_task_and_deadline_event(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(
        intent="add_task",
        confidence=0.9,
        task=ExtractedTask(title="Finish CAD", deadline_text="tomorrow night"),
    )
    outcome = engine.apply_intent(db_session, user=user, intent=intent, raw_text="finish cad tomorrow night")
    db_session.commit()

    task = db_session.execute(select(Task).where(Task.title == "Finish CAD")).scalars().first()
    assert task is not None
    assert task.next_step is not None
    assert outcome.response_goal == "acknowledge_new_task"


def test_add_task_persists_multiple_tasks_subtasks_and_dependencies(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()

    intent = IntentResult(
        intent="add_task",
        confidence=0.95,
        tasks=[
            ExtractedTask(
                title="Send recruiter email",
                blockers=["need to fix portfolio website first"],
                subtasks=[ExtractedTask(title="Export work sample PDF", priority=4)],
            ),
            ExtractedTask(title="Finish CAD", priority=5),
        ],
    )

    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="need to send recruiter email, export work sample pdf for it, and finish cad",
    )
    db_session.commit()

    email_task = db_session.execute(select(Task).where(Task.title == "Send recruiter email")).scalars().one()
    subtask = db_session.execute(select(Task).where(Task.title == "Export work sample PDF")).scalars().one()
    finish_cad = db_session.execute(select(Task).where(Task.title == "Finish CAD")).scalars().one()
    prerequisite = db_session.execute(select(Task).where(Task.title == "Fix portfolio website")).scalars().one()
    dependencies = db_session.execute(
        select(TaskDependency).where(TaskDependency.successor_task_id == email_task.id)
    ).scalars().all()

    assert finish_cad is not None
    assert subtask.parent_task_id == email_task.id
    assert email_task.status == TaskStatus.blocked
    assert {dependency.predecessor_task_id for dependency in dependencies} == {subtask.id, prerequisite.id}
    assert outcome.is_multi_task_turn is True
    assert any("i'd start with" in fact.lower() for fact in outcome.key_facts_to_include)
    assert any("broke one of those into 1 smaller step" in fact for fact in outcome.key_facts_to_include)
    assert any("dependency" in fact for fact in outcome.key_facts_to_include)


def test_context_signal_creates_schedule_block(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(intent="context_signal", confidence=0.8, context_signal="in class rn")
    engine.apply_intent(db_session, user=user, intent=intent, raw_text="in class rn")
    db_session.commit()

    block = db_session.execute(select(ScheduleBlock)).scalars().first()
    assert block is not None
    assert block.block_type == "in_class"


def test_add_task_with_context_signal_captures_placeholder_and_backs_off(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()
    intent = IntentResult(
        intent="add_task",
        confidence=0.84,
        context_signal="prof just dropped another assignment and i'm in class rn",
        task=ExtractedTask(
            title="New assignment from professor",
            confidence=0.56,
            next_step="send me the assignment details once you're free",
        ),
    )
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="prof just dropped another assignment and i'm in class rn",
    )
    db_session.commit()

    task = db_session.execute(select(Task).where(Task.title == "New assignment from professor")).scalars().first()
    block = db_session.execute(select(ScheduleBlock).order_by(ScheduleBlock.created_at.desc())).scalars().first()

    assert task is not None
    assert block is not None
    assert block.block_type == "in_class"
    assert outcome.should_push_for_action is False
    assert outcome.suggested_next_step is None
    assert outcome.should_ask_question is True
    assert outcome.question_if_needed is not None
    assert any("class first" in fact for fact in outcome.key_facts_to_include)


def test_assignment_detail_followup_reuses_and_refines_existing_placeholder_task(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()
    first = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.84,
            context_signal="prof just dropped another assignment and i'm in class rn",
            task=ExtractedTask(
                title="New assignment from professor",
                confidence=0.56,
                next_step="send me the assignment details once you're free",
            ),
        ),
        raw_text="prof just dropped another assignment and i'm in class rn",
    )
    assert first.response_goal == "acknowledge_new_task"

    followup = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.68,
            task=ExtractedTask(
                title="New assignment from studio",
                deadline_text="tuesday night",
            ),
        ),
        raw_text="actually its for studio and due tuesday night",
    )
    db_session.commit()

    tasks = db_session.execute(select(Task).where(Task.user_id == user.id)).scalars().all()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.title == "New assignment from studio"
    assert task.deadline_source_phrase == "tuesday night"
    assert followup.response_goal == "acknowledge_new_task"
    assert any("studio's in for tuesday night" in fact.lower() for fact in followup.key_facts_to_include)


def test_status_query_meta_gets_direct_explanation(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(intent="status_query", confidence=0.9)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="are these canned responses or live ai generated?",
    )
    assert outcome.response_goal == "answer_question"
    assert any("live-generated" in fact or "live" in fact for fact in outcome.key_facts_to_include)


def test_status_query_live_now_includes_direct_status_fact(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(intent="status_query", confidence=0.9)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="are you actually live now?",
    )
    assert any("yes, i'm live right now" in fact for fact in outcome.key_facts_to_include)


def test_general_progress_message_maps_to_progress_response_goal(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(intent="general_chat", confidence=0.7)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="lowk making good progress now",
    )
    assert outcome.response_goal == "react_to_progress"


def test_single_add_task_does_not_force_checkpoint_question(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(
        intent="add_task",
        confidence=0.9,
        task=ExtractedTask(title="Submit scout job application", deadline_text="tomorrow morning"),
    )
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="tomorrow morning i need to submit my scout job application",
    )
    assert outcome.response_goal == "acknowledge_new_task"
    assert outcome.should_ask_question is False
    assert outcome.question_if_needed is None
    assert "final proofread" in (outcome.suggested_next_step or "").lower()


def test_quick_message_windowed_task_does_not_push_fake_prep_step(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()
    tomorrow_morning = (datetime.now(tz=ZoneInfo(user.timezone)) + timedelta(days=1)).replace(
        hour=8,
        minute=0,
        second=0,
        microsecond=0,
    )
    intent = IntentResult(
        intent="add_task",
        confidence=0.9,
        task=ExtractedTask(
            title="Text roommate back",
            deadline_text="tomorrow morning",
            start_after=tomorrow_morning,
            action_kind="quick_message",
            next_step=None,
        ),
    )
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="need to text my roommate tomorrow morning",
    )
    assert outcome.response_goal == "acknowledge_new_task"
    assert outcome.should_push_for_action is False
    assert outcome.suggested_next_step is None
    assert any("i'll remind you" in fact.lower() for fact in outcome.key_facts_to_include)


def test_new_task_links_to_source_message_for_followup_context(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    inbound = create_message(
        db_session,
        user_id=user.id,
        direction=MessageDirection.inbound,
        body="dont let me forget to email the scout recruiter tmr morning",
        external_id="SM_TEST_FOLLOWUP",
    )
    engine = StateEngine()
    intent = IntentResult(
        intent="add_task",
        confidence=0.9,
        task=ExtractedTask(
            title="Email the scout recruiter",
            deadline_text="tomorrow morning",
            action_kind="quick_message",
        ),
    )
    engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="dont let me forget to email the scout recruiter tmr morning",
        source_message_id=inbound.id,
    )
    db_session.commit()

    stored = db_session.execute(select(Task).where(Task.title == "Email the scout recruiter")).scalars().one()
    assert stored.source_message_id == inbound.id


def test_time_only_followup_updates_recent_relevant_task_instead_of_creating_new_one(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    first_inbound = create_message(
        db_session,
        user_id=user.id,
        direction=MessageDirection.inbound,
        body="dont let me forget to email the scout recruiter tmr morning",
        external_id="SM_TEST_FIRST",
    )
    engine = StateEngine()
    add_intent = IntentResult(
        intent="add_task",
        confidence=0.92,
        task=ExtractedTask(
            title="Email the scout recruiter",
            deadline_text="tomorrow morning",
            action_kind="quick_message",
        ),
    )
    engine.apply_intent(
        db_session,
        user=user,
        intent=add_intent,
        raw_text="dont let me forget to email the scout recruiter tmr morning",
        source_message_id=first_inbound.id,
    )
    second_inbound = create_message(
        db_session,
        user_id=user.id,
        direction=MessageDirection.inbound,
        body="actually monday morning",
        external_id="SM_TEST_SECOND",
    )
    update_intent = IntentResult(
        intent="update_task",
        confidence=0.9,
        time_reference="monday morning",
        task_updates={"action": "reschedule"},
    )
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=update_intent,
        raw_text="actually monday morning",
        source_message_id=second_inbound.id,
    )
    db_session.commit()

    tasks = db_session.execute(select(Task).where(Task.user_id == user.id)).scalars().all()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.deadline_source_phrase == "monday morning"
    assert task.source_message_id == second_inbound.id
    assert outcome.response_goal == "confirm_update"
    assert outcome.key_facts_to_include == ["bet, switched it"]


def test_day_only_followup_preserves_existing_part_of_day_window(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    first_inbound = create_message(
        db_session,
        user_id=user.id,
        direction=MessageDirection.inbound,
        body="dont let me forget to email the scout recruiter tmr morning",
        external_id="SM_TEST_DAY_ONLY_FIRST",
    )
    engine = StateEngine()
    add_intent = IntentResult(
        intent="add_task",
        confidence=0.92,
        task=ExtractedTask(
            title="Email the scout recruiter",
            deadline_text="tomorrow morning",
            action_kind="quick_message",
        ),
    )
    engine.apply_intent(
        db_session,
        user=user,
        intent=add_intent,
        raw_text="dont let me forget to email the scout recruiter tmr morning",
        source_message_id=first_inbound.id,
    )
    second_inbound = create_message(
        db_session,
        user_id=user.id,
        direction=MessageDirection.inbound,
        body="actually do monday instead",
        external_id="SM_TEST_DAY_ONLY_SECOND",
    )
    update_intent = IntentResult(
        intent="update_task",
        confidence=0.9,
        time_reference="monday",
        task_updates={"action": "reschedule"},
    )
    engine.apply_intent(
        db_session,
        user=user,
        intent=update_intent,
        raw_text="actually do monday instead",
        source_message_id=second_inbound.id,
    )
    db_session.commit()

    task = db_session.execute(select(Task).where(Task.user_id == user.id)).scalars().one()
    assert task.deadline_source_phrase == "monday morning"
    assert task.deadline_granularity == "part_of_day"
    assert task.start_after is not None


def test_named_task_reschedule_updates_existing_deadline_from_embedded_phrase(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()
    first_inbound = create_message(
        db_session,
        user_id=user.id,
        direction=MessageDirection.inbound,
        body="update my portfolio bullets by wednesday",
        external_id="SM_TEST_PORTFOLIO_FIRST",
    )
    engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.9,
            task=ExtractedTask(title="Update my portfolio bullets", deadline_text="wednesday"),
        ),
        raw_text="update my portfolio bullets by wednesday",
        source_message_id=first_inbound.id,
    )
    second_inbound = create_message(
        db_session,
        user_id=user.id,
        direction=MessageDirection.inbound,
        body="actually portfolio bullets can wait till friday",
        external_id="SM_TEST_PORTFOLIO_SECOND",
    )
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="update_task",
            confidence=0.88,
            time_reference="friday",
            task_updates={"action": "reschedule"},
        ),
        raw_text="actually portfolio bullets can wait till friday",
        source_message_id=second_inbound.id,
    )
    db_session.commit()

    task = db_session.execute(select(Task).where(Task.user_id == user.id)).scalars().one()
    assert task.deadline_source_phrase == "friday"
    assert task.source_message_id == second_inbound.id
    assert outcome.key_facts_to_include == ["bet, switched it"]


def test_default_next_step_does_not_repeat_submit_for_submit_titles():
    step = StateEngine._default_next_step("Submit my scout job application")
    lowered = step.lower()
    assert "submit submit" not in lowered
    assert lowered.startswith("do a final proofread, then submit my scout job application")


def test_add_task_with_later_time_keeps_it_soft_without_immediate_clarification(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(
        intent="add_task",
        confidence=0.9,
        time_reference="later",
        time_confidence=0.35,
        needs_clarification=False,
        task=ExtractedTask(title="Send email update", deadline_text="later"),
    )
    outcome = engine.apply_intent(db_session, user=user, intent=intent, raw_text="need to send that email later")
    assert outcome.response_goal == "acknowledge_new_task"
    assert outcome.should_ask_question is False
    assert outcome.question_if_needed is None
    assert all("circle back around" not in fact for fact in outcome.key_facts_to_include)


def test_bulk_clear_archives_active_tasks_and_pending_reminders(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    task_a = Task(user_id=user.id, title="Task A")
    task_b = Task(user_id=user.id, title="Task B", status=TaskStatus.blocked)
    db_session.add_all([task_a, task_b])
    db_session.flush()
    db_session.add(
        Reminder(
            user_id=user.id,
            task_id=task_a.id,
            status=ReminderStatus.pending,
            scheduled_for=datetime.now(tz=ZoneInfo(user.timezone)),
        )
    )
    db_session.commit()

    engine = StateEngine()
    intent = IntentResult(intent="update_task", confidence=0.95, task_updates={"bulk_action": "clear_active_tasks"})
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="clear all tasks",
    )
    db_session.commit()

    refreshed = db_session.execute(select(Task).where(Task.user_id == user.id)).scalars().all()
    assert all(task.status == TaskStatus.archived for task in refreshed)
    reminder = db_session.execute(select(Reminder).where(Reminder.user_id == user.id)).scalars().first()
    assert reminder is not None
    assert reminder.status == ReminderStatus.skipped
    assert outcome.response_goal == "confirm_update"
    assert any("cleared everything out" in fact for fact in outcome.key_facts_to_include)


def test_timeline_query_tomorrow_morning_routes_to_specific_window(db_session):
    user = db_session.execute(select(User)).scalars().first()
    engine = StateEngine()
    intent = IntentResult(intent="timeline_query", confidence=0.9)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="what do i need to get done tomorrow morning",
    )
    assert outcome.response_goal == "timeline_summary"
    assert any("tomorrow morning" in fact.lower() for fact in outcome.key_facts_to_include)


def test_project_plan_query_returns_project_view_without_creating_task(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    create_task(db_session, user_id=user.id, title="Finish the CAD for the enclosure", priority=5)
    create_task(db_session, user_id=user.id, title="Send enclosure update email", priority=4)
    db_session.commit()

    engine = StateEngine()
    intent = IntentResult(intent="timeline_query", confidence=0.9)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="what's the plan for the enclosure project?",
    )

    assert outcome.response_goal == "timeline_summary"
    summary = " ".join(outcome.key_facts_to_include).lower()
    assert "enclosure" in summary
    assert "finish the cad for the enclosure" in summary
    assert "captured:" not in summary


def test_update_task_blocker_creates_prerequisite_and_dependency(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    create_task(db_session, user_id=user.id, title="Send recruiter email", priority=4)
    db_session.commit()

    engine = StateEngine()
    intent = IntentResult(
        intent="update_task",
        confidence=0.88,
        blockers=["need to fix website first"],
        task_updates={"status": "blocked"},
    )
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="send recruiter email is blocked because i need to fix website first",
    )
    db_session.commit()

    blocked_task = db_session.execute(select(Task).where(Task.title == "Send recruiter email")).scalars().one()
    prerequisite = db_session.execute(select(Task).where(Task.title == "Fix website")).scalars().one()
    dependency = db_session.execute(
        select(TaskDependency).where(
            TaskDependency.predecessor_task_id == prerequisite.id,
            TaskDependency.successor_task_id == blocked_task.id,
        )
    ).scalars().first()

    assert blocked_task.status == TaskStatus.blocked
    assert dependency is not None
    assert outcome.response_goal == "replan_blocker"
    assert outcome.should_ask_question is False
    assert "fix website" in (outcome.suggested_next_step or "").lower()


def test_archive_task_action_really_archives_task_and_skips_reminders(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    task = create_task(db_session, user_id=user.id, title="Fix website", priority=4)
    blocked = create_task(
        db_session,
        user_id=user.id,
        title="Send recruiter email",
        priority=5,
        blocked_reason="need website fixed first",
        blocked_at=datetime.now(tz=ZoneInfo(user.timezone)),
    )
    blocked.status = TaskStatus.blocked
    create_task_dependency(
        db_session,
        user_id=user.id,
        predecessor_task_id=task.id,
        successor_task_id=blocked.id,
    )
    db_session.add(
        Reminder(
            user_id=user.id,
            task_id=task.id,
            status=ReminderStatus.pending,
            scheduled_for=datetime.now(tz=ZoneInfo(user.timezone)),
        )
    )
    db_session.commit()

    engine = StateEngine()
    intent = IntentResult(intent="update_task", confidence=0.95, task_updates={"action": "archive"})
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="delete the website task",
    )
    db_session.commit()

    refreshed_task = db_session.get(Task, task.id)
    refreshed_blocked = db_session.get(Task, blocked.id)
    reminder = db_session.execute(select(Reminder).where(Reminder.task_id == task.id)).scalars().one()

    assert refreshed_task is not None
    assert refreshed_task.status == TaskStatus.archived
    assert refreshed_blocked is not None
    assert refreshed_blocked.status == TaskStatus.active
    assert reminder.status == ReminderStatus.skipped
    assert any(fact == "bet, i took fix website out" for fact in outcome.key_facts_to_include)


def test_complete_task_adds_more_human_followup_for_next_work(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    finished = create_task(db_session, user_id=user.id, title="Pay rent", priority=4)
    next_task = create_task(
        db_session,
        user_id=user.id,
        title="Finish enclosure CAD",
        priority=5,
        next_step="finish the enclosure cad",
    )
    db_session.commit()

    engine = StateEngine()
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(intent="complete_task", confidence=0.9),
        raw_text="bet rent is done",
    )

    assert outcome.response_goal == "react_to_progress"
    assert outcome.key_facts_to_include[0] == "bet"
    assert any("if you could also get the enclosure cad done next that'd be huge" == fact for fact in outcome.key_facts_to_include)


def test_stack_sequence_falls_back_to_human_hold_fact_when_followup_is_future_windowed(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()
    tomorrow_morning = (datetime.now(tz=ZoneInfo(user.timezone)) + timedelta(days=1)).replace(
        hour=8,
        minute=0,
        second=0,
        microsecond=0,
    )
    tomorrow_night = (datetime.now(tz=ZoneInfo(user.timezone)) + timedelta(days=1)).replace(
        hour=21,
        minute=0,
        second=0,
        microsecond=0,
    )
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.9,
            tasks=[
                ExtractedTask(title="Finish enclosure CAD", deadline_at=tomorrow_night, action_kind="project_chunk"),
                ExtractedTask(
                    title="Text roommate back",
                    deadline_text="tomorrow morning",
                    start_after=tomorrow_morning,
                    action_kind="quick_message",
                ),
                ExtractedTask(
                    title="Send scout followup",
                    deadline_text="monday morning",
                    start_after=tomorrow_morning + timedelta(days=2),
                    action_kind="quick_message",
                ),
            ],
        ),
        raw_text="finish the enclosure cad by tomorrow night and text my roommate tomorrow morning and send scout followup monday morning",
    )
    assert any(
        "you can just text roommate back tomorrow morning" in fact.lower()
        or "text roommate back can happen tomorrow morning" in fact.lower()
        for fact in outcome.key_facts_to_include
    )
    assert outcome.should_ask_question is False


def test_update_task_without_concrete_change_asks_clarifier_instead_of_bluffing(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    create_task(db_session, user_id=user.id, title="Finish CAD", priority=5)
    db_session.commit()

    engine = StateEngine()
    intent = IntentResult(intent="update_task", confidence=0.8)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="about the cad task",
    )

    assert outcome.response_goal == "confirm_update"
    assert outcome.should_ask_question is True
    assert outcome.question_if_needed is not None
    assert "what do you want me to change" in outcome.question_if_needed.lower()


def test_complete_task_unblocks_successor_and_surfaces_next_action(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    prerequisite = create_task(db_session, user_id=user.id, title="Export resume PDF", priority=4)
    successor = create_task(
        db_session,
        user_id=user.id,
        title="Send recruiter email",
        priority=5,
        blocked_reason="need resume pdf first",
        blocked_at=datetime.now(tz=ZoneInfo(user.timezone)),
    )
    successor.status = TaskStatus.blocked
    create_task_dependency(
        db_session,
        user_id=user.id,
        predecessor_task_id=prerequisite.id,
        successor_task_id=successor.id,
    )
    db_session.commit()

    engine = StateEngine()
    intent = IntentResult(intent="complete_task", confidence=0.92)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="finished exporting the resume pdf",
    )
    db_session.commit()

    refreshed_successor = db_session.execute(select(Task).where(Task.id == successor.id)).scalars().one()
    assert refreshed_successor.status == TaskStatus.active
    assert any("that opens up Send recruiter email" == fact for fact in outcome.key_facts_to_include)
    assert "send recruiter email" in (outcome.suggested_next_step or "").lower()


def test_complete_task_with_uncertain_match_asks_softer_followup(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(intent="complete_task", confidence=0.9),
        raw_text="bet website fix is done",
    )

    assert outcome.response_goal == "react_to_progress"
    assert outcome.key_facts_to_include == ["bet"]
    assert outcome.should_ask_question is True
    assert outcome.question_if_needed == "which one was that?"


def test_multi_task_add_with_limited_timing_asks_one_prioritization_question(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()
    intent = IntentResult(
        intent="add_task",
        confidence=0.91,
        tasks=[
            ExtractedTask(title="Clean up portfolio bullets"),
            ExtractedTask(title="Pay rent"),
            ExtractedTask(title="Check in with team lead"),
        ],
    )

    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="tonight i need to clean up portfolio bullets, pay rent, and check in with my team lead",
    )

    assert outcome.response_goal == "acknowledge_new_task"
    assert outcome.should_ask_question is True
    assert outcome.question_if_needed == "which one gets ugly first if it slips?"


def test_multi_task_add_with_clear_mixed_timing_sequences_without_kicking_back_priority(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    tz = ZoneInfo(user.timezone)
    now = datetime.now(tz=tz)
    tomorrow_night = (now + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
    tonight = now.replace(hour=21, minute=0, second=0, microsecond=0)
    tomorrow_morning = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    monday_morning = (now + timedelta(days=((7 - now.weekday()) % 7 or 7))).replace(hour=9, minute=0, second=0, microsecond=0)

    engine = StateEngine()
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.94,
            tasks=[
                ExtractedTask(title="Finish the enclosure CAD", deadline_at=tomorrow_night, action_kind="project_chunk"),
                ExtractedTask(title="Send the scout followup", deadline_at=monday_morning, start_after=monday_morning, action_kind="quick_message"),
                ExtractedTask(title="Pay rent", deadline_at=tonight, action_kind="quick_admin"),
                ExtractedTask(title="Text my roommate", deadline_at=tomorrow_morning, start_after=tomorrow_morning, action_kind="quick_message"),
            ],
        ),
        raw_text="i need to finish the enclosure cad by tomorrow night and send the scout followup monday morning and pay rent tonight and text my roommate tomorrow morning",
    )

    assert outcome.response_goal == "acknowledge_new_task"
    assert outcome.should_ask_question is False
    assert any("i'd start with the enclosure cad" in fact.lower() for fact in outcome.key_facts_to_include)
    assert any("pay rent can wait till tonight" in fact.lower() or "then just pay rent" in fact.lower() for fact in outcome.key_facts_to_include)
    assert not any("deadline coming up" in fact.lower() for fact in outcome.key_facts_to_include)
    assert outcome.should_push_for_action is True
    assert "enclosure cad" in (outcome.suggested_next_step or "").lower()


def test_assignment_detail_followup_with_due_date_does_not_keep_asking_for_details(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    engine = StateEngine()

    first_outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.84,
            context_signal="prof just dropped another assignment and im in class rn",
            task=ExtractedTask(title="New assignment from studio"),
        ),
        raw_text="prof just dropped another assignment and im in class rn",
    )
    assert first_outcome.should_ask_question is True

    second_outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.88,
            task=ExtractedTask(title="New assignment from studio", deadline_text="tuesday night"),
        ),
        raw_text="actually its for studio and due tuesday night",
    )

    assert second_outcome.should_ask_question is False
    assert second_outcome.should_push_for_action is False
    lowered_facts = [fact.lower() for fact in second_outcome.key_facts_to_include]
    assert sum("due" in fact or "deadline" in fact for fact in lowered_facts) == 1
    assert not any("assignment details" in fact for fact in lowered_facts)
    assert second_outcome.suggested_next_step is None


def test_repeat_task_mention_reuses_existing_active_task(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    existing = create_task(db_session, user_id=user.id, title="Pay rent", priority=2)
    db_session.commit()

    engine = StateEngine()
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="add_task",
            confidence=0.9,
            task=ExtractedTask(title="Pay rent", deadline_text="tonight", action_kind="quick_admin"),
        ),
        raw_text="need to pay rent tonight",
    )
    db_session.commit()

    tasks = db_session.execute(select(Task).where(Task.title == "Pay rent")).scalars().all()
    assert len(tasks) == 1
    assert tasks[0].id == existing.id
    assert outcome.response_goal == "acknowledge_new_task"


def test_blocker_update_targets_blocked_work_not_blocker_task_itself(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    website = create_task(db_session, user_id=user.id, title="Fix the website", priority=3)
    cad = create_task(db_session, user_id=user.id, title="Finish the enclosure cad", priority=4)
    db_session.commit()

    engine = StateEngine()
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=IntentResult(
            intent="update_task",
            confidence=0.86,
            blockers=["i need to fix the website first"],
            task_updates={"status": "blocked"},
        ),
        raw_text="i keep getting distracted because i need to fix the website first",
    )
    db_session.commit()

    website = db_session.get(Task, website.id)
    cad = db_session.get(Task, cad.id)
    assert website is not None and cad is not None
    assert website.status == TaskStatus.active
    assert cad.status == TaskStatus.blocked
    assert cad.blocked_reason is not None
    assert "fix the website" in cad.blocked_reason.lower()
    assert outcome.response_goal == "replan_blocker"
    assert outcome.should_ask_question is False
    assert "fix the website" in (outcome.suggested_next_step or "").lower()


def test_reflection_records_slip_reason_on_task_and_memory(db_session):
    user = db_session.execute(select(User)).scalars().first()
    assert user is not None
    create_task(db_session, user_id=user.id, title="Finish CAD", priority=5)
    db_session.commit()

    engine = StateEngine()
    intent = IntentResult(intent="reflection", confidence=0.86)
    outcome = engine.apply_intent(
        db_session,
        user=user,
        intent=intent,
        raw_text="missed finish cad because i got distracted by another assignment",
    )
    db_session.commit()

    task = db_session.execute(select(Task).where(Task.title == "Finish CAD")).scalars().one()
    notes = db_session.execute(select(PlanningNote).order_by(PlanningNote.created_at.asc())).scalars().all()

    assert task.slip_count == 1
    assert task.last_slip_reason == "missed finish cad because i got distracted by another assignment"
    assert any(note.note_type == "slip_reason" and note.related_task_id == task.id for note in notes)
    assert any(note.note_type == "behavior_pattern" for note in notes)
    assert outcome.response_goal == "replan_blocker"
    assert "finish cad" in (outcome.suggested_next_step or "").lower()
