from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import ProfileStyle, User, UserProfile


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    user = User(phone_number="+15550001111", name="Test", timezone="America/New_York")
    session.add(user)
    session.flush()
    session.add(
        UserProfile(
            user_id=user.id,
            style=ProfileStyle.casual_cool,
            planning_preferences={},
            bio="",
        )
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()

