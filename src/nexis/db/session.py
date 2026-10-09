"""Engine and session factory. One env var picks the database.

NEXIS_DATABASE_URL examples:
    postgresql+psycopg://nexis:nexis@localhost:5432/nexis   (target)
    sqlite:///data/nexis.db                                  (default, no server)
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from nexis.db.models import Base

DEFAULT_URL = "sqlite:///data/nexis.db"


def database_url() -> str:
    return os.environ.get("NEXIS_DATABASE_URL", DEFAULT_URL)


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    engine = create_engine(url, future=True, pool_pre_ping=True)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")  # evidence must reference a real alert
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


def init_db(url: str | None = None, drop: bool = False) -> Engine:
    engine = get_engine(url)
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """One unit of work: commit on success, roll back everything on error.

    This is what makes rule 6 hold: an alert and its evidence are added in the
    same scope, so either both are committed or neither is.
    """
    session = sessionmaker(bind=get_engine(url), expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
