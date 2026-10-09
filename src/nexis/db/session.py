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


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a local .env into os.environ (existing vars win).

    Lets `cp .env.example .env` configure the API without touching the shell.
    The file is gitignored; secrets never belong in the repository.
    """
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def database_url() -> str:
    load_dotenv()
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
