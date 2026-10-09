"""Relational schema: transactions, alerts, evidence, rings, drift, model runs.

Implements Concept Mastery §17.1 (core schema), §17.1.1 (two timestamps:
occurred_at and ingested_at), §17.2 (indexes matching the query patterns) and
§17.3 (transactions and isolation: an alert and its evidence commit together).

PostgreSQL is the target (`NEXIS_DATABASE_URL=postgresql+psycopg://...`);
SQLite runs the same schema for tests and single-machine demos. JSON columns map
to JSONB on PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    tx_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    # §17.1.1: when the system learned of it. IBM AML records only occurrence,
    # so the replay sets ingested_at = occurred_at and says so in the dataset card.
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    src_account: Mapped[str] = mapped_column(String(40), nullable=False)
    dst_account: Mapped[str] = mapped_column(String(40), nullable=False)
    amount_usd: Mapped[float] = mapped_column(Float, nullable=False)
    amount_paid: Mapped[float] = mapped_column(Float, nullable=False)
    pay_currency: Mapped[str] = mapped_column(String(32), nullable=False)
    payment_format: Mapped[str] = mapped_column(String(32), nullable=False)
    is_cross_bank: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_cross_currency: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        Index("idx_tx_occurred", "occurred_at"),
        # Neighbourhood assembly (§17.2.1): composite, time-ordered.
        Index("idx_tx_src_time", "src_account", "occurred_at"),
        Index("idx_tx_dst_time", "dst_account", "occurred_at"),
    )


class Alert(Base):
    __tablename__ = "alerts"

    alert_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tx_id: Mapped[str] = mapped_column(ForeignKey("transactions.tx_id"), nullable=False, unique=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    disposition_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    disposed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    top_reasons: Mapped[Any] = mapped_column(JSONType, nullable=False, default=list)

    evidence: Mapped[Evidence] = relationship(back_populates="alert", uselist=False)
    transaction: Mapped[Transaction] = relationship()

    __table_args__ = (Index("idx_alerts_status_score", "status", "risk_score"),)


class Evidence(Base):
    __tablename__ = "evidence"

    evidence_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[int] = mapped_column(
        ForeignKey("alerts.alert_id", ondelete="CASCADE"), nullable=False, unique=True
    )
    packet: Mapped[Any] = mapped_column(JSONType, nullable=False)
    source_ids: Mapped[Any] = mapped_column(JSONType, nullable=False)

    alert: Mapped[Alert] = relationship(back_populates="evidence")


class Ring(Base):
    __tablename__ = "rings"

    ring_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    members: Mapped[Any] = mapped_column(JSONType, nullable=False)
    tx_ids: Mapped[Any] = mapped_column(JSONType, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class DriftWindow(Base):
    __tablename__ = "drift_windows"

    window_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    report: Mapped[Any] = mapped_column(JSONType, nullable=False)


class DriftEvent(Base):
    __tablename__ = "drift_events"

    drift_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)


class ModelRun(Base):
    __tablename__ = "model_runs"

    model_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    code_commit: Mapped[str] = mapped_column(String(40), nullable=False)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)
    train_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    train_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    hyperparams: Mapped[Any] = mapped_column(JSONType, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[Any] = mapped_column(JSONType, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_serving: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
