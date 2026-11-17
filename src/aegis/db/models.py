"""知识库、父块、履约工单与检索审计的持久化契约。"""

from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from aegis.db.base import Base, TimestampMixin


class DocumentType(StrEnum):
    SKU_SPEC = "SKU_SPEC"
    AFTER_SALES = "AFTER_SALES"
    WAREHOUSE_SOP = "WAREHOUSE_SOP"
    FINANCE_RULE = "FINANCE_RULE"


class WorkOrderPriority(StrEnum):
    P0_EMERGENCY = "P0_EMERGENCY"
    P1_HIGH = "P1_HIGH"
    P2_NORMAL = "P2_NORMAL"


class BadcaseCategory(StrEnum):
    NONE = "NONE"
    MISSED_RECALL = "MISSED_RECALL"
    LOW_RANKING = "LOW_RANKING"
    HALLUCINATION = "HALLUCINATION"


class KBDocument(TimestampMixin, Base):
    __tablename__ = "kb_documents"
    __table_args__ = (
        Index("ix_kb_documents_acl_status", "dept_scope", "min_role_level", "status"),
        Index("ix_kb_documents_version_status", "index_version", "status"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"},
    )

    doc_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(255))
    doc_type: Mapped[DocumentType] = mapped_column(
        Enum(
            DocumentType,
            native_enum=False,
            length=32,
            create_constraint=True,
            validate_strings=True,
            name="document_type",
        )
    )
    dept_scope: Mapped[str] = mapped_column(String(64), index=True)
    min_role_level: Mapped[int] = mapped_column(Integer, index=True, default=1, server_default="1")
    sha256_hash: Mapped[str] = mapped_column(String(64), index=True)
    index_version: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE", server_default="ACTIVE")


class KBParentChunk(TimestampMixin, Base):
    __tablename__ = "kb_parent_chunks"
    __table_args__ = (
        Index("ix_kb_parent_chunks_doc_sku", "doc_id", "is_sku_table"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"},
    )

    parent_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    doc_id: Mapped[int] = mapped_column(
        ForeignKey("kb_documents.id", ondelete="RESTRICT"), index=True
    )
    header_path: Mapped[str] = mapped_column(String(512))
    content_text: Mapped[str] = mapped_column(Text)
    is_sku_table: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("0"))
    sku_rows_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class OpsWorkOrder(TimestampMixin, Base):
    __tablename__ = "ops_work_orders"
    __table_args__ = (
        Index("ix_ops_work_orders_status_created", "status", "created_at"),
        Index("ix_ops_work_orders_session_created", "session_id", "created_at"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"},
    )

    ticket_no: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    priority: Mapped[WorkOrderPriority] = mapped_column(
        Enum(
            WorkOrderPriority,
            native_enum=False,
            length=32,
            create_constraint=True,
            validate_strings=True,
            name="work_order_priority",
        )
    )
    trigger_source: Mapped[str] = mapped_column(String(64))
    slots_payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    completeness_score: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0")
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", server_default="PENDING")


class RAGAuditLog(TimestampMixin, Base):
    __tablename__ = "rag_audit_logs"
    __table_args__ = (
        Index("ix_rag_audit_logs_badcase_created", "badcase_category", "created_at"),
        Index("ix_rag_audit_logs_trace_created", "trace_id", "created_at"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"},
    )

    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    query_text: Mapped[str] = mapped_column(Text)
    recalled_parent_ids: Mapped[list[str]] = mapped_column(JSON)
    guardrail_action: Mapped[str] = mapped_column(String(64))
    badcase_category: Mapped[BadcaseCategory] = mapped_column(
        Enum(
            BadcaseCategory,
            native_enum=False,
            length=32,
            create_constraint=True,
            validate_strings=True,
            name="badcase_category",
        ),
        default=BadcaseCategory.NONE,
        server_default="NONE",
    )
    latency_ms: Mapped[int] = mapped_column(Integer)
    degraded_flag: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("0"))
    degraded_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)


class OpsTicketOutbox(TimestampMixin, Base):
    """与工单一起提交的派单事件，避免提交后进程退出导致事件永久丢失。"""

    __tablename__ = "ops_ticket_outbox"
    __table_args__ = (
        Index("ix_ops_ticket_outbox_pending", "published", "stream_name", "id"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"},
    )

    event_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("ops_work_orders.id", ondelete="CASCADE"), unique=True, index=True
    )
    stream_name: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict[str, str]] = mapped_column(JSON)
    published: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("0"))
    redis_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
