"""初始化知识文档、父块、履约工单和检索审计四张核心表。"""

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def audit_columns() -> list[sa.Column]:
    """冻结本版迁移的公共字段，不依赖随业务变化的 ORM。"""
    return [
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            mysql.DATETIME(fsp=6),
            server_default=sa.text("CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)"),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "kb_documents",
        *audit_columns(),
        sa.Column("doc_code", sa.String(64), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("doc_type", sa.String(32), nullable=False),
        sa.Column("dept_scope", sa.String(64), nullable=False),
        sa.Column("min_role_level", sa.Integer(), server_default="1", nullable=False),
        sa.Column("sha256_hash", sa.String(64), nullable=False),
        sa.Column("index_version", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), server_default="ACTIVE", nullable=False),
        sa.CheckConstraint(
            "doc_type IN ('SKU_SPEC', 'AFTER_SALES', 'WAREHOUSE_SOP', 'FINANCE_RULE')",
            name=op.f("ck_kb_documents_document_type"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_kb_documents")),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    for column in ("doc_code", "dept_scope", "min_role_level", "sha256_hash"):
        op.create_index(
            f"ix_kb_documents_{column}", "kb_documents", [column], unique=column == "doc_code"
        )
    op.create_index(
        "ix_kb_documents_acl_status", "kb_documents", ["dept_scope", "min_role_level", "status"]
    )
    op.create_index("ix_kb_documents_version_status", "kb_documents", ["index_version", "status"])

    op.create_table(
        "kb_parent_chunks",
        *audit_columns(),
        sa.Column("parent_id", sa.String(64), nullable=False),
        sa.Column("doc_id", sa.BigInteger(), nullable=False),
        sa.Column("header_path", sa.String(512), nullable=False),
        sa.Column("content_text", sa.Text(), nullable=False),
        sa.Column("is_sku_table", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.Column("sku_rows_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["doc_id"],
            ["kb_documents.id"],
            name=op.f("fk_kb_parent_chunks_doc_id_kb_documents"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_kb_parent_chunks")),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_kb_parent_chunks_parent_id", "kb_parent_chunks", ["parent_id"], unique=True)
    op.create_index("ix_kb_parent_chunks_doc_id", "kb_parent_chunks", ["doc_id"])
    op.create_index("ix_kb_parent_chunks_doc_sku", "kb_parent_chunks", ["doc_id", "is_sku_table"])

    op.create_table(
        "ops_work_orders",
        *audit_columns(),
        sa.Column("ticket_no", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("priority", sa.String(32), nullable=False),
        sa.Column("trigger_source", sa.String(64), nullable=False),
        sa.Column("slots_payload", sa.JSON(), nullable=False),
        sa.Column("completeness_score", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("context_summary", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), server_default="PENDING", nullable=False),
        sa.CheckConstraint(
            "priority IN ('P0_EMERGENCY', 'P1_HIGH', 'P2_NORMAL')",
            name=op.f("ck_ops_work_orders_work_order_priority"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ops_work_orders")),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    for column in ("ticket_no", "idempotency_key", "session_id", "trace_id"):
        op.create_index(
            f"ix_ops_work_orders_{column}",
            "ops_work_orders",
            [column],
            unique=column in {"ticket_no", "idempotency_key"},
        )
    op.create_index(
        "ix_ops_work_orders_status_created", "ops_work_orders", ["status", "created_at"]
    )
    op.create_index(
        "ix_ops_work_orders_session_created", "ops_work_orders", ["session_id", "created_at"]
    )

    op.create_table(
        "rag_audit_logs",
        *audit_columns(),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("recalled_parent_ids", sa.JSON(), nullable=False),
        sa.Column("guardrail_action", sa.String(64), nullable=False),
        sa.Column("badcase_category", sa.String(32), server_default="NONE", nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("degraded_flag", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.Column("degraded_stage", sa.String(64), nullable=True),
        sa.CheckConstraint(
            "badcase_category IN ('NONE', 'MISSED_RECALL', 'LOW_RANKING', 'HALLUCINATION')",
            name=op.f("ck_rag_audit_logs_badcase_category"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rag_audit_logs")),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_rag_audit_logs_trace_id", "rag_audit_logs", ["trace_id"])
    op.create_index("ix_rag_audit_logs_session_id", "rag_audit_logs", ["session_id"])
    op.create_index(
        "ix_rag_audit_logs_badcase_created", "rag_audit_logs", ["badcase_category", "created_at"]
    )
    op.create_index("ix_rag_audit_logs_trace_created", "rag_audit_logs", ["trace_id", "created_at"])


def downgrade() -> None:
    op.drop_table("rag_audit_logs")
    op.drop_table("ops_work_orders")
    op.drop_table("kb_parent_chunks")
    op.drop_table("kb_documents")
