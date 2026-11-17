"""无需数据库服务即可校验核心表结构与 MySQL DDL 契约。"""

import pytest
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from aegis.db.base import Base
from aegis.db.models import KBDocument, KBParentChunk, OpsTicketOutbox, OpsWorkOrder, RAGAuditLog


@pytest.mark.parametrize(
    "model", [KBDocument, KBParentChunk, OpsTicketOutbox, OpsWorkOrder, RAGAuditLog]
)
def test_core_table_audit_columns_and_indexes(model):
    table = model.__table__
    assert {"id", "created_at", "updated_at"} <= set(table.columns.keys())
    assert table.c.id.primary_key
    assert not table.c.created_at.nullable
    assert not table.c.updated_at.nullable
    assert any(len(index.columns) > 1 for index in table.indexes)
    ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
    assert "utf8mb4" in ddl
    assert "CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)" in ddl


def test_core_table_count_and_foreign_key():
    assert set(Base.metadata.tables) == {
        "kb_documents",
        "kb_parent_chunks",
        "ops_work_orders",
        "rag_audit_logs",
        "ops_ticket_outbox",
    }
    foreign_key = next(iter(KBParentChunk.__table__.c.doc_id.foreign_keys))
    assert foreign_key.target_fullname == "kb_documents.id"


def test_work_order_idempotency_is_unique_in_database():
    indexes = OpsWorkOrder.__table__.indexes
    assert any(
        index.unique and list(index.columns.keys()) == ["idempotency_key"] for index in indexes
    )


@pytest.mark.parametrize(
    ("model", "column", "allowed"),
    [
        (KBDocument, "doc_type", {"SKU_SPEC", "AFTER_SALES", "WAREHOUSE_SOP", "FINANCE_RULE"}),
        (OpsWorkOrder, "priority", {"P0_EMERGENCY", "P1_HIGH", "P2_NORMAL"}),
        (
            RAGAuditLog,
            "badcase_category",
            {"NONE", "MISSED_RECALL", "LOW_RANKING", "HALLUCINATION"},
        ),
    ],
)
def test_enum_contracts_use_varchar_and_checks(model, column, allowed):
    data_type = model.__table__.c[column].type
    assert data_type.length == 32
    assert not data_type.native_enum
    assert data_type.create_constraint
    assert data_type.validate_strings
    assert set(data_type.enums) == allowed
