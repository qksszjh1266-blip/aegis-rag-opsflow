"""新增工单事务发件箱，保证 MySQL 提交后的派单事件可恢复发布。"""

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "ops_ticket_outbox",
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
        sa.Column("event_id", sa.String(64), nullable=False),
        sa.Column("ticket_id", sa.BigInteger(), nullable=False),
        sa.Column("stream_name", sa.String(128), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("published", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.Column("redis_message_id", sa.String(64), nullable=True),
        sa.ForeignKeyConstraint(
            ["ticket_id"],
            ["ops_work_orders.id"],
            name=op.f("fk_ops_ticket_outbox_ticket_id_ops_work_orders"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ops_ticket_outbox")),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_ops_ticket_outbox_event_id", "ops_ticket_outbox", ["event_id"], unique=True)
    op.create_index(
        "ix_ops_ticket_outbox_ticket_id", "ops_ticket_outbox", ["ticket_id"], unique=True
    )
    op.create_index(
        "ix_ops_ticket_outbox_pending", "ops_ticket_outbox", ["published", "stream_name", "id"]
    )


def downgrade() -> None:
    op.drop_table("ops_ticket_outbox")
