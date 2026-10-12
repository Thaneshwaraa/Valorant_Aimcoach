"""mouse settings on profile

Revision ID: a1c3d5e7f901
Revises: e5441ec2f8db
Create Date: 2026-10-12 15:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1c3d5e7f901"
down_revision: Union[str, Sequence[str], None] = "e5441ec2f8db"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("profiles", schema=None) as batch_op:
        batch_op.add_column(sa.Column("dpi", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(
            sa.Column("sensitivity", sa.Float(), nullable=False, server_default="0")
        )


def downgrade() -> None:
    with op.batch_alter_table("profiles", schema=None) as batch_op:
        batch_op.drop_column("sensitivity")
        batch_op.drop_column("dpi")
