"""Initial security monitoring schema

Revision ID: 20260304_0001
Revises:
Create Date: 2026-03-04 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260304_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "incidents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("source_ip", sa.String(length=64), nullable=False),
        sa.Column("destination_ip", sa.String(length=64), nullable=False),
        sa.Column("attack_type", sa.String(length=100), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_incidents_id"), "incidents", ["id"], unique=False)
    op.create_index(op.f("ix_incidents_timestamp"), "incidents", ["timestamp"], unique=False)
    op.create_index(op.f("ix_incidents_source_ip"), "incidents", ["source_ip"], unique=False)
    op.create_index(op.f("ix_incidents_destination_ip"), "incidents", ["destination_ip"], unique=False)
    op.create_index(op.f("ix_incidents_attack_type"), "incidents", ["attack_type"], unique=False)
    op.create_index(op.f("ix_incidents_severity"), "incidents", ["severity"], unique=False)
    op.create_index(op.f("ix_incidents_status"), "incidents", ["status"], unique=False)

    op.create_table(
        "traffic_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("source_ip", sa.String(length=64), nullable=False),
        sa.Column("destination_ip", sa.String(length=64), nullable=False),
        sa.Column("protocol", sa.String(length=20), nullable=False),
        sa.Column("source_port", sa.Integer(), nullable=False),
        sa.Column("destination_port", sa.Integer(), nullable=False),
        sa.Column("packet_size", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_traffic_logs_id"), "traffic_logs", ["id"], unique=False)
    op.create_index(op.f("ix_traffic_logs_timestamp"), "traffic_logs", ["timestamp"], unique=False)
    op.create_index(op.f("ix_traffic_logs_source_ip"), "traffic_logs", ["source_ip"], unique=False)
    op.create_index(op.f("ix_traffic_logs_destination_ip"), "traffic_logs", ["destination_ip"], unique=False)
    op.create_index(op.f("ix_traffic_logs_protocol"), "traffic_logs", ["protocol"], unique=False)

    op.create_table(
        "threat_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("source_ip", sa.String(length=64), nullable=False),
        sa.Column("destination_ip", sa.String(length=64), nullable=False),
        sa.Column("attack_type", sa.String(length=100), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("threat_score", sa.Integer(), nullable=False),
        sa.Column("protocol", sa.String(length=20), nullable=False),
        sa.Column("destination_port", sa.Integer(), nullable=False),
        sa.Column("incident_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_threat_events_id"), "threat_events", ["id"], unique=False)
    op.create_index(op.f("ix_threat_events_timestamp"), "threat_events", ["timestamp"], unique=False)
    op.create_index(op.f("ix_threat_events_source_ip"), "threat_events", ["source_ip"], unique=False)
    op.create_index(op.f("ix_threat_events_destination_ip"), "threat_events", ["destination_ip"], unique=False)
    op.create_index(op.f("ix_threat_events_attack_type"), "threat_events", ["attack_type"], unique=False)
    op.create_index(op.f("ix_threat_events_severity"), "threat_events", ["severity"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_threat_events_severity"), table_name="threat_events")
    op.drop_index(op.f("ix_threat_events_attack_type"), table_name="threat_events")
    op.drop_index(op.f("ix_threat_events_destination_ip"), table_name="threat_events")
    op.drop_index(op.f("ix_threat_events_source_ip"), table_name="threat_events")
    op.drop_index(op.f("ix_threat_events_timestamp"), table_name="threat_events")
    op.drop_index(op.f("ix_threat_events_id"), table_name="threat_events")
    op.drop_table("threat_events")

    op.drop_index(op.f("ix_traffic_logs_protocol"), table_name="traffic_logs")
    op.drop_index(op.f("ix_traffic_logs_destination_ip"), table_name="traffic_logs")
    op.drop_index(op.f("ix_traffic_logs_source_ip"), table_name="traffic_logs")
    op.drop_index(op.f("ix_traffic_logs_timestamp"), table_name="traffic_logs")
    op.drop_index(op.f("ix_traffic_logs_id"), table_name="traffic_logs")
    op.drop_table("traffic_logs")

    op.drop_index(op.f("ix_incidents_status"), table_name="incidents")
    op.drop_index(op.f("ix_incidents_severity"), table_name="incidents")
    op.drop_index(op.f("ix_incidents_attack_type"), table_name="incidents")
    op.drop_index(op.f("ix_incidents_destination_ip"), table_name="incidents")
    op.drop_index(op.f("ix_incidents_source_ip"), table_name="incidents")
    op.drop_index(op.f("ix_incidents_timestamp"), table_name="incidents")
    op.drop_index(op.f("ix_incidents_id"), table_name="incidents")
    op.drop_table("incidents")
