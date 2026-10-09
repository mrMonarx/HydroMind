"""Ma'lumotlar bazasi jadvallari."""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def utcnow() -> datetime:
    # SQLite vaqt zonasini saqlamaydi, shuning uchun hamma vaqtni UTC deb hisoblab, zonasiz saqlaymiz
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Reading(Base):
    """Bitta sensor tugunidan kelgan bitta o'lchov (keng format: bir qatorda hamma ko'rsatkich)."""

    __tablename__ = "readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    device_id: Mapped[str] = mapped_column(String(64), index=True)
    zone_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    soil_moisture_pct: Mapped[float] = mapped_column(Float)
    soil_temp_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    air_temp_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    air_humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    battery_v: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (Index("ix_readings_zone_time", "zone_id", "timestamp"),)
