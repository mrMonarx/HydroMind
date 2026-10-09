"""API orqali kiruvchi va chiquvchi ma'lumot shakllari (validatsiya bilan)."""
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator


def to_utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class ReadingIn(BaseModel):
    """Sensor tugunidan keladigan ma'lumot. Real bo'lmagan qiymatlar rad etiladi."""

    device_id: str = Field(min_length=1, max_length=64, examples=["esp32_zone1_node1"])
    zone_id: str = Field(min_length=1, max_length=64, examples=["zone_1"])
    soil_moisture_pct: float = Field(ge=0, le=100, examples=[34.7])
    soil_temp_c: float | None = Field(default=None, ge=-30, le=70, examples=[24.1])
    air_temp_c: float | None = Field(default=None, ge=-50, le=80, examples=[31.2])
    air_humidity_pct: float | None = Field(default=None, ge=0, le=100, examples=[42.5])
    battery_v: float | None = Field(default=None, ge=0, le=30, examples=[3.78])
    timestamp: datetime | None = Field(
        default=None, description="Ko'rsatilmasa, server vaqti (UTC) qo'yiladi"
    )

    @field_validator("timestamp")
    @classmethod
    def normalize_ts(cls, v):
        return to_utc_naive(v) if v else v


class ReadingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    device_id: str
    zone_id: str
    timestamp: datetime
    soil_moisture_pct: float
    soil_temp_c: float | None
    air_temp_c: float | None
    air_humidity_pct: float | None
    battery_v: float | None

    @field_validator("timestamp")
    @classmethod
    def mark_utc(cls, v: datetime):
        # Bazadan zonasiz keladi; javobda UTC ekanini aniq ko'rsatamiz
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v


class BulkResult(BaseModel):
    inserted: int
