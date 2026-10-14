import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import pandas as pd

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models, schemas
from .database import Base, engine, get_db
from .ml.config import ZONES
from .ml.model import MoistureForecaster
from .ml.service import build_recommendation

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/moisture_model.joblib"))
_model_cache: dict = {"mtime": None, "model": None}


def get_forecaster() -> MoistureForecaster | None:
    """Modelni yuklaydi (fayl yangilansa, serverni qayta ishga tushirmasdan qayta yuklaydi)."""
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if _model_cache["mtime"] != mtime:
        _model_cache.update(mtime=mtime, model=MoistureForecaster.load(MODEL_PATH))
    return _model_cache["model"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Server ishga tushganda jadvallar yo'q bo'lsa, yaratadi
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="HydroMind Agro API",
    description="Sug'orish ma'lumotlari va tavsiyalar uchun backend",
    version="0.5.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Server ishlayotganini tekshirish uchun."""
    return {"status": "ok", "project": "HydroMind Agro", "version": app.version}


@app.post("/readings", response_model=schemas.ReadingOut, status_code=201, tags=["readings"])
def create_reading(payload: schemas.ReadingIn, db: Session = Depends(get_db)):
    """Sensor tugunidan kelgan o'lchovni bazaga yozadi."""
    data = payload.model_dump(exclude_none=True)
    reading = models.Reading(**data)  # timestamp bo'lmasa, model o'zi UTC vaqtni qo'yadi
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


@app.post("/readings/bulk", response_model=schemas.BulkResult, status_code=201, tags=["readings"])
def create_readings_bulk(
    payload: Annotated[list[schemas.ReadingIn], Body(min_length=1, max_length=1000)],
    db: Session = Depends(get_db),
):
    """Bir nechta o'lchovni (1000 tagacha) bitta so'rovda yozadi. Birortasi noto'g'ri bo'lsa, hech biri yozilmaydi."""
    db.add_all([models.Reading(**item.model_dump(exclude_none=True)) for item in payload])
    db.commit()
    return {"inserted": len(payload)}


@app.get("/readings", response_model=list[schemas.ReadingOut], tags=["readings"])
def list_readings(
    zone_id: str | None = None,
    device_id: str | None = None,
    since: datetime | None = Query(default=None, description="Shu vaqtdan keyingilar (UTC)"),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """O'lchovlar ro'yxati (eng yangisi birinchi). zone_id, device_id va vaqt bo'yicha filtrlash mumkin."""
    stmt = select(models.Reading)
    if zone_id:
        stmt = stmt.where(models.Reading.zone_id == zone_id)
    if device_id:
        stmt = stmt.where(models.Reading.device_id == device_id)
    if since:
        stmt = stmt.where(models.Reading.timestamp >= schemas.to_utc_naive(since))
    stmt = stmt.order_by(models.Reading.timestamp.desc()).limit(limit)
    return db.scalars(stmt).all()


@app.get("/zones/{zone_id}/latest", response_model=schemas.ReadingOut, tags=["zones"])
def latest_for_zone(zone_id: str, db: Session = Depends(get_db)):
    """Zonaning eng oxirgi o'lchovi."""
    stmt = (
        select(models.Reading)
        .where(models.Reading.zone_id == zone_id)
        .order_by(models.Reading.timestamp.desc())
        .limit(1)
    )
    reading = db.scalars(stmt).first()
    if reading is None:
        raise HTTPException(status_code=404, detail=f"'{zone_id}' zonasi uchun ma'lumot topilmadi")
    return reading


@app.get("/zones/{zone_id}/recommendation", response_model=schemas.RecommendationOut, tags=["zones"])
def recommendation(
    zone_id: str,
    autonomy: int = Query(default=0, ge=0, le=2, description="0: hammasi tasdiq bilan, 1: low xavf avtomatik, 2: low+medium avtomatik"),
    as_of: datetime | None = Query(default=None, description="Shu vaqt holatida tavsiya olish (UTC). Demo/tekshiruv uchun; ko'rsatilmasa, hozir."),
    db: Session = Depends(get_db),
):
    """ML prognozi + xavfsizlik qoidalari asosida zonaga sug'orish tavsiyasi."""
    cfg = ZONES.get(zone_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"'{zone_id}' zonasi uchun sozlama topilmadi (backend/app/ml/config.py)")

    now = schemas.to_utc_naive(as_of) if as_of else models.utcnow()
    stmt = (
        select(models.Reading)
        .where(models.Reading.zone_id == zone_id, models.Reading.timestamp >= now - timedelta(hours=48),
               models.Reading.timestamp <= now)
        .order_by(models.Reading.timestamp)
    )
    rows = db.scalars(stmt).all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"'{zone_id}' zonasi uchun so'nggi 48 soatda ma'lumot yo'q")

    raw = pd.DataFrame([{
        "zone_id": r.zone_id, "timestamp": r.timestamp, "soil_moisture_pct": r.soil_moisture_pct,
        "soil_temp_c": r.soil_temp_c, "air_temp_c": r.air_temp_c, "air_humidity_pct": r.air_humidity_pct,
    } for r in rows])
    raw["timestamp"] = raw["timestamp"].astype("datetime64[ns]")

    result = build_recommendation(raw, cfg, pd.Timestamp(now), get_forecaster(), autonomy)
    result["as_of"] = now.replace(tzinfo=timezone.utc)
    return result
