from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models, schemas
from .database import Base, engine, get_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Server ishga tushganda jadvallar yo'q bo'lsa, yaratadi
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="HydroMind Agro API",
    description="Sug'orish ma'lumotlari va tavsiyalar uchun backend",
    version="0.3.0",
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
