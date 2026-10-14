"""Xom o'lchovlardan tayyor tavsiya: tozalash -> belgilar -> prognoz -> qaror dvigateli.
API ham, notebook ham aynan shu funksiyani ishlatadi (ikki joyda ikki xil natija chiqmasligi uchun)."""
from datetime import timedelta

import pandas as pd

from .cleaning import MOIST_SPIKE_THR, clean_readings, detect_events, irrigation_context
from .config import TZ_OFFSET_H, ZoneConfig
from .decision import Context, decide
from .features import STATE_FEATURES, add_features


def build_recommendation(raw: pd.DataFrame, cfg: ZoneConfig, now: pd.Timestamp, forecaster, autonomy: int = 0) -> dict:
    """raw: shu zonaning so'nggi ~48 soatlik xom o'lchovlari (zone_id, timestamp UTC, soil_moisture_pct, air_temp_c, ...).
    now: hozirgi vaqt (UTC, zonasiz). forecaster: MoistureForecaster yoki None (faqat qoidalar rejimi)."""
    df = clean_readings(raw)
    events = detect_events(df)
    feat = add_features(df)

    notes: list[str] = []
    last = feat.iloc[-1]
    # Oxirgi o'lchovni "keyingi qo'shni" bilan tekshirib bo'lmaydi. Bir qadamda fizik jihatdan mumkin bo'lmagan sakrash bo'lsa,
    # oldingi o'lchovdan foydalanamiz.
    if len(feat) > 1 and abs(last["moisture"] - feat.iloc[-2]["moisture"]) > MOIST_SPIKE_THR:
        last = feat.iloc[-2]
        notes.append("Oxirgi o'lchov shubhali (keskin sakrash), oldingi o'lchovdan foydalanildi.")

    age = (now - last["timestamp"]).total_seconds() / 60
    state_ok = bool(last[STATE_FEATURES].notna().all())

    forecast: list[dict] = []
    if forecaster is not None and state_ok and age <= cfg.max_data_age_min:
        forecast = forecaster.forecast(last)
    elif forecaster is None:
        notes.append("Model topilmadi: faqat qoidalar asosida (prognozsiz) qaror. Modelni o'rgating: python scripts/train_model.py")
    elif not state_ok:
        notes.append("Prognoz uchun tarix yetarli emas (kamida 3 soat uzluksiz ma'lumot kerak).")

    hours_since, used = irrigation_context(events, cfg.zone_id, now)
    ctx = Context(
        now_local=(now + timedelta(hours=TZ_OFFSET_H)).to_pydatetime(),
        current_moisture=float(last["moisture"]),
        data_age_min=float(age),
        forecast_low=[(p["hours"], p["moisture_low"]) for p in forecast],
        hours_since_irrigation=hours_since,
        minutes_last_24h=used,
    )
    d = decide(cfg, ctx, autonomy)
    return {
        "zone_id": cfg.zone_id,
        "as_of": now,
        "current_moisture": round(ctx.current_moisture, 1),
        "data_age_min": round(age, 1),
        "action": d.action,
        "minutes": d.minutes,
        "risk": d.risk,
        "requires_confirmation": d.requires_confirmation,
        "reasons": d.reasons + notes,
        "hours_to_threshold": d.hours_to_threshold,
        "forecast": [{"hours": p["hours"], "moisture": round(p["moisture"], 1), "moisture_low": round(p["moisture_low"], 1)}
                     for p in forecast],
        "model_used": bool(forecast),
    }
