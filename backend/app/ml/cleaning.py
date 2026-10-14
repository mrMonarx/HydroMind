"""Xom sensor ma'lumotini tozalash va sug'orish hodisalarini topish (4-qadam notebook'idagi mantiq)."""
import numpy as np
import pandas as pd

from .config import STEP_MIN

MOIST_SPIKE_THR = 10.0   # namlik: ikkala qo'shnidan 10 punktdan ko'p farq
AIR_SPIKE_THR = 6.0      # havo harorati: 6 darajadan ko'p farq
RISE_THRESHOLD = 2.5     # 15 daqiqada namlik o'sishi shundan ko'p bo'lsa, sug'orish deb hisoblanadi


def spike_flags(s: pd.Series, thr: float) -> pd.Series:
    """Bitta nuqta ikkala qo'shnisidan ham thr dan ko'p va bir tomonga farq qilsa, True."""
    d_prev = s - s.shift(1)
    d_next = s - s.shift(-1)
    return (d_prev.abs() > thr) & (d_next.abs() > thr) & (np.sign(d_prev) == np.sign(d_next))


def clean_readings(raw: pd.DataFrame) -> pd.DataFrame:
    """Xom o'lchovlardan tozalangan jadval yasaydi.

    Kirish ustunlari: zone_id, timestamp (UTC), soil_moisture_pct, air_temp_c, (ixtiyoriy) soil_temp_c, air_humidity_pct, battery_v.
    Qo'shiladigan ustunlar: moisture, air_temp (xatolari tuzatilgan), anomaly_detected, dm, irr_step.
    """
    df = raw.sort_values(["zone_id", "timestamp"]).reset_index(drop=True).copy()
    g = df.groupby("zone_id")
    moist_spike = g["soil_moisture_pct"].transform(lambda s: spike_flags(s, MOIST_SPIKE_THR))
    air_spike = g["air_temp_c"].transform(lambda s: spike_flags(s, AIR_SPIKE_THR))
    df["anomaly_detected"] = (moist_spike | air_spike).astype(bool)

    df["moisture"] = df["soil_moisture_pct"].where(~moist_spike)
    df["air_temp"] = df["air_temp_c"].where(~air_spike)
    for col in ["moisture", "air_temp"]:
        df[col] = df.groupby("zone_id")[col].transform(lambda s: s.interpolate(limit=3, limit_area="inside"))

    df["dm"] = df.groupby("zone_id")["moisture"].diff()
    df["irr_step"] = (df["dm"] >= RISE_THRESHOLD).astype(bool)
    return df


def detect_events(df: pd.DataFrame) -> pd.DataFrame:
    """Namlikning keskin ko'tarilishidan sug'orish hodisalarini ajratadi (clean_readings natijasi kerak)."""
    cols = ["zone_id", "start", "minutes", "moisture_before", "moisture_after"]
    out = []
    for zone, gz in df.groupby("zone_id"):
        gz = gz.reset_index(drop=True)
        run_id = (gz["irr_step"] != gz["irr_step"].shift()).cumsum()
        for _, idx in gz[gz["irr_step"]].groupby(run_id[gz["irr_step"]]).groups.items():
            i0, i1 = idx[0], idx[-1]
            before = gz.loc[i0 - 1, "moisture"] if i0 > 0 else np.nan
            next_val = gz.loc[min(i1 + 1, len(gz) - 1), "moisture"]
            if not (next_val - before >= RISE_THRESHOLD):   # baland qolmasa, bu sensor shovqini
                continue
            out.append({
                "zone_id": zone,
                "start": gz.loc[i0, "timestamp"],
                "minutes": (i1 - i0 + 1) * STEP_MIN,
                "moisture_before": before,
                "moisture_after": gz.loc[i1, "moisture"],
            })
    return pd.DataFrame(out, columns=cols)


def irrigation_context(events: pd.DataFrame, zone_id: str, now: pd.Timestamp):
    """(oxirgi sug'orishdan beri o'tgan soat yoki None, so'nggi 24 soatdagi jami sug'orish daqiqalari)."""
    e = events[(events["zone_id"] == zone_id) & (events["start"] <= now)]
    if e.empty:
        return None, 0.0
    last_end = (e["start"] + pd.to_timedelta(e["minutes"], unit="m")).max()
    hours_since = max((now - last_end).total_seconds() / 3600, 0.0)
    recent = e[e["start"] >= now - pd.Timedelta(hours=24)]
    return float(hours_since), float(recent["minutes"].sum())


def load_raw(engine, zone_id: str | None = None, since=None, until=None) -> pd.DataFrame:
    """Bazadan xom o'lchovlarni o'qiydi."""
    clauses, params = [], {}
    if zone_id:
        clauses.append("zone_id = :zone_id"); params["zone_id"] = zone_id
    if since is not None:
        clauses.append("timestamp >= :since"); params["since"] = since
    if until is not None:
        clauses.append("timestamp <= :until"); params["until"] = until
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    from sqlalchemy import text
    q = text(f"SELECT * FROM readings {where} ORDER BY zone_id, timestamp")
    df = pd.read_sql(q, engine, params=params, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    return df
