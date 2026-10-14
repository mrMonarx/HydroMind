"""Model uchun belgilar (features) va o'qitish jadvalini yasash."""
import numpy as np
import pandas as pd

from .config import STEP_MIN, STEPS_PER_HOUR, TZ_OFFSET_H, ZONE_CODE

STEP = pd.Timedelta(minutes=STEP_MIN)

# Hozirgi holat belgilari (prognoz vaqtida ma'lum bo'lgan narsalar, kelajak emas)
STATE_FEATURES = [
    "moisture", "air_temp", "air_humidity_pct", "soil_temp_c",
    "hour_sin", "hour_cos",
    "d_moist_1h", "d_moist_3h", "air_temp_mean_3h", "d_air_temp_1h",
    "zone_code",
]
MODEL_FEATURES = STATE_FEATURES + ["horizon_h"]


def _lagged(df: pd.DataFrame, col: str, steps: int) -> pd.Series:
    """col ning `steps` qadam oldingi qiymati. Vaqt bo'shlig'i bo'lsa (yo'qolgan o'lchovlar), NaN."""
    g = df.groupby("zone_id")
    val = g[col].shift(steps)
    gap_ok = (df["timestamp"] - g["timestamp"].shift(steps)) == steps * STEP
    return val.where(gap_ok)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """clean_readings natijasiga belgilar qo'shadi."""
    out = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True).copy()
    local = out["timestamp"] + pd.Timedelta(hours=TZ_OFFSET_H)
    hour = local.dt.hour + local.dt.minute / 60
    out["hour_local"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    out["d_moist_1h"] = out["moisture"] - _lagged(out, "moisture", STEPS_PER_HOUR)
    out["d_moist_3h"] = out["moisture"] - _lagged(out, "moisture", 3 * STEPS_PER_HOUR)
    out["d_air_temp_1h"] = out["air_temp"] - _lagged(out, "air_temp", STEPS_PER_HOUR)

    # oxirgi 3 soat (12 ta o'lchov) o'rtacha harorati; bo'shliq bo'lsa NaN
    roll = out.groupby("zone_id")["air_temp"].transform(lambda s: s.rolling(12, min_periods=12).mean())
    span_ok = (out["timestamp"] - out.groupby("zone_id")["timestamp"].shift(11)) == 11 * STEP
    out["air_temp_mean_3h"] = roll.where(span_ok)

    out["zone_code"] = out["zone_id"].map(ZONE_CODE)
    return out


def future_arrays(df: pd.DataFrame, steps: int):
    """Har qator uchun keyingi `steps` ta namlik qiymati va sug'orish bayrog'i.

    Qaytaradi: (F namlik (n, steps), irr_any (n,) — [t, t+steps] oralig'ida sug'orish bo'lganmi, valid (n,) — kelajak to'liq va bo'shliqsizmi).
    df zona va vaqt bo'yicha tartiblangan bo'lishi kerak (add_features shuni kafolatlaydi).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(df)
    F = np.full((n, steps), np.nan)
    irr_any = np.ones(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    pos = 0
    for _, gz in df.groupby("zone_id", sort=False):
        m = gz["moisture"].to_numpy(dtype=float)
        irr = gz["irr_step"].to_numpy(dtype=float)
        ts = gz["timestamp"].to_numpy()
        k = len(gz)
        if k > steps:
            fut = sliding_window_view(m[1:], steps)                       # (k-steps, steps)
            irr_win = sliding_window_view(irr, steps + 1).max(axis=1)      # i..i+steps
            span = ts[steps:] - ts[: k - steps]
            ok = span == np.timedelta64(steps * STEP_MIN, "m")
            sl = slice(pos, pos + k - steps)
            F[sl] = fut
            irr_any[sl] = irr_win > 0
            valid[sl] = ok & ~np.isnan(fut).any(axis=1)
        pos += k
    return F, irr_any, valid


def make_training_set(df_feat: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Uzun jadval: har (vaqt, zona, gorizont) uchun maqsad = namlik pasayishi (hozirgi - h soatdan keyingi).

    Faqat shu oraliqda sug'orish bo'lmagan oynalar olinadi: model "sug'orilmasa nima bo'ladi" savoliga javob berishni o'rganadi.
    """
    parts = []
    base_ok = df_feat[STATE_FEATURES].notna().all(axis=1).to_numpy()
    for h in horizons:
        steps = h * STEPS_PER_HOUR
        F, irr_any, valid = future_arrays(df_feat, steps)
        ok = base_ok & valid & ~irr_any
        part = df_feat.loc[ok, ["timestamp", "zone_id", "hour_local"] + STATE_FEATURES].copy()
        part["horizon_h"] = h
        part["drop"] = part["moisture"].to_numpy() - F[ok, steps - 1]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
