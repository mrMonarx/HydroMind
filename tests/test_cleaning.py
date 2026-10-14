"""Tozalash, hodisa topish va belgilar testlari (sintetik ma'lumot bilan)."""
import numpy as np
import pandas as pd

from backend.app.ml.cleaning import clean_readings, detect_events, irrigation_context
from backend.app.ml.features import add_features
from backend.app.ml.model import curves_from_drops


def frame(moisture, zone="zone_1", start="2026-10-01 00:00", air=None):
    n = len(moisture)
    return pd.DataFrame({
        "zone_id": zone,
        "timestamp": pd.date_range(start, periods=n, freq="15min"),
        "soil_moisture_pct": moisture,
        "air_temp_c": air if air is not None else [25.0] * n,
        "soil_temp_c": 22.0, "air_humidity_pct": 40.0, "battery_v": 3.9,
    })


def test_single_point_spike_is_removed_and_interpolated():
    m = [30.0] * 20
    m[10] = 95.0
    df = clean_readings(frame(m))
    assert df.loc[10, "anomaly_detected"]
    assert abs(df.loc[10, "moisture"] - 30.0) < 1e-6
    assert df["anomaly_detected"].sum() == 1


def test_irrigation_ramp_is_not_treated_as_anomaly():
    m = [24.0] * 5 + [29.0, 34.0, 38.0] + [38.0] * 8
    df = clean_readings(frame(m))
    assert df["anomaly_detected"].sum() == 0


def test_irrigation_event_detected_with_duration():
    m = [24.0] * 5 + [29.0, 34.0, 38.0] + [37.8] * 8
    ev = detect_events(clean_readings(frame(m)))
    assert len(ev) == 1
    assert ev.loc[0, "minutes"] >= 30 and ev.loc[0, "moisture_after"] > ev.loc[0, "moisture_before"] + 10


def test_single_sensor_noise_blip_is_not_an_event():
    m = [30.0] * 5 + [33.0] + [30.0] * 10        # bitta sakrash va darrov qaytish
    ev = detect_events(clean_readings(frame(m)))
    assert len(ev) == 0


def test_irrigation_context_hours_and_minutes():
    m = [24.0] * 5 + [29.0, 34.0, 38.0] + [37.8] * 20
    df = clean_readings(frame(m))
    ev = detect_events(df)
    now = df["timestamp"].iloc[-1]
    hours, minutes = irrigation_context(ev, "zone_1", now)
    assert minutes >= 30 and hours > 0
    assert irrigation_context(ev, "zone_2", now) == (None, 0.0)


def test_features_are_nan_across_time_gap():
    df = frame([30.0 - 0.1 * i for i in range(30)])
    df = df.drop(index=range(14, 20)).reset_index(drop=True)      # 1.5 soatlik bo'shliq
    feat = add_features(clean_readings(df))
    after_gap = feat[feat["timestamp"] == pd.Timestamp("2026-10-01 05:00")]
    assert after_gap["d_moist_1h"].isna().all()


def test_curves_are_monotonic_and_never_rise():
    drops = np.array([[0.5, 0.4, -0.2, 1.0]])
    curve = curves_from_drops(np.array([30.0]), drops)[0]
    assert all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1))
    assert curve[0] <= 30.0
