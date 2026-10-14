"""build_recommendation: butun zanjir (tozalash -> prognoz -> qaror), soxta prognozchi bilan."""
import pandas as pd

from backend.app.ml.config import ZONES
from backend.app.ml.service import build_recommendation


class StubForecaster:
    """Doimiy tezlikda quriydigan soxta model."""
    def __init__(self, per_hour):
        self.per_hour = per_hour

    def forecast(self, row):
        m = float(row["moisture"])
        return [{"hours": h, "moisture": m - self.per_hour * h, "moisture_low": m - self.per_hour * h - 0.5} for h in range(1, 13)]


def raw_frame(moisture, end="2026-10-01 13:00", air=30.0):
    n = len(moisture)
    ts = pd.date_range(end=end, periods=n, freq="15min")
    return pd.DataFrame({"zone_id": "zone_1", "timestamp": ts, "soil_moisture_pct": moisture, "air_temp_c": air,
                         "soil_temp_c": 22.0, "air_humidity_pct": 35.0})


CFG = ZONES["zone_1"]
NOW = pd.Timestamp("2026-10-01 13:00")   # 18:00 mahalliy = sug'orish oynasi


def test_full_chain_recommends_irrigation_when_dry_and_in_window():
    raw = raw_frame([30 - 0.05 * i for i in range(60)][::-1][::-1])    # 60 ta o'lchov (15 soat), ~27 ga tushgan
    r = build_recommendation(raw, CFG, NOW, StubForecaster(0.6))
    assert r["model_used"] and r["action"] == "irrigate" and r["minutes"] % 15 == 0


def test_suspicious_last_reading_uses_previous_value():
    m = [27.0] * 59 + [0.0]
    r = build_recommendation(raw_frame(m), CFG, NOW, StubForecaster(0.1))
    assert r["current_moisture"] == 27.0 and any("shubhali" in x for x in r["reasons"])


def test_rules_only_mode_without_model():
    r = build_recommendation(raw_frame([22.0] * 60), CFG, NOW, None)
    assert not r["model_used"] and r["action"] == "irrigate"      # past namlik: modelsiz ham ishlaydi


def test_stale_data_returns_alert_without_forecast():
    r = build_recommendation(raw_frame([27.0] * 60, end="2026-10-01 08:00"), CFG, NOW, StubForecaster(0.5))
    assert r["action"] == "alert" and r["forecast"] == []
