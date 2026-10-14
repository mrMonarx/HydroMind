"""Xavfsizlik qoidalari testlari. Bu testlar o'tmasa, tizimni dalaga qo'ymang."""
import math
from datetime import datetime

import pytest

from backend.app.ml.config import ZONES
from backend.app.ml.decision import Context, decide

CFG = ZONES["zone_1"]   # min 25, critical 20, target 33, max 36, rise 0.32/daq, oynalar 5-9 va 18-21


def flat(v, hours=12):
    return [(h, v) for h in range(1, hours + 1)]


def falling(start, per_hour, hours=12):
    return [(h, start - per_hour * h) for h in range(1, hours + 1)]


def ctx(hour=18.5, m=27.0, age=5.0, fc=None, since=None, used=0.0):
    now = datetime(2026, 10, 12, int(hour), int(round((hour % 1) * 60)))
    return Context(now_local=now, current_moisture=m, data_age_min=age,
                   forecast_low=fc if fc is not None else flat(m), hours_since_irrigation=since, minutes_last_24h=used)


# --- ma'lumot ishonchliligi
def test_stale_data_blocks_decision():
    d = decide(CFG, ctx(age=120, m=22))
    assert d.action == "alert" and d.minutes == 0 and not d.requires_confirmation


@pytest.mark.parametrize("bad", [-5.0, 120.0, float("nan")])
def test_invalid_moisture_blocks_decision(bad):
    d = decide(CFG, ctx(m=bad))
    assert d.action == "alert" and d.minutes == 0


# --- sug'orish kerak emas
def test_saturated_soil_never_irrigates_even_if_forecast_says_dry():
    d = decide(CFG, ctx(m=37, fc=flat(10)))
    assert d.action == "wait" and d.minutes == 0


def test_no_need_when_forecast_stays_above_threshold():
    d = decide(CFG, ctx(m=30, fc=flat(28)))
    assert d.action == "wait" and d.minutes == 0 and d.hours_to_threshold is None


def test_no_forecast_and_moisture_ok_waits():
    assert decide(CFG, ctx(m=30, fc=[])).action == "wait"


# --- oddiy sug'orish
def test_irrigate_in_window_when_threshold_will_be_crossed():
    d = decide(CFG, ctx(hour=18.5, m=26, fc=falling(26, 0.5)))
    assert d.action == "irrigate"
    assert d.minutes == 30                      # (33-26)/0.32 = 21.9 -> 30 daqiqa
    assert d.risk == "low" and d.hours_to_threshold == 3.0   # 2 soatda 25.0 (chegaraning o'zi), 3 soatda 24.5 (pastda)


def test_no_forecast_but_below_threshold_in_window_irrigates():
    d = decide(CFG, ctx(hour=6.0, m=22.0, fc=[]))
    assert d.action == "irrigate" and d.risk == "medium"


# --- tasdiq va avtonomlik
def test_autonomy_levels_for_low_and_medium_risk():
    low = ctx(m=26, fc=falling(26, 0.5))
    assert decide(CFG, low, autonomy=0).requires_confirmation is True
    assert decide(CFG, low, autonomy=1).requires_confirmation is False
    medium = ctx(m=23, fc=flat(22))
    assert decide(CFG, medium, autonomy=1).requires_confirmation is True
    assert decide(CFG, medium, autonomy=2).requires_confirmation is False


@pytest.mark.parametrize("autonomy", [0, 1, 2])
def test_high_risk_always_requires_confirmation(autonomy):
    d = decide(CFG, ctx(hour=12.0, m=19.0, fc=falling(19, 0.5)), autonomy=autonomy)
    assert d.action == "alert" and d.risk == "high" and d.minutes > 0 and d.requires_confirmation is True


# --- vaqt oynasi
def test_outside_window_waits_for_next_window_when_not_critical():
    d = decide(CFG, ctx(hour=12.0, m=27, fc=falling(27, 0.7)))
    assert d.action == "wait" and d.minutes == 0
    assert any("oyna" in r for r in d.reasons)


def test_outside_window_emergency_when_critical_before_next_window():
    # 12:00, keyingi oyna 18:00 (6 soat). Kritik (20) ~3 soatda kutilmoqda.
    d = decide(CFG, ctx(hour=12.0, m=24, fc=falling(24, 1.5)))
    assert d.action == "alert" and d.risk == "high" and d.requires_confirmation


# --- cheklovlar
def test_min_interval_blocks_irrigation():
    d = decide(CFG, ctx(m=23, fc=flat(22), since=3.0))
    assert d.action == "wait" and d.minutes == 0


def test_min_interval_with_critical_moisture_escalates_to_human():
    d = decide(CFG, ctx(m=18, fc=flat(17), since=3.0))
    assert d.action == "alert" and d.minutes == 0


def test_daily_cap_shortens_irrigation():
    d = decide(CFG, ctx(m=26, fc=falling(26, 0.5), used=130))   # qolgan 20 daqiqa -> 15
    assert d.action == "irrigate" and d.minutes == 15 and d.risk == "medium"


def test_daily_cap_exhausted_blocks_irrigation():
    d = decide(CFG, ctx(m=23, fc=flat(22), used=145))
    assert d.action == "wait" and d.minutes == 0


@pytest.mark.parametrize("m", [5, 12, 18, 22, 24, 25.5, 30])
def test_duration_always_valid(m):
    d = decide(CFG, ctx(hour=19.0, m=m, fc=flat(m - 1)))
    if d.minutes:
        assert d.minutes % 15 == 0
        assert 15 <= d.minutes <= CFG.max_minutes_per_event
