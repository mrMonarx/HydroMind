"""Qaror dvigateli: ML prognozi + QAT'IY xavfsizlik qoidalari + xavf darajasi.

Prinsip: model faqat "nima bo'lishi mumkin" deydi. Sug'orish kerakmi, qancha va tasdiq kerakmi, buni
shu fayldagi oddiy, tekshirib bo'ladigan qoidalar hal qiladi. Model xato qilsa ham qoidalar chegarasidan chiqib bo'lmaydi.

Xavf darajalari:
  low    - oddiy sug'orish, hamma chegara ichida
  medium - sug'orish bor, lekin qisqartirilgan yoki namlik allaqachon chegaradan past
  high   - oynadan tashqari favqulodda sug'orish yoki sensor/ma'lumot muammosi

Avtonomlik (autonomy):
  0 - hamma sug'orish fermer tasdig'i bilan (boshlang'ich holat)
  1 - faqat low xavfli sug'orish avtomatik
  2 - low va medium avtomatik. high HAR DOIM tasdiq talab qiladi.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime

from .config import ZoneConfig

SLOT_MIN = 15  # sug'orish davomiyligi shu qadam bilan yaxlitlanadi


@dataclass
class Context:
    now_local: datetime
    current_moisture: float
    data_age_min: float
    forecast_low: list = field(default_factory=list)   # [(soat, pessimistik namlik %)] model prognozi minus xato zaxirasi
    hours_since_irrigation: float | None = None
    minutes_last_24h: float = 0.0


@dataclass
class Decision:
    action: str                       # "irrigate" | "wait" | "alert"
    minutes: int
    risk: str                         # "low" | "medium" | "high"
    requires_confirmation: bool
    reasons: list
    hours_to_threshold: float | None = None


def _hour(dt: datetime) -> float:
    return dt.hour + dt.minute / 60


def _in_window(cfg: ZoneConfig, hour: float) -> bool:
    return any(a <= hour < b for a, b in cfg.windows)


def _hours_to_next_window(cfg: ZoneConfig, hour: float) -> float:
    return min((a - hour) % 24 for a, _ in cfg.windows)


def _windows_text(cfg: ZoneConfig) -> str:
    return " va ".join(f"{a}:00-{b}:00" for a, b in cfg.windows)


def _needs_confirmation(action: str, minutes: int, risk: str, autonomy: int) -> bool:
    if action == "wait" or minutes == 0:
        return False
    if risk == "high":
        return True
    if autonomy <= 0:
        return True
    if autonomy == 1:
        return risk != "low"
    return False


def _make(action, minutes, risk, reasons, autonomy, h_thr=None) -> Decision:
    return Decision(action, minutes, risk, _needs_confirmation(action, minutes, risk, autonomy), reasons, h_thr)


def decide(cfg: ZoneConfig, ctx: Context, autonomy: int = 0) -> Decision:
    m = ctx.current_moisture

    # 1) Ma'lumot ishonchliligi: shubhali ma'lumotga asoslanib qaror chiqarilmaydi
    if m is None or (isinstance(m, float) and math.isnan(m)) or not (0 <= m <= 100):
        return _make("alert", 0, "high", [f"Namlik qiymati noto'g'ri ({m}). Sensorni tekshiring; qaror chiqarilmadi."], autonomy)
    if ctx.data_age_min > cfg.max_data_age_min:
        return _make("alert", 0, "high", [
            f"Sensor ma'lumoti {ctx.data_age_min:.0f} daqiqa oldingi (ruxsat etilgan: {cfg.max_data_age_min}). "
            "Qurilma yoki aloqani tekshiring; qaror chiqarilmadi."], autonomy)

    # 2) To'yingan tuproq: hech qanday prognoz bunga qaramay sug'ortirmaydi
    if m >= cfg.max_moisture:
        return _make("wait", 0, "low", [f"Tuproq yetarli nam ({m:.1f}% >= {cfg.max_moisture:.0f}%). Sug'orilmaydi."], autonomy)

    # 3) Kerakmi?
    fc = list(ctx.forecast_low)
    h_cross = next((h for h, v in fc if v < cfg.min_moisture), None)
    h_crit = next((h for h, v in fc if v < cfg.critical_moisture), None)
    below_now = m < cfg.min_moisture
    horizon = max((h for h, _ in fc), default=0)

    if not below_now and h_cross is None:
        if fc:
            lowest = min(v for _, v in fc)
            txt = f"Keyingi {horizon} soatda namlik chegaradan ({cfg.min_moisture:.0f}%) pastga tushmaydi (eng past prognoz {lowest:.1f}%)."
        else:
            txt = f"Namlik chegaradan yuqori ({m:.1f}%). Prognoz mavjud emas, faqat joriy namlik tekshirildi."
        return _make("wait", 0, "low", [txt], autonomy)

    need_txt = (f"Namlik chegaradan past ({m:.1f}% < {cfg.min_moisture:.0f}%)." if below_now
                else f"Namlik ~{h_cross} soatdan keyin chegaradan ({cfg.min_moisture:.0f}%) pastga tushishi kutilmoqda.")
    h_thr = 0.0 if below_now else float(h_cross)
    critical_now = m < cfg.critical_moisture

    # 4) Qat'iy cheklovlar
    if ctx.hours_since_irrigation is not None and ctx.hours_since_irrigation < cfg.min_hours_between:
        why = (f"Oxirgi sug'orishdan beri {ctx.hours_since_irrigation:.1f} soat o'tgan "
               f"(minimal oraliq {cfg.min_hours_between:.0f} soat).")
        if critical_now:
            return _make("alert", 0, "high", [need_txt, why, f"Lekin namlik kritik darajadan ({cfg.critical_moisture:.0f}%) past. Inson tekshiruvi kerak."], autonomy, h_thr)
        return _make("wait", 0, "low", [need_txt, why], autonomy, h_thr)

    remaining = cfg.max_minutes_per_day - ctx.minutes_last_24h
    if remaining < SLOT_MIN:
        why = f"So'nggi 24 soatda {ctx.minutes_last_24h:.0f} daqiqa sug'orilgan (limit {cfg.max_minutes_per_day})."
        if critical_now:
            return _make("alert", 0, "high", [need_txt, why, "Lekin namlik kritik darajadan past. Inson tekshiruvi kerak."], autonomy, h_thr)
        return _make("wait", 0, "low", [need_txt, why], autonomy, h_thr)

    # 5) Davomiylik: maqsadli namlikka yetish uchun kerakli vaqt, chegaralar ichida
    wanted = max(SLOT_MIN, int(math.ceil(max(cfg.target_moisture - m, 0) / cfg.rise_per_min / SLOT_MIN) * SLOT_MIN))
    limit = int(min(cfg.max_minutes_per_event, remaining) // SLOT_MIN * SLOT_MIN)
    minutes = min(wanted, limit)
    capped = minutes < wanted

    # 6) Vaqt oynasi
    hour = _hour(ctx.now_local)
    if _in_window(cfg, hour):
        risk = "medium" if (capped or below_now) else "low"
        reasons = [need_txt, f"Hozir sug'orish oynasida ({_windows_text(cfg)}). {minutes} daqiqa tavsiya etiladi "
                             f"(maqsad {cfg.target_moisture:.0f}%)."]
        if capped:
            reasons.append(f"Davomiylik chegaralar tufayli qisqartirildi (kerak edi: {wanted} daqiqa).")
        return _make("irrigate", minutes, risk, reasons, autonomy, h_thr)

    next_w = _hours_to_next_window(cfg, hour)
    emergency = critical_now or (h_crit is not None and h_crit < next_w)
    if emergency:
        why = (f"Namlik kritik darajadan ({cfg.critical_moisture:.0f}%) past." if critical_now
               else f"Kritik darajaga ({cfg.critical_moisture:.0f}%) ~{h_crit} soatda yetishi kutilmoqda, navbatdagi oyna esa {next_w:.1f} soatdan keyin.")
        return _make("alert", minutes, "high", [
            need_txt, why,
            f"Hozir sug'orish oynasidan tashqarida ({_windows_text(cfg)}). Favqulodda {minutes} daqiqa sug'orish faqat fermer tasdig'i bilan."
        ], autonomy, h_thr)

    return _make("wait", 0, "low", [
        need_txt,
        f"Navbatdagi sug'orish oynasi {next_w:.1f} soatdan keyin. Shunda taxminan {minutes} daqiqa sug'orish rejalashtiriladi."
    ], autonomy, h_thr)
