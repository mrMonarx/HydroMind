"""Zonalar sozlamalari va umumiy konstantalar.

MUHIM: bu qiymatlar agronom/fermer bilan kelishilishi kerak. Ular ML model emas, inson belgilaydigan chegaralar:
tizim ularni hech qachon o'zi o'zgartirmaydi.
"""
from dataclasses import dataclass

TZ_OFFSET_H = 5          # Toshkent: UTC+5
STEP_MIN = 15            # o'lchovlar oralig'i
STEPS_PER_HOUR = 60 // STEP_MIN
HORIZONS_H = list(range(1, 13))   # prognoz: 1..12 soat oldinga


@dataclass(frozen=True)
class ZoneConfig:
    zone_id: str
    min_moisture: float        # namlik shundan pastga tushmasligi kerak (%)
    critical_moisture: float   # favqulodda daraja: bundan pastda ekin zarar ko'ra boshlaydi (%)
    target_moisture: float     # sug'orishdan keyin erishilishi kerak bo'lgan namlik (%)
    max_moisture: float        # shundan yuqori bo'lsa, sug'orilmaydi (%)
    rise_per_min: float        # sug'orishda namlik o'sish tezligi (punkt/daqiqa), hodisalardan baholangan
    max_minutes_per_event: int = 90     # bitta sug'orishning maksimal davomiyligi
    max_minutes_per_day: int = 150      # so'nggi 24 soatdagi maksimal jami davomiylik
    min_hours_between: float = 10.0     # ikki sug'orish orasidagi minimal vaqt
    windows: tuple = ((5, 9), (18, 21)) # ruxsat etilgan sug'orish oynalari (mahalliy soat, [boshi, oxiri))
    max_data_age_min: int = 45          # sensor ma'lumoti shundan eski bo'lsa, qaror chiqarilmaydi


ZONES = {
    "zone_1": ZoneConfig("zone_1", min_moisture=25, critical_moisture=20, target_moisture=33, max_moisture=36, rise_per_min=0.32),
    "zone_2": ZoneConfig("zone_2", min_moisture=28, critical_moisture=23, target_moisture=40, max_moisture=43, rise_per_min=0.32),
    "zone_3": ZoneConfig("zone_3", min_moisture=32, critical_moisture=27, target_moisture=47, max_moisture=50, rise_per_min=0.32),
}
ZONE_CODE = {z: i for i, z in enumerate(ZONES)}
