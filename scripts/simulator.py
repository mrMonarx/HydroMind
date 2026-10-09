"""HydroMind Agro sensor simulyatori.

Hardware bo'lmaganda realistik sensor ma'lumotlarini yaratadi va API ga (yoki CSV ga) yozadi.

Misollar:
    python scripts/simulator.py backfill --days 30              # 30 kunlik tarix, API ga yozadi
    python scripts/simulator.py backfill --days 30 --csv data/sim.csv --no-api   # faqat CSV
    python scripts/simulator.py live --interval 2                # jonli demo (dashboard uchun)

Fizika (soddalashtirilgan, lekin yo'nalishi to'g'ri):
  - Havo harorati va namligi kunlik sikl bilan o'zgaradi (Toshkent mahalliy vaqti, UTC+5)
  - Tuproq namligi bug'lanish (issiq, quruq, quyoshli payt tezroq) tufayli kamayadi
  - Zonalar tuprog'i har xil: zone_1 qumloq (tez quriydi), zone_2 o'rta, zone_3 gil (suvni ushlaydi)
  - "Fermer" namlik chegaradan pastga tushsa, ertalab (5-9) yoki kechqurun (18-21) sug'oradi
  - Ba'zan sensor xatosi (sakrash yoki 0 ga tushish) qo'shiladi, tahlil bosqichida topish uchun
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import error, request

TZ_OFFSET_H = 5  # Toshkent: UTC+5
IRRIGATION_RATE_PER_H = 20.0  # sug'orish paytida namlik o'sishi (foiz punkt / soat)

CSV_COLUMNS = [
    "timestamp", "zone_id", "device_id", "soil_moisture_pct", "soil_temp_c",
    "air_temp_c", "air_humidity_pct", "battery_v", "valve_open", "is_anomaly",
]
API_FIELDS = [
    "device_id", "zone_id", "timestamp", "soil_moisture_pct", "soil_temp_c",
    "air_temp_c", "air_humidity_pct", "battery_v",
]


@dataclass
class Zone:
    zone_id: str
    device_id: str
    dry_factor: float       # tuproq qanchalik tez quriydi
    field_capacity: float   # to'yingan tuproq namligi (%)
    irrigate_below: float   # "fermer" shu darajadan pastda sug'orishni boshlaydi
    moisture: float         # joriy haqiqiy namlik
    battery_offset: float = 0.0
    irrigation_left_min: float = 0.0
    was_open: bool = False


def make_zones(n: int) -> list[Zone]:
    zones = [
        Zone("zone_1", "esp32_zone1_node1", 1.5, 38.0, 25.0, 33.0, battery_offset=0.00),
        Zone("zone_2", "esp32_zone2_node1", 1.0, 45.0, 28.0, 40.0, battery_offset=0.03),
        Zone("zone_3", "esp32_zone3_node1", 0.7, 52.0, 32.0, 46.0, battery_offset=-0.02),
    ]
    return zones[:n]


class Simulator:
    def __init__(self, start_utc: datetime, step_min: int, seed: int, anomaly_rate: float,
                 mean_temp: float, n_zones: int):
        self.t = start_utc
        self.start = start_utc
        self.step_min = step_min
        self.rng = random.Random(seed)
        self.anomaly_rate = anomaly_rate
        self.mean_temp = mean_temp
        self.zones = make_zones(n_zones)
        self._days: dict[int, tuple[float, float]] = {}
        self.irrigation_events = {z.zone_id: 0 for z in self.zones}
        self.anomaly_count = {z.zone_id: 0 for z in self.zones}

    def local_hour(self) -> float:
        lt = self.t + timedelta(hours=TZ_OFFSET_H)
        return lt.hour + lt.minute / 60.0

    def day_params(self) -> tuple[float, float]:
        """Kun uchun (harorat siljishi, bulutlilik koeffitsiyenti). Ketma-ket kunlar bog'liq."""
        d = (self.t + timedelta(hours=TZ_OFFSET_H)).toordinal()
        if d not in self._days:
            prev_off = self._days.get(d - 1, (0.0, 0.9))[0]
            off = 0.7 * prev_off + self.rng.gauss(0, 1.8)
            cloud = min(1.0, max(0.55, self.rng.gauss(0.92, 0.1)))
            self._days[d] = (off, cloud)
        return self._days[d]

    def weather(self) -> dict:
        h = self.local_hour()
        off, cloud = self.day_params()
        temp = self.mean_temp + off + 8.0 * math.sin(2 * math.pi * (h - 9) / 24) + self.rng.gauss(0, 0.3)
        rh = 45.0 - 1.5 * off - 22.0 * math.sin(2 * math.pi * (h - 9) / 24) + self.rng.gauss(0, 1.5)
        rh = min(95.0, max(12.0, rh))
        sun = max(0.0, math.sin(math.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
        sun *= cloud
        soil_t = self.mean_temp - 4 + 0.6 * off + 5.0 * math.sin(2 * math.pi * (h - 11) / 24) + self.rng.gauss(0, 0.15)
        return {"temp": temp, "rh": rh, "sun": sun, "soil_t": soil_t}

    def evaporation_per_hour(self, z: Zone, w: dict) -> float:
        heat = max(0.1, 0.15 + 0.035 * (w["temp"] - 10))
        dryness = max(0.2, 1.25 - w["rh"] / 100)
        return z.dry_factor * (0.04 + 0.9 * w["sun"]) * heat * dryness

    def step(self) -> list[dict]:
        """Vaqtni bir qadam (step_min) oldinga suradi va har zona uchun bitta qator qaytaradi."""
        dt = self.step_min
        w = self.weather()
        h = self.local_hour()
        days_elapsed = (self.t - self.start).total_seconds() / 86400
        rows = []
        for z in self.zones:
            if z.irrigation_left_min > 0:
                z.moisture = min(z.field_capacity, z.moisture + IRRIGATION_RATE_PER_H * dt / 60)
                z.irrigation_left_min -= dt
                valve = True
            else:
                z.moisture = max(5.0, z.moisture - self.evaporation_per_hour(z, w) * dt / 60)
                valve = False
                in_window = 5 <= h < 9 or 18 <= h < 21
                if in_window and z.moisture < z.irrigate_below + self.rng.gauss(0, 1.5):
                    z.irrigation_left_min = self.rng.choice([30, 45, 60, 75])
                    valve = True
            if valve and not z.was_open:
                self.irrigation_events[z.zone_id] += 1
            z.was_open = valve

            moisture = z.moisture + self.rng.gauss(0, 0.4)
            air_t, soil_t = w["temp"], w["soil_t"]
            is_anom = False
            if self.rng.random() < self.anomaly_rate:
                is_anom = True
                kind = self.rng.choice(["spike", "dropout", "air_spike"])
                if kind == "spike":
                    moisture = self.rng.uniform(85, 100)
                elif kind == "dropout":
                    moisture = 0.0
                else:
                    air_t += self.rng.uniform(15, 25)
                self.anomaly_count[z.zone_id] += 1

            battery = 3.85 + 0.30 * w["sun"] - 0.0015 * days_elapsed + z.battery_offset + self.rng.gauss(0, 0.005)
            rows.append({
                "timestamp": self.t.isoformat() + "Z",
                "zone_id": z.zone_id,
                "device_id": z.device_id,
                "soil_moisture_pct": round(min(100.0, max(0.0, moisture)), 1),
                "soil_temp_c": round(soil_t, 1),
                "air_temp_c": round(air_t, 1),
                "air_humidity_pct": round(w["rh"], 1),
                "battery_v": round(battery, 2),
                "valve_open": valve,
                "is_anomaly": is_anom,
            })
        self.t += timedelta(minutes=dt)
        return rows


def post_json(url: str, payload, timeout: int = 60):
    data = json.dumps(payload).encode()
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except error.HTTPError as e:
        print(f"\nAPI xatosi {e.code}: {e.read().decode()[:300]}")
        sys.exit(1)
    except error.URLError as e:
        print(f"\nAPI ga ulanib bo'lmadi ({url}): {e.reason}")
        print("Server ishlayaptimi? Boshqa terminalda: uvicorn backend.app.main:app --reload")
        sys.exit(1)


def api_payload(row: dict) -> dict:
    return {k: row[k] for k in API_FIELDS}


def build_sim(args, start: datetime) -> Simulator:
    return Simulator(start, args.step_min, args.seed, args.anomaly_rate, args.mean_temp, args.zones)


def run_backfill(args):
    step = args.step_min
    end = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
    end -= timedelta(minutes=end.minute % step)
    start = end - timedelta(days=args.days)
    sim = build_sim(args, start)
    n_steps = int(args.days * 1440 / step)

    rows = []
    for _ in range(n_steps):
        rows.extend(sim.step())
    print(f"Yaratildi: {len(rows)} ta o'lchov ({args.days} kun, {len(sim.zones)} zona, har {step} daqiqada)")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"CSV saqlandi: {path}")

    if not args.no_api:
        url = args.api.rstrip("/") + "/readings/bulk"
        sent = 0
        for i in range(0, len(rows), 500):
            chunk = [api_payload(r) for r in rows[i:i + 500]]
            _, res = post_json(url, chunk)
            sent += res["inserted"]
            print(f"\rAPI ga yozildi: {sent}/{len(rows)}", end="", flush=True)
        print()

    print("\nXulosa:")
    for z in sim.zones:
        zr = [r for r in rows if r["zone_id"] == z.zone_id and not r["is_anomaly"]]
        m = [r["soil_moisture_pct"] for r in zr]
        print(f"  {z.zone_id}: sug'orishlar {sim.irrigation_events[z.zone_id]} marta, "
              f"namlik {min(m):.1f}-{max(m):.1f}%, sensor xatolari {sim.anomaly_count[z.zone_id]} ta")


def run_live(args):
    start = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
    sim = build_sim(args, start)
    url = args.api.rstrip("/") + "/readings"
    print(f"Jonli rejim: har {args.interval}s da bitta o'lchov (simulyatsiya vaqti har safar {args.step_min} daqiqa oldinga). Ctrl+C - to'xtatish.")
    try:
        while True:
            for row in sim.step():
                payload = api_payload(row)
                payload.pop("timestamp")  # jonli rejimda server o'z vaqtini qo'yadi
                _, res = post_json(url, payload, timeout=10)
                print(f"  {res['zone_id']}: namlik {res['soil_moisture_pct']}%  "
                      f"havo {res['air_temp_c']}C  klapan {'OCHIQ' if row['valve_open'] else 'yopiq'}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nTo'xtatildi.")


def main():
    p = argparse.ArgumentParser(description="HydroMind Agro sensor simulyatori")
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        sp.add_argument("--api", default="http://127.0.0.1:8000", help="API manzili")
        sp.add_argument("--seed", type=int, default=42, help="bir xil natija uchun (default 42)")
        sp.add_argument("--step-min", type=int, default=15, help="o'lchovlar oralig'i, daqiqa")
        sp.add_argument("--anomaly-rate", type=float, default=0.003, help="sensor xatosi ehtimoli (0 - o'chirish)")
        sp.add_argument("--mean-temp", type=float, default=28.0, help="o'rtacha havo harorati, C")
        sp.add_argument("--zones", type=int, default=3, choices=[1, 2, 3], help="zonalar soni")

    b = sub.add_parser("backfill", help="o'tgan davr uchun tarix yaratish")
    common(b)
    b.add_argument("--days", type=int, default=30)
    b.add_argument("--csv", help="CSV fayl yo'li (haqiqiy klapan holati va xato belgisi ham yoziladi)")
    b.add_argument("--no-api", action="store_true", help="API ga yozmasdan, faqat CSV")
    b.set_defaults(func=run_backfill)

    l = sub.add_parser("live", help="jonli demo rejimi")
    common(l)
    l.add_argument("--interval", type=float, default=2.0, help="o'lchovlar orasidagi real soniyalar")
    l.set_defaults(func=run_live)

    args = p.parse_args()
    if args.mode == "backfill" and args.no_api and not args.csv:
        p.error("--no-api bilan birga --csv ham ko'rsating")
    args.func(args)


if __name__ == "__main__":
    main()
