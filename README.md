# HydroMind Agro

IoT sensorlar va machine learning orqali sug'orishni ma'lumotga asoslangan qarorga aylantiruvchi tizim.
Holat: ishlab chiqish boshlang'ich bosqichida.

## Tuzilma

```
hydromind-agro/
  backend/app/      FastAPI server (ma'lumot qabul qilish, API)
  scripts/          check_env.py (muhit tekshiruvi), simulator.py (sensor simulyatori)
  notebooks/        tahlil notebook'lari (04_eda_va_tozalash.ipynb)
  data/             simulyator va tahlil yaratgan CSV fayllar (GitHub'ga yuklanmaydi)
  docs/             hujjatlar
  requirements.txt  Python kutubxonalari
  .env.example      sozlamalar namunasi
```

## O'rnatish

Talab: Python 3.10 yoki yangiroq, Git.

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python scripts/check_env.py
uvicorn backend.app.main:app --reload
```
Agar Activate.ps1 xato bersa, bir marta: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/check_env.py
uvicorn backend.app.main:app --reload
```

## Tekshirish
Brauzerda http://127.0.0.1:8000/health oching. `{"status":"ok", ...}` chiqsa, hammasi tayyor.
Avtomatik hujjat: http://127.0.0.1:8000/docs

## API (2-qadam)

| Metod | Yo'l | Vazifasi |
|---|---|---|
| POST | `/readings` | Sensor o'lchovini bazaga yozadi |
| POST | `/readings/bulk` | 1000 tagacha o'lchovni bir so'rovda yozadi (xato bo'lsa, hech biri yozilmaydi) |
| GET | `/readings` | O'lchovlar ro'yxati (`zone_id`, `device_id`, `since`, `limit` bilan filtrlash) |
| GET | `/zones/{zone_id}/latest` | Zonaning eng oxirgi o'lchovi |
| GET | `/health` | Server holati |

Namuna so'rov (`POST /readings`):
```json
{
  "device_id": "esp32_z1",
  "zone_id": "zone_1",
  "soil_moisture_pct": 34.7,
  "soil_temp_c": 24.1,
  "air_temp_c": 31.2,
  "air_humidity_pct": 42.5,
  "battery_v": 3.78
}
```
Majburiy faqat `device_id`, `zone_id` va `soil_moisture_pct`. Vaqt (`timestamp`) ko'rsatilmasa, server UTC vaqtini qo'yadi.
Real bo'lmagan qiymatlar (masalan namlik 140%) rad etiladi (HTTP 422).

Ma'lumotlar `hydromind.db` faylida saqlanadi (SQLite). Bazani nolga tushirish uchun serverni to'xtatib, shu faylni o'chiring.

## Sensor simulyatori (3-qadam)

Hardware bo'lmaganda realistik ma'lumot yaratadi. Server ishlab turishi kerak (yoki `--no-api` bilan faqat CSV).

```bash
# 30 kunlik tarixni (3 zona, har 15 daqiqada) bazaga yozish
python scripts/simulator.py backfill --days 30

# Faqat CSV ga (haqiqiy klapan holati va sensor xatosi belgisi bilan), API siz
python scripts/simulator.py backfill --days 30 --csv data/sim.csv --no-api

# Jonli demo: har 2 soniyada yangi o'lchov (dashboard uchun)
python scripts/simulator.py live --interval 2
```

Foydali parametrlar: `--zones 1..3`, `--seed 42` (bir xil natija), `--mean-temp 28`, `--anomaly-rate 0.003` (sensor xatolari, 0 bilan o'chadi).

Modellashtirilgan: havo harorati/namligining kunlik sikli (Toshkent vaqti), bug'lanish orqali tuproq namligi kamayishi,
zonalar tuprog'ining farqi (zone_1 qumloq, zone_3 gil), ertalab/kechqurun sug'orish va ba'zan sensor xatolari.

Eslatma: `backfill` ni ikki marta ishga tushirsangiz, bazada ma'lumot ikki baravar bo'ladi. Noldan boshlash uchun
serverni to'xtatib, `hydromind.db` ni o'chiring.

CSV dagi `valve_open` va `is_anomaly` ustunlari faqat simulyatorda bor (haqiqiy tizimda ular noma'lum bo'ladi),
ular keyingi qadamlarda tahlil va modelni tekshirish uchun "to'g'ri javob" vazifasini bajaradi.

## Ma'lumot tahlili (4-qadam)

Notebook: `notebooks/04_eda_va_tozalash.ipynb`. Sensor xatolarini topadi va tuzatadi, sug'orish hodisalarini ajratib oladi,
zonalarni solishtiradi, quruyish tezligini harorat/namlik/soat bilan bog'laydi va qurilma batareyasini kuzatadi.

Tayyorgarlik (yangi kutubxonalar uchun bir marta): `pip install -r requirements.txt`

Natijani "to'g'ri javob" bilan tekshirish uchun bazani CSV bilan birga to'ldiring (oldin `hydromind.db` ni o'chiring):
```bash
python scripts/simulator.py backfill --days 30 --csv data/sim.csv
```

Ochish (venv aktiv, loyiha papkasida):
```bash
jupyter lab
```
yoki VS Code'da `.ipynb` faylni ochib, kernel sifatida `.venv` ni tanlang. Keyin yuqoridan pastga hamma katakni ishga tushiring (Run All).

Natija: `data/clean_readings.csv` (tozalangan ma'lumot) va `data/irrigation_events.csv` (sug'orish hodisalari). Ular 5-qadamda (ML) ishlatiladi.
