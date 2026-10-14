# HydroMind Agro

IoT sensorlar va machine learning orqali sug'orishni ma'lumotga asoslangan qarorga aylantiruvchi tizim.
Holat: ishlab chiqish boshlang'ich bosqichida.

## Tuzilma

```
hydromind-agro/
  backend/app/      FastAPI server (ma'lumot qabul qilish, API)
  scripts/          check_env.py (muhit tekshiruvi), simulator.py (sensor simulyatori)
  backend/app/ml/   ML: sozlamalar, tozalash, belgilar, model, qaror dvigateli
  notebooks/        tahlil notebook'lari (04: EDA, 05: ML va qaror)
  models/           o'qitilgan model (GitHub'ga yuklanmaydi)
  tests/            avtomatik testlar (xavfsizlik qoidalari)
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
| GET | `/zones/{zone_id}/recommendation` | ML prognozi va xavfsizlik qoidalari asosida sug'orish tavsiyasi |
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

## ML va qaror dvigateli (5-qadam)

Tizim ikki qismdan iborat: **model** keyingi 1-12 soatda namlik qanchaga pasayishini prognoz qiladi,
**qaror dvigateli** (oddiy qoidalar) esa sug'orish kerakmi, qancha va tasdiq kerakmi, buni hal qiladi.
Model xato qilsa ham, qoidalar belgilagan chegaradan chiqib bo'lmaydi.

### Ishga tushirish

```bash
pip install -r requirements.txt                       # yangi kutubxonalar (bir marta)
python scripts/simulator.py backfill --days 90        # kamida 30, tavsiya 90 kun (oldin hydromind.db ni o'chiring)
python scripts/train_model.py                         # modelni o'qitadi: models/moisture_model.joblib
uvicorn backend.app.main:app --reload
```

Tavsiya olish (brauzerda `/docs` orqali yoki):
```
GET /zones/zone_1/recommendation                      # hozirgi holat uchun
GET /zones/zone_1/recommendation?autonomy=1           # low xavfli sug'orish avtomatik
GET /zones/zone_1/recommendation?as_of=2026-10-03T13:00:00Z   # o'tgan lahza uchun (demo/tekshiruv)
```
Eslatma: ma'lumot 45 daqiqadan eski bo'lsa, tizim ataylab qaror chiqarmaydi (alert). Demo uchun `as_of` ishlating yoki
`python scripts/simulator.py live` bilan jonli ma'lumot yuboring.

### Javob (qisqacha)

| Maydon | Ma'nosi |
|---|---|
| `action` | `irrigate` (sug'orish), `wait` (kutish), `alert` (inson e'tibori kerak) |
| `minutes` | tavsiya etilgan davomiylik (15 daqiqa qadam bilan) |
| `risk` | `low`, `medium`, `high` |
| `requires_confirmation` | fermer tasdig'i kerakmi (avtonomlik darajasiga bog'liq) |
| `reasons` | qaror sabablari (o'zbek tilida) |
| `forecast` | 1-12 soatlik namlik prognozi va pessimistik bahosi |

### Xavfsizlik qoidalari (backend/app/ml/decision.py)

1. Ma'lumot shubhali yoki eskirgan bo'lsa, qaror chiqarilmaydi
2. Tuproq to'yingan bo'lsa (max_moisture), sug'orilmaydi
3. Ikki sug'orish orasida minimal vaqt, kunlik jami limit, bitta sug'orishning maksimal davomiyligi
4. Sug'orish faqat ruxsat etilgan oynalarda (standart: 5-9 va 18-21); oynadan tashqarida faqat favqulodda holatda va **faqat fermer tasdig'i bilan**
5. Avtonomlik: `0` hammasi tasdiq bilan (boshlang'ich), `1` faqat low xavf avtomatik, `2` low va medium avtomatik. `high` har doim tasdiq talab qiladi.

Chegaralar (`min_moisture`, `critical_moisture`, `target_moisture`, ...) `backend/app/ml/config.py` da. Bular ML natijasi emas,
**agronom yoki fermer bilan kelishilishi kerak bo'lgan qiymatlar**; tizim ularni o'zi o'zgartirmaydi.

### Testlar

```bash
python -m pytest -q
```
Testlar xavfsizlik qoidalarini tekshiradi. O'tmasa, tizimni dalaga qo'ymang.

### Cheklov

Hamma narsa simulyatorda sinalgan. Haqiqiy sensor ma'lumotida natija pastroq bo'lishi kutiladi, shuning uchun pilotni
**tavsiya rejimida** (`autonomy=0`, har bir sug'orish fermer tasdig'i bilan) boshlang.
