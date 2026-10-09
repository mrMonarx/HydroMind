# HydroMind Agro

IoT sensorlar va machine learning orqali sug'orishni ma'lumotga asoslangan qarorga aylantiruvchi tizim.
Holat: ishlab chiqish boshlang'ich bosqichida.

## Tuzilma

```
hydromind-agro/
  backend/app/      FastAPI server (ma'lumot qabul qilish, API)
  scripts/          yordamchi skriptlar (muhitni tekshirish, simulyator)
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
