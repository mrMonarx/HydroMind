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


Testlar xavfsizlik qoidalarini tekshiradi. O'tmasa, tizimni dalaga qo'ymang.

### Cheklov

Hamma narsa simulyatorda sinalgan. Haqiqiy sensor ma'lumotida natija pastroq bo'lishi kutiladi, shuning uchun pilotni
**tavsiya rejimida** (`autonomy=0`, har bir sug'orish fermer tasdig'i bilan) boshlang.
