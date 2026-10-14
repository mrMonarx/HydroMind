"""Modelni bazadagi ma'lumot bilan o'qitadi va models/ papkasiga saqlaydi.

Ishga tushirish (loyiha ildizidan):  python scripts/train_model.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from backend.app.ml.cleaning import clean_readings, load_raw
from backend.app.ml.config import HORIZONS_H
from backend.app.ml.features import add_features
from backend.app.ml.model import fit_final, run_experiment

load_dotenv(ROOT / ".env")
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./hydromind.db")
if DB_URL.startswith("sqlite:///./"):
    DB_URL = "sqlite:///" + str(ROOT / DB_URL.replace("sqlite:///./", ""))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "moisture_model.joblib"))


def main():
    pd.set_option("display.width", 160)
    raw = load_raw(create_engine(DB_URL))
    if raw.empty:
        sys.exit("Baza bo'sh. Avval: python scripts/simulator.py backfill --days 90")
    days = (raw["timestamp"].max() - raw["timestamp"].min()).days
    print(f"Ma'lumot: {len(raw)} qator, {days} kun")
    if days < 30:
        print("OGOHLANTIRISH: 30 kundan kam ma'lumot. Natija ishonchsiz bo'lishi mumkin (tavsiya: 60-90 kun).")

    df = add_features(clean_readings(raw))
    print("Tajriba (dastlabki 70% vaqt - o'qitish, qolgan 30% - sinov)...")
    ex = run_experiment(df)

    print("\nPrognoz xatosi (MAE, namlik punkti), model va oddiy usullar:")
    print(ex["mae"].loc[[1, 3, 6, 9, 12]].round(2).to_string())
    print("\nQaror darajasi ('12 soat ichida chegaradan pastga tushadimi?'):")
    cols = ["n", "positives", "precision", "recall", "f1", "time_mae_h"]
    print(ex["decision"][cols].round(3).to_string())

    print("\nHamma ma'lumot bilan yakuniy modelni o'qitish...")
    fc = fit_final(df, ex["mae"]["model"].to_dict(), HORIZONS_H, meta={"days": days})
    MODEL_PATH.parent.mkdir(exist_ok=True)
    fc.save(MODEL_PATH)

    metrics = {
        "trained_at": fc.meta["trained_at"], "days": days, "rows": int(len(raw)),
        "mae_by_horizon": {int(k): round(float(v), 3) for k, v in ex["mae"]["model"].items()},
        "baseline_mae_by_horizon": {int(k): round(float(v), 3) for k, v in ex["mae"]["zona-soat o'rtachasi"].items()},
        "decision": json.loads(ex["decision"][cols].round(4).to_json(orient="index")),
    }
    (MODEL_PATH.parent / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaqlandi: {MODEL_PATH}")


if __name__ == "__main__":
    main()
