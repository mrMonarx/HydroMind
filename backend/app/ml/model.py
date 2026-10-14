"""Namlik pasayishi prognozi: o'qitish, baholash, baseline'lar va prognozchi (forecaster)."""
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import HORIZONS_H, STEPS_PER_HOUR, ZONES
from .features import MODEL_FEATURES, STATE_FEATURES, future_arrays, make_training_set

EVAL_HORIZON_H = 12


# ----------------------------------------------------------------------------- bo'lish
def split_cutoff(df: pd.DataFrame, frac: float = 0.7) -> pd.Timestamp:
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    return t0 + (t1 - t0) * frac


def time_split(long: pd.DataFrame, cutoff: pd.Timestamp):
    """Vaqt bo'yicha bo'lish. O'qitish qatorlarining kelajagi (t + h) ham chegaradan oldin tugashi shart,
    aks holda sinov davridagi qiymatlar o'qitishga "sizib" kiradi."""
    train = long[long["timestamp"] + pd.to_timedelta(long["horizon_h"], unit="h") <= cutoff]
    test = long[long["timestamp"] >= cutoff]
    return train, test


# ----------------------------------------------------------------------------- modellar
def fit_model(train: pd.DataFrame, seed: int = 42) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=300, max_leaf_nodes=24, min_samples_leaf=60,
        l2_regularization=1.0, categorical_features=["zone_code"], random_state=seed,
    )
    model.fit(train[MODEL_FEATURES], train["drop"])
    return model


class ZoneHourBaseline:
    """Oddiy taqqoslash usuli: shu zona, shu soat va shu gorizont uchun tarixiy o'rtacha pasayish."""

    def fit(self, train: pd.DataFrame):
        t = train.assign(hour_bin=np.floor(train["hour_local"]).astype(int))
        self.table = t.groupby(["zone_id", "hour_bin", "horizon_h"])["drop"].mean()
        self.fallback = t.groupby(["zone_id", "horizon_h"])["drop"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        hb = np.floor(df["hour_local"]).astype(int)
        pred = self.table.reindex(pd.MultiIndex.from_arrays([df["zone_id"], hb, df["horizon_h"]])).to_numpy()
        miss = np.isnan(pred)
        if miss.any():
            pred[miss] = self.fallback.reindex(
                pd.MultiIndex.from_arrays([df["zone_id"][miss], df["horizon_h"][miss]])).to_numpy()
        return pred


def persistence_predict(df: pd.DataFrame) -> np.ndarray:
    """Eng sodda usul: so'nggi 3 soatdagi quruyish tezligi o'zgarmay davom etadi."""
    rate = np.maximum(-df["d_moist_3h"].to_numpy() / 3.0, 0.0)
    return rate * df["horizon_h"].to_numpy()


def mae_by_horizon(test: pd.DataFrame, pred: np.ndarray) -> pd.Series:
    err = np.abs(test["drop"].to_numpy() - pred)
    return pd.Series(err, index=test.index).groupby(test["horizon_h"]).mean()


# ----------------------------------------------------------------------------- prognozchi
class MoistureForecaster:
    """O'qitilgan model + har gorizont uchun kutilayotgan xato. Saqlanadi va API tomonidan yuklanadi."""

    def __init__(self, model, horizons: list[int], mae_by_h: dict[int, float], meta: dict | None = None):
        self.model = model
        self.horizons = list(horizons)
        self.mae_by_h = {int(k): float(v) for k, v in mae_by_h.items()}
        self.meta = meta or {}

    def predict_drops(self, state: pd.DataFrame) -> np.ndarray:
        """(n_qator, n_gorizont) pasayish. Sug'orishsiz holat uchun: manfiy emas va vaqt bilan kamaymaydi."""
        n, H = len(state), len(self.horizons)
        X = state[STATE_FEATURES].reset_index(drop=True)
        Xr = X.loc[X.index.repeat(H)].reset_index(drop=True)
        Xr["horizon_h"] = np.tile(self.horizons, n)
        d = self.model.predict(Xr[MODEL_FEATURES]).reshape(n, H)
        return np.maximum.accumulate(np.maximum(d, 0.0), axis=1)

    def forecast(self, state_row: pd.Series) -> list[dict]:
        drops = self.predict_drops(state_row.to_frame().T.infer_objects())[0]
        m = float(state_row["moisture"])
        return [
            {"hours": h, "moisture": m - d, "moisture_low": m - d - self.mae_by_h.get(h, 0.0)}
            for h, d in zip(self.horizons, drops)
        ]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "MoistureForecaster":
        return joblib.load(path)


# ----------------------------------------------------------------------------- qaror darajasidagi baholash
def build_eval_rows(df_feat: pd.DataFrame, cutoff: pd.Timestamp, horizon_h: int = EVAL_HORIZON_H):
    """Sinov qatorlari: sinov davrida, keyingi 12 soatda sug'orish bo'lmagan va namlik hozir chegaradan yuqori bo'lgan paytlar.
    Maqsad: "chegaradan pastga tushadimi?" savoliga oldindan javob berish."""
    steps = horizon_h * STEPS_PER_HOUR
    F, irr_any, valid = future_arrays(df_feat, steps)
    thr = df_feat["zone_id"].map({z: c.min_moisture for z, c in ZONES.items()}).to_numpy(float)
    ok = (
        valid & ~irr_any
        & df_feat[STATE_FEATURES].notna().all(axis=1).to_numpy()
        & (df_feat["timestamp"] >= cutoff).to_numpy()
        & (df_feat["moisture"].to_numpy() >= thr)
    )
    return df_feat.loc[ok].reset_index(drop=True), F[ok], thr[ok]


def crossing_metrics(thr: np.ndarray, F: np.ndarray, pred_m: np.ndarray) -> dict:
    """F: haqiqiy keyingi 12 soat (n, 48, har 15 daqiqa); pred_m: prognoz namlik (n, 12, har soat)."""
    true_cross = (F < thr[:, None]).any(axis=1)
    pred_cross = (pred_m < thr[:, None]).any(axis=1)
    tp = int((true_cross & pred_cross).sum())
    fp = int((~true_cross & pred_cross).sum())
    fn = int((true_cross & ~pred_cross).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else float("nan")
    both = true_cross & pred_cross
    true_h = (np.argmax(F < thr[:, None], axis=1) + 1) / STEPS_PER_HOUR
    pred_h = np.argmax(pred_m < thr[:, None], axis=1) + 1
    time_mae = float(np.abs(pred_h[both] - true_h[both]).mean()) if both.any() else float("nan")
    return {"n": len(thr), "positives": int(true_cross.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "time_mae_h": time_mae}


def curves_from_drops(m_now: np.ndarray, drops: np.ndarray) -> np.ndarray:
    return m_now[:, None] - np.maximum.accumulate(np.maximum(drops, 0.0), axis=1)


def baseline_drops(base: ZoneHourBaseline, rows: pd.DataFrame, horizons: list[int]) -> np.ndarray:
    cols = [base.predict(rows[["zone_id", "hour_local"]].assign(horizon_h=h)) for h in horizons]
    return np.column_stack(cols)


# ----------------------------------------------------------------------------- to'liq tajriba
def run_experiment(df_feat: pd.DataFrame, horizons: list[int] = HORIZONS_H, train_frac: float = 0.7) -> dict:
    """O'qitish (dastlabki 70% vaqt) -> sinov (qolgan 30%) -> baseline'lar bilan solishtirish."""
    long = make_training_set(df_feat, horizons)
    cutoff = split_cutoff(df_feat, train_frac)
    train, test = time_split(long, cutoff)

    model = fit_model(train)
    base = ZoneHourBaseline().fit(train)
    preds = {
        "model": model.predict(test[MODEL_FEATURES]),
        "zona-soat o'rtachasi": base.predict(test),
        "oxirgi tezlik": persistence_predict(test),
    }
    mae = pd.DataFrame({name: mae_by_horizon(test, p) for name, p in preds.items()})

    fc = MoistureForecaster(model, horizons, mae["model"].to_dict())
    rows, F, thr = build_eval_rows(df_feat, cutoff)
    m_now = rows["moisture"].to_numpy()
    ml_m = curves_from_drops(m_now, fc.predict_drops(rows))
    base_m = curves_from_drops(m_now, baseline_drops(base, rows, horizons))
    margin = np.array([fc.mae_by_h[h] for h in horizons])[None, :]
    decision = pd.DataFrame({
        "model": crossing_metrics(thr, F, ml_m),
        "model (xato zaxirasi bilan)": crossing_metrics(thr, F, ml_m - margin),
        "zona-soat o'rtachasi": crossing_metrics(thr, F, base_m),
        "hozirgi namlikka qarab (oldindan ko'rmaydi)": crossing_metrics(thr, F, np.repeat(m_now[:, None], len(horizons), axis=1)),
    }).T

    return {"long": long, "cutoff": cutoff, "train": train, "test": test, "model": model, "baseline": base,
            "mae": mae, "forecaster": fc, "eval_rows": rows, "eval_F": F, "eval_thr": thr, "decision": decision}


def fit_final(df_feat: pd.DataFrame, mae_by_h: dict, horizons: list[int] = HORIZONS_H, meta: dict | None = None):
    """Hamma ma'lumot bilan qayta o'qitib, saqlashga tayyor prognozchi qaytaradi."""
    long = make_training_set(df_feat, horizons)
    model = fit_model(long)
    meta = {**(meta or {}), "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_rows": int(len(long))}
    return MoistureForecaster(model, horizons, mae_by_h, meta)
