"""Прогноз урожая по результатам подсчёта.

Два метода:
  * physical  — масса = количество x поправка на невидимые плоды x средняя масса
                (когда известна средняя масса плодов и калибровка по полевому контролю);
  * regression — линейная/ридж-модель массы дерева по количеству плодов,
                обучаемая на CSV (tree_id, count, mass_kg) полевого контроля.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config

Z_95 = 1.959963984540054  # квантиль нормального распределения для 95%


def z_for_confidence(confidence: float) -> float:
    """Квантиль нормального распределения для заданной доверительной вероятности."""
    # аппроксимация inverse erf через разложение (достаточно для 0.80..0.99)
    p = min(max(float(confidence), 0.5), 0.9999)
    # используем таблицу: точность 1e-4
    table = {
        0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758,
    }
    if p in table:
        return table[p]
    # бинарный поиск по «эфистову» обратному erf
    lo, hi = 0.0, 6.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class YieldPoint:
    tree_id: int
    count: int
    mass_kg: float
    sigma_kg: float
    ci_lo: float
    ci_hi: float


@dataclass
class YieldForecast:
    method: str
    points: List[YieldPoint] = field(default_factory=list)
    total_kg: float = 0.0
    total_sigma: float = 0.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0
    params: dict = field(default_factory=dict)

    def per_tree(self) -> Dict[int, float]:
        return {p.tree_id: p.mass_kg for p in self.points}

    def as_rows(self) -> List[dict]:
        return [
            {
                "tree_id": p.tree_id,
                "count": p.count,
                "mass_kg": round(p.mass_kg, 2),
                "sigma_kg": round(p.sigma_kg, 2),
                "ci_lo": round(p.ci_lo, 2),
                "ci_hi": round(p.ci_hi, 2),
            }
            for p in self.points
        ]


def forecast_counts(per_tree: Dict[int, int], cfg: Config) -> YieldForecast:
    """Физический метод: count x visibility x mean_mass."""
    y = cfg.get("yield", {}) or {}
    method = str(y.get("method", "physical"))
    if method == "regression":
        return _forecast_regression(per_tree, cfg)
    return _forecast_physical(per_tree, cfg)


def _forecast_physical(per_tree: Dict[int, int], cfg: Config) -> YieldForecast:
    y = cfg.get("yield", {}) or {}
    mean_mass = float(y.get("mean_mass_g", 180.0))
    alpha = float(y.get("visibility_factor", 1.0))
    sigma_alpha = float(y.get("visibility_sigma", 0.08))
    sigma_mass = float(y.get("mass_sigma_g", 25.0))
    z = z_for_confidence(float(y.get("confidence", 0.95)))

    points: List[YieldPoint] = []
    total = 0.0
    total_var = 0.0
    for tree_id, count in sorted(per_tree.items()):
        n = int(count)
        mass = n * alpha * mean_mass / 1000.0
        # дисперсия: Пуассоновский счёт + неопределённости массы и калибровки
        var = (
            (alpha * mean_mass / 1000.0) ** 2 * n
            + (n * mean_mass / 1000.0) ** 2 * sigma_alpha ** 2
            + (n * alpha / 1000.0) ** 2 * sigma_mass ** 2
        )
        sigma = math.sqrt(max(var, 0.0))
        points.append(
            YieldPoint(
                tree_id=tree_id,
                count=n,
                mass_kg=mass,
                sigma_kg=sigma,
                ci_lo=max(0.0, mass - z * sigma),
                ci_hi=mass + z * sigma,
            )
        )
        total += mass
        total_var += var

    total_sigma = math.sqrt(total_var)
    return YieldForecast(
        method="physical",
        points=points,
        total_kg=total,
        total_sigma=total_sigma,
        ci_lo=max(0.0, total - z * total_sigma),
        ci_hi=total + z * total_sigma,
        params={
            "mean_mass_g": mean_mass,
            "visibility_factor": alpha,
            "visibility_sigma": sigma_alpha,
            "mass_sigma_g": sigma_mass,
            "confidence": float(y.get("confidence", 0.95)),
            "z": z,
        },
    )


# ---------------------------------------------------------------------------
# Регрессионный метод
# ---------------------------------------------------------------------------
def fit_regression(csv_path: str, ridge_lambda: float = 1e-3) -> dict:
    """Ридж-регрессия mass_kg ~ a * count + b по CSV (tree_id,count,mass_kg)."""
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    count_col = cols.get("count") or cols.get("n") or cols.get("fruit_count")
    mass_col = cols.get("mass_kg") or cols.get("kg") or cols.get("mass")
    if count_col is None or mass_col is None:
        raise ValueError(f"в {csv_path} нужны колонки count и mass_kg; найдено: {list(df.columns)}")

    x = df[count_col].to_numpy(dtype=float)
    y = df[mass_col].to_numpy(dtype=float)
    if len(x) < 3:
        raise ValueError("для регрессии нужно минимум 3 наблюдения")

    X = np.stack([x, np.ones_like(x)], axis=1)
    lam = float(ridge_lambda)
    penalty = np.diag([lam, 0.0])  # не регуляризируем сдвиг
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)

    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.abs(y - pred).mean())
    rmse = float(np.sqrt(((y - pred) ** 2).mean()))

    residuals = y - pred
    dof = max(1, len(x) - 2)
    sigma2 = ss_res / dof
    sigma = math.sqrt(sigma2)
    return {
        "a": float(beta[0]),
        "b": float(beta[1]),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "sigma_kg": sigma,
        "n": int(len(x)),
        "lambda": lam,
    }


def _forecast_regression(per_tree: Dict[int, int], cfg: Config) -> YieldForecast:
    y = cfg.get("yield", {}) or {}
    csv_path = y.get("regression_csv")
    if not csv_path:
        raise ValueError("yield.method=regression требует yield.regression_csv (tree_id,count,mass_kg)")
    model = fit_regression(csv_path, float(y.get("ridge_lambda", 1e-3)))
    z = z_for_confidence(float(y.get("confidence", 0.95)))

    points: List[YieldPoint] = []
    total = 0.0
    total_var = 0.0
    for tree_id, count in sorted(per_tree.items()):
        n = int(count)
        mass = model["a"] * n + model["b"]
        # неопределённость: дисперсия остатков модели + дисперсия счёта Пуассона
        var = model["sigma_kg"] ** 2 + (model["a"] ** 2) * max(n, 1) * 0.05 ** 2
        sigma = math.sqrt(max(var, 0.0))
        points.append(
            YieldPoint(
                tree_id=tree_id,
                count=n,
                mass_kg=mass,
                sigma_kg=sigma,
                ci_lo=max(0.0, mass - z * sigma),
                ci_hi=mass + z * sigma,
            )
        )
        total += mass
        total_var += var

    total_sigma = math.sqrt(total_var)
    return YieldForecast(
        method="regression",
        points=points,
        total_kg=total,
        total_sigma=total_sigma,
        ci_lo=max(0.0, total - z * total_sigma),
        ci_hi=total + z * total_sigma,
        params={**model, "confidence": float(y.get("confidence", 0.95)), "z": z},
    )
