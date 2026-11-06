import math

import numpy as np
import pytest

from applecounter.config import Config
from applecounter.yieldpred.model import (
    fit_regression,
    forecast_counts,
    z_for_confidence,
)


def test_z_for_confidence():
    assert z_for_confidence(0.95) == pytest.approx(1.96, abs=0.01)
    assert z_for_confidence(0.99) == pytest.approx(2.576, abs=0.01)
    assert z_for_confidence(0.90) == pytest.approx(1.645, abs=0.01)


def test_physical_forecast():
    cfg = Config({"yield": {"method": "physical", "mean_mass_g": 200.0,
                            "visibility_factor": 1.0, "visibility_sigma": 0.0,
                            "mass_sigma_g": 0.0, "confidence": 0.95}})
    f = forecast_counts({1: 10, 2: 20}, cfg)
    assert f.method == "physical"
    assert f.total_kg == pytest.approx((10 + 20) * 0.2, rel=1e-6)
    assert [p.tree_id for p in f.points] == [1, 2]
    assert f.points[1].mass_kg == pytest.approx(4.0)
    assert f.ci_lo <= f.total_kg <= f.ci_hi


def test_physical_forecast_uncertainty_positive():
    cfg = Config({"yield": {"method": "physical", "mean_mass_g": 180.0,
                            "visibility_sigma": 0.1, "mass_sigma_g": 30.0}})
    f = forecast_counts({1: 50}, cfg)
    assert f.total_sigma > 0
    assert f.ci_hi > f.total_kg > f.ci_lo >= 0


def test_regression_fit_and_forecast(tmp_path):
    counts = np.array([10, 20, 30, 40, 50], dtype=float)
    mass = 0.2 * counts + 1.5
    csv = tmp_path / "field.csv"
    lines = ["tree_id,count,mass_kg"]
    lines += [f"{i + 1},{int(c)},{m:.3f}" for i, (c, m) in enumerate(zip(counts, mass))]
    csv.write_text("\n".join(lines), encoding="utf-8")

    model = fit_regression(str(csv))
    assert model["a"] == pytest.approx(0.2, abs=1e-4)
    assert model["b"] == pytest.approx(1.5, abs=1e-4)
    assert model["r2"] == pytest.approx(1.0, abs=1e-6)

    cfg = Config({"yield": {"method": "regression", "regression_csv": str(csv)}})
    f = forecast_counts({1: 25}, cfg)
    assert f.method == "regression"
    assert f.points[0].mass_kg == pytest.approx(0.2 * 25 + 1.5, rel=1e-6)
    assert f.total_sigma >= 0
    assert math.isfinite(f.ci_hi)


def test_regression_requires_csv():
    cfg = Config({"yield": {"method": "regression", "regression_csv": None}})
    with pytest.raises(ValueError):
        forecast_counts({1: 10}, cfg)


def test_visibility_fraction_corrects_hidden_fruit():
    cfg = Config({"yield": {"method": "physical", "mean_mass_g": 200.0,
                            "visibility_factor": 0.5, "visibility_sigma": 0.0,
                            "mass_sigma_g": 0.0}})
    forecast = forecast_counts({1: 10}, cfg)
    assert forecast.total_kg == pytest.approx(4.0)
    assert forecast.estimate_type == "illustrative_unless_field_calibrated"
    with pytest.raises(ValueError):
        forecast_counts({1: 10}, Config({"yield": {"visibility_factor": 0.0}}))


def test_regression_external_holdout(tmp_path):
    train = tmp_path / "train.csv"
    train.write_text("tree_id,count,mass_kg\n1,10,2\n2,20,4\n3,30,6\n4,40,8\n",
                     encoding="utf-8")
    holdout = tmp_path / "holdout.csv"
    holdout.write_text("tree_id,count,mass_kg\n5,15,3\n6,35,7\n", encoding="utf-8")
    model = fit_regression(str(train), validation_csv=str(holdout))
    assert model["validation"]["method"] == "external_holdout"
    assert model["validation"]["n"] == 2
    assert model["validation"]["mae"] == pytest.approx(0.0, abs=1e-3)
    loo = fit_regression(str(train))
    assert loo["validation"]["method"] == "leave_one_out"
