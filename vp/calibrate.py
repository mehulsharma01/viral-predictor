"""Step 3: learn from your own posted reels, and measure how often the predictor is right.

Each posted reel is logged with its prediction (made BEFORE posting) and its real
Insights. Accuracy is measured on those stored predictions, so it is honest:
the model never sees a reel's real numbers before predicting it.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from datetime import date
from pathlib import Path

import numpy as np

from .analyze import DATA, MODEL_PATH

OUTCOMES = DATA / "outcomes.csv"
FEATURES = ["action_start", "early_flashes", "hook_shots", "static_tail_share", "quiet_start",
            "duration", "ai_hook_score", "ai_movement_1s", "ai_curiosity_gap", "ai_share_trigger", "ai_original"]
COLUMNS = (["name", "logged_on", "skip_rate", "views", "likes", "comments", "shares", "saves", "reposts",
            "non_followers_pct", "drop_start", "drop_end", "predicted_skip", "predicted_overall",
            "finding_ranges"] + FEATURES)
MIN_FIT = 8


def read_rows() -> list[dict]:
    if not OUTCOMES.exists():
        return []
    with OUTCOMES.open() as f:
        return list(csv.DictReader(f))


def log(report_path: str, skip_rate: float, drop: str | None = None, **metrics) -> dict:
    r = json.loads(Path(report_path).read_text())
    ranges = ";".join(f'{f["start"]}-{f["end"]}' for f in r["findings"] if f["severity"] >= 2)
    ds, de = (drop.split("-") + [""])[:2] if drop else ("", "")
    row = {"name": r["name"], "logged_on": date.today().isoformat(), "skip_rate": skip_rate,
           "drop_start": ds, "drop_end": de, "predicted_skip": r["predicted"]["skip_rate"],
           "predicted_overall": r["scores"]["overall"], "finding_ranges": ranges,
           **{k: metrics.get(k, "") for k in ("views", "likes", "comments", "shares", "saves", "reposts",
                                               "non_followers_pct")},
           **{k: r["features"].get(k, "") for k in FEATURES}}
    new = not OUTCOMES.exists()
    OUTCOMES.parent.mkdir(parents=True, exist_ok=True)
    with OUTCOMES.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def accuracy(rows: list[dict]) -> dict:
    """The three numbers that tell you whether to trust the predictor."""
    out = {"reels": len(rows)}
    pairs = [(_f(r["predicted_skip"]), _f(r["skip_rate"])) for r in rows]
    pairs = [(p, a) for p, a in pairs if p is not None and a is not None]
    if pairs:
        out["skip_within_10pts"] = round(sum(abs(p - a) <= 10 for p, a in pairs) / len(pairs) * 100)
    ordered = [(a, b) for a, b in itertools.combinations(pairs, 2) if abs(a[1] - b[1]) >= 5]
    if ordered:
        out["which_is_better_correct"] = round(
            sum((a[0] - b[0]) * (a[1] - b[1]) > 0 for a, b in ordered) / len(ordered) * 100)
        out["pairs_compared"] = len(ordered)
    hits = []
    for r in rows:
        ds, de = _f(r["drop_start"]), _f(r["drop_end"])
        if ds is None or de is None:
            continue
        ranges = [tuple(map(float, x.split("-"))) for x in r["finding_ranges"].split(";") if x]
        hits.append(any(s < de and e > ds for s, e in ranges))
    if hits:
        out["reason_matches_real_drop"] = round(sum(hits) / len(hits) * 100)
        out["reels_with_drop_marked"] = len(hits)
    return out


def fit(rows: list[dict], lam: float = 1.0) -> dict | None:
    usable = [r for r in rows if _f(r["skip_rate"]) is not None]
    if len(usable) < MIN_FIT:
        return None
    X = np.array([[_f(r[k]) or 0.0 for k in FEATURES] for r in usable])
    y = np.array([min(max(_f(r["skip_rate"]) / 100, 0.02), 0.98) for r in usable])
    z = np.log(y / (1 - y))
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + lam * np.eye(len(FEATURES))
    coef = np.linalg.solve(A, Xs.T @ (z - z.mean()))
    return {"intercept": float(z.mean()), "coef": dict(zip(FEATURES, map(float, coef))),
            "mean": dict(zip(FEATURES, map(float, mu))), "std": dict(zip(FEATURES, map(float, sd)))}


def loo_check(rows: list[dict]) -> dict | None:
    """Leave-one-out: predict each reel with a model fitted on all the others."""
    usable = [r for r in rows if _f(r["skip_rate"]) is not None]
    if len(usable) < MIN_FIT + 1:
        return None
    sim = []
    for i, r in enumerate(usable):
        m = fit(usable[:i] + usable[i + 1:])
        z = m["intercept"] + sum(m["coef"][k] * ((_f(r[k]) or 0) - m["mean"][k]) / m["std"][k] for k in FEATURES)
        sim.append(dict(r, predicted_skip=round(100 / (1 + math.exp(-z)), 1)))
    return accuracy(sim)


def calibrate() -> dict:
    rows = read_rows()
    result = {"logged_predictions": accuracy(rows)}
    m = fit(rows)
    if m:
        MODEL_PATH.write_text(json.dumps({"n": len(rows), "skip_model": m}, indent=1))
        result["calibrated_model_leave_one_out"] = loo_check(rows)
        result["model"] = str(MODEL_PATH)
    else:
        result["model"] = f"not fitted: need {MIN_FIT} reels with a skip rate, have {len(rows)}"
    return result
