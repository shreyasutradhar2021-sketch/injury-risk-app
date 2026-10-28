"""
Predictive model wrapper: score one player-week, explain it, and run what-if checks.
"""

import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_LABELS, FEATURES, INPUT_FIELDS, features_from_inputs

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.joblib")


def load_model(path: str = MODEL_PATH) -> Dict:
    """Load the saved model; if it is missing or was saved with another scikit-learn
    version (e.g. on Streamlit Cloud), retrain it from the CSV (takes a few seconds)."""
    import sklearn
    try:
        bundle = joblib.load(path)
        if bundle.get("sklearn_version") != sklearn.__version__:
            raise ValueError("scikit-learn version changed")
        return bundle
    except Exception:  # noqa: BLE001
        from train_model import fit_final
        raw = os.path.join(os.path.dirname(__file__), "data", "week_approach_maskedID_timeseries.csv")
        bundle = fit_final(raw)
        try:
            joblib.dump(bundle, path)
        except Exception:  # noqa: BLE001  (read-only file system)
            pass
        return bundle


def validate_inputs(rec: Dict) -> List[str]:
    """Input guardrail: every field present, numeric, inside its range."""
    errors = []
    for key, label, lo, hi, _default, _step in INPUT_FIELDS:
        val = rec.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).strip() == "":
            errors.append(f"{label}: missing")
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            errors.append(f"{label}: not a number ({val!r})")
            continue
        if v < lo or v > hi:
            errors.append(f"{label}: {v:g} is outside the allowed range {lo:g}-{hi:g}")
    # Cross-field checks
    try:
        if float(rec.get("hi_km", 0)) > float(rec.get("km_this_week", 0)):
            errors.append("High-intensity distance cannot be more than total distance this week")
    except (TypeError, ValueError):
        pass
    return errors


def score(model: Dict, rec: Dict, borderline_band: float = 0.10) -> Dict:
    """Score one validated record. Returns everything the UI, protocol and LLM need."""
    feats = features_from_inputs(rec)
    x = np.array([[feats[f] for f in FEATURES]])
    p = float(model["pipeline"].predict_proba(x)[0, 1])
    thr = float(model["threshold"])
    level = "High" if p >= thr else "Low"
    borderline = abs(p - thr) / thr <= borderline_band

    # Percentile among training player-weeks
    q = np.array(model["train_score_quantiles"])
    pct = int(np.clip(np.searchsorted(q, p), 0, 100))

    # Reasons: contribution of each feature = coefficient x standardised value
    contrib = {}
    for f in FEATURES:
        z = (feats[f] - model["means"][f]) / model["scales"][f]
        contrib[f] = model["coefficients"][f] * z
    up = sorted([(f, c) for f, c in contrib.items() if c > 0], key=lambda t: -t[1])[:3]
    reasons = [describe_reason(f, feats, rec) for f, _ in up]

    return {
        "probability": p,
        "threshold": thr,
        "risk_level": level,
        "borderline": borderline,
        "risk_index": p / model["base_rate"],   # x times the average player-week
        "percentile": pct,
        "features": feats,
        "contributions": contrib,
        "reasons": reasons,
    }


def describe_reason(f: str, feats: Dict, rec: Dict) -> str:
    v = feats[f]
    if f == "acwr":
        return f"ACWR of {v:.2f} (this week {rec['km_this_week']:g} km vs 3-week average)"
    if f == "hi_share":
        return f"{v * 100:.0f}% of distance at high intensity"
    if f == "rest_days":
        n = int(v)
        return "no rest day this week" if n == 0 else ("only 1 rest day this week" if n == 1 else f"{n} rest days this week")
    if f == "strength_sessions":
        return f"{int(v)} strength session{'s' if int(v) != 1 else ''} this week"
    if f == "exertion":
        return f"average perceived exertion {rec['exertion']:g}/10"
    if f == "training_success":
        return f"low perceived training success ({rec['training_success']:g}/10)"
    if f == "success_not_recorded":
        return "training success not recorded this week"
    if f == "prior_injury_12m":
        return "time-loss injury in the past 12 months"
    return FEATURE_LABELS.get(f, f)


def facts_for_protocol(rec: Dict, result: Dict) -> Dict:
    """The numbers the protocol's `Applies when:` lines are checked against."""
    f = result["features"]
    return {
        "risk_level": result["risk_level"],
        "borderline": "yes" if result["borderline"] else "no",
        "acwr": round(f["acwr"], 2),
        "hi_share": round(f["hi_share"], 3),
        "rest_days": int(f["rest_days"]),
        "exertion": float(rec["exertion"]),
        "sleep_hours": float(rec["sleep_hours"]),
        "prior_injury_12m": int(rec["prior_injury_12m"]),
    }


def what_if(model: Dict, rec: Dict, km_cut_pct: float = 0, extra_rest: int = 0,
            hi_cut_pct: float = 0) -> Dict:
    """Risk if next week's load is reduced (shifts the 3-week window forward)."""
    nxt = dict(rec)
    new_km = rec["km_this_week"] * (1 - km_cut_pct / 100)
    nxt["km_two_weeks_ago"] = rec["km_last_week"]
    nxt["km_last_week"] = rec["km_this_week"]
    nxt["km_this_week"] = new_km
    nxt["hi_km"] = min(rec["hi_km"] * (1 - hi_cut_pct / 100), new_km)
    nxt["rest_days"] = min(7, int(rec["rest_days"]) + extra_rest)
    return score(model, nxt)
