"""
Feature definitions shared by training (train_model.py) and the app (app.py).

The model is trained on a REAL dataset: Lövdal, den Hartigh & Azzopardi (2021),
"Injury Prediction in Competitive Runners With Machine Learning", 74 elite
runners, 7 years of daily training logs, real injuries, CC0 licence.

One row = one athlete on one day, with the three weeks of training before it.
In the raw file, columns without a suffix are the most recent week (week 0),
".1" is the week before, ".2" is two weeks before (checked in train_model.py).
"""

import numpy as np
import pandas as pd

# The features the model uses, in order. Every one of them can be typed in by
# a sports scientist at the end of a week (see INPUT_FIELDS below).
FEATURES = [
    "acwr",
    "hi_share",
    "rest_days",
    "strength_sessions",
    "exertion",
    "prior_injury_12m",
]
# "Perceived training success" was dropped: 19 of the 74 athletes never logged it (stored
# as 0), so the model was reading blanks as "very low success". Filling the blanks and adding
# a "not recorded" flag scored 5-fold AUC 0.672; dropping the feature scored 0.665 (inside the
# fold-to-fold variation of about 0.01), so the simpler, cleaner 6-feature model is used.
TS_FILL = 0.69  # median recorded training success (kept only for the training table)
# Feature selection: a 12-feature version scored the same (5-fold AUC 0.669 vs
# 0.669), so the simpler 7-feature model is used because it is easier to explain.
# Note: the dataset also has a "perceived recovery" score. It is left out because
# removing it did not lower accuracy (5-fold AUC 0.669 without vs 0.663 with) and
# the dataset does not document which direction the scale runs.

# Plain-English names used in the "top 3 reasons" shown to staff.
FEATURE_LABELS = {
    "km_this_week": "weekly running distance",
    "acwr": "acute:chronic workload ratio (ACWR)",
    "week_change": "load change vs last week",
    "hi_km": "high-intensity distance",
    "hi_share": "share of distance at high intensity",
    "sessions": "number of sessions",
    "rest_days": "number of rest days",
    "tough_sessions": "number of very hard sessions",
    "strength_sessions": "number of strength sessions",
    "exertion": "average perceived exertion",
    "training_success": "average perceived training success",
    "success_not_recorded": "training success not recorded",
    "prior_injury_12m": "injury in the past 12 months",
}

# What staff type in each week, with the allowed range (input validation guardrail).
# (field, label, min, max, default, step)
INPUT_FIELDS = [
    ("km_this_week", "Total distance this week (km, GPS)", 0.0, 250.0, 45.0, 0.5),
    ("km_last_week", "Total distance last week (km)", 0.0, 250.0, 45.0, 0.5),
    ("km_two_weeks_ago", "Total distance two weeks ago (km)", 0.0, 250.0, 45.0, 0.5),
    ("hi_km", "High-intensity / sprint distance this week (km)", 0.0, 80.0, 3.0, 0.1),
    ("rest_days", "Rest days this week", 0, 7, 1, 1),
    ("strength_sessions", "Strength sessions this week", 0, 9, 1, 1),
    ("exertion", "Average perceived exertion (0-10)", 0.0, 10.0, 3.0, 0.5),
    ("prior_injury_12m", "Time-loss injury in the past 12 months (1 = yes, 0 = no)", 0, 1, 0, 1),
    ("sleep_hours", "Average sleep per night (hours) - protocol only, not in model", 0.0, 14.0, 8.0, 0.5),
]


def safe_ratio(num, den, empty_value):
    """num/den, with empty_value when den is 0 and num is 0, capped at 3."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.where(den > 0, num / np.where(den > 0, den, 1), np.where(num > 0, 3.0, empty_value))
    return np.clip(out, 0, 3.0)


def features_from_inputs(rec: dict) -> dict:
    """Turn one staff-entered weekly record into model features (same maths as training)."""
    km0 = float(rec["km_this_week"])
    km1 = float(rec["km_last_week"])
    km2 = float(rec["km_two_weeks_ago"])
    chronic = (km0 + km1 + km2) / 3.0
    hi = float(rec["hi_km"])
    return {
        "km_this_week": km0,
        "acwr": float(safe_ratio(km0, chronic, 1.0)),
        "week_change": float(safe_ratio(km0, km1, 1.0)),
        "hi_km": hi,
        "hi_share": float(min(hi / km0, 1.0)) if km0 > 0 else 0.0,
        "rest_days": float(rec["rest_days"]),
        "strength_sessions": float(rec["strength_sessions"]),
        # The dataset stores perceived scores on a 0-1 scale; the app asks for 0-10.
        "exertion": float(rec["exertion"]) / 10.0,
        "prior_injury_12m": float(rec["prior_injury_12m"]),
    }


def build_training_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Build features + label from the raw runners file."""
    df = raw.copy()
    km0, km1, km2 = df["total kms"], df["total kms.1"], df["total kms.2"]
    chronic = (km0 + km1 + km2) / 3.0

    out = pd.DataFrame({
        "athlete_id": df["Athlete ID"].astype(int),
        "day": df["Date"].astype(int),
        "km_this_week": km0,
        "km_last_week": km1,
        "km_two_weeks_ago": km2,
        "acwr": safe_ratio(km0, chronic, 1.0),
        "week_change": safe_ratio(km0, km1, 1.0),
        "hi_km": df["total km Z5-T1-T2"],
        "hi_share": np.where(km0 > 0, np.minimum(df["total km Z5-T1-T2"] / km0.replace(0, 1), 1.0), 0.0),
        "sessions": df["nr. sessions"],
        "rest_days": df["nr. rest days"],
        "tough_sessions": df["nr. tough sessions (effort in Z5, T1 or T2)"],
        "strength_sessions": df["nr. strength trainings"],
        "exertion": df["avg exertion"],
        "recovery": df["avg recovery"],
        "training_success": df["avg training success"].where(df["avg training success"] > 0, TS_FILL),
        "success_not_recorded": (df["avg training success"] <= 0).astype(int),
        "injury_today": df["injury"].astype(int),
    })

    # Label: a time-loss injury in the next 7 days (this day or the 6 after it).
    # Prior injury: any injury in the 365 days before this day (known to staff).
    out["injury_next_7d"] = 0
    out["prior_injury_12m"] = 0
    for aid, grp in out.groupby("athlete_id"):
        inj_days = np.sort(grp.loc[grp["injury_today"] == 1, "day"].to_numpy())
        days = grp["day"].to_numpy()
        if len(inj_days) == 0:
            continue
        # next-7-day label
        idx = np.searchsorted(inj_days, days, side="left")
        nxt = np.where(idx < len(inj_days), inj_days[np.minimum(idx, len(inj_days) - 1)], 10**9)
        out.loc[grp.index, "injury_next_7d"] = ((nxt - days) <= 6).astype(int)
        # prior injury in last 365 days (strictly before today)
        idx_prev = np.searchsorted(inj_days, days, side="left") - 1
        prev = np.where(idx_prev >= 0, inj_days[np.maximum(idx_prev, 0)], -10**9)
        out.loc[grp.index, "prior_injury_12m"] = ((days - prev) <= 365).astype(int)

    return out
