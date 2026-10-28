"""
Train and evaluate the injury-risk model on the real runners' dataset.

Run:  python train_model.py
Writes: model/model.joblib, eval/model_metrics.json, data/demo_squad.csv

Evaluation is done on athletes the model has NEVER seen (held out by athlete),
because the club will use the model on its own players, who are not in the
training data.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, build_training_table

SEED = 42
TARGET_RECALL = 0.50  # cut-off set to catch half of the injuries in the TRAINING data
FLAG_SHARE = 0.20     # used only for the gradient-boosting comparison
RAW = "data/week_approach_maskedID_timeseries.csv"


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def make_gb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=200, random_state=SEED)


def split_train_test(t):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(t, t["injury_today"], t["athlete_id"]))
    return t.iloc[tr_idx], t.iloc[te_idx]


def fit_final(raw_path=RAW):
    """Fit the deployed model (logistic regression on training athletes) and return the bundle.
    Also used by the app to retrain if the saved model can't be loaded on another sklearn version."""
    import sklearn
    t = build_training_table(pd.read_csv(raw_path))
    train, _ = split_train_test(t)
    Xtr, ytr = train[FEATURES].to_numpy(), train["injury_today"].to_numpy()
    model = make_lr().fit(Xtr, ytr)
    p_tr = model.predict_proba(Xtr)[:, 1]
    threshold = float(np.quantile(p_tr[ytr == 1], 1 - TARGET_RECALL))
    scaler = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    return {
        "pipeline": model,
        "features": FEATURES,
        "threshold": threshold,
        "base_rate": float(ytr.mean()),
        "means": dict(zip(FEATURES, scaler.mean_.tolist())),
        "scales": dict(zip(FEATURES, scaler.scale_.tolist())),
        "coefficients": {f: round(float(c), 4) for f, c in zip(FEATURES, lr.coef_[0])},
        "train_score_quantiles": np.quantile(p_tr, np.linspace(0, 1, 101)).tolist(),
        "sklearn_version": sklearn.__version__,
    }


def scores_at(y, flag):
    tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
    base = y.mean()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "flag_rate": round(float(flag.mean()), 4),
        "recall": round(float(tp / (tp + fn)) if (tp + fn) else 0.0, 4),
        "precision": round(float(prec), 4),
        "lift_vs_base_rate": round(float(prec / base), 2) if base else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    os.makedirs("model", exist_ok=True)
    os.makedirs("eval", exist_ok=True)

    t = build_training_table(pd.read_csv(RAW))
    y_all = t["injury_today"].to_numpy()
    g_all = t["athlete_id"].to_numpy()
    X_all = t[FEATURES].to_numpy()

    # ---- 5-fold cross-validation, grouped by athlete (stability check) ----
    cv = {"logistic_regression": [], "gradient_boosting": [], "acwr_only": []}
    for tr, te in GroupKFold(5).split(X_all, y_all, g_all):
        for name, mk in [("logistic_regression", make_lr), ("gradient_boosting", make_gb)]:
            m = mk().fit(X_all[tr], y_all[tr])
            cv[name].append(roc_auc_score(y_all[te], m.predict_proba(X_all[te])[:, 1]))
        cv["acwr_only"].append(roc_auc_score(y_all[te], t["acwr"].to_numpy()[te]))
    cv_summary = {k: {"mean_auc": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3)}
                  for k, v in cv.items()}

    # ---- Single held-out test: 20% of athletes never seen in training ----
    train, test = split_train_test(t)
    Xtr, ytr = train[FEATURES].to_numpy(), train["injury_today"].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(), test["injury_today"].to_numpy()

    model = make_lr().fit(Xtr, ytr)
    gb = make_gb().fit(Xtr, ytr)
    p_tr = model.predict_proba(Xtr)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    p_gb = gb.predict_proba(Xte)[:, 1]

    # Cut-off chosen on TRAINING data only: the score that catches 50% of training
    # injuries. A missed injury costs far more than a false alarm (see app, Evaluation tab).
    threshold = float(np.quantile(p_tr[ytr == 1], 1 - TARGET_RECALL))
    base_rate = float(ytr.mean())

    results = {
        "Baseline: always 'Low risk'": {"auc": 0.5, **scores_at(yte, np.zeros_like(yte))},
        "Baseline: flag if ACWR >= 1.5": {
            "auc": round(float(roc_auc_score(yte, test["acwr"])), 3),
            **scores_at(yte, (test["acwr"].to_numpy() >= 1.5).astype(int))},
        "Baseline: flag if injured in last 12 months": {
            "auc": round(float(roc_auc_score(yte, test["prior_injury_12m"])), 3),
            **scores_at(yte, test["prior_injury_12m"].to_numpy().astype(int))},
        "Logistic regression, default cut-off 0.50": {
            "auc": round(float(roc_auc_score(yte, p_te)), 3), **scores_at(yte, (p_te >= 0.5).astype(int))},
        f"Logistic regression, chosen cut-off {threshold:.4f}": {
            "auc": round(float(roc_auc_score(yte, p_te)), 3), **scores_at(yte, (p_te >= threshold).astype(int))},
        "Gradient boosting (comparison, same flag share)": {
            "auc": round(float(roc_auc_score(yte, p_gb)), 3),
            **scores_at(yte, (p_gb >= np.quantile(gb.predict_proba(Xtr)[:, 1], 1 - FLAG_SHARE)).astype(int))},
    }

    # Coefficients (on standardised features) - used for the "top 3 reasons".
    scaler = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    coefs = {f: round(float(c), 4) for f, c in zip(FEATURES, lr.coef_[0])}

    metrics = {
        "dataset": {
            "name": "Injury Prediction in Competitive Runners (Lövdal, den Hartigh & Azzopardi, 2021)",
            "licence": "CC0",
            "athletes": int(t["athlete_id"].nunique()),
            "rows": int(len(t)),
            "injuries": int(y_all.sum()),
            "row_definition": "One athlete on one day, with the 3 weeks of training before it",
            "label": "Time-loss injury on the day right after the 3-week window",
            "train_athletes": int(train["athlete_id"].nunique()),
            "test_athletes": int(test["athlete_id"].nunique()),
            "test_rows": int(len(test)),
            "test_injuries": int(yte.sum()),
        },
        "cv_5fold_by_athlete": cv_summary,
        "test_results": results,
        "threshold": threshold,
        "target_recall_on_train": TARGET_RECALL,
        "base_rate": base_rate,
        "coefficients": coefs,
    }
    with open("eval/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    bundle = fit_final(RAW)
    assert abs(bundle["threshold"] - threshold) < 1e-12
    joblib.dump(bundle, "model/model.joblib")

    # Test-set scores, so the app can redraw the trade-off for any cut-off.
    pd.DataFrame({"score": p_te, "injury": yte, "acwr": test["acwr"].to_numpy(),
                  "prior_injury_12m": test["prior_injury_12m"].to_numpy()}).to_csv(
        "eval/test_predictions.csv", index=False)

    # ---- Demo squad: 30 REAL held-out athletes, one recent week each ----
    test = test.assign(score=p_te)
    rng = np.random.default_rng(SEED)
    rows = []
    for aid, grp in test.groupby("athlete_id"):
        active = grp[(grp["km_this_week"] > 0) & (grp["sessions"] > 0)]
        if len(active):
            rows.append(active.iloc[rng.integers(len(active))])
    squad = pd.DataFrame(rows)
    # Top up from train-set athletes' weeks if fewer than 30 test athletes.
    need = 30 - len(squad)
    if need > 0:
        pool = test[(test["km_this_week"] > 0) & ~test.index.isin(squad.index)]
        squad = pd.concat([squad, pool.sample(need, random_state=SEED)])
    # Make sure the demo has a clear high-risk and a borderline case (real rows).
    hi = test[test["score"] >= np.quantile(p_te, 0.98)].sample(1, random_state=SEED)
    border = test[(test["score"] - threshold).abs() < 0.0008].sample(1, random_state=SEED)
    squad = pd.concat([squad.iloc[: 30 - 2], hi, border])
    keep = ["athlete_id", "day", "km_this_week", "km_last_week", "km_two_weeks_ago", "hi_km",
            "rest_days", "strength_sessions", "exertion",
            "prior_injury_12m", "injury_today"]
    squad = squad[keep].reset_index(drop=True)
    squad.insert(0, "player_id", [f"P-{i + 1:02d}" for i in range(len(squad))])
    # Convert 0-1 perceived scores back to the 0-10 scale staff use in the app.
    squad["exertion"] = (squad["exertion"] * 10).round(1)
    # Sleep is not in the dataset: demo values are set for protocol testing only.
    squad["sleep_hours"] = np.clip(rng.normal(7.8, 0.7, len(squad)).round(1), 5.5, 9.5)
    squad.loc[squad.index[-2], "sleep_hours"] = 6.5   # high-risk demo player also sleeps badly
    squad = squad.rename(columns={"athlete_id": "source_athlete_id", "day": "source_day",
                                  "injury_today": "actual_injury_next_day"})
    squad.to_csv("data/demo_squad.csv", index=False)

    print(json.dumps({"cv": cv_summary, "threshold": threshold, "base_rate": base_rate}, indent=2))
    for k, v in results.items():
        print(f"{k:55s} {v}")
    print("coefficients:", coefs)


if __name__ == "__main__":
    main()
