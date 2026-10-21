"""Baselines, feature-necessity review, three tuned algorithms, threshold selection, evaluation, MLflow.

Protocol (continuity with Module 3):
  train      -> hyper-parameter search with StratifiedGroupKFold (grouped by patient_key) + final fit
  validation -> champion selection, operating thresholds (70 % / 90 % recall), group thresholds for mitigation
  test       -> reported once, never used for any choice
"""
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from m4 import config as C

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------------------------------------------------ helpers
def make_preprocessor(features=None):
    features = features or C.FEATURES
    num = [f for f in C.NUMERIC if f in features]
    binary = [f for f in C.BINARY if f in features]
    cat = [f for f in C.CATEGORICAL if f in features]
    return ColumnTransformer(
        [("num", StandardScaler(), num), ("bin", "passthrough", binary),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        verbose_feature_names_out=False)


def make_cv():
    return StratifiedGroupKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)


def candidate_models(y_train):
    pos = y_train.mean()
    spw = (1 - pos) / pos
    models = {
        "logistic_regression": (
            LogisticRegression(class_weight="balanced", max_iter=3000, solver="liblinear"),
            {"clf__C": loguniform(1e-3, 1e1), "clf__penalty": ["l1", "l2"]}),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=C.RANDOM_STATE),
            {"clf__n_estimators": [200, 300], "clf__max_depth": [8, 12, 16, None],
             "clf__min_samples_leaf": [5, 10, 20, 50], "clf__max_features": ["sqrt", 0.5]}),
    }
    if HAS_XGB:
        models["xgboost"] = (
            XGBClassifier(eval_metric="logloss", tree_method="hist", scale_pos_weight=spw,
                          random_state=C.RANDOM_STATE, n_jobs=1),   # single-threaded inside parallel CV workers (Windows stability)
            {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [3, 4, 5, 6],
             "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.7, 0.85, 1.0],
             "clf__colsample_bytree": [0.7, 0.85, 1.0], "clf__min_child_weight": [1, 5, 10]})
    else:
        models["hist_gradient_boosting"] = (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=C.RANDOM_STATE),
            {"clf__max_iter": [200, 400], "clf__max_depth": [3, 5, None],
             "clf__learning_rate": [0.03, 0.05, 0.1], "clf__min_samples_leaf": [20, 50, 100]})
    return models


def _fmt(v):
    if isinstance(v, float):
        return float(f"{v:.4g}")
    return v if isinstance(v, (int, str)) or v is None else str(v)


def threshold_for_recall(y, p, target):
    """Highest threshold whose recall is still >= target (i.e. the most precise operating point)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred), "f1": f1_score(y, pred), "pr_auc": average_precision_score(y, p),
           "flag_rate": float(pred.mean()), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    try:
        out["roc_auc"] = roc_auc_score(y, p)
        out["brier"] = brier_score_loss(y, np.clip(p, 0, 1))
    except ValueError:
        out["roc_auc"], out["brier"] = np.nan, np.nan
    return out


TRUSTED_SKOPS = ["sklearn.tree._tree.Tree", "xgboost.sklearn.XGBClassifier", "xgboost.core.Booster",
                 "numpy.random._pickle.__generator_ctor", "numpy.random._pcg64.PCG64",
                 "numpy.random.bit_generator.BitGenerator"]


def _log_model(pipe, registered_name=None):
    """Log the fitted pipeline as an MLflow model artefact. MLflow >= 3.15 defaults to the skops format,
    which refuses tree models (sklearn.tree._tree.Tree) unless trusted, so we try cloudpickle first,
    then skops with the known-safe types, then older signatures. Failure is non-fatal."""
    base = {"registered_model_name": registered_name} if registered_name else {}
    attempts = [
        {"name": "model", "serialization_format": "cloudpickle"},
        {"name": "model", "skops_trusted_types": TRUSTED_SKOPS},
        {"artifact_path": "model", "serialization_format": "cloudpickle"},
        {"name": "model"},
        {"artifact_path": "model"},
    ]
    last = None
    for kw in attempts:
        try:
            return mlflow.sklearn.log_model(pipe, **kw, **base)
        except Exception as e:          # TypeError (unknown kwarg) or MlflowException (untrusted types)
            last = e
    print(f"  WARNING: could not log the model artefact to MLflow ({str(last)[:120]}); "
          "metrics/params are logged, the joblib file is still written.")
    return None


def _plot_confusion(y, p, t, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay(confusion_matrix(y, (p >= t).astype(int)),
                           display_labels=["Attended", "No-show"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ------------------------------------------------------------------ necessity review (Module 3 open item)
def necessity_review(train):
    """Module 3 kept `alcoholism` and `has_disability` 'for bias audit; necessity review in Module 4'.
    Data-minimisation test: does a fast model lose measurable skill without each flag?
    A flag is kept as an INPUT only if removing it costs more than NECESSITY_TOLERANCE CV PR-AUC;
    it is always kept for the fairness audit."""
    y, g = train[C.TARGET].values, train[C.GROUP_KEY].values
    cv = make_cv()

    def score(features, dropped=()):
        from m4.data import apply_minimisation
        X = apply_minimisation(train, dropped)
        pipe = Pipeline([("prep", make_preprocessor(features)),
                         ("clf", HistGradientBoostingClassifier(class_weight="balanced", max_iter=200,
                                                                learning_rate=0.05, random_state=C.RANDOM_STATE))])
        s = cross_val_score(pipe, X[features], y, groups=g, cv=cv, scoring="average_precision", n_jobs=-1)
        return float(s.mean()), float(s.std())

    full, full_sd = score(C.FEATURES)
    rows, keep = [], []
    for f in C.SENSITIVE_HEALTH_FLAGS:
        without, sd = score([x for x in C.FEATURES if x != f], dropped=(f,))
        delta = full - without
        decision = "keep as input" if delta > C.NECESSITY_TOLERANCE else "drop as input (audit only)"
        rows.append({"feature": f, "cv_pr_auc_with": full, "cv_pr_auc_without": without,
                     "delta": delta, "decision": decision})
        if delta > C.NECESSITY_TOLERANCE:
            keep.append(f)
    dropped = [f for f in C.SENSITIVE_HEALTH_FLAGS if f not in keep]
    final = [f for f in C.FEATURES if f not in dropped]
    return {"table": rows, "baseline_cv_pr_auc": full, "baseline_sd": full_sd,
            "dropped": dropped, "final_features": final, "tolerance": C.NECESSITY_TOLERANCE}


# ------------------------------------------------------------------ training
def train_all(parts, data_info, features):
    C.FIG.mkdir(parents=True, exist_ok=True); C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = parts["train"], parts["validation"], parts["test"]
    Xtr, ytr, gtr = train[features], train[C.TARGET].values, train[C.GROUP_KEY].values
    Xva, yva = val[features], val[C.TARGET].values
    Xte, yte = test[features], test[C.TARGET].values
    cv = make_cv()

    if HAS_MLFLOW:
        mlflow.set_tracking_uri(C.MLFLOW_URI)
        mlflow.set_experiment(C.EXPERIMENT)
    common_tags = {"data_md5": data_info["data_md5"], "data_route": data_info["route"],
                   "split": "Module 3 patient-grouped 70/15/15 (seed 42)",
                   "project": "BAN6800 JJUSHYCSH no-show", "module": "4", "author": C.AUTHOR,
                   "features": ",".join(features)}

    results, probs, fitted = {}, {}, {}

    # ---------- Baselines ----------
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    p = dummy.predict_proba(Xte)[:, 1]
    results["baseline_majority"] = {**metrics_at(yte, p, 0.5), "val_pr_auc": np.nan, "cv_pr_auc": np.nan,
                                    "cv_roc_auc": 0.5, "threshold": 0.5, "params": {"strategy": "most_frequent"}}
    probs["baseline_majority"] = p
    # Transparent rule a scheduler could apply without ML: flag long lead times. Cut chosen on VALIDATION
    # so that the rule reaches the same 70 % recall target as the models (like-for-like comparison).
    rule_t = threshold_for_recall(yva, Xva["lead_days"].values.astype(float), C.TARGET_RECALL)
    rule_t = max(rule_t, 1.0)
    rule_score = Xte["lead_days"].values.astype(float)
    r = metrics_at(yte, rule_score, rule_t); r["brier"] = np.nan
    results["baseline_lead_time_rule"] = {**r, "val_pr_auc": float(average_precision_score(yva, Xva["lead_days"])),
                                          "cv_pr_auc": np.nan, "cv_roc_auc": float(roc_auc_score(ytr, Xtr["lead_days"])),
                                          "threshold": rule_t,
                                          "params": {"rule": f"flag if lead_days >= {rule_t:.0f}"}}
    probs["baseline_lead_time_rule"] = rule_score / max(rule_score.max(), 1)
    if HAS_MLFLOW:
        for name in ["baseline_majority", "baseline_lead_time_rule"]:
            with mlflow.start_run(run_name=name):
                mlflow.set_tags({**common_tags, "model_type": "baseline"})
                mlflow.log_params(results[name]["params"])
                mlflow.log_metrics({f"test_{k}": float(v) for k, v in results[name].items()
                                    if isinstance(v, (int, float, np.floating)) and not np.isnan(v)})

    # ---------- Tuned models ----------
    for name, (est, space) in candidate_models(ytr).items():
        t0 = time.time()
        pipe = Pipeline([("prep", make_preprocessor(features)), ("clf", est)])
        search = RandomizedSearchCV(pipe, space, n_iter=C.N_ITER, cv=cv, n_jobs=-1,
                                    scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall"},
                                    refit="pr_auc", random_state=C.RANDOM_STATE, verbose=0)
        search.fit(Xtr, ytr, groups=gtr)
        best = search.best_estimator_
        p_val = best.predict_proba(Xva)[:, 1]
        t_high = threshold_for_recall(yva, p_val, C.TARGET_RECALL)
        t_med = threshold_for_recall(yva, p_val, C.MEDIUM_RECALL)
        p = best.predict_proba(Xte)[:, 1]
        i = search.best_index_
        res = {**metrics_at(yte, p, t_high),
               "cv_pr_auc": float(search.cv_results_["mean_test_pr_auc"][i]),
               "cv_pr_auc_std": float(search.cv_results_["std_test_pr_auc"][i]),
               "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][i]),
               "val_pr_auc": float(average_precision_score(yva, p_val)),
               "val_roc_auc": float(roc_auc_score(yva, p_val)),
               "val_recall_at_threshold": float(recall_score(yva, (p_val >= t_high).astype(int))),
               "val_precision_at_threshold": float(precision_score(yva, (p_val >= t_high).astype(int), zero_division=0)),
               "threshold": t_high, "threshold_medium": t_med,
               "fit_seconds": round(time.time() - t0, 1),
               "params": {k.replace("clf__", ""): _fmt(v) for k, v in search.best_params_.items()}}
        results[name], probs[name] = res, p
        fitted[name] = {"pipe": best, "p_val": p_val, "t_high": t_high, "t_med": t_med, "search": search}
        cm_path = C.FIG / f"confusion_{name}.png"
        _plot_confusion(yte, p, t_high, f"{name.replace('_', ' ').title()} (t={t_high:.3f})", cm_path)

        if HAS_MLFLOW:
            with mlflow.start_run(run_name=name) as run:
                mlflow.set_tags({**common_tags, "model_type": name, "search": "RandomizedSearchCV"})
                mlflow.log_params({**{k: str(v) for k, v in res["params"].items()},
                                   "n_iter": C.N_ITER, "cv_folds": C.CV_FOLDS, "cv": "StratifiedGroupKFold(patient_key)",
                                   "target_recall": C.TARGET_RECALL, "threshold_high": round(t_high, 5),
                                   "threshold_medium": round(t_med, 5), "n_features": len(features)})
                mlflow.log_metrics({"cv_pr_auc": res["cv_pr_auc"], "cv_roc_auc": res["cv_roc_auc"],
                                    "val_pr_auc": res["val_pr_auc"], "val_roc_auc": res["val_roc_auc"],
                                    "val_recall_at_threshold": res["val_recall_at_threshold"],
                                    **{f"test_{k}": float(res[k]) for k in
                                       ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "flag_rate"]}})
                mlflow.log_artifact(str(cm_path))
                info = _log_model(best)
                fitted[name]["run_id"] = run.info.run_id
                fitted[name]["model_uri"] = getattr(info, "model_uri", f"runs:/{run.info.run_id}/model")
                fitted[name]["model_logged"] = info is not None
                cvr = search.cv_results_          # every candidate as a nested run = timestamped tuning trail
                for j in range(len(cvr["params"])):
                    with mlflow.start_run(run_name=f"{name}_cand{j:02d}", nested=True):
                        mlflow.log_params({k.replace("clf__", ""): str(v) for k, v in cvr["params"][j].items()})
                        mlflow.log_metrics({"cv_pr_auc": float(cvr["mean_test_pr_auc"][j]),
                                            "cv_roc_auc": float(cvr["mean_test_roc_auc"][j]),
                                            "cv_recall_default_t": float(cvr["mean_test_recall"][j])})
        print(f"  {name}: cv PR-AUC={res['cv_pr_auc']:.3f} val PR-AUC={res['val_pr_auc']:.3f} "
              f"test ROC-AUC={res['roc_auc']:.3f} recall={res['recall']:.3f} ({res['fit_seconds']}s)", flush=True)

    # Champion = best VALIDATION PR-AUC (test set untouched by any decision).
    tuned = list(fitted)
    champ = max(tuned, key=lambda k: results[k]["val_pr_auc"])
    registry = register_champion(champ, fitted[champ]) if HAS_MLFLOW else {"status": "mlflow not installed"}

    plot_curves(yte, probs, tuned + ["baseline_lead_time_rule"])
    _plot_confusion(yte, probs[champ], fitted[champ]["t_high"],
                    f"Champion: {champ.replace('_', ' ').title()}", C.FIG / "confusion_champion.png")
    plot_calibration(yte, probs[champ], champ)
    return results, fitted, champ, registry


def temporal_check(champ_pipe, parts, features):
    """Forward-in-time robustness check (supports the monitoring plan): the champion configuration is
    refitted on the earliest 80 % of train+validation appointment dates and scored on the latest 20 %."""
    df = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    cut = df[C.DATE_COL].quantile(0.80)
    early, late = df[df[C.DATE_COL] < cut], df[df[C.DATE_COL] >= cut]
    if len(late) < 200 or late[C.TARGET].nunique() < 2:
        return {"status": "not enough later appointments"}
    pipe = clone(champ_pipe).fit(early[features], early[C.TARGET].values)
    p = pipe.predict_proba(late[features])[:, 1]
    p_early = pipe.predict_proba(early[features])[:, 1]
    t = threshold_for_recall(early[C.TARGET].values, p_early, C.TARGET_RECALL)
    m = metrics_at(late[C.TARGET].values, p, t)
    return {"status": "ok", "cut_date": str(pd.Timestamp(cut).date()), "n_early": int(len(early)),
            "n_late": int(len(late)), "late_no_show_rate": float(late[C.TARGET].mean()),
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "recall_at_early_threshold": m["recall"],
            "precision": m["precision"], "flag_rate": m["flag_rate"]}


def register_champion(name, f):
    info = {"model_name": C.REGISTERED_MODEL}
    if not f.get("model_logged"):
        info["error"] = "model artefact was not logged to MLflow, so it could not be registered"
        return info
    try:
        client = MlflowClient()
        mv = mlflow.register_model(f["model_uri"], C.REGISTERED_MODEL)
        info["version"] = int(mv.version)
        try:
            client.set_registered_model_alias(C.REGISTERED_MODEL, "champion", mv.version)
            info["alias"] = "champion"
        except Exception as e:
            info["alias_error"] = str(e)[:120]
        try:
            client.transition_model_version_stage(C.REGISTERED_MODEL, mv.version, "Staging")
            info["stage"] = "Staging"
        except Exception:
            client.set_model_version_tag(C.REGISTERED_MODEL, mv.version, "stage", "Staging")
            info["stage"] = "Staging (tag)"
        client.set_model_version_tag(C.REGISTERED_MODEL, mv.version, "algorithm", name)
        client.set_model_version_tag(C.REGISTERED_MODEL, mv.version, "validated_on", "Module 3 test split")
    except Exception as e:
        info["error"] = str(e)[:200]
    return info


def plot_curves(y, probs, names):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for n in names:
        fpr, tpr, _ = roc_curve(y, probs[n]); prec, rec, _ = precision_recall_curve(y, probs[n])
        lab = n.replace("_", " ").title()
        axes[0].plot(fpr, tpr, label=f"{lab} (AUC {roc_auc_score(y, probs[n]):.3f})",
                     ls="--" if n.startswith("baseline") else "-")
        axes[1].plot(rec, prec, label=f"{lab} (AP {average_precision_score(y, probs[n]):.3f})",
                     ls="--" if n.startswith("baseline") else "-")
    axes[0].plot([0, 1], [0, 1], color="grey", lw=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve (Module 3 test split)")
    axes[1].axhline(y.mean(), color="grey", lw=0.8)
    axes[1].axvline(C.TARGET_RECALL, color="red", lw=0.8, ls=":")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (Module 3 test split)")
    for a in axes:
        a.legend(fontsize=7.5, loc="lower right" if a is axes[0] else "upper right")
    fig.tight_layout(); fig.savefig(C.FIG / "roc_pr_curves.png", dpi=160); plt.close(fig)


def plot_calibration(y, p, name):
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    obs = [y[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    pred = [p[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax.plot(pred, obs, marker="o")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed no-show rate",
           title=f"Calibration: {name.replace('_', ' ').title()}", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout(); fig.savefig(C.FIG / "calibration_champion.png", dpi=160); plt.close(fig)


def export_mlflow_runs():
    if not HAS_MLFLOW:
        return None
    try:
        df = mlflow.search_runs(experiment_names=[C.EXPERIMENT])
        keep = [c for c in df.columns if c in ("run_id", "start_time", "tags.mlflow.runName")
                or c.startswith("metrics.")]
        df[keep].to_csv(C.OUT / "mlflow_runs_export.csv", index=False)
        return len(df)
    except Exception as e:
        print("MLflow export failed:", e)
        return None
