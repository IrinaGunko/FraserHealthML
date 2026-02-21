#!/usr/bin/env python3

import argparse
import json
import logging
import os
import tempfile
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def union_feature_flags(global_cfg: dict, models_cfg: dict) -> dict:
    union = dict(global_cfg["feature_flags"])
    for variants in models_cfg.values():
        for model_cfg in variants.values():
            for flag, val in model_cfg.get("feature_flags", {}).items():
                if val:
                    union[flag] = True
    return union

_ALPHA_PREFIXES     = ("alpha_peak_cf", "alpha_peak_pw", "alpha_peak_bw")
_APERIODIC_PREFIXES = ("aperiodic_offset", "aperiodic_exponent",
                       "r_squared", "fit_error")
_GRADIENT_PREFIXES  = (
    "T1T2ratio", "G1.fMRI", "Evolution.Expansion",
    "AllometricScaling.PNC20mm", "PET.AG", "CBF",
    "PC1.AHBA", "PC1.Neurosynth", "BigBrain.Histology", "Cortical.Thickness",
)
_RANK_PREFIXES = (
    "averagerank.wholebrain", "finalrank.wholebrain",
    "averagerank.hemisphere", "finalrank.hemisphere",
)
_STAT_SUFFIXES = ("_median", "_p5", "_p95")


def _matches(col: str, prefixes: tuple, suffixes: tuple) -> bool:
    return (any(col.startswith(p) for p in prefixes) and
            any(col.endswith(s)   for s in suffixes))


def build_features(df: pd.DataFrame, flags: dict) -> list:
    all_cols = set(df.select_dtypes(include=[np.number]).columns)
    cols: set = set()

    rules = [
        ("USE_GLOBAL_ALPHA_MEDIAN_P5_P95",     _ALPHA_PREFIXES,     _STAT_SUFFIXES),
        ("USE_GLOBAL_APERIODIC_MEDIAN_P5_P95", _APERIODIC_PREFIXES, _STAT_SUFFIXES),
        ("USE_GLOBAL_GRADIENT_MEDIAN_P5_P95",  _GRADIENT_PREFIXES,  _STAT_SUFFIXES),
        ("USE_GLOBAL_RANK_MEDIAN_P5_P95",      _RANK_PREFIXES,      _STAT_SUFFIXES),
    ]
    for flag, prefixes, suffixes in rules:
        if flags.get(flag):
            cols |= {c for c in all_cols if _matches(c, prefixes, suffixes)}

    if flags.get("USE_NETWORK_COUNTS"):
        cols |= {c for c in all_cols
                 if (c.startswith("net17_") or c.startswith("net8_"))
                 and c.endswith("_count")}
    if flags.get("USE_NETWORK_PCTS"):
        cols |= {c for c in all_cols
                 if (c.startswith("net17_") or c.startswith("net8_"))
                 and c.endswith("_pct")}
    if flags.get("USE_NET8_MEDIAN_P5_P95"):
        cols |= {c for c in all_cols
                 if c.startswith("net8_")
                 and any(c.endswith(s) for s in _STAT_SUFFIXES)}
    if flags.get("USE_NET17_MEDIAN_P5_P95"):
        cols |= {c for c in all_cols
                 if c.startswith("net17_")
                 and any(c.endswith(s) for s in _STAT_SUFFIXES)}

    return sorted(cols)

def make_labels(series: pd.Series, label_map: str) -> tuple[pd.Series, pd.Series]:
    """Return (mask, binary_y) for a given label_map string."""
    if label_map == "1_vs_4":
        mask = series.isin([1, 4])
        y    = (series[mask] == 4).astype(int)
    elif label_map == "12_vs_34":
        mask = series.isin([1, 2, 3, 4])
        y    = (series[mask] >= 3).astype(int)
    else:
        raise ValueError(f"Unknown label_map: {label_map!r}")
    return mask, y

def get_resampler(name: str, random_state: int):
    factories = {
        "SMOTE":      lambda: SMOTE(random_state=random_state),
        "ADASYN":     lambda: ADASYN(random_state=random_state),
        "SMOTETomek": lambda: SMOTETomek(random_state=random_state),
        "none":       lambda: None,
    }
    if name not in factories:
        raise ValueError(
            f"Unknown resampler: {name!r}. Valid options: {list(factories)}")
    return factories[name]()


def resolve_resampler_name(model_cfg: dict, global_cfg: dict) -> str:
    """Per-model 'resampler' key takes priority over global default."""
    return model_cfg.get("resampler", global_cfg.get("resampler", "none"))


def impute(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    X_shap:  pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.Series]:

    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test  = X_test.fillna(train_medians)
    if X_shap is not None:
        X_shap = X_shap.fillna(train_medians)
    return X_train, X_test, X_shap, train_medians

def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   mode: str,
                   recall_targets: list[float],
                   min_precision: float,
                   cost_fn: float = 1.0,
                   cost_fp: float = 1.0,
                   target_precision: float = 0.60) -> float:

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if mode == "f1":
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            pred  = (y_prob >= t).astype(int)
            tp    = int(((pred == 1) & (y_true == 1)).sum())
            fp    = int(((pred == 1) & (y_true == 0)).sum())
            fn    = int(((pred == 0) & (y_true == 1)).sum())
            denom = 2 * tp + fp + fn
            if denom == 0:
                continue
            f1 = 2 * tp / denom
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t

    elif mode == "youden":
        # J = TPR - FPR  (= sensitivity + specificity - 1)
        j   = tpr - fpr
        idx = int(np.argmax(j))
        return float(thresholds[idx])

    elif mode == "min_cost":
        # Total cost = cost_fn * FN + cost_fp * FP
        best_t, best_cost = 0.5, float("inf")
        for t in thresholds:
            pred = (y_prob >= t).astype(int)
            fn   = int(((pred == 0) & (y_true == 1)).sum())
            fp   = int(((pred == 1) & (y_true == 0)).sum())
            cost = cost_fn * fn + cost_fp * fp
            if cost < best_cost:
                best_cost, best_t = cost, float(t)
        return best_t

    elif mode == "target_precision":

        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)
        candidates = [
            (rec_arr[i], float(thresh_arr[i]))
            for i in range(len(thresh_arr))
            if prec_arr[i] >= target_precision
        ]
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]
        log.warning(
            "target_precision=%.2f never achieved — falling back to 0.5", target_precision)
        return 0.5

    elif mode == "max_recall":
        candidates = []
        for t in thresholds:
            pred = (y_prob >= t).astype(int)
            tp   = int(((pred == 1) & (y_true == 1)).sum())
            fp   = int(((pred == 1) & (y_true == 0)).sum())
            fn   = int(((pred == 0) & (y_true == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if prec >= min_precision:
                candidates.append((rec, float(t)))
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]
        target_recall = min(recall_targets)
        _, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)
        idx = np.argmin(np.abs(rec_arr[:-1] - target_recall))
        return float(thresh_arr[idx])

    else:
        raise ValueError(
            f"Unknown threshold mode: {mode!r}. "
            f"Valid options: 'f1', 'youden', 'min_cost', "
            f"'target_precision', 'max_recall'"
        )


def compute_metrics(target: str, variant: str,
                    y_true: np.ndarray, y_prob: np.ndarray,
                    y_pred: np.ndarray, threshold: float,
                    n_train: int, n_test: int,
                    n_pos_train: int, n_pos_test: int,
                    resampler_name: str) -> dict:
    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Abnormal"],
        output_dict=True,
        zero_division=0,
    )

    try:
        test_auc = round(roc_auc_score(y_true, y_prob), 4)
    except ValueError:
        log.warning("roc_auc_score failed for %s [%s] — only one class in y_test.",
                    target, variant)
        test_auc = float("nan")

    try:
        test_avg_precision = round(average_precision_score(y_true, y_prob), 4)
    except ValueError:
        log.warning("average_precision_score failed for %s [%s] — only one class in y_test.",
                    target, variant)
        test_avg_precision = float("nan")

    return {
        "target":             target,
        "variant":            variant,
        "resampler":          resampler_name,
        "n_train":            n_train,
        "n_test":             n_test,
        "n_pos_train":        n_pos_train,
        "n_pos_test":         n_pos_test,
        "test_auc":           test_auc,
        "test_avg_precision": test_avg_precision,
        "accuracy":           round(report["accuracy"], 4),
        "abnormal_precision": round(report["Abnormal"]["precision"], 4),
        "abnormal_recall":    round(report["Abnormal"]["recall"], 4),
        "abnormal_f1":        round(report["Abnormal"]["f1-score"], 4),
        "threshold":          round(threshold, 4),
    }

def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_roc_single(ax: plt.Axes, y_true, y_prob, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, lw=2, color="#2166ac", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_pr_single(ax: plt.Axes, y_true, y_prob, title: str) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(rec, prec, lw=2, color="#d73027", label=f"AP = {ap:.3f}")
    ax.axhline(y_true.mean(), linestyle="--", color="gray", lw=1,
               label=f"Base = {y_true.mean():.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_confusion_single(ax: plt.Axes, y_true, y_pred,
                          title: str, mode: str = "pct") -> None:
    cm     = confusion_matrix(y_true, y_pred)
    matrix = (cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
              if mode == "pct" else cm.copy())
    im     = ax.imshow(matrix, cmap="Blues",
                       vmin=0, vmax=(100 if mode == "pct" else None))
    labels = ["Normal", "Abnormal"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=9)
    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            val  = matrix[i, j]
            text = f"{val:.1f}%" if mode == "pct" else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val > thresh else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


TARGETS = ["Abnormality", "Focal_Epi", "Focal_Non_epi", "Gen_Epi", "Gen_Non_epi"]


def plot_panel_roc(results: dict, variant: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
    fig.suptitle(f"ROC Curves — {variant.upper()} labels", fontsize=13, y=1.02)
    for ax, target in zip(axes, TARGETS):
        key = (target, variant)
        if key not in results:
            ax.set_title(f"{target}\n(no data)", fontsize=9); ax.axis("off"); continue
        plot_roc_single(ax, results[key]["y_test"], results[key]["y_prob"], target)
    plt.tight_layout()
    _save(fig, out_path)
    log.info("Saved %s", out_path)


def plot_panel_pr(results: dict, variant: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
    fig.suptitle(f"PR Curves — {variant.upper()} labels", fontsize=13, y=1.02)
    for ax, target in zip(axes, TARGETS):
        key = (target, variant)
        if key not in results:
            ax.set_title(f"{target}\n(no data)", fontsize=9); ax.axis("off"); continue
        plot_pr_single(ax, results[key]["y_test"], results[key]["y_prob"], target)
    plt.tight_layout()
    _save(fig, out_path)
    log.info("Saved %s", out_path)


def plot_panel_confusion(results: dict, variant: str, out_path: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.suptitle(f"Confusion Matrices — {variant.upper()} labels",
                 fontsize=13, y=1.01)
    for col, target in enumerate(TARGETS):
        key = (target, variant)
        if key not in results:
            for row in range(2):
                axes[row, col].set_title(f"{target}\n(no data)", fontsize=9)
                axes[row, col].axis("off")
            continue
        r = results[key]
        plot_confusion_single(axes[0, col], r["y_test"], r["y_pred"],
                              f"{target} (counts)", mode="raw")
        plot_confusion_single(axes[1, col], r["y_test"], r["y_pred"],
                              f"{target} (row %)",  mode="pct")
    plt.tight_layout()
    _save(fig, out_path)
    log.info("Saved %s", out_path)


def plot_panel_shap(models: dict, X_tests: dict, variant: str,
                    out_path: str, shap_samples: int,
                    random_state: int) -> None:

    tmp_paths: dict = {}

    for target in TARGETS:
        key = (target, variant)
        if key not in models:
            continue
        try:
            sample = X_tests[key].sample(
                min(shap_samples, len(X_tests[key])),
                random_state=random_state)
            explainer   = shap.TreeExplainer(models[key])
            shap_values = explainer(sample)

            plt.figure(figsize=(5, 6))
            shap.plots.beeswarm(shap_values, max_display=10,
                                show=False, plot_size=None)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            plt.savefig(tmp.name, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close("all")
            tmp_paths[key] = tmp.name
        except Exception as e:
            log.warning("SHAP failed for %s [%s]: %s", target, variant, e)
            plt.close("all")

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    fig.suptitle(f"SHAP Beeswarm — {variant.upper()} labels",
                 fontsize=13, y=1.02)
    for ax, target in zip(axes, TARGETS):
        key = (target, variant)
        ax.axis("off")
        if key in tmp_paths:
            ax.imshow(mpimg.imread(tmp_paths[key]))
            ax.set_title(target, fontsize=9, pad=4)
        else:
            ax.set_title(f"{target}\n(no data / SHAP failed)", fontsize=9)

    plt.tight_layout()
    _save(fig, out_path)
    log.info("Saved %s", out_path)

    for p in tmp_paths.values():
        try:
            os.remove(p)
        except OSError:
            pass


def plot_metrics_summary_bar(all_metrics: list[dict], out_path: str) -> None:
    df     = pd.DataFrame(all_metrics)
    df["label"] = df["target"] + "\n[" + df["variant"] + "]"
    labels = df["label"].tolist()
    x      = np.arange(len(labels))
    width  = 0.22

    metrics_to_show = [
        ("test_auc",        "AUC",    "#2166ac"),
        ("abnormal_f1",     "F1",     "#d73027"),
        ("abnormal_recall", "Recall", "#1a9850"),
    ]
    offsets = np.linspace(-(len(metrics_to_show) - 1) / 2,
                           (len(metrics_to_show) - 1) / 2,
                           len(metrics_to_show)) * width

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.6), 5))
    for (mkey, mlabel, mcolor), off in zip(metrics_to_show, offsets):
        bars = ax.bar(x + off, df[mkey].values, width,
                      label=mlabel, color=mcolor, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("Model Summary — AUC / F1 / Recall across all targets & variants")
    ax.legend(frameon=False, fontsize=9)
    ax.axhline(0.5, linestyle="--", color="lightgray", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_path)
    log.info("Saved %s", out_path)


XGB_FIXED = dict(
    objective             = "binary:logistic",
    eval_metric           = "auc",
    tree_method           = "hist",
    n_jobs                = -1,
    verbosity             = 0,
    early_stopping_rounds = 30,
)


def train_model(
    target: str,
    variant: str,
    X: pd.DataFrame,       # raw (NaN still present), superset of all columns
    y: np.ndarray,
    model_cfg: dict,
    global_cfg: dict,
) -> tuple[xgb.XGBClassifier, dict, dict, pd.DataFrame, pd.Series]:

    rs  = global_cfg["random_state"]
    tfr = global_cfg["train_fraction"]

    log.info("Training %s [%s] — samples=%d  pos=%d  neg=%d",
             target, variant, len(y), (y == 1).sum(), (y == 0).sum())

    flags        = model_cfg.get("feature_flags", global_cfg["feature_flags"])
    feature_cols = build_features(X, flags)
    X_feat       = X[feature_cols]   # still contains NaN — impute after split

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=1 - tfr, stratify=y, random_state=rs)

    X_train, X_test, _, train_medians = impute(X_train, X_test)

    resampler_name = resolve_resampler_name(model_cfg, global_cfg)
    resampler      = get_resampler(resampler_name, rs)
    if resampler is not None:
        log.info("  Resampling with %s  (before: pos=%d  neg=%d)",
                 resampler_name, (y_train == 1).sum(), (y_train == 0).sum())
        X_train_arr, y_train = resampler.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train_arr, columns=feature_cols)
        log.info("  After resampling:   pos=%d  neg=%d",
                 (y_train == 1).sum(), (y_train == 0).sum())
    else:
        log.info("  Resampling: none")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=rs)

    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = float(n_neg / n_pos) if n_pos > 0 else 1.0

    params = {
        **XGB_FIXED,
        **model_cfg["hyperparams"],
        "scale_pos_weight": spw,
        "random_state":     rs,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    log.info("  best_iteration=%d  scale_pos_weight=%.3f",
             model.best_iteration, spw)

    thr        = model_cfg.get("threshold", global_cfg["threshold"])
    y_val_prob = model.predict_proba(X_val)[:, 1]
    threshold  = tune_threshold(
        y_val, y_val_prob,
        mode             = thr["mode"],
        recall_targets   = thr.get("recall_targets",               [0.60]),
        min_precision    = thr.get("min_precision_for_max_recall", 0.60),
        cost_fn          = thr.get("cost_fn",                      1.0),
        cost_fp          = thr.get("cost_fp",                      1.0),
        target_precision = thr.get("target_precision",             0.60),
    )
    log.info("  threshold=%.4f (mode=%s)", threshold, thr["mode"])

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(
        target, variant,
        y_test, y_prob, y_pred, threshold,
        n_train        = len(y_train),
        n_test         = len(y_test),
        n_pos_train    = int((y_train == 1).sum()),
        n_pos_test     = int((y_test  == 1).sum()),
        resampler_name = resampler_name,
    )
    log.info("  AUC=%.3f  F1=%.3f  Recall=%.3f  Precision=%.3f",
             metrics["test_auc"], metrics["abnormal_f1"],
             metrics["abnormal_recall"], metrics["abnormal_precision"])

    preds = {"y_test": y_test, "y_prob": y_prob, "y_pred": y_pred}

    return model, metrics, preds, X_test, train_medians


def save_models(
    trained_models: dict,
    X_tests: dict,
    all_metrics: list[dict],
    medians_store: dict,
    global_cfg: dict,
    out_dir: str,
) -> None:

    models_dir = global_cfg.get("models_dir", os.path.join(out_dir, "models"))
    os.makedirs(models_dir, exist_ok=True)

    for (target, variant), model in trained_models.items():
        model_name = f"{target}_{variant}"
        path = os.path.join(models_dir, f"{model_name}.joblib")
        joblib.dump(model, path)
        log.info("  Saved model: %s", path)

    feature_map = {
        f"{target}_{variant}": X_tests[(target, variant)].columns.tolist()
        for (target, variant) in trained_models
    }
    feat_path = os.path.join(models_dir, "feature_columns.json")
    with open(feat_path, "w") as f:
        json.dump(feature_map, f, indent=2)
    log.info("  Saved feature columns: %s", feat_path)

    medians_serializable = {
        f"{target}_{variant}": medians.to_dict()
        for (target, variant), medians in medians_store.items()
    }
    medians_path = os.path.join(models_dir, "train_medians.json")
    with open(medians_path, "w") as f:
        json.dump(medians_serializable, f, indent=2)
    log.info("  Saved train medians: %s", medians_path)

    thresholds = {
        f"{m['target']}_{m['variant']}": m["threshold"]
        for m in all_metrics
    }
    thresh_path = os.path.join(models_dir, "thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info("  Saved thresholds: %s", thresh_path)

    log.info("All model artifacts saved to: %s", models_dir)


def main(config_path: str) -> None:
    cfg        = load_config(config_path)
    global_cfg = cfg["global"]
    models_cfg = cfg["models"]
    out_dir    = global_cfg["output_dir"]
    do_plots   = global_cfg.get("model_stats_plots", True)
    do_save    = global_cfg.get("save_models", False)
    rs         = global_cfg["random_state"]
    shap_n     = global_cfg["shap_samples"]

    os.makedirs(out_dir, exist_ok=True)

    import shutil as _shutil
    config_snapshot = os.path.join(out_dir, "config_used.json")
    _shutil.copy2(config_path, config_snapshot)
    log.info("Config snapshot saved: %s", config_snapshot)

    log.info("Loading CSV: %s", global_cfg["csv_path"])
    df = pd.read_csv(global_cfg["csv_path"], low_memory=False)
    log.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    superset_flags = union_feature_flags(global_cfg, models_cfg)
    feature_cols   = build_features(df, superset_flags)
    X_all          = df[feature_cols]   # NaN retained — imputed inside train_model
    log.info("Feature pool (union of all model flags): %d columns",
             len(feature_cols))

    all_metrics: list[dict]  = []
    trained_models: dict     = {}  # (target, variant) → model
    test_data: dict          = {}  # (target, variant) → {y_test, y_prob, y_pred}
    X_tests: dict            = {}  # (target, variant) → X_test (imputed, model features)
    medians_store: dict      = {}  # (target, variant) → train_medians pd.Series

    for target, variants in models_cfg.items():
        if target not in df.columns:
            log.warning("Column '%s' not in CSV — skipping.", target)
            continue

        raw = df[target]

        for variant, model_cfg in variants.items():
            mask, y_series = make_labels(raw, model_cfg["label_map"])
            if mask.sum() < 100:
                log.warning("Not enough samples for %s [%s] — skipping.",
                            target, variant)
                continue

            X_sub = X_all.loc[mask]
            y     = y_series.values
            key   = (target, variant)

            model, metrics, preds, X_test_imputed, train_medians = train_model(
                target, variant, X_sub, y, model_cfg, global_cfg)

            all_metrics.append(metrics)
            trained_models[key] = model
            test_data[key]      = preds
            X_tests[key]        = X_test_imputed  # imputed, correct feature columns
            medians_store[key]  = train_medians    # for saving / future inference

    df_summary   = pd.DataFrame(all_metrics)
    summary_path = os.path.join(out_dir, "summary_all.csv")
    df_summary.to_csv(summary_path, index=False)
    log.info("Summary saved: %s", summary_path)

    show_cols = ["target", "variant", "resampler", "test_auc", "abnormal_f1",
                 "abnormal_recall", "accuracy", "test_avg_precision", "threshold"]
    log.info("\n%s", df_summary[show_cols].to_string(index=False))

    if do_save:
        log.info("Saving models (save_models=true)...")
        save_models(trained_models, X_tests, all_metrics,
                    medians_store, global_cfg, out_dir)

    if do_plots:
        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for variant in ["strict", "relaxed"]:
            plot_panel_roc(
                test_data, variant,
                os.path.join(plots_dir, f"panel_roc_{variant}.png"))
            plot_panel_pr(
                test_data, variant,
                os.path.join(plots_dir, f"panel_pr_{variant}.png"))
            plot_panel_confusion(
                test_data, variant,
                os.path.join(plots_dir, f"panel_confusion_{variant}.png"))
            plot_panel_shap(
                trained_models, X_tests, variant,
                os.path.join(plots_dir, f"panel_shap_{variant}.png"),
                shap_samples=shap_n, random_state=rs)

        plot_metrics_summary_bar(
            all_metrics,
            os.path.join(plots_dir, "summary_bar.png"))

    log.info("Done. All outputs in: %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost training pipeline")
    parser.add_argument(
        "--config", default="config_xgb.json",
        help="Path to config JSON (default: config_xgb.json)")
    args = parser.parse_args()
    main(args.config)