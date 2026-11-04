#!/usr/bin/env python3
"""
Train an interpretable learned adaptive-horizon policy from OSM-aware planner rollouts.

Purpose
-------
This script upgrades the adaptive-horizon component from manually selected fixed gains to a
learned, bounded, uncertainty-conditioned policy.

It reads existing OSM-aware rollout metrics produced by Script 09, derives outcome-guided
horizon targets from safety/comfort/efficiency performance, and fits a low-dimensional
bounded logistic model:

    H_i(t) = H_min,i + (H_max,i - H_min,i) * sigmoid(theta_i^T x_t)

where i is pedestrian or vehicle and x_t contains interpretable uncertainty/context features:
- camera degradation fraction
- normalized OSM crosswalk positional error
- mean crossing proximity
- actual camera drop fraction if available
- degradation mode indicators
- interaction terms

This is intentionally not a black-box neural network. It is a learned but interpretable
policy that can be reported as an offline-learned uncertainty-conditioned horizon module.

Inputs
------
Default:
- data/planner_osm_aware/<scene_name>/per_run_metrics.csv

Optional:
- data/planner_perception_only/<scene_name>/per_run_metrics.csv

Outputs
-------
Default root: data/learned_horizon_policy/<scene_name>/
- adaptive_horizon_policy.json
- learned_policy_training_dataset.csv
- learned_policy_predictions.csv
- learned_policy_training_summary.json
- plots/pedestrian_target_vs_prediction.png
- plots/vehicle_target_vs_prediction.png
- plots/learned_theta_coefficients.png

Typical usage
-------------
1) Run Script 09 once with the rule-based fallback policy to produce per_run_metrics.csv.
2) Train the learned policy:

   python 09a_train_adaptive_horizon_policy.py --scene-name scene-0061

3) Rerun the updated Script 09. It will automatically load:

   data/learned_horizon_policy/<scene_name>/adaptive_horizon_policy.json

Notes
-----
- The target horizons are not ground-truth labels from humans. They are outcome-guided
  simulation targets derived from safety, comfort, and efficiency metrics.
- This makes the policy learned from closed-loop rollout evidence while remaining
  interpretable and bounded.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-9
FEATURE_NAMES = [
    "bias",
    "degradation_fraction",
    "normalized_osm_error",
    "crossing_proximity_factor",
    "actual_camera_drop_fraction",
    "is_random_dropout",
    "is_burst_dropout",
    "is_distance_based_dropout",
    "degradation_x_osm_error",
    "degradation_x_proximity",
    "osm_error_x_proximity",
    "drop_fraction_x_proximity",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


DEFAULT_BATCH_SCENES = ["scene-1019", "scene-0061", "scene-0383"]


# ---------------------------------------------------------------------------
# Geometry-specific learning profiles
# ---------------------------------------------------------------------------
# Script 10 remains one training script, but the target-generation weights are
# allowed to reflect the scenario geometry. This keeps the paper method unified
# while avoiding a learned policy that is tuned only to scene-0061.
GEOMETRY_LEARNING_PROFILES: Dict[str, Dict[str, Any]] = {
    "scene-1019": {
        "geometry_case": "straight_approach",
        "rationale": "Straight approach should gain the most from early OSM/camera uncertainty anticipation, so safety risk receives a stronger target-horizon gain.",
        "overrides": {
            "safety_gain_ped_s": 1.30,
            "conservatism_penalty_ped_s": 0.22,
            "safety_gain_veh_s": 0.42,
            "conservatism_penalty_veh_s": 0.12,
            "ridge_lambda": 0.12,
            "ped_horizon_max_s": 5.0,
            "veh_horizon_max_s": 2.8,
        },
    },
    "scene-0061": {
        "geometry_case": "moderate_curvature_approach",
        "rationale": "Reference moderate-curvature case; keep the balanced safety/efficiency weighting that already worked.",
        "overrides": {
            "safety_gain_ped_s": 0.95,
            "conservatism_penalty_ped_s": 0.32,
            "safety_gain_veh_s": 0.35,
            "conservatism_penalty_veh_s": 0.18,
            "ridge_lambda": 0.15,
        },
    },
    "scene-0383": {
        "geometry_case": "high_curvature_circular_approach",
        "rationale": "High-curvature interactive crossing: raise caution under uncertainty, but penalize travel-time/jerk inflation when no measurable safety benefit appears.",
        "overrides": {
            "safety_gain_ped_s": 1.10,
            "conservatism_penalty_ped_s": 0.58,
            "safety_gain_veh_s": 0.24,
            "conservatism_penalty_veh_s": 0.26,
            "ridge_lambda": 0.22,
            "ped_horizon_max_s": 5.0,
            "veh_horizon_max_s": 2.4,
        },
    },
}


def apply_geometry_learning_profile(scene_name: str, args: argparse.Namespace, disabled: bool = False) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Return a per-scene copy of args with learning-profile overrides applied."""
    import copy
    scene = normalize_scene_name_token(scene_name)
    local_args = copy.copy(args)
    if disabled:
        return local_args, {"scene_name": scene, "profile_used": False, "reason": "disabled_by_cli", "overrides": {}}
    profile = GEOMETRY_LEARNING_PROFILES.get(scene)
    if not profile:
        return local_args, {"scene_name": scene, "profile_used": False, "reason": "no_profile_for_scene", "overrides": {}}
    applied: Dict[str, Any] = {}
    for key, value in dict(profile.get("overrides", {})).items():
        if hasattr(local_args, key):
            setattr(local_args, key, value)
            applied[key] = value
    return local_args, {
        "scene_name": scene,
        "profile_used": True,
        "geometry_case": profile.get("geometry_case", ""),
        "rationale": profile.get("rationale", ""),
        "overrides": applied,
    }


def normalize_scene_name_token(token: str) -> str:
    """Accept 0383, scene-0383, comma tokens, or all."""
    t = str(token).strip()
    if not t:
        return ""
    if t.lower() in {"all", "three", "selected"}:
        return "all"
    if t.startswith("scene-"):
        return t
    digits = "".join(ch for ch in t if ch.isdigit())
    if digits:
        return f"scene-{digits.zfill(4)}"
    return t


def parse_scene_name_list(scene_name_text: str, scene_names_text: str) -> List[str]:
    primary = str(scene_name_text or "").strip()
    if primary.lower() in {"", "all", "three", "selected"}:
        raw_items = [x.strip() for x in str(scene_names_text or "").split(",") if x.strip()]
    else:
        raw_items = [x.strip() for x in primary.split(",") if x.strip()]

    out: List[str] = []
    for item in raw_items:
        norm = normalize_scene_name_token(item)
        if norm == "all":
            out.extend(DEFAULT_BATCH_SCENES)
        elif norm:
            out.append(norm)

    if not out:
        out = list(DEFAULT_BATCH_SCENES)

    seen = set()
    final: List[str] = []
    for scene in out:
        if scene not in seen:
            final.append(scene)
            seen.add(scene)
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train learned adaptive prediction-horizon policies from planner rollouts.")
    p.add_argument(
        "--scene-name",
        default="all",
        help="Scene to train, short scene number, comma-separated scenes, or 'all'. Default: all current project scenes.",
    )
    p.add_argument(
        "--scene-names",
        default=",".join(DEFAULT_BATCH_SCENES),
        help="Comma-separated scene list used when --scene-name all. Default: scene-1019,scene-0061,scene-0383.",
    )
    p.add_argument("--osm-results-root", default="data/planner_osm_aware")
    p.add_argument("--baseline-results-root", default="data/planner_perception_only")
    p.add_argument("--output-root", default="data/learned_horizon_policy")
    p.add_argument("--ridge-lambda", type=float, default=0.15)
    p.add_argument("--safety-gain-ped-s", type=float, default=0.90)
    p.add_argument("--conservatism-penalty-ped-s", type=float, default=0.35)
    p.add_argument("--safety-gain-veh-s", type=float, default=0.35)
    p.add_argument("--conservatism-penalty-veh-s", type=float, default=0.18)
    p.add_argument("--ped-horizon-min-s", type=float, default=1.0)
    p.add_argument("--ped-horizon-max-s", type=float, default=5.0)
    p.add_argument("--veh-horizon-min-s", type=float, default=1.0)
    p.add_argument("--veh-horizon-max-s", type=float, default=2.8)
    p.add_argument("--critical-ttc-threshold-s", type=float, default=1.5)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="For batch training, skip scenes with missing Script 09 metrics instead of stopping immediately.",
    )
    p.add_argument(
        "--disable-geometry-learning-profiles",
        action="store_true",
        help="Disable scene-specific target-generation weights and train all scenes with shared weights.",
    )
    p.add_argument(
        "--print-geometry-learning-profile",
        action="store_true",
        help="Print the geometry learning profile used for each scene.",
    )
    return p.parse_args()


def ensure_dirs(scene_out: Path) -> Dict[str, Path]:
    paths = {"root": scene_out, "plots": scene_out / "plots"}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (8.6, 5.4),
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def safe_numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def robust01(series: pd.Series, invert: bool = False) -> pd.Series:
    """Robustly normalize a metric to [0, 1]. Higher returned value means more penalty/risk."""
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    fill = float(x.median()) if np.isfinite(float(x.median())) else 0.0
    x = x.fillna(fill)
    lo = float(np.nanpercentile(x.to_numpy(dtype=float), 5))
    hi = float(np.nanpercentile(x.to_numpy(dtype=float), 95))
    if hi <= lo + EPS:
        z = pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    else:
        z = ((x - lo) / (hi - lo)).clip(0.0, 1.0)
    if invert:
        z = 1.0 - z
    return z.astype(float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(y: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), 1e-4, 1.0 - 1e-4)
    return np.log(y / (1.0 - y))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    d = (safe_numeric(df, "degradation_level_pct", 0.0) / 100.0).clip(0.0, 1.0)
    e = safe_numeric(df, "normalized_osm_error", 0.0).clip(0.0, 1.0)
    p = safe_numeric(df, "mean_crossing_proximity_factor", np.nan)
    if p.isna().all():
        p = safe_numeric(df, "crossing_proximity_factor", 0.0)
    p = p.fillna(0.0).clip(0.0, 1.0)
    a = safe_numeric(df, "actual_camera_drop_fraction", np.nan)
    if a.isna().all():
        a = safe_numeric(df, "mean_frame_camera_drop_fraction", np.nan)
    a = a.fillna(d).clip(0.0, 1.0)
    mode = df.get("degradation_mode", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()

    out["bias"] = 1.0
    out["degradation_fraction"] = d
    out["normalized_osm_error"] = e
    out["crossing_proximity_factor"] = p
    out["actual_camera_drop_fraction"] = a
    out["is_random_dropout"] = mode.eq("random_dropout").astype(float)
    out["is_burst_dropout"] = mode.eq("burst_dropout").astype(float)
    out["is_distance_based_dropout"] = mode.eq("distance_based_dropout").astype(float)
    out["degradation_x_osm_error"] = d * e
    out["degradation_x_proximity"] = d * p
    out["osm_error_x_proximity"] = e * p
    out["drop_fraction_x_proximity"] = a * p
    return out[FEATURE_NAMES].astype(float)


def fallback_rule_horizons(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.Series, pd.Series]:
    x = build_features(df)
    d = x["degradation_fraction"]
    e = x["normalized_osm_error"]
    p = x["crossing_proximity_factor"]
    ped = (2.0 + 1.0 * d + 1.5 * e + 1.0 * p).clip(args.ped_horizon_min_s, args.ped_horizon_max_s)
    veh = (1.5 + 0.6 * d + 0.4 * e + 0.3 * p).clip(args.veh_horizon_min_s, args.veh_horizon_max_s)
    return ped.astype(float), veh.astype(float)


def derive_outcome_targets(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    ped_current, veh_current = fallback_rule_horizons(out, args)
    if "mean_adaptive_ped_horizon_s" in out.columns:
        ped_current = safe_numeric(out, "mean_adaptive_ped_horizon_s", np.nan).fillna(ped_current)
    if "mean_adaptive_vehicle_horizon_s" in out.columns:
        veh_current = safe_numeric(out, "mean_adaptive_vehicle_horizon_s", np.nan).fillna(veh_current)

    # Higher values below mean higher safety risk.
    risk_components = []
    for col in ["unsafe_frames_total", "ped_conflict_zone_violation_frames", "critical_ped_ttc_frames", "hard_brake_frames"]:
        if col in out.columns:
            risk_components.append(robust01(out[col]))

    if "min_ped_ttc_s" in out.columns:
        ttc = safe_numeric(out, "min_ped_ttc_s", np.nan)
        ttc_risk = ((args.critical_ttc_threshold_s - ttc) / max(args.critical_ttc_threshold_s, EPS)).clip(0.0, 1.0).fillna(0.0)
        risk_components.append(ttc_risk)

    if "pet_critical_flag" in out.columns:
        risk_components.append(safe_numeric(out, "pet_critical_flag", 0.0).fillna(0.0).clip(0.0, 1.0))

    if risk_components:
        safety_risk = sum(risk_components) / float(len(risk_components))
    else:
        safety_risk = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)

    # Higher values below mean more conservatism/comfort/efficiency cost.
    cost_components = []
    for col in ["stop_time_s", "travel_time_s", "comfort_jerk_rms_mps3", "max_braking_mps2"]:
        if col in out.columns:
            cost_components.append(robust01(out[col]))
    if cost_components:
        conservatism_cost = sum(cost_components) / float(len(cost_components))
    else:
        conservatism_cost = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)

    safety_improvement_components: List[pd.Series] = []
    for col in ["unsafe_frames_total", "critical_ped_ttc_frames", "ped_conflict_zone_violation_frames"]:
        bcol = f"{col}_baseline"
        if col in out.columns and bcol in out.columns:
            delta = safe_numeric(out, bcol, np.nan) - safe_numeric(out, col, np.nan)
            safety_improvement_components.append(robust01(delta))
    if "min_ped_ttc_s" in out.columns and "min_ped_ttc_s_baseline" in out.columns:
        delta_ttc = safe_numeric(out, "min_ped_ttc_s", np.nan) - safe_numeric(out, "min_ped_ttc_s_baseline", np.nan)
        safety_improvement_components.append(robust01(delta_ttc))
    if safety_improvement_components:
        safety_improvement_score = sum(safety_improvement_components) / float(len(safety_improvement_components))
    else:
        safety_improvement_score = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)

    travel_time_delta_penalty = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
    if "travel_time_s" in out.columns and "travel_time_s_baseline" in out.columns:
        travel_time_delta_penalty = robust01((safe_numeric(out, "travel_time_s", np.nan) - safe_numeric(out, "travel_time_s_baseline", np.nan)).clip(lower=0.0))
    jerk_delta_penalty = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
    if "comfort_jerk_rms_mps3" in out.columns and "comfort_jerk_rms_mps3_baseline" in out.columns:
        jerk_delta_penalty = robust01((safe_numeric(out, "comfort_jerk_rms_mps3", np.nan) - safe_numeric(out, "comfort_jerk_rms_mps3_baseline", np.nan)).clip(lower=0.0))
    no_safety_benefit_cost = (0.6 * travel_time_delta_penalty + 0.4 * jerk_delta_penalty) * (1.0 - safety_improvement_score.clip(0.0, 1.0))

    # Outcome-guided targets: raise horizon for true safety risk/benefit, and penalize
    # conservatism when efficiency/comfort worsens without measurable safety gains.
    out["safety_risk_score"] = safety_risk.clip(0.0, 1.0)
    out["conservatism_cost_score"] = conservatism_cost.clip(0.0, 1.0)
    out["safety_improvement_score"] = safety_improvement_score.clip(0.0, 1.0)
    out["no_safety_benefit_cost_score"] = no_safety_benefit_cost.clip(0.0, 1.0)
    out["target_ped_horizon_s"] = (
        ped_current
        + args.safety_gain_ped_s * (0.55 * out["safety_risk_score"] + 0.45 * out["safety_improvement_score"])
        - args.conservatism_penalty_ped_s * out["conservatism_cost_score"]
        - 0.55 * out["no_safety_benefit_cost_score"]
    ).clip(args.ped_horizon_min_s, args.ped_horizon_max_s)
    out["target_vehicle_horizon_s"] = (
        veh_current
        + args.safety_gain_veh_s * (0.60 * out["safety_risk_score"] + 0.40 * out["safety_improvement_score"])
        - args.conservatism_penalty_veh_s * out["conservatism_cost_score"]
        - 0.35 * out["no_safety_benefit_cost_score"]
    ).clip(args.veh_horizon_min_s, args.veh_horizon_max_s)
    out["current_ped_horizon_s"] = ped_current.astype(float)
    out["current_vehicle_horizon_s"] = veh_current.astype(float)
    return out


def fit_bounded_logistic_policy(
    X: pd.DataFrame,
    y_horizon: pd.Series,
    h_min: float,
    h_max: float,
    ridge_lambda: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    y = pd.to_numeric(y_horizon, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < max(4, X.shape[1] // 2):
        raise ValueError("Not enough finite training rows to fit learned horizon policy.")
    Xv = X.to_numpy(dtype=float)[valid]
    yv = y[valid]
    y01 = (yv - float(h_min)) / max(float(h_max) - float(h_min), EPS)
    z_target = logit(y01)

    # Ridge regression in logit space. Do not regularize the intercept/bias strongly.
    lam = max(float(ridge_lambda), 0.0)
    reg = lam * np.eye(Xv.shape[1])
    if Xv.shape[1] > 0:
        reg[0, 0] = 0.02 * lam
    theta = np.linalg.pinv(Xv.T @ Xv + reg) @ Xv.T @ z_target
    pred01 = sigmoid(Xv @ theta)
    pred_h = float(h_min) + (float(h_max) - float(h_min)) * pred01

    residual = pred_h - yv
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual * residual)))
    denom = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = float(1.0 - np.sum(residual * residual) / denom) if denom > EPS else float("nan")
    metrics = {
        "num_training_rows": int(valid.sum()),
        "mae_s": mae,
        "rmse_s": rmse,
        "r2": r2,
        "target_mean_s": float(np.mean(yv)),
        "prediction_mean_s": float(np.mean(pred_h)),
        "target_min_s": float(np.min(yv)),
        "target_max_s": float(np.max(yv)),
    }
    full_pred = np.full(len(X), np.nan, dtype=float)
    full_pred[valid] = pred_h
    return theta.astype(float), full_pred, metrics


def save_scatter_plot(actual: pd.Series, pred: pd.Series, title: str, out_path: Path, dpi: int) -> None:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    if valid.sum() == 0:
        return
    fig, ax = plt.subplots()
    ax.scatter(a[valid], p[valid], s=34, alpha=0.78, edgecolor="white", linewidth=0.4)
    lo = float(min(np.min(a[valid]), np.min(p[valid])))
    hi = float(max(np.max(a[valid]), np.max(p[valid])))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.6, label="ideal")
    ax.set_xlabel("Outcome-guided target horizon (s)")
    ax.set_ylabel("Learned policy prediction (s)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_coefficients_plot(theta_ped: np.ndarray, theta_veh: np.ndarray, out_path: Path, dpi: int) -> None:
    idx = np.arange(len(FEATURE_NAMES))
    width = 0.38
    fig_h = max(5.8, 0.38 * len(FEATURE_NAMES) + 2.0)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    ax.barh(idx - width / 2, theta_ped, height=width, label="pedestrian")
    ax.barh(idx + width / 2, theta_veh, height=width, label="vehicle")
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(idx)
    ax.set_yticklabels(FEATURE_NAMES)
    ax.set_xlabel("Learned coefficient in logistic horizon policy")
    ax.set_title("Learned adaptive-horizon policy coefficients")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def maybe_merge_baseline_features(osm_df: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    """
    Optional helper: if baseline metrics exist, attach baseline values by mode/level/seed.
    The learned target currently does not require them, but storing them supports auditability.
    """
    if not baseline_path.is_file():
        return osm_df
    try:
        b = pd.read_csv(baseline_path)
    except Exception:
        return osm_df
    keys = ["degradation_mode", "degradation_level_pct", "seed"]
    if not set(keys).issubset(osm_df.columns) or not set(keys).issubset(b.columns):
        return osm_df
    metric_cols = [
        c
        for c in [
            "unsafe_frames_total",
            "critical_ped_ttc_frames",
            "ped_conflict_zone_violation_frames",
            "min_ped_ttc_s",
            "stop_time_s",
            "travel_time_s",
            "comfort_jerk_rms_mps3",
        ]
        if c in b.columns
    ]
    if not metric_cols:
        return osm_df
    keep = keys + metric_cols
    b2 = b[keep].copy().groupby(keys, dropna=False).mean(numeric_only=True).reset_index()
    return osm_df.merge(b2, on=keys, how="left", suffixes=("", "_baseline"))


def train_scene(scene: str, args: argparse.Namespace) -> Dict[str, Any]:
    scene = normalize_scene_name_token(scene)
    args, geometry_learning_profile_info = apply_geometry_learning_profile(
        scene, args, disabled=bool(getattr(args, "disable_geometry_learning_profiles", False))
    )
    if bool(getattr(args, "print_geometry_learning_profile", False)) or geometry_learning_profile_info.get("profile_used", False):
        print(f"[geometry-learning-profile] {scene}: {geometry_learning_profile_info.get('geometry_case', geometry_learning_profile_info.get('reason', 'none'))}")
        if bool(getattr(args, "print_geometry_learning_profile", False)):
            print(json.dumps(geometry_learning_profile_info, indent=2))

    osm_path = Path(args.osm_results_root) / scene / "per_run_metrics.csv"
    baseline_path = Path(args.baseline_results_root) / scene / "per_run_metrics.csv"
    out_dir = Path(args.output_root) / scene
    paths = ensure_dirs(out_dir)

    if not osm_path.is_file():
        raise FileNotFoundError(
            f"Missing OSM-aware per-run metrics: {osm_path}. Run Script 09 first with the rule-based fallback outputs."
        )

    raw = pd.read_csv(osm_path)
    if len(raw) == 0:
        raise ValueError(f"No rows found in {osm_path}")
    raw = maybe_merge_baseline_features(raw, baseline_path)
    train_df = derive_outcome_targets(raw, args)
    X = build_features(train_df)

    theta_ped, pred_ped, ped_fit = fit_bounded_logistic_policy(
        X, train_df["target_ped_horizon_s"], args.ped_horizon_min_s, args.ped_horizon_max_s, args.ridge_lambda
    )
    theta_veh, pred_veh, veh_fit = fit_bounded_logistic_policy(
        X, train_df["target_vehicle_horizon_s"], args.veh_horizon_min_s, args.veh_horizon_max_s, args.ridge_lambda
    )

    pred_df = train_df.copy()
    for name in FEATURE_NAMES:
        pred_df[f"feature_{name}"] = X[name].to_numpy(dtype=float)
    pred_df["learned_pred_ped_horizon_s"] = pred_ped
    pred_df["learned_pred_vehicle_horizon_s"] = pred_veh
    pred_df["ped_horizon_residual_s"] = pred_df["learned_pred_ped_horizon_s"] - pred_df["target_ped_horizon_s"]
    pred_df["vehicle_horizon_residual_s"] = pred_df["learned_pred_vehicle_horizon_s"] - pred_df["target_vehicle_horizon_s"]

    training_cols = [
        c for c in [
            "osm_error_case", "offset_direction", "osm_error_signed_m", "osm_error_abs_m", "normalized_osm_error",
            "degradation_mode", "degradation_level_pct", "seed", "actual_camera_drop_fraction",
            "mean_crossing_proximity_factor", "safety_risk_score", "conservatism_cost_score",
            "current_ped_horizon_s", "target_ped_horizon_s", "learned_pred_ped_horizon_s",
            "current_vehicle_horizon_s", "target_vehicle_horizon_s", "learned_pred_vehicle_horizon_s",
        ] if c in pred_df.columns
    ]
    train_df.to_csv(paths["root"] / "learned_policy_training_dataset.csv", index=False)
    pred_df[training_cols + [f"feature_{n}" for n in FEATURE_NAMES]].to_csv(paths["root"] / "learned_policy_predictions.csv", index=False)

    save_scatter_plot(
        pred_df["target_ped_horizon_s"],
        pred_df["learned_pred_ped_horizon_s"],
        "Pedestrian horizon: outcome-guided target vs learned policy",
        paths["plots"] / "pedestrian_target_vs_prediction.png",
        args.dpi,
    )
    save_scatter_plot(
        pred_df["target_vehicle_horizon_s"],
        pred_df["learned_pred_vehicle_horizon_s"],
        "Vehicle horizon: outcome-guided target vs learned policy",
        paths["plots"] / "vehicle_target_vs_prediction.png",
        args.dpi,
    )
    save_coefficients_plot(theta_ped, theta_veh, paths["plots"] / "learned_theta_coefficients.png", args.dpi)

    policy = {
        "policy_type": "bounded_logistic_linear",
        "policy_name": "learned_uncertainty_conditioned_horizon_policy",
        "created_utc": utc_now_iso(),
        "scene_name": scene,
        "geometry_learning_profile": geometry_learning_profile_info,
        "training_source": str(osm_path),
        "baseline_source_optional": str(baseline_path) if baseline_path.is_file() else "",
        "feature_names": FEATURE_NAMES,
        "model_equation": "H_i = H_min_i + (H_max_i - H_min_i) * sigmoid(theta_i^T x)",
        "feature_definition": {
            "degradation_fraction": "degradation_level_pct / 100",
            "normalized_osm_error": "OSM crosswalk positional error normalized by crossing length/bounds",
            "crossing_proximity_factor": "mean ego proximity to crossing prior, 0 far and 1 near",
            "actual_camera_drop_fraction": "observed degraded-camera drop fraction when available; otherwise degradation_fraction",
            "mode_indicators": "one-hot indicators for random, burst, and distance-based dropout",
            "interaction_terms": "products capturing joint map-perception-context uncertainty",
        },
        "target_generation": {
            "description": "Outcome-guided targets are derived from closed-loop rollout metrics. Safety risk increases target horizon; conservatism/comfort/efficiency cost mildly decreases it.",
            "safety_gain_ped_s": args.safety_gain_ped_s,
            "conservatism_penalty_ped_s": args.conservatism_penalty_ped_s,
            "safety_gain_veh_s": args.safety_gain_veh_s,
            "conservatism_penalty_veh_s": args.conservatism_penalty_veh_s,
            "critical_ttc_threshold_s": args.critical_ttc_threshold_s,
        },
        "pedestrian": {
            "horizon_min_s": args.ped_horizon_min_s,
            "horizon_max_s": args.ped_horizon_max_s,
            "theta": [float(x) for x in theta_ped.tolist()],
            "fit_metrics": ped_fit,
        },
        "vehicle": {
            "horizon_min_s": args.veh_horizon_min_s,
            "horizon_max_s": args.veh_horizon_max_s,
            "theta": [float(x) for x in theta_veh.tolist()],
            "fit_metrics": veh_fit,
        },
        "deployment_notes": [
            "Place this JSON at data/learned_horizon_policy/<scene>/adaptive_horizon_policy.json.",
            "Script 09 loads it automatically unless --disable-learned-horizon-policy is used.",
            "If this file is absent, Script 09 falls back to the rule-based bounded linear horizon model.",
        ],
    }
    policy_path = paths["root"] / "adaptive_horizon_policy.json"
    policy_path.write_text(json.dumps(policy, indent=2, allow_nan=True), encoding="utf-8")

    summary = {
        "scene_name": scene,
        "geometry_learning_profile": geometry_learning_profile_info,
        "num_training_rows": int(len(pred_df)),
        "mean_target_pedestrian_horizon_s": float(pd.to_numeric(pred_df["target_ped_horizon_s"], errors="coerce").mean()),
        "mean_predicted_pedestrian_horizon_s": float(pd.to_numeric(pred_df["learned_pred_ped_horizon_s"], errors="coerce").mean()),
        "crossing_proximity_factor_nonzero": bool((safe_numeric(pred_df, "mean_crossing_proximity_factor", 0.0).fillna(0.0) > 0.0).any()),
        "normalized_osm_error_varies": bool(safe_numeric(pred_df, "normalized_osm_error", 0.0).nunique(dropna=True) > 1),
        "has_high_degradation_rows_ge_70_pct": bool((safe_numeric(pred_df, "degradation_level_pct", 0.0).fillna(0.0) >= 70.0).any()),
        "feature_names": FEATURE_NAMES,
        "feature_weights": {
            "pedestrian": {name: float(theta_ped[i]) for i, name in enumerate(FEATURE_NAMES)},
            "vehicle": {name: float(theta_veh[i]) for i, name in enumerate(FEATURE_NAMES)},
        },
        "pedestrian_fit": ped_fit,
        "vehicle_fit": veh_fit,
        "training_source": str(osm_path),
        "baseline_source_optional": str(baseline_path) if baseline_path.is_file() else "",
        "output_policy": str(policy_path),
    }
    (paths["root"] / "learned_policy_training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    print("learned_horizon_policy_training_summary:")
    print(f"  scene = {scene}")
    print(f"  rows = {len(pred_df)}")
    print(f"  pedestrian_mae_s = {ped_fit['mae_s']:.4f}")
    print(f"  vehicle_mae_s = {veh_fit['mae_s']:.4f}")
    print(f"  output_policy = {policy_path}")
    print("[done] Learned adaptive horizon policy saved.")

    return summary


def main() -> None:
    apply_plot_style()
    args = parse_args()
    scenes = parse_scene_name_list(args.scene_name, args.scene_names)

    print("batch_learned_horizon_training_start")
    print(f"  scenes: {scenes}")
    print(f"  osm_results_root: {args.osm_results_root}")
    print(f"  output_root: {args.output_root}")

    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for scene in scenes:
        try:
            summaries.append(train_scene(scene, args))
        except Exception as exc:
            failures.append({"scene_name": str(scene), "error": str(exc)})
            print(f"[failed] {scene}: {exc}")
            if not args.skip_missing:
                raise

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_payload = {
        "generated_utc": utc_now_iso(),
        "script": Path(__file__).name,
        "scenes_requested": scenes,
        "scenes_completed": [s["scene_name"] for s in summaries],
        "failures": failures,
        "osm_results_root": str(args.osm_results_root),
        "baseline_results_root": str(args.baseline_results_root),
        "output_root": str(args.output_root),
        "scene_summaries": summaries,
    }
    (output_root / "all_scenes_learned_horizon_training_summary.json").write_text(
        json.dumps(batch_payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    pd.DataFrame(summaries + failures).to_csv(output_root / "all_scenes_learned_horizon_training_summary.csv", index=False)

    print("\nbatch_learned_horizon_training_done")
    print(f"  completed: {[s['scene_name'] for s in summaries]}")
    print(f"  failures: {failures}")
    print(f"  batch_summary: {output_root / 'all_scenes_learned_horizon_training_summary.json'}")
    if failures and not args.skip_missing:
        raise RuntimeError(f"Learned horizon training failed for scenes: {[f['scene_name'] for f in failures]}")


if __name__ == "__main__":
    main()
