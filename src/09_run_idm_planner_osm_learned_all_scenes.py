#!/usr/bin/env python3
"""
Run OSM-error-aware adaptive-horizon closed-loop IDM longitudinal planning
using degraded camera-based perception inputs.

This script replaces the older reliability-aware OSM planner logic with an
explicit OSM crosswalk positional-error formulation.

It uses:
- degraded camera detections from Script 04,
- calibrated IDM parameters from Script 05,
- extracted OSM context from Script 07,
- OSM positional-error cases from Script 08,
- lightweight constant-velocity prediction for surrounding agents.

Main contribution:
For each frame, the planner computes an adaptive prediction horizon for
pedestrians and vehicles as a function of camera degradation level, OSM
crosswalk positional error, and ego proximity to the OSM crossing prior.

The goal is to mitigate the combined effect of camera detection dropout and
OSM crosswalk positional error while preserving safety, comfort, and efficiency.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUPPORTED_MODES = {"random_dropout", "burst_dropout", "distance_based_dropout"}
EPS = 1e-9


def parse_int_list_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 5.8),
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.2,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )


@dataclass
class IDMConfig:
    # Base calibrated IDM parameters. These are overwritten by Script 05 outputs when available.
    v0_mps: float = 8.5
    s0_m: float = 2.0
    T_s: float = 1.5
    a_max_mps2: float = 2.0
    b_comfort_mps2: float = 2.4
    delta: float = 4.0
    hard_brake_limit_mps2: float = 6.0
    max_speed_mps: float = 12.0

    # Runtime conflict gates.
    ped_stop_buffer_m: float = 3.8
    lead_lateral_threshold_m: float = 2.4
    ped_lateral_threshold_m: float = 3.0
    ped_future_path_threshold_m: float = 2.7
    ped_conflict_longitudinal_max_m: float = 22.0
    ped_conflict_lateral_max_m: float = 2.8
    ped_conflict_future_path_max_m: float = 2.7
    max_relevant_pedestrians_for_reporting: int = 3
    max_relevant_pedestrians_for_planning: int = 1
    lookahead_horizon_s: float = 3.0
    lookahead_step_m: float = 1.0

    # Safety / comfort metrics.
    unsafe_ped_distance_m: float = 4.0
    unsafe_veh_distance_m: float = 5.0
    stopped_speed_threshold_mps: float = 0.2
    critical_ped_ttc_threshold_s: float = 1.5
    critical_pet_threshold_s: float = 1.5
    hard_brake_epsilon_mps2: float = 0.05

    # OSM crossing-prior behavior.
    osm_crossing_min_radius_m: float = 4.5
    osm_signal_association_radius_m: float = 14.0
    osm_crossing_approach_distance_m: float = 38.0
    osm_crossing_core_distance_m: float = 10.0
    osm_crossing_past_buffer_m: float = 3.0
    osm_caution_speed_mps: float = 5.2
    osm_signal_extra_caution_factor: float = 0.85
    osm_low_speed_floor_mps: float = 2.5
    osm_ped_stop_buffer_extra_m: float = 1.2
    osm_ped_relevance_distance_m: float = 9.0
    osm_ped_relevance_bonus: float = 6.0
    osm_crossing_rel_long_bonus_m: float = 2.0

    # Lightweight prediction layer.
    prediction_dt_s: float = 0.5
    pedestrian_speed_cap_mps: float = 2.8
    vehicle_speed_cap_mps: float = 18.0
    pedestrian_prediction_base_radius_m: float = 0.8
    pedestrian_prediction_radius_growth_mps: float = 0.4
    vehicle_prediction_base_radius_m: float = 1.0
    vehicle_prediction_radius_growth_mps: float = 0.5
    predicted_path_conflict_threshold_m: float = 2.0
    predicted_crossing_zone_threshold_m: float = 2.5
    predicted_conflict_time_bonus: float = 7.0
    predicted_conflict_bonus: float = 9.0
    predicted_crossing_bonus: float = 6.0
    prediction_longitudinal_activation_m: float = 28.0
    prediction_min_speed_mps: float = 0.2

    # Adaptive prediction-horizon model.
    ped_horizon_min_s: float = 1.0
    ped_horizon_base_s: float = 2.0
    ped_horizon_max_s: float = 5.0
    ped_horizon_degradation_gain_s: float = 1.0
    ped_horizon_osm_error_gain_s: float = 1.5
    ped_horizon_proximity_gain_s: float = 1.0

    vehicle_horizon_min_s: float = 1.0
    vehicle_horizon_base_s: float = 1.5
    vehicle_horizon_max_s: float = 2.8
    vehicle_horizon_degradation_gain_s: float = 0.6
    vehicle_horizon_osm_error_gain_s: float = 0.4
    vehicle_horizon_proximity_gain_s: float = 0.3

    adaptive_horizon_proximity_distance_m: float = 35.0

    # PET-style crossing-zone metrics.
    pet_crossing_zone_radius_m: float = 3.0
    ego_crossing_zone_s_window_m: float = 1.0


@dataclass
class EgoState:
    frame_index: int
    timestamp_s: float
    s_progress_m: float
    speed_mps: float
    accel_mps2: float
    x: float
    y: float
    yaw_rad: float


@dataclass
class CrossingPrior:
    crossing_id: str
    center_x: float
    center_y: float
    radius_m: float
    s_crossing_m: float
    has_signal: bool
    num_points: int


@dataclass
class OSMErrorCase:
    osm_error_case: str
    offset_direction: str
    offset_m: float
    injected_offset_signed_m: float
    injected_offset_abs_m: float
    osm_error_signed_m: float
    osm_error_abs_m: float
    normalized_osm_error: float
    perturbed_crossing_s_m: float
    perturbed_crossing_x: float
    perturbed_crossing_y: float
    perturbed_crossing_yaw_rad: float
    crossing_length_m: float
    crossing_width_m: float
    clean_osm_feature_id: str
    clean_osm_feature_type: str
    clean_osm_crossing_s_m: float
    clean_osm_distance_to_reference_xy_m: float
    clean_osm_distance_to_route_m: float


class PathReference:
    def __init__(self, ego_df: pd.DataFrame) -> None:
        df = ego_df.sort_values("frame_index").reset_index(drop=True)
        self.s = pd.to_numeric(df["cumulative_distance_m"], errors="coerce").to_numpy(dtype=float)
        self.x = pd.to_numeric(df["ego_x"], errors="coerce").to_numpy(dtype=float)
        self.y = pd.to_numeric(df["ego_y"], errors="coerce").to_numpy(dtype=float)
        self.yaw = np.unwrap(pd.to_numeric(df["ego_yaw_rad"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        valid = np.isfinite(self.s) & np.isfinite(self.x) & np.isfinite(self.y)
        self.s = self.s[valid]
        self.x = self.x[valid]
        self.y = self.y[valid]
        self.yaw = self.yaw[valid]
        if len(self.s) < 2:
            raise ValueError("ego_timeline.csv must contain at least 2 valid points.")
        order = np.argsort(self.s)
        self.s = self.s[order]
        self.x = self.x[order]
        self.y = self.y[order]
        self.yaw = self.yaw[order]
        keep = np.concatenate([[True], np.diff(self.s) > 1e-6])
        self.s = self.s[keep]
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.yaw = self.yaw[keep]
        self.s_min = float(self.s[0])
        self.s_max = float(self.s[-1])

    def interp_pose(self, s_query: float) -> Tuple[float, float, float]:
        s_q = float(np.clip(float(s_query), self.s_min, self.s_max))
        x = float(np.interp(s_q, self.s, self.x))
        y = float(np.interp(s_q, self.s, self.y))
        yaw = float(np.interp(s_q, self.s, self.yaw))
        return x, y, yaw

    def future_polyline(self, s_start: float, speed_mps: float, horizon_s: float, step_m: float) -> np.ndarray:
        horizon_m = max(10.0, max(0.5, float(speed_mps)) * max(0.1, float(horizon_s)))
        n = max(2, int(horizon_m / max(step_m, 0.5)))
        ss = np.linspace(float(s_start), min(self.s_max, float(s_start) + horizon_m), n)
        return np.column_stack([np.interp(ss, self.s, self.x), np.interp(ss, self.s, self.y)])


def load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


TRUE_OBJECT_FILE_CANDIDATES = [
    "objects_timeline.csv",
    "objects_timeline_refined.csv",
    "refined_objects_timeline.csv",
    "agents_timeline.csv",
    "annotations_timeline.csv",
    "pedestrians_timeline.csv",
    "tracked_objects.csv",
    "objects.csv",
]


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_true_objects_table(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """
    Normalize the full/refined object table used only for PET evaluation.

    The planner still uses the degraded perception table. This true-object table
    is used only to decide whether the real pedestrian trajectory occupies the
    shifted OSM crossing zone for PET computation.
    """
    out = df.copy()

    if "frame_index" not in out.columns:
        raise ValueError(f"True-object file is missing frame_index: {source_path}")

    if "obj_x" not in out.columns:
        x_col = _first_existing_column(out, ["x", "local_x", "center_x", "translation_x", "annotation_x", "ego_local_x"])
        if x_col is None:
            raise ValueError(f"True-object file is missing obj_x/local x coordinate: {source_path}")
        out["obj_x"] = pd.to_numeric(out[x_col], errors="coerce")

    if "obj_y" not in out.columns:
        y_col = _first_existing_column(out, ["y", "local_y", "center_y", "translation_y", "annotation_y", "ego_local_y"])
        if y_col is None:
            raise ValueError(f"True-object file is missing obj_y/local y coordinate: {source_path}")
        out["obj_y"] = pd.to_numeric(out[y_col], errors="coerce")

    if "coarse_class" not in out.columns:
        cls_col = _first_existing_column(out, ["object_class", "class", "category", "category_name", "type", "label", "obj_class"])
        if cls_col is None:
            raise ValueError(f"True-object file is missing coarse_class/category column: {source_path}")
        raw = out[cls_col].astype(str).str.lower()
        out["coarse_class"] = np.where(
            raw.str.contains("pedestrian|human|person|adult|child", regex=True, na=False),
            "pedestrian",
            np.where(
                raw.str.contains("vehicle|car|truck|bus|trailer|construction|motorcycle|bicycle", regex=True, na=False),
                "vehicle",
                raw,
            ),
        )

    out["frame_index"] = pd.to_numeric(out["frame_index"], errors="coerce")
    out["obj_x"] = pd.to_numeric(out["obj_x"], errors="coerce")
    out["obj_y"] = pd.to_numeric(out["obj_y"], errors="coerce")
    out = out[np.isfinite(out["frame_index"]) & np.isfinite(out["obj_x"]) & np.isfinite(out["obj_y"])].copy()
    out["frame_index"] = out["frame_index"].astype(int)
    out["coarse_class"] = out["coarse_class"].astype(str).str.lower()
    return out.sort_values(["frame_index"]).reset_index(drop=True)


def _candidate_true_object_paths(refined_scene_dir: Path, requested_file: str = "") -> List[Path]:
    paths: List[Path] = []
    if requested_file and str(requested_file).strip():
        req = Path(str(requested_file).strip())
        paths.append(req if req.is_absolute() else refined_scene_dir / req)

    # Default refined-scene candidates.
    for name in TRUE_OBJECT_FILE_CANDIDATES:
        paths.append(refined_scene_dir / name)

    # Also allow the same scene under data/raw if refined does not contain a full object table.
    # Example: refined_scene_dir = data/refined/scene-0061 -> data/raw/scene-0061.
    raw_scene_dir = refined_scene_dir.parent.parent / "raw" / refined_scene_dir.name
    for name in TRUE_OBJECT_FILE_CANDIDATES:
        paths.append(raw_scene_dir / name)

    # Flexible fallback scan. Header validation below prevents sample/ego CSVs from being used.
    for base in [refined_scene_dir, raw_scene_dir]:
        if base.is_dir():
            for pattern in ["*object*.csv", "*agent*.csv", "*annotation*.csv", "*.csv"]:
                paths.extend(sorted(base.glob(pattern)))

    # Preserve order while removing duplicates.
    unique: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def load_true_objects_for_pet(refined_scene_dir: Path, requested_file: str = "") -> Tuple[pd.DataFrame, Path]:
    """
    Load the full/refined object table used for PET evaluation.

    This table should not be degraded. It is used only for the pedestrian-zone
    occupancy flag in PET. The planner itself continues to use degraded camera
    detections.
    """
    attempted: List[str] = []
    for path in _candidate_true_object_paths(refined_scene_dir, requested_file):
        if not path.is_file():
            continue
        attempted.append(str(path))
        try:
            head = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        header = set(head.columns)
        has_frame = "frame_index" in header
        has_xy = ("obj_x" in header and "obj_y" in header) or (
            bool(header & {"x", "local_x", "center_x", "translation_x", "annotation_x"})
            and bool(header & {"y", "local_y", "center_y", "translation_y", "annotation_y"})
        )
        has_class = bool(header & {"coarse_class", "object_class", "class", "category", "category_name", "type", "label", "obj_class"})
        if not (has_frame and has_xy and has_class):
            continue
        df = pd.read_csv(path)
        norm = normalize_true_objects_table(df, path)
        if len(norm) == 0:
            continue
        if not norm["coarse_class"].eq("pedestrian").any():
            # This file is probably not the full dynamic-object table we need.
            continue
        return norm, path

    expected = ", ".join(TRUE_OBJECT_FILE_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a full/refined true object trajectory file for PET evaluation. "
        f"Looked under {refined_scene_dir} and the matching raw scene folder. "
        f"Expected one of: {expected}. You can pass --true-objects-file <path-or-name>. "
        f"Candidate files inspected: {attempted[:12]}"
    )


def load_calibrated_idm_config(scene_name: str, calibration_root: Path) -> Tuple[IDMConfig, Optional[Dict[str, Any]]]:
    cfg = IDMConfig()
    params_path = calibration_root / scene_name / "idm_parameters.json"
    payload = load_optional_json(params_path)
    if not payload:
        return cfg, None
    fitted = payload.get("fitted_parameters", {})
    for field in ("v0_mps", "s0_m", "T_s", "a_max_mps2", "b_comfort_mps2", "delta"):
        if field in fitted:
            setattr(cfg, field, float(fitted[field]))
    return cfg, payload


def osm_error_case_from_row(row: pd.Series) -> OSMErrorCase:
    def f(name: str, default: float = 0.0) -> float:
        val = row.get(name, default)
        return float(val) if pd.notna(val) else float(default)

    def s(name: str, default: str = "") -> str:
        val = row.get(name, default)
        return str(val) if pd.notna(val) else default

    return OSMErrorCase(
        osm_error_case=s("osm_error_case"),
        offset_direction=s("offset_direction"),
        offset_m=f("offset_m"),
        injected_offset_signed_m=f("injected_offset_signed_m"),
        injected_offset_abs_m=f("injected_offset_abs_m"),
        osm_error_signed_m=f("osm_error_signed_m"),
        osm_error_abs_m=f("osm_error_abs_m"),
        normalized_osm_error=float(np.clip(f("normalized_osm_error"), 0.0, 1.0)),
        perturbed_crossing_s_m=f("perturbed_crossing_s_m"),
        perturbed_crossing_x=f("perturbed_crossing_x"),
        perturbed_crossing_y=f("perturbed_crossing_y"),
        perturbed_crossing_yaw_rad=f("perturbed_crossing_yaw_rad"),
        crossing_length_m=max(f("crossing_length_m", 10.0), 1e-6),
        crossing_width_m=max(f("crossing_width_m", 5.0), 1e-6),
        clean_osm_feature_id=s("clean_osm_feature_id"),
        clean_osm_feature_type=s("clean_osm_feature_type"),
        clean_osm_crossing_s_m=f("clean_osm_crossing_s_m"),
        clean_osm_distance_to_reference_xy_m=f("clean_osm_distance_to_reference_xy_m"),
        clean_osm_distance_to_route_m=f("clean_osm_distance_to_route_m"),
    )


def load_osm_error_cases(osm_scene_dir: Path, requested_cases: Optional[List[str]] = None) -> List[OSMErrorCase]:
    path = osm_scene_dir / "error_cases" / "osm_error_cases.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing OSM error cases: {path}. Run 08_generate_osm_error_cases.py first.")
    df = pd.read_csv(path)
    if requested_cases:
        wanted = set(requested_cases)
        df = df[df["osm_error_case"].astype(str).isin(wanted)].copy()
        missing = sorted(wanted - set(df["osm_error_case"].astype(str).unique()))
        if missing:
            raise ValueError(f"Requested OSM error cases not found in {path}: {missing}")
    if len(df) == 0:
        raise ValueError(f"No OSM error cases available in {path}.")
    return [osm_error_case_from_row(r) for _, r in df.iterrows()]


def infer_signal_near_crossing(osm_local_df: pd.DataFrame, x: float, y: float, cfg: IDMConfig) -> bool:
    if osm_local_df is None or len(osm_local_df) == 0 or "feature_type" not in osm_local_df.columns:
        return False
    sig = osm_local_df[osm_local_df["feature_type"].astype(str).eq("traffic_signal")].copy()
    if len(sig) == 0:
        return False
    sx = pd.to_numeric(sig["local_x"], errors="coerce").to_numpy(dtype=float)
    sy = pd.to_numeric(sig["local_y"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(sx) & np.isfinite(sy)
    if not valid.any():
        return False
    d = np.sqrt((sx[valid] - float(x)) ** 2 + (sy[valid] - float(y)) ** 2)
    return bool(np.any(d <= cfg.osm_signal_association_radius_m))


def build_crossing_prior_from_error_case(case: OSMErrorCase, osm_local_df: pd.DataFrame, cfg: IDMConfig) -> CrossingPrior:
    has_signal = infer_signal_near_crossing(osm_local_df, case.perturbed_crossing_x, case.perturbed_crossing_y, cfg)
    radius = max(float(cfg.osm_crossing_min_radius_m), 0.5 * float(case.crossing_width_m) + 1.5)
    return CrossingPrior(
        crossing_id=str(case.osm_error_case),
        center_x=float(case.perturbed_crossing_x),
        center_y=float(case.perturbed_crossing_y),
        radius_m=float(radius),
        s_crossing_m=float(case.perturbed_crossing_s_m),
        has_signal=bool(has_signal),
        num_points=1,
    )


def save_crossing_priors(rows: List[Dict[str, Any]], priors_root: Path) -> None:
    priors_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(priors_root / "derived_osm_error_priors.csv", index=False)
    (priors_root / "derived_osm_error_priors.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def nearest_crossing_for_ego(priors: List[CrossingPrior], ego_s: float, ego_x: float, ego_y: float, cfg: IDMConfig) -> Dict[str, Any]:
    if not priors:
        return {
            "crossing_ahead": False,
            "crossing_core": False,
            "caution_active": False,
            "distance_along_path_m": np.nan,
            "nearest_crossing_id": "",
            "nearest_center_x": np.nan,
            "nearest_center_y": np.nan,
            "nearest_has_signal": False,
            "nearest_euclidean_m": np.inf,
        }
    nearest = min(priors, key=lambda p: (ego_x - p.center_x) ** 2 + (ego_y - p.center_y) ** 2)
    ds = float(nearest.s_crossing_m - ego_s)
    dxy = float(math.hypot(ego_x - nearest.center_x, ego_y - nearest.center_y))
    approaching = bool((-cfg.osm_crossing_past_buffer_m <= ds) and (ds <= cfg.osm_crossing_approach_distance_m))
    return {
        "crossing_ahead": bool(ds > 0.0),
        "crossing_approaching": approaching,
        "crossing_core": bool(abs(ds) <= max(nearest.radius_m, cfg.osm_crossing_core_distance_m)),
        "caution_active": False,
        "distance_along_path_m": ds,
        "gap_to_crossing_m": ds,
        "nearest_crossing_id": nearest.crossing_id,
        "nearest_center_x": nearest.center_x,
        "nearest_center_y": nearest.center_y,
        "nearest_has_signal": bool(nearest.has_signal),
        "nearest_euclidean_m": dxy,
    }


def crossing_proximity_factor(gap_to_crossing_m: float, cfg: IDMConfig) -> float:
    if not np.isfinite(gap_to_crossing_m):
        return 0.0
    if not (-cfg.osm_crossing_past_buffer_m <= float(gap_to_crossing_m) <= cfg.adaptive_horizon_proximity_distance_m):
        return 0.0
    return float(
        1.0
        - np.clip(
            float(gap_to_crossing_m) / max(cfg.adaptive_horizon_proximity_distance_m, 1e-6),
            0.0,
            1.0,
        )
    )


def sigmoid_stable(z: float) -> float:
    z = float(np.clip(z, -60.0, 60.0))
    return float(1.0 / (1.0 + math.exp(-z)))


def build_horizon_policy_features(
    level_pct: int,
    normalized_osm_error: float,
    crossing_proximity: float,
    degradation_mode: str,
    actual_camera_drop_fraction: float,
) -> Dict[str, float]:
    """
    Compact uncertainty/context state used by the learned horizon policy.

    The learned model is intentionally low-dimensional and interpretable. It maps
    camera degradation, OSM positional error, ego-crossing proximity, and mode
    indicators to bounded pedestrian/vehicle prediction horizons.
    """
    d = float(np.clip(float(level_pct) / 100.0, 0.0, 1.0))
    e = float(np.clip(float(normalized_osm_error), 0.0, 1.0))
    p = float(np.clip(float(crossing_proximity), 0.0, 1.0))
    a = float(actual_camera_drop_fraction) if np.isfinite(actual_camera_drop_fraction) else d
    a = float(np.clip(a, 0.0, 1.0))
    mode = str(degradation_mode or "").strip().lower()
    return {
        "bias": 1.0,
        "degradation_fraction": d,
        "normalized_osm_error": e,
        "crossing_proximity_factor": p,
        "actual_camera_drop_fraction": a,
        "is_random_dropout": 1.0 if mode == "random_dropout" else 0.0,
        "is_burst_dropout": 1.0 if mode == "burst_dropout" else 0.0,
        "is_distance_based_dropout": 1.0 if mode == "distance_based_dropout" else 0.0,
        "degradation_x_osm_error": d * e,
        "degradation_x_proximity": d * p,
        "osm_error_x_proximity": e * p,
        "drop_fraction_x_proximity": a * p,
    }


def load_learned_horizon_policy(policy_path: Path) -> Optional[Dict[str, Any]]:
    """Load a learned bounded-logistic horizon policy if it exists."""
    if not policy_path.is_file():
        return None
    payload = load_optional_json(policy_path)
    if not isinstance(payload, dict):
        return None
    if str(payload.get("policy_type", "")) != "bounded_logistic_linear":
        raise ValueError(
            f"Unsupported horizon policy_type in {policy_path}: {payload.get('policy_type')!r}. "
            "Expected 'bounded_logistic_linear'."
        )
    features = payload.get("feature_names", [])
    if not isinstance(features, list) or not features:
        raise ValueError(f"Learned horizon policy is missing feature_names: {policy_path}")
    for actor in ("pedestrian", "vehicle"):
        block = payload.get(actor, {})
        theta = block.get("theta", []) if isinstance(block, dict) else []
        if not isinstance(theta, list) or len(theta) != len(features):
            raise ValueError(
                f"Learned horizon policy {actor} theta length does not match feature_names in {policy_path}."
            )
    payload["_loaded_from"] = str(policy_path)
    return payload


def learned_horizon_from_policy(
    policy: Dict[str, Any],
    actor: str,
    feature_values: Dict[str, float],
    fallback_min: float,
    fallback_max: float,
) -> Tuple[float, float]:
    """Return (bounded_horizon, latent_score) from a learned linear-logistic policy."""
    features = [str(x) for x in policy.get("feature_names", [])]
    block = policy.get(actor, {})
    if not isinstance(block, dict):
        raise ValueError(f"Policy missing actor block: {actor}")
    theta = np.asarray(block.get("theta", []), dtype=float)
    x = np.asarray([float(feature_values.get(name, 0.0)) for name in features], dtype=float)
    if len(theta) != len(x):
        raise ValueError(f"Policy theta/features mismatch for {actor}: {len(theta)} vs {len(x)}")
    h_min = float(block.get("horizon_min_s", fallback_min))
    h_max = float(block.get("horizon_max_s", fallback_max))
    if not np.isfinite(h_min) or not np.isfinite(h_max) or h_max <= h_min:
        h_min, h_max = fallback_min, fallback_max
    z = float(np.dot(theta, x))
    y = sigmoid_stable(z)
    return float(h_min + (h_max - h_min) * y), z


def compute_adaptive_horizons(
    level_pct: int,
    normalized_osm_error: float,
    gap_to_crossing_m: float,
    cfg: IDMConfig,
    degradation_mode: str = "",
    actual_camera_drop_fraction: float = float("nan"),
    learned_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, float | str | bool]:
    degradation_fraction = float(np.clip(float(level_pct) / 100.0, 0.0, 1.0))
    err = float(np.clip(normalized_osm_error, 0.0, 1.0))
    prox = crossing_proximity_factor(gap_to_crossing_m, cfg)

    # Fallback/manual policy: deterministic bounded linear rule.
    ped_h_rule = (
        cfg.ped_horizon_base_s
        + cfg.ped_horizon_degradation_gain_s * degradation_fraction
        + cfg.ped_horizon_osm_error_gain_s * err
        + cfg.ped_horizon_proximity_gain_s * prox
    )
    veh_h_rule = (
        cfg.vehicle_horizon_base_s
        + cfg.vehicle_horizon_degradation_gain_s * degradation_fraction
        + cfg.vehicle_horizon_osm_error_gain_s * err
        + cfg.vehicle_horizon_proximity_gain_s * prox
    )
    ped_h = float(np.clip(ped_h_rule, cfg.ped_horizon_min_s, cfg.ped_horizon_max_s))
    veh_h = float(np.clip(veh_h_rule, cfg.vehicle_horizon_min_s, cfg.vehicle_horizon_max_s))
    policy_type = "rule_based_bounded_linear"
    learned_used = False
    ped_score = float("nan")
    veh_score = float("nan")

    if learned_policy is not None:
        features = build_horizon_policy_features(
            level_pct=level_pct,
            normalized_osm_error=err,
            crossing_proximity=prox,
            degradation_mode=degradation_mode,
            actual_camera_drop_fraction=actual_camera_drop_fraction,
        )
        ped_h, ped_score = learned_horizon_from_policy(
            learned_policy, "pedestrian", features, cfg.ped_horizon_min_s, cfg.ped_horizon_max_s
        )
        veh_h, veh_score = learned_horizon_from_policy(
            learned_policy, "vehicle", features, cfg.vehicle_horizon_min_s, cfg.vehicle_horizon_max_s
        )
        ped_h = float(np.clip(ped_h, cfg.ped_horizon_min_s, cfg.ped_horizon_max_s))
        veh_h = float(np.clip(veh_h, cfg.vehicle_horizon_min_s, cfg.vehicle_horizon_max_s))
        policy_type = str(learned_policy.get("policy_name", "learned_bounded_logistic"))
        learned_used = True

    return {
        "degradation_fraction": degradation_fraction,
        "normalized_osm_error": err,
        "crossing_proximity_factor": prox,
        "actual_camera_drop_fraction_used": float(actual_camera_drop_fraction) if np.isfinite(actual_camera_drop_fraction) else degradation_fraction,
        "adaptive_ped_prediction_horizon_s": ped_h,
        "adaptive_vehicle_prediction_horizon_s": veh_h,
        "adaptive_ped_horizon_rule_fallback_s": float(np.clip(ped_h_rule, cfg.ped_horizon_min_s, cfg.ped_horizon_max_s)),
        "adaptive_vehicle_horizon_rule_fallback_s": float(np.clip(veh_h_rule, cfg.vehicle_horizon_min_s, cfg.vehicle_horizon_max_s)),
        "adaptive_horizon_policy_type": policy_type,
        "learned_horizon_policy_used_flag": bool(learned_used),
        "learned_policy_latent_score_ped": ped_score,
        "learned_policy_latent_score_vehicle": veh_score,
    }


def osm_context_weight(level_pct: int, normalized_osm_error: float, proximity: float) -> float:
    """
    Context weight used only to blend OSM caution and map-prior conflict bonuses.
    It is not a reliability score. It increases near the crossing and under higher
    modeled error/degradation, making the planner more cautious without changing
    the calibrated IDM parameters.
    """
    deg = float(np.clip(float(level_pct) / 100.0, 0.0, 1.0))
    err = float(np.clip(normalized_osm_error, 0.0, 1.0))
    prox = float(np.clip(proximity, 0.0, 1.0))
    return float(np.clip(prox * (0.45 + 0.35 * err + 0.20 * deg), 0.0, 1.0))


def combined_uncertainty_score(
    degradation_fraction: float,
    normalized_osm_error: float,
    actual_camera_drop_fraction: float,
    is_burst_dropout: float,
    route_based_crossing_proximity_factor: float,
) -> float:
    a = float(actual_camera_drop_fraction) if np.isfinite(actual_camera_drop_fraction) else float(degradation_fraction)
    return float(
        np.clip(
            0.35 * float(np.clip(degradation_fraction, 0.0, 1.0))
            + 0.25 * float(np.clip(normalized_osm_error, 0.0, 1.0))
            + 0.20 * float(np.clip(a, 0.0, 1.0))
            + 0.10 * float(np.clip(is_burst_dropout, 0.0, 1.0))
            + 0.10 * float(np.clip(route_based_crossing_proximity_factor, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )


def map_aware_v0(cfg: IDMConfig, crossing_ctx: Dict[str, Any], context_weight: float, route_proximity: float) -> Tuple[float, bool]:
    ds = crossing_ctx["distance_along_path_m"]
    if not np.isfinite(ds) or context_weight <= 0.0 or route_proximity <= 0.0:
        return cfg.v0_mps, False
    base_target = cfg.v0_mps
    caution = False
    if -cfg.osm_crossing_past_buffer_m <= ds <= cfg.osm_crossing_approach_distance_m:
        base_target = min(cfg.v0_mps, cfg.osm_caution_speed_mps)
        caution = True
    elif bool(crossing_ctx["crossing_core"]):
        base_target = min(cfg.v0_mps, cfg.osm_caution_speed_mps * 0.9)
        caution = True
    if crossing_ctx["nearest_has_signal"]:
        base_target *= cfg.osm_signal_extra_caution_factor
    v0_used = (1.0 - context_weight) * cfg.v0_mps + context_weight * base_target
    v0_used = max(cfg.osm_low_speed_floor_mps, float(v0_used))
    return float(v0_used), bool(caution)


def init_ego(sample_df: pd.DataFrame, ego_ref_df: pd.DataFrame, pref: PathReference) -> EgoState:
    t0 = float(sample_df.iloc[0]["timestamp_s"])
    s0 = float(ego_ref_df.iloc[0]["cumulative_distance_m"])
    vs = pd.to_numeric(ego_ref_df["ego_speed_mps"], errors="coerce").dropna()
    v0 = float(vs.iloc[0]) if len(vs) else 0.0
    x0, y0, yaw0 = pref.interp_pose(s0)
    return EgoState(0, t0, s0, v0, 0.0, x0, y0, yaw0)


def compute_object_velocity(df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        out = df.copy()
        out["vx_est_mps"] = []
        out["vy_est_mps"] = []
        return out
    ts_map = sample_df.set_index("frame_index")["timestamp_s"].to_dict()
    work = df.copy().sort_values(["instance_token", "frame_index"]).reset_index(drop=True)
    work["vx_est_mps"] = 0.0
    work["vy_est_mps"] = 0.0
    for _, g in work.groupby("instance_token"):
        idx = g.index.to_numpy()
        gx = pd.to_numeric(g["obj_x"], errors="coerce").to_numpy(dtype=float)
        gy = pd.to_numeric(g["obj_y"], errors="coerce").to_numpy(dtype=float)
        gf = pd.to_numeric(g["frame_index"], errors="coerce").to_numpy(dtype=int)
        t = np.array([float(ts_map.get(int(fi), np.nan)) for fi in gf], dtype=float)
        vx = np.zeros(len(g), dtype=float)
        vy = np.zeros(len(g), dtype=float)
        for i in range(len(g)):
            if len(g) == 1 or not np.isfinite(t[i]):
                continue
            if i == 0:
                dt = max(1e-3, t[i + 1] - t[i])
                vx[i] = (gx[i + 1] - gx[i]) / dt
                vy[i] = (gy[i + 1] - gy[i]) / dt
            elif i == len(g) - 1:
                dt = max(1e-3, t[i] - t[i - 1])
                vx[i] = (gx[i] - gx[i - 1]) / dt
                vy[i] = (gy[i] - gy[i - 1]) / dt
            else:
                dt = max(1e-3, t[i + 1] - t[i - 1])
                vx[i] = (gx[i + 1] - gx[i - 1]) / dt
                vy[i] = (gy[i + 1] - gy[i - 1]) / dt
        work.loc[idx, "vx_est_mps"] = vx
        work.loc[idx, "vy_est_mps"] = vy
    return work


def point_to_polyline_distance(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    if len(poly) == 0:
        return np.full(len(px), np.inf, dtype=float)
    if len(poly) < 2:
        return np.sqrt((px - poly[0, 0]) ** 2 + (py - poly[0, 1]) ** 2)
    pts = np.column_stack([px, py])
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)
    out = np.full(len(pts), np.inf, dtype=float)
    for i, p in enumerate(pts):
        ap = p - a
        t = np.clip(np.sum(ap * ab, axis=1) / np.maximum(ab2, 1e-9), 0.0, 1.0)
        proj = a + ab * t[:, None]
        out[i] = float(np.min(np.sqrt(np.sum((proj - p) ** 2, axis=1))))
    return out


def count_true_episodes(flags: pd.Series) -> int:
    arr = flags.fillna(False).astype(bool).to_numpy()
    if len(arr) == 0:
        return 0
    starts = arr & np.concatenate(([True], ~arr[:-1]))
    return int(starts.sum())


def safe_min(series: pd.Series, default: float = float("inf")) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace(np.inf, np.nan).to_numpy(dtype=float)
    return float(np.nanmin(vals)) if np.isfinite(vals).any() else float(default)


def finite_or_nan(value: Any) -> float:
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) and np.isfinite(x) else float("nan")


def compute_runtime_features(frame_df: pd.DataFrame, ego: EgoState, pref: PathReference, cfg: IDMConfig, crossing_priors: List[CrossingPrior]) -> pd.DataFrame:
    out = frame_df.copy()
    if len(out) == 0:
        return out
    dx = pd.to_numeric(out["obj_x"], errors="coerce").to_numpy(dtype=float) - ego.x
    dy = pd.to_numeric(out["obj_y"], errors="coerce").to_numpy(dtype=float) - ego.y
    c = float(math.cos(ego.yaw_rad))
    s = float(math.sin(ego.yaw_rad))
    rel_long = dx * c + dy * s
    rel_lat = -dx * s + dy * c
    dist = np.sqrt(dx * dx + dy * dy)
    ahead = rel_long > 0.0
    v_long = out["vx_est_mps"].to_numpy(dtype=float) * c + out["vy_est_mps"].to_numpy(dtype=float) * s
    future = pref.future_polyline(ego.s_progress_m, ego.speed_mps, cfg.lookahead_horizon_s, cfg.lookahead_step_m)
    fdist = point_to_polyline_distance(out["obj_x"].to_numpy(dtype=float), out["obj_y"].to_numpy(dtype=float), future)

    out["runtime_distance_to_ego_m"] = dist
    out["runtime_rel_long_m"] = rel_long
    out["runtime_rel_lat_m"] = rel_lat
    out["runtime_ahead_flag"] = ahead
    out["runtime_future_path_distance_m"] = fdist
    out["runtime_near_future_path_flag"] = fdist <= cfg.ped_future_path_threshold_m
    out["runtime_v_long_mps"] = v_long

    if crossing_priors:
        cxy = np.array([[p.center_x, p.center_y] for p in crossing_priors], dtype=float)
        ox = out["obj_x"].to_numpy(dtype=float)[:, None]
        oy = out["obj_y"].to_numpy(dtype=float)[:, None]
        dd = np.sqrt((ox - cxy[:, 0][None, :]) ** 2 + (oy - cxy[:, 1][None, :]) ** 2)
        out["osm_crossing_euclidean_m"] = np.min(dd, axis=1)
    else:
        out["osm_crossing_euclidean_m"] = np.inf
    return out


def predict_agent_conflicts(
    runtime_df: pd.DataFrame,
    ego: EgoState,
    pref: PathReference,
    cfg: IDMConfig,
    crossing_priors: List[CrossingPrior],
    ped_horizon_s: float,
    vehicle_horizon_s: float,
) -> pd.DataFrame:
    out = runtime_df.copy()
    if len(out) == 0:
        return out

    max_h = max(float(ped_horizon_s), float(vehicle_horizon_s), cfg.prediction_dt_s)
    ego_future = pref.future_polyline(ego.s_progress_m, max(ego.speed_mps, 1.0), max_h, cfg.lookahead_step_m)
    crossing_centers = np.array([[p.center_x, p.center_y] for p in crossing_priors], dtype=float) if crossing_priors else np.zeros((0, 2), dtype=float)
    step_times_all = np.arange(cfg.prediction_dt_s, max_h + 1e-9, cfg.prediction_dt_s, dtype=float)

    vals = {
        k: []
        for k in [
            "pred_horizon_s_used",
            "pred_min_path_distance_m",
            "pred_min_crossing_distance_m",
            "pred_conflict_any_flag",
            "pred_crossing_any_flag",
            "pred_first_conflict_t_s",
            "pred_first_crossing_t_s",
            "pred_uncertainty_radius_final_m",
            "predicted_conflict_bonus",
        ]
    }

    for _, row in out.iterrows():
        cc = str(row.get("coarse_class", ""))
        is_ped = cc == "pedestrian"
        is_veh = cc == "vehicle"
        if not (is_ped or is_veh):
            for k, v in zip(vals.keys(), [0.0, np.nan, np.nan, False, False, np.nan, np.nan, np.nan, 0.0]):
                vals[k].append(v)
            continue

        horizon = float(ped_horizon_s if is_ped else vehicle_horizon_s)
        times = step_times_all[step_times_all <= horizon + 1e-9]
        if len(times) == 0:
            times = np.array([cfg.prediction_dt_s], dtype=float)

        x0 = float(row["obj_x"])
        y0 = float(row["obj_y"])
        vx = float(row["vx_est_mps"])
        vy = float(row["vy_est_mps"])
        speed = float(math.hypot(vx, vy))
        speed_cap = cfg.pedestrian_speed_cap_mps if is_ped else cfg.vehicle_speed_cap_mps
        if speed > speed_cap and speed > 1e-6:
            scale = speed_cap / speed
            vx *= scale
            vy *= scale
            speed = speed_cap
        if speed < cfg.prediction_min_speed_mps:
            vx = 0.0
            vy = 0.0

        rel_long = float(row["runtime_rel_long_m"])
        if not np.isfinite(rel_long) or rel_long > cfg.prediction_longitudinal_activation_m:
            for k, v in zip(vals.keys(), [horizon, np.nan, np.nan, False, False, np.nan, np.nan, np.nan, 0.0]):
                vals[k].append(v)
            continue

        px = x0 + vx * times
        py = y0 + vy * times
        path_dist = point_to_polyline_distance(px, py, ego_future)
        if is_ped:
            unc = cfg.pedestrian_prediction_base_radius_m + cfg.pedestrian_prediction_radius_growth_mps * times
        else:
            unc = cfg.vehicle_prediction_base_radius_m + cfg.vehicle_prediction_radius_growth_mps * times

        conflict_mask = (path_dist - unc) <= cfg.predicted_path_conflict_threshold_m
        conflict_any = bool(np.any(conflict_mask))
        first_conflict = float(times[np.argmax(conflict_mask)]) if conflict_any else np.nan

        if len(crossing_centers):
            dd = np.sqrt((px[:, None] - crossing_centers[:, 0][None, :]) ** 2 + (py[:, None] - crossing_centers[:, 1][None, :]) ** 2)
            cross_dist = np.min(dd, axis=1)
            cross_mask = (cross_dist - unc) <= cfg.predicted_crossing_zone_threshold_m
            cross_any = bool(np.any(cross_mask))
            first_cross = float(times[np.argmax(cross_mask)]) if cross_any else np.nan
            min_cross = float(np.min(cross_dist))
        else:
            cross_any = False
            first_cross = np.nan
            min_cross = np.nan

        bonus = 0.0
        if conflict_any:
            bonus += cfg.predicted_conflict_bonus + cfg.predicted_conflict_time_bonus / max(first_conflict, cfg.prediction_dt_s)
        if is_ped and cross_any:
            bonus += cfg.predicted_crossing_bonus

        for k, v in zip(
            vals.keys(),
            [
                horizon,
                float(np.min(path_dist)) if len(path_dist) else np.nan,
                min_cross,
                conflict_any,
                cross_any,
                first_conflict,
                first_cross,
                float(unc[-1]) if len(unc) else np.nan,
                float(bonus),
            ],
        ):
            vals[k].append(v)

    for k in vals:
        out[k] = vals[k]
    return out


def get_relevant_conflict_pedestrians(runtime_df: pd.DataFrame, cfg: IDMConfig, crossing_ctx: Dict[str, Any], context_weight: float) -> pd.DataFrame:
    if len(runtime_df) == 0 or "coarse_class" not in runtime_df.columns:
        return runtime_df.iloc[0:0].copy()
    ped = runtime_df[runtime_df["coarse_class"].eq("pedestrian")].copy()
    if len(ped) == 0:
        return ped

    base_gate = (
        ped["runtime_ahead_flag"].astype(bool)
        & (ped["runtime_rel_long_m"] > 0.0)
        & (ped["runtime_rel_long_m"] <= cfg.ped_conflict_longitudinal_max_m)
        & (ped["runtime_rel_lat_m"].abs() <= cfg.ped_conflict_lateral_max_m)
    )
    current_or_predicted = (
        ped["runtime_near_future_path_flag"].astype(bool)
        | ped["pred_conflict_any_flag"].fillna(False).astype(bool)
        | ped["pred_crossing_any_flag"].fillna(False).astype(bool)
    )
    ped = ped[base_gate & current_or_predicted].copy()
    if len(ped) == 0:
        return ped

    ped["osm_near_crossing_flag"] = ped["osm_crossing_euclidean_m"].to_numpy(dtype=float) <= cfg.osm_ped_relevance_distance_m
    ped["ped_conflict_score_used"] = (
        4.0 * ped["runtime_future_path_distance_m"].fillna(cfg.ped_future_path_threshold_m + 5.0)
        + 2.0 * ped["runtime_rel_lat_m"].abs()
        + 1.0 * ped["runtime_rel_long_m"]
    )
    ped.loc[ped["osm_near_crossing_flag"].astype(bool), "ped_conflict_score_used"] -= context_weight * float(cfg.osm_ped_relevance_bonus)
    ped["ped_conflict_score_used"] -= context_weight * ped["predicted_conflict_bonus"].fillna(0.0)
    if bool(crossing_ctx.get("caution_active", False)):
        ped["ped_conflict_score_used"] -= context_weight * float(cfg.osm_crossing_rel_long_bonus_m) / np.maximum(ped["runtime_rel_long_m"].to_numpy(dtype=float), 1.0)

    return ped.sort_values(
        ["ped_conflict_score_used", "runtime_future_path_distance_m", "pred_first_conflict_t_s", "runtime_rel_lat_m", "runtime_rel_long_m"],
        key=lambda c: c.abs() if c.name == "runtime_rel_lat_m" else c,
    ).reset_index(drop=True)


def select_active_constraint(
    runtime_df: pd.DataFrame,
    relevant_peds_df: pd.DataFrame,
    ego_s_progress_m: float,
    cfg: IDMConfig,
    context_weight: float,
    ped_horizon_s: float,
    osm_virtual_gate_active: bool = False,
    osm_virtual_gate_s: float = float("nan"),
) -> Dict[str, Any]:
    default = {
        "lead_type": "none",
        "lead_token": "",
        "lead_gap_m": np.inf,
        "lead_speed_mps": 0.0,
        "ped_conflict_present": False,
        "num_relevant_conflict_pedestrians_raw": 0,
        "num_relevant_conflict_pedestrians_used": 0,
        "num_relevant_conflict_pedestrians": 0,
        "closest_relevant_ped_distance_m": np.inf,
        "strongest_relevant_ped_token": "",
        "strongest_relevant_ped_rel_long_m": np.nan,
        "strongest_relevant_ped_rel_lat_m": np.nan,
        "strongest_relevant_ped_future_path_distance_m": np.nan,
        "strongest_relevant_ped_distance_to_ego_m": np.nan,
        "strongest_relevant_ped_pred_first_conflict_t_s": np.nan,
        "strongest_relevant_ped_pred_crossing_any_flag": False,
        "ped_conflict_score_used": np.nan,
        "conflict_gate_s": np.nan,
        "osm_crossing_related": False,
        "osm_crossing_related_stop_buffer_m": cfg.ped_stop_buffer_m,
        "osm_virtual_crossing_lead_active_flag": False,
    }
    if len(runtime_df) == 0:
        return default

    ped_candidates_raw = relevant_peds_df.copy()
    n_raw = int(len(ped_candidates_raw))
    ped_candidates_planning = ped_candidates_raw.head(max(1, int(cfg.max_relevant_pedestrians_for_planning))).copy()
    ped_conflict = len(ped_candidates_planning) > 0
    ped_choice = None
    ped_gap = np.inf
    conflict_gate_s = np.nan
    used_stop_buffer = cfg.ped_stop_buffer_m
    nearest_relevant_ped_m = float(ped_candidates_raw["runtime_distance_to_ego_m"].min()) if len(ped_candidates_raw) else np.inf

    if ped_conflict:
        stop_buffer = np.where(
            ped_candidates_planning["osm_near_crossing_flag"].astype(bool).to_numpy(),
            cfg.ped_stop_buffer_m + context_weight * cfg.osm_ped_stop_buffer_extra_m,
            cfg.ped_stop_buffer_m,
        )
        pred_t = ped_candidates_planning["pred_first_conflict_t_s"].to_numpy(dtype=float)
        soon_bonus = np.where(np.isfinite(pred_t), np.clip((ped_horizon_s - pred_t) / max(ped_horizon_s, 1e-6), 0.0, 1.0), 0.0)
        stop_buffer = stop_buffer + context_weight * 0.8 * soon_bonus
        ped_candidates_planning["virtual_stop_buffer_m"] = stop_buffer
        ped_candidates_planning["virtual_stop_gap_m"] = ped_candidates_planning["runtime_rel_long_m"] - ped_candidates_planning["virtual_stop_buffer_m"]
        ped_candidates_planning = ped_candidates_planning.sort_values("virtual_stop_gap_m")
        ped_choice = ped_candidates_planning.iloc[0]
        ped_gap = max(0.15, float(ped_choice["virtual_stop_gap_m"]))
        used_stop_buffer = float(ped_choice["virtual_stop_buffer_m"])
        conflict_gate_s = float(ego_s_progress_m + float(ped_choice["runtime_rel_long_m"]) - used_stop_buffer)

    veh_candidates = runtime_df[
        runtime_df["coarse_class"].eq("vehicle")
        & runtime_df["runtime_ahead_flag"].astype(bool)
        & (np.abs(runtime_df["runtime_rel_lat_m"]) <= cfg.lead_lateral_threshold_m)
    ].copy()
    veh_choice = None
    veh_gap = np.inf
    if len(veh_candidates):
        bonus = np.where(veh_candidates["pred_conflict_any_flag"].fillna(False).astype(bool).to_numpy(), 0.7, 0.0)
        veh_candidates["vehicle_lead_score"] = veh_candidates["runtime_rel_long_m"] - bonus
        veh_candidates = veh_candidates.sort_values("vehicle_lead_score")
        veh_choice = veh_candidates.iloc[0]
        veh_gap = max(0.15, float(veh_choice["runtime_rel_long_m"]))

    if ped_conflict and ped_gap <= veh_gap:
        return {
            **default,
            "lead_type": "ped_virtual_stop",
            "lead_token": str(ped_choice["instance_token"]) if ped_choice is not None else "",
            "lead_gap_m": float(ped_gap),
            "ped_conflict_present": True,
            "num_relevant_conflict_pedestrians_raw": n_raw,
            "num_relevant_conflict_pedestrians_used": int(len(ped_candidates_planning)),
            "num_relevant_conflict_pedestrians": int(len(ped_candidates_planning)),
            "closest_relevant_ped_distance_m": nearest_relevant_ped_m,
            "strongest_relevant_ped_token": str(ped_choice["instance_token"]) if ped_choice is not None else "",
            "strongest_relevant_ped_rel_long_m": float(ped_choice["runtime_rel_long_m"]) if ped_choice is not None else np.nan,
            "strongest_relevant_ped_rel_lat_m": float(ped_choice["runtime_rel_lat_m"]) if ped_choice is not None else np.nan,
            "strongest_relevant_ped_future_path_distance_m": float(ped_choice["runtime_future_path_distance_m"]) if ped_choice is not None else np.nan,
            "strongest_relevant_ped_distance_to_ego_m": float(ped_choice["runtime_distance_to_ego_m"]) if ped_choice is not None else np.nan,
            "strongest_relevant_ped_pred_first_conflict_t_s": float(ped_choice["pred_first_conflict_t_s"]) if ped_choice is not None and np.isfinite(ped_choice["pred_first_conflict_t_s"]) else np.nan,
            "strongest_relevant_ped_pred_crossing_any_flag": bool(ped_choice["pred_crossing_any_flag"]) if ped_choice is not None else False,
            "ped_conflict_score_used": float(ped_choice["ped_conflict_score_used"]) if ped_choice is not None else np.nan,
            "conflict_gate_s": conflict_gate_s,
            "osm_crossing_related": bool(ped_choice["osm_near_crossing_flag"]) if ped_choice is not None else False,
            "osm_crossing_related_stop_buffer_m": used_stop_buffer,
        }

    if veh_choice is not None:
        return {
            **default,
            "lead_type": "vehicle",
            "lead_token": str(veh_choice["instance_token"]),
            "lead_gap_m": float(veh_gap),
            "lead_speed_mps": float(veh_choice["runtime_v_long_mps"]),
            "ped_conflict_present": ped_conflict,
            "num_relevant_conflict_pedestrians_raw": n_raw,
            "num_relevant_conflict_pedestrians_used": int(len(ped_candidates_planning)),
            "num_relevant_conflict_pedestrians": int(len(ped_candidates_planning)),
            "closest_relevant_ped_distance_m": nearest_relevant_ped_m,
        }

    if bool(osm_virtual_gate_active) and np.isfinite(osm_virtual_gate_s) and float(osm_virtual_gate_s) > float(ego_s_progress_m):
        return {
            **default,
            "lead_type": "osm_virtual_crossing_lead",
            "lead_token": "osm_crossing_prior_virtual_gate",
            "lead_gap_m": max(0.15, float(osm_virtual_gate_s) - float(ego_s_progress_m)),
            "lead_speed_mps": 0.0,
            "ped_conflict_present": ped_conflict,
            "num_relevant_conflict_pedestrians_raw": n_raw,
            "num_relevant_conflict_pedestrians_used": int(len(ped_candidates_planning)),
            "num_relevant_conflict_pedestrians": int(len(ped_candidates_planning)),
            "closest_relevant_ped_distance_m": nearest_relevant_ped_m,
            "conflict_gate_s": float(osm_virtual_gate_s),
            "osm_crossing_related": True,
            "osm_crossing_related_stop_buffer_m": float(cfg.ped_stop_buffer_m + cfg.osm_ped_stop_buffer_extra_m),
            "osm_virtual_crossing_lead_active_flag": True,
        }

    return {
        **default,
        "ped_conflict_present": ped_conflict,
        "num_relevant_conflict_pedestrians_raw": n_raw,
        "num_relevant_conflict_pedestrians_used": int(len(ped_candidates_planning)),
        "num_relevant_conflict_pedestrians": int(len(ped_candidates_planning)),
        "closest_relevant_ped_distance_m": nearest_relevant_ped_m,
    }


def idm_acceleration(v_ego: float, gap_m: float, v_lead: float, cfg: IDMConfig, v0_override_mps: Optional[float] = None) -> float:
    v = max(0.0, float(v_ego))
    s = max(0.1, float(gap_m))
    dv = v - float(v_lead)
    a = cfg.a_max_mps2
    b = max(1e-3, cfg.b_comfort_mps2)
    s_star = cfg.s0_m + max(0.0, v * cfg.T_s + (v * dv) / (2.0 * math.sqrt(a * b)))
    v0 = float(v0_override_mps) if v0_override_mps is not None else cfg.v0_mps
    free_term = 1.0 - (v / max(v0, 0.1)) ** cfg.delta
    interact_term = (s_star / s) ** 2
    return float(np.clip(a * (free_term - interact_term), -cfg.hard_brake_limit_mps2, cfg.a_max_mps2))


def step_ego(ego: EgoState, frame_index: int, timestamp_s: float, dt: float, acc_cmd: float, pref: PathReference, cfg: IDMConfig) -> EgoState:
    """Advance the longitudinal ego state without allowing non-physical backward motion.

    Earlier versions used the raw constant-acceleration displacement
    ``v*dt + 0.5*a*dt^2`` even when braking drove the speed below zero inside
    the time step. That can make ``s_progress_m`` decrease after a full stop.
    For the paper runs we treat the ego as a non-reversing longitudinal vehicle:
    if the commanded braking would stop the vehicle within the step, integrate
    only until the stopping time and then hold position for the rest of the step.
    """
    dt_eff = max(1e-3, float(dt))
    a = float(np.clip(acc_cmd, -cfg.hard_brake_limit_mps2, cfg.a_max_mps2))
    v0 = max(0.0, float(ego.speed_mps))

    if a < 0.0 and v0 + a * dt_eff <= 0.0:
        # Stop occurs inside this step. Distance traveled until stop is v^2/(2|a|).
        t_stop = v0 / max(-a, 1e-6)
        ds = v0 * t_stop + 0.5 * a * t_stop * t_stop
        v_next = 0.0
    else:
        ds = v0 * dt_eff + 0.5 * a * dt_eff * dt_eff
        v_next = float(np.clip(v0 + a * dt_eff, 0.0, cfg.max_speed_mps))

    ds = max(0.0, float(ds))
    s_next = float(np.clip(float(ego.s_progress_m) + ds, pref.s_min, pref.s_max))
    x, y, yaw = pref.interp_pose(s_next)
    return EgoState(int(frame_index), float(timestamp_s), s_next, v_next, a, x, y, yaw)


def compute_ttc_ped(relevant_peds_df: pd.DataFrame, ego_speed: float) -> float:
    if len(relevant_peds_df) == 0:
        return float("nan")
    rel_speed = ego_speed - relevant_peds_df["runtime_v_long_mps"].to_numpy(dtype=float)
    gap = relevant_peds_df["runtime_rel_long_m"].to_numpy(dtype=float)
    valid = rel_speed > 1e-3
    ttc = np.full(len(relevant_peds_df), np.nan, dtype=float)
    ttc[valid] = gap[valid] / rel_speed[valid]
    ttc = np.where((ttc > 0.0) & (gap > 0.0), ttc, np.nan)
    return float(np.nanmin(ttc)) if np.isfinite(ttc).any() else float("nan")


def estimate_ego_crossing_time_from_rollout(feat_df: pd.DataFrame, ego_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Estimate the time at which the ego reaches the current shifted OSM crossing.

    Why this exists:
    - per_frame_features.csv stores the ego state before each IDM step.
    - ego_rollout.csv stores the post-step ego state.
    - PET needs the actual time the ego enters/passes the shifted crossing.

    The most accurate estimate is therefore segment-based:
        s_before = feature-row ego_s_progress_m
        s_after  = rollout-row s_progress_m
        crossing time = timestamp_s + alpha * delta_t_s

    This keeps PET tied to the shifted OSM crossing while avoiding missed ego
    crossing events caused by using only pre-step ego states.
    """
    empty = {
        "ego_crossing_time_s": float("nan"),
        "ego_crossing_frame_index": -1,
        "ego_crossing_detection_source": "not_found",
        "ego_crossing_segment_fraction": float("nan"),
        "ego_crossing_s_m": float("nan"),
        "ego_crossing_zone_flag_count": 0,
    }
    if feat_df is None or len(feat_df) == 0:
        return empty

    d = feat_df.copy().sort_values("timestamp_s").reset_index(drop=True)
    if "perturbed_crossing_s_m" not in d.columns:
        return empty

    crossing_vals = pd.to_numeric(d["perturbed_crossing_s_m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not crossing_vals.notna().any():
        return empty
    crossing_s = float(np.nanmedian(crossing_vals.to_numpy(dtype=float)))

    # Preferred method: pre-step feature state + post-step rollout state.
    if (
        ego_df is not None
        and len(ego_df) > 0
        and "frame_index" in d.columns
        and "frame_index" in ego_df.columns
        and "ego_s_progress_m" in d.columns
        and "s_progress_m" in ego_df.columns
        and "timestamp_s" in d.columns
    ):
        pre_cols = ["frame_index", "timestamp_s", "ego_s_progress_m"]
        pre = d[pre_cols].copy()
        post_cols = ["frame_index", "s_progress_m"]
        if "delta_t_s" in ego_df.columns:
            post_cols.append("delta_t_s")
        if "timestamp_s" in ego_df.columns:
            post_cols.append("timestamp_s")
        post = ego_df[post_cols].copy()
        merged = pre.merge(post, on="frame_index", how="inner", suffixes=("_pre", "_post"))
        if len(merged):
            merged = merged.sort_values("timestamp_s_pre" if "timestamp_s_pre" in merged.columns else "timestamp_s").reset_index(drop=True)
            for _, row in merged.iterrows():
                s0 = finite_or_nan(row.get("ego_s_progress_m", np.nan))
                s1 = finite_or_nan(row.get("s_progress_m", np.nan))
                if not (np.isfinite(s0) and np.isfinite(s1)):
                    continue
                t0 = finite_or_nan(row.get("timestamp_s_pre", row.get("timestamp_s", np.nan)))
                if not np.isfinite(t0):
                    t0 = finite_or_nan(row.get("timestamp_s_post", np.nan))
                dt = finite_or_nan(row.get("delta_t_s", np.nan))
                if not np.isfinite(dt) or dt < 0.0:
                    dt = 0.0
                frame_index = int(row.get("frame_index", -1)) if pd.notna(row.get("frame_index", np.nan)) else -1

                # Already at/past the crossing before the step.
                if s0 >= crossing_s:
                    out = dict(empty)
                    out.update({
                        "ego_crossing_time_s": float(t0),
                        "ego_crossing_frame_index": frame_index,
                        "ego_crossing_detection_source": "pre_step_already_at_or_past_shifted_crossing",
                        "ego_crossing_segment_fraction": 0.0,
                        "ego_crossing_s_m": crossing_s,
                    })
                    return out

                # Crossing occurs during this IDM step.
                if s0 < crossing_s <= s1:
                    if abs(s1 - s0) < EPS:
                        frac = 0.0
                    else:
                        frac = float(np.clip((crossing_s - s0) / (s1 - s0), 0.0, 1.0))
                    out = dict(empty)
                    out.update({
                        "ego_crossing_time_s": float(t0 + frac * dt),
                        "ego_crossing_frame_index": frame_index,
                        "ego_crossing_detection_source": "interpolated_pre_to_post_step_shifted_crossing",
                        "ego_crossing_segment_fraction": frac,
                        "ego_crossing_s_m": crossing_s,
                    })
                    return out

    # Fallback 1: use the feature-row in-zone flag.
    if "ego_in_osm_crossing_zone_flag" in d.columns and "timestamp_s" in d.columns:
        ego_zone = d["ego_in_osm_crossing_zone_flag"].fillna(False).astype(bool).to_numpy()
        zone_count = int(ego_zone.sum())
        if ego_zone.any():
            idx = int(np.argmax(ego_zone))
            out = dict(empty)
            out.update({
                "ego_crossing_time_s": float(pd.to_numeric(d.loc[idx, "timestamp_s"], errors="coerce")),
                "ego_crossing_frame_index": int(d.loc[idx, "frame_index"]) if "frame_index" in d.columns and pd.notna(d.loc[idx, "frame_index"]) else idx,
                "ego_crossing_detection_source": "feature_ego_zone_flag",
                "ego_crossing_segment_fraction": 0.0,
                "ego_crossing_s_m": crossing_s,
                "ego_crossing_zone_flag_count": zone_count,
            })
            return out
        empty["ego_crossing_zone_flag_count"] = zone_count

    # Fallback 2: use rollout post-step state only.
    if ego_df is not None and len(ego_df) and "s_progress_m" in ego_df.columns and "timestamp_s" in ego_df.columns:
        e = ego_df.copy().sort_values("timestamp_s").reset_index(drop=True)
        es = pd.to_numeric(e["s_progress_m"], errors="coerce").to_numpy(dtype=float)
        et = pd.to_numeric(e["timestamp_s"], errors="coerce").to_numpy(dtype=float)
        passed = np.isfinite(es) & np.isfinite(et) & (es >= crossing_s)
        if passed.any():
            idx = int(np.argmax(passed))
            out = dict(empty)
            out.update({
                "ego_crossing_time_s": float(et[idx]),
                "ego_crossing_frame_index": int(e.loc[idx, "frame_index"]) if "frame_index" in e.columns and pd.notna(e.loc[idx, "frame_index"]) else idx,
                "ego_crossing_detection_source": "post_step_rollout_first_passed_shifted_crossing",
                "ego_crossing_segment_fraction": float("nan"),
                "ego_crossing_s_m": crossing_s,
            })
            return out

    empty["ego_crossing_s_m"] = crossing_s
    return empty


def compute_pet_metrics(feat_df: pd.DataFrame, ego_df: pd.DataFrame, cfg: IDMConfig) -> Dict[str, Any]:
    """
    Compute PET for the current shifted OSM crossing case.

    Final evaluation logic:
    - The PET zone remains the corresponding shifted/perturbed OSM crossing case.
    - Pedestrian occupancy prefers the full/refined true pedestrian trajectory.
    - Ego crossing time is computed from the ego rollout segment, not only the
      pre-step feature state, so we do not miss crossing events after an IDM step.
    """
    empty = {
        "post_encroachment_time_s": float("nan"),
        "pet_available_flag": False,
        "pet_critical_flag": False,
        "ego_crossing_time_s": float("nan"),
        "last_ped_zone_time_before_ego_s": float("nan"),
        "first_ped_zone_time_after_ego_s": float("nan"),
        "pet_pedestrian_source": "none",
        "pet_uses_true_ped_path_flag": False,
        "num_frames_true_ped_in_osm_zone": 0,
        "num_frames_degraded_ped_in_osm_zone": 0,
        "ego_crossing_frame_index": -1,
        "ego_crossing_detection_source": "not_found",
        "ego_crossing_segment_fraction": float("nan"),
        "ego_crossing_s_m": float("nan"),
        "ego_crossing_zone_flag_count": 0,
    }
    if feat_df is None or len(feat_df) == 0 or "timestamp_s" not in feat_df.columns:
        return empty

    d = feat_df.copy().sort_values("timestamp_s").reset_index(drop=True)
    t = pd.to_numeric(d["timestamp_s"], errors="coerce").to_numpy(dtype=float)

    degraded_ped_zone = d.get("any_ped_in_osm_crossing_zone_flag", pd.Series(False, index=d.index)).fillna(False).astype(bool).to_numpy()
    true_ped_col = "any_true_ped_in_osm_crossing_zone_flag"
    if true_ped_col in d.columns:
        true_ped_zone = d[true_ped_col].fillna(False).astype(bool).to_numpy()
        ped_zone = true_ped_zone
        ped_source = "true_refined_pedestrian_path_shifted_osm_zone"
        uses_true = True
    else:
        true_ped_zone = np.zeros(len(d), dtype=bool)
        ped_zone = degraded_ped_zone
        ped_source = "degraded_camera_detections_shifted_osm_zone"
        uses_true = False

    ego_crossing = estimate_ego_crossing_time_from_rollout(d, ego_df)
    ego_time = finite_or_nan(ego_crossing.get("ego_crossing_time_s", np.nan))

    if not np.isfinite(ego_time):
        out = dict(empty)
        out.update({
            "pet_pedestrian_source": ped_source,
            "pet_uses_true_ped_path_flag": uses_true,
            "num_frames_true_ped_in_osm_zone": int(true_ped_zone.sum()),
            "num_frames_degraded_ped_in_osm_zone": int(degraded_ped_zone.sum()),
            **ego_crossing,
        })
        return out

    # Identify the closest feature row at/just before the ego crossing time.
    finite_t = np.isfinite(t)
    if finite_t.any():
        before_or_at = np.where(finite_t & (t <= ego_time))[0]
        if len(before_or_at):
            ego_idx = int(before_or_at[-1])
        else:
            ego_idx = int(np.nanargmin(np.abs(t - ego_time)))
    else:
        ego_idx = -1

    ego_overlaps_ped_zone = bool(0 <= ego_idx < len(ped_zone) and ped_zone[ego_idx])
    if ego_overlaps_ped_zone:
        pet = 0.0
        last_before = ego_time
        first_after = ego_time
    else:
        before = t[(t < ego_time) & ped_zone & np.isfinite(t)]
        after = t[(t > ego_time) & ped_zone & np.isfinite(t)]
        last_before = float(np.max(before)) if len(before) else float("nan")
        first_after = float(np.min(after)) if len(after) else float("nan")
        candidates = []
        if np.isfinite(last_before):
            candidates.append(ego_time - last_before)
        if np.isfinite(first_after):
            candidates.append(first_after - ego_time)
        pet = float(min(candidates)) if candidates else float("nan")

    return {
        "post_encroachment_time_s": pet,
        "pet_available_flag": bool(np.isfinite(pet)),
        "pet_critical_flag": bool(np.isfinite(pet) and pet < cfg.critical_pet_threshold_s),
        "ego_crossing_time_s": ego_time,
        "last_ped_zone_time_before_ego_s": last_before,
        "first_ped_zone_time_after_ego_s": first_after,
        "pet_pedestrian_source": ped_source,
        "pet_uses_true_ped_path_flag": uses_true,
        "num_frames_true_ped_in_osm_zone": int(true_ped_zone.sum()),
        "num_frames_degraded_ped_in_osm_zone": int(degraded_ped_zone.sum()),
        **ego_crossing,
    }

def compute_run_metrics(ego_df: pd.DataFrame, feat_df: pd.DataFrame, cfg: IDMConfig) -> Dict[str, Any]:
    min_ped = safe_min(feat_df["nearest_ped_m"], default=float("inf"))
    min_veh = safe_min(feat_df["nearest_vehicle_m"], default=float("inf"))
    min_dyn = safe_min(feat_df["nearest_dynamic_m"], default=float("inf"))
    min_rel_ped = safe_min(feat_df["nearest_relevant_ped_m"], default=float("inf"))
    min_active_gap = safe_min(feat_df["active_lead_gap_m"], default=float("inf"))

    unsafe_ped_flag = feat_df["unsafe_ped_flag"].fillna(False).astype(bool)
    unsafe_veh_flag = feat_df["unsafe_vehicle_flag"].fillna(False).astype(bool)
    unsafe_any_flag = feat_df["unsafe_any_flag"].fillna(False).astype(bool)
    critical_ttc_flag = feat_df["critical_ped_ttc_flag"].fillna(False).astype(bool)
    hard_brake_flag = feat_df["hard_brake_flag"].fillna(False).astype(bool)
    ped_conflict_flag = feat_df["ped_conflict_present"].fillna(False).astype(bool)
    ped_violation_flag = feat_df["ped_conflict_zone_violation"].fillna(False).astype(bool)

    dt = ego_df["delta_t_s"].fillna(0.0).to_numpy(dtype=float) if len(ego_df) else np.array([], dtype=float)
    speed = ego_df["speed_mps"].to_numpy(dtype=float) if len(ego_df) else np.array([], dtype=float)
    acc = ego_df["accel_mps2"].to_numpy(dtype=float) if len(ego_df) else np.array([], dtype=float)
    travel = float(np.sum(dt)) if len(dt) else 0.0
    stop_time = float(np.sum(dt[speed <= cfg.stopped_speed_threshold_mps])) if len(speed) and len(dt) == len(speed) else 0.0
    avg_speed = float(np.mean(speed)) if len(speed) else 0.0
    max_braking = float(np.max(np.maximum(0.0, -acc))) if len(acc) else 0.0
    braking_magnitudes = np.maximum(0.0, -acc) if len(acc) else np.array([], dtype=float)
    mean_braking_during_brake = float(np.mean(braking_magnitudes[acc < 0.0])) if len(acc) and np.any(acc < 0.0) else 0.0
    jerk = np.diff(acc) / np.maximum(dt[1:], 1e-3) if len(acc) > 1 and len(dt) == len(acc) else np.array([0.0])
    comfort = float(np.sqrt(np.mean(jerk * jerk))) if len(jerk) else 0.0

    pet = compute_pet_metrics(feat_df, ego_df, cfg)

    out = {
        "min_ped_distance_m": min_ped,
        "min_vehicle_distance_m": min_veh,
        "min_dynamic_distance_m": min_dyn,
        "min_relevant_ped_distance_m": min_rel_ped,
        "min_active_constraint_distance_m": min_active_gap,
        "unsafe_ped_frames": int(unsafe_ped_flag.sum()),
        "unsafe_vehicle_frames": int(unsafe_veh_flag.sum()),
        "unsafe_frames_total": int(unsafe_any_flag.sum()),
        "unsafe_ped_episodes": count_true_episodes(unsafe_ped_flag),
        "unsafe_vehicle_episodes": count_true_episodes(unsafe_veh_flag),
        "unsafe_total_episodes": count_true_episodes(unsafe_any_flag),
        "min_ped_ttc_s": safe_min(feat_df["min_ttc_ped_s"], default=float("nan")),
        "critical_ped_ttc_frames": int(critical_ttc_flag.sum()),
        "critical_ped_ttc_episodes": count_true_episodes(critical_ttc_flag),
        "travel_time_s": travel,
        "stop_time_s": stop_time,
        "average_speed_mps": avg_speed,
        "max_braking_mps2": max_braking,
        "mean_braking_mps2_during_brake": mean_braking_during_brake,
        "hard_brake_frames": int(hard_brake_flag.sum()),
        "hard_brake_episodes": count_true_episodes(hard_brake_flag),
        "comfort_jerk_rms_mps3": comfort,
        "active_ped_conflict_frames": int(ped_conflict_flag.sum()),
        "active_ped_conflict_episodes": count_true_episodes(ped_conflict_flag),
        "ped_conflict_zone_violation_frames": int(ped_violation_flag.sum()),
        "ped_conflict_zone_violation_episodes": count_true_episodes(ped_violation_flag),
        "mean_osm_speed_limit_v0_mps_used": float(feat_df["osm_speed_limit_v0_mps_used"].mean()) if len(feat_df) else cfg.v0_mps,
        "mean_osm_context_weight": float(feat_df["osm_context_weight_used"].mean()) if "osm_context_weight_used" in feat_df.columns and len(feat_df) else 0.0,
        "mean_predicted_conflict_pedestrians": float(feat_df["num_predicted_conflict_pedestrians"].mean()) if "num_predicted_conflict_pedestrians" in feat_df.columns and len(feat_df) else 0.0,
        "mean_predicted_crossing_pedestrians": float(feat_df["num_predicted_crossing_pedestrians"].mean()) if "num_predicted_crossing_pedestrians" in feat_df.columns and len(feat_df) else 0.0,
        "mean_adaptive_ped_horizon_s": float(feat_df["adaptive_ped_prediction_horizon_s"].mean()) if len(feat_df) else float("nan"),
        "min_adaptive_ped_horizon_s": float(feat_df["adaptive_ped_prediction_horizon_s"].min()) if len(feat_df) else float("nan"),
        "max_adaptive_ped_horizon_s": float(feat_df["adaptive_ped_prediction_horizon_s"].max()) if len(feat_df) else float("nan"),
        "mean_adaptive_vehicle_horizon_s": float(feat_df["adaptive_vehicle_prediction_horizon_s"].mean()) if len(feat_df) else float("nan"),
        "max_adaptive_vehicle_horizon_s": float(feat_df["adaptive_vehicle_prediction_horizon_s"].max()) if len(feat_df) else float("nan"),
        "mean_crossing_proximity_factor": float(feat_df["crossing_proximity_factor"].mean()) if len(feat_df) else 0.0,
    }
    out.update(pet)
    return out


def pedestrian_zone_counts(run_df: pd.DataFrame, crossing_prior: CrossingPrior, cfg: IDMConfig) -> Tuple[int, bool]:
    """Count pedestrians inside the current shifted OSM crossing zone."""
    if len(run_df) == 0 or "coarse_class" not in run_df.columns:
        return 0, False
    cls = run_df["coarse_class"].astype(str).str.lower()
    ped = run_df[cls.eq("pedestrian") | cls.str.contains("pedestrian|human|person", regex=True, na=False)].copy()
    if len(ped) == 0:
        return 0, False
    if "obj_x" not in ped.columns or "obj_y" not in ped.columns:
        return 0, False
    px = pd.to_numeric(ped["obj_x"], errors="coerce").to_numpy(dtype=float)
    py = pd.to_numeric(ped["obj_y"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(px) & np.isfinite(py)
    if not valid.any():
        return 0, False
    d = np.sqrt((px[valid] - crossing_prior.center_x) ** 2 + (py[valid] - crossing_prior.center_y) ** 2)
    radius = max(cfg.pet_crossing_zone_radius_m, 0.5 * crossing_prior.radius_m)
    count = int(np.sum(d <= radius))
    return count, bool(count > 0)


def run_single(
    sample_df: pd.DataFrame,
    ego_ref_df: pd.DataFrame,
    objects_deg_df: pd.DataFrame,
    objects_true_df: pd.DataFrame,
    osm_case: OSMErrorCase,
    degradation_mode: str,
    level: int,
    seed: int,
    out_dir: Path,
    cfg: IDMConfig,
    crossing_prior: CrossingPrior,
    degradation_audit: Optional[Dict[str, Any]],
    learned_horizon_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pref = PathReference(ego_ref_df)
    ego = init_ego(sample_df, ego_ref_df, pref)
    dt_candidates = sample_df["delta_t_s"][sample_df["delta_t_s"] > 1e-4]
    dt_fallback = float(dt_candidates.median()) if len(dt_candidates) else 0.5
    obj_vel_df = compute_object_velocity(objects_deg_df, sample_df)
    crossing_priors = [crossing_prior]
    actual_camera_drop_fraction = float("nan")
    if degradation_audit:
        for _k in ("actual_camera_drop_fraction", "mean_frame_camera_drop_fraction"):
            if _k in degradation_audit:
                try:
                    actual_camera_drop_fraction = float(degradation_audit[_k])
                    break
                except Exception:
                    pass

    ego_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    feat_rows: List[Dict[str, Any]] = []
    constraint_rows: List[Dict[str, Any]] = []

    for _, sr in sample_df.iterrows():
        frame = int(sr["frame_index"])
        ts = float(sr["timestamp_s"])
        dt = float(sr["delta_t_s"]) if float(sr["delta_t_s"]) > 1e-4 else dt_fallback

        crossing_ctx = nearest_crossing_for_ego(crossing_priors, ego.s_progress_m, ego.x, ego.y, cfg)
        ds = crossing_ctx["distance_along_path_m"]
        route_gap_to_crossing_m = float(ds) if np.isfinite(ds) else float("nan")
        horizon_info = compute_adaptive_horizons(
            level_pct=level,
            normalized_osm_error=osm_case.normalized_osm_error,
            gap_to_crossing_m=route_gap_to_crossing_m,
            cfg=cfg,
            degradation_mode=degradation_mode,
            actual_camera_drop_fraction=actual_camera_drop_fraction,
            learned_policy=learned_horizon_policy,
        )
        ped_horizon_s = horizon_info["adaptive_ped_prediction_horizon_s"]
        veh_horizon_s = horizon_info["adaptive_vehicle_prediction_horizon_s"]
        route_based_proximity = float(horizon_info["crossing_proximity_factor"])
        context_w = osm_context_weight(level, osm_case.normalized_osm_error, route_based_proximity)

        is_burst_dropout = 1.0 if str(degradation_mode).strip().lower() == "burst_dropout" else 0.0
        if degradation_audit and is_burst_dropout > 0.0:
            b0 = degradation_audit.get("burst_start_frame", None)
            b1 = degradation_audit.get("burst_end_frame", None)
            if b0 is not None and b1 is not None:
                try:
                    is_burst_dropout = 1.0 if int(b0) <= frame <= int(b1) else 0.0
                except Exception:
                    pass
        combined_unc = combined_uncertainty_score(
            degradation_fraction=float(horizon_info["degradation_fraction"]),
            normalized_osm_error=float(osm_case.normalized_osm_error),
            actual_camera_drop_fraction=float(horizon_info["actual_camera_drop_fraction_used"]),
            is_burst_dropout=float(is_burst_dropout),
            route_based_crossing_proximity_factor=route_based_proximity,
        )
        osm_virtual_gate_active = False
        osm_virtual_gate_s = float(crossing_prior.s_crossing_m - (cfg.ped_stop_buffer_m + cfg.osm_ped_stop_buffer_extra_m))

        crossing_ctx["caution_active"] = bool(
            context_w > 0.0
            and np.isfinite(ds)
            and route_based_proximity > 0.0
            and ((-cfg.osm_crossing_past_buffer_m <= ds <= cfg.osm_crossing_approach_distance_m) or bool(crossing_ctx["crossing_core"]))
        )
        v0_used, caution_active = map_aware_v0(cfg, crossing_ctx, context_w, route_based_proximity)

        frame_df = obj_vel_df[obj_vel_df["frame_index"] == frame].copy()
        run_df = compute_runtime_features(frame_df, ego, pref, cfg, crossing_priors)
        run_df = predict_agent_conflicts(run_df, ego, pref, cfg, crossing_priors, ped_horizon_s, veh_horizon_s)
        visible_relevant_ped_count = int(len(run_df[run_df["coarse_class"].eq("pedestrian")])) if len(run_df) and "coarse_class" in run_df.columns else 0
        expected_ped_missing_flag = bool(route_based_proximity > 0.20 and visible_relevant_ped_count <= 0)
        osm_virtual_gate_active = bool(
            route_based_proximity > 0.20
            and combined_unc > 0.25
            and expected_ped_missing_flag
            and np.isfinite(route_gap_to_crossing_m)
            and route_gap_to_crossing_m >= 0.0
            and route_gap_to_crossing_m <= cfg.osm_crossing_approach_distance_m
        )
        relevant_peds_df = get_relevant_conflict_pedestrians(run_df, cfg, crossing_ctx, context_w)
        constraint = select_active_constraint(
            run_df,
            relevant_peds_df,
            ego.s_progress_m,
            cfg,
            context_w,
            ped_horizon_s,
            osm_virtual_gate_active=osm_virtual_gate_active,
            osm_virtual_gate_s=osm_virtual_gate_s,
        )

        lead_gap = 1e6 if constraint["lead_type"] == "none" else float(constraint["lead_gap_m"])
        lead_speed = v0_used if constraint["lead_type"] == "none" else float(constraint["lead_speed_mps"])
        acc_cmd = idm_acceleration(ego.speed_mps, lead_gap, lead_speed, cfg, v0_override_mps=v0_used)
        next_ego = step_ego(ego, frame, ts, dt, acc_cmd, pref, cfg)

        min_ttc_ped = compute_ttc_ped(relevant_peds_df, ego.speed_mps)
        if len(run_df) and "coarse_class" in run_df.columns:
            dyn_mask = run_df["coarse_class"].isin(["pedestrian", "vehicle"])
            nearest_dyn = float(run_df[dyn_mask]["runtime_distance_to_ego_m"].min()) if dyn_mask.any() else float("inf")
            nearest_ped = float(run_df[run_df["coarse_class"] == "pedestrian"]["runtime_distance_to_ego_m"].min()) if int((run_df["coarse_class"] == "pedestrian").sum()) else float("inf")
            nearest_veh = float(run_df[run_df["coarse_class"] == "vehicle"]["runtime_distance_to_ego_m"].min()) if int((run_df["coarse_class"] == "vehicle").sum()) else float("inf")
            num_dyn = int(dyn_mask.sum())
            num_predicted_conflict_peds = int(run_df[(run_df["coarse_class"] == "pedestrian") & run_df["pred_conflict_any_flag"].fillna(False).astype(bool)].shape[0])
            num_predicted_crossing_peds = int(run_df[(run_df["coarse_class"] == "pedestrian") & run_df["pred_crossing_any_flag"].fillna(False).astype(bool)].shape[0])
            num_peds_in_zone, any_ped_in_zone = pedestrian_zone_counts(run_df, crossing_prior, cfg)
        else:
            nearest_dyn = nearest_ped = nearest_veh = float("inf")
            num_dyn = 0
            num_predicted_conflict_peds = 0
            num_predicted_crossing_peds = 0
            num_peds_in_zone = 0
            any_ped_in_zone = False

        # PET evaluation uses the true/refined pedestrian path at the same shifted OSM crossing.
        # This keeps PET independent from camera dropout while preserving the OSM-error case.
        if objects_true_df is not None and len(objects_true_df):
            true_frame_df = objects_true_df[objects_true_df["frame_index"] == frame].copy()
            num_true_peds_in_zone, any_true_ped_in_zone = pedestrian_zone_counts(true_frame_df, crossing_prior, cfg)
        else:
            # Should normally not happen because main() loads the true object table.
            # Fallback preserves backwards compatibility but PET will again depend on degraded detections.
            num_true_peds_in_zone, any_true_ped_in_zone = num_peds_in_zone, any_ped_in_zone

        nearest_relevant_ped = float(relevant_peds_df["runtime_distance_to_ego_m"].min()) if len(relevant_peds_df) else float("inf")
        conflict_gate_s = float(constraint["conflict_gate_s"]) if np.isfinite(constraint["conflict_gate_s"]) else float("nan")
        ego_has_passed_conflict_gate = bool(np.isfinite(conflict_gate_s) and ego.s_progress_m >= conflict_gate_s)
        ped_violation = bool(ego_has_passed_conflict_gate and len(relevant_peds_df) > 0)
        critical_ttc_flag = bool(np.isfinite(min_ttc_ped) and min_ttc_ped < cfg.critical_ped_ttc_threshold_s)
        unsafe_ped_flag = bool(np.isfinite(nearest_ped) and nearest_ped < cfg.unsafe_ped_distance_m)
        unsafe_vehicle_flag = bool(np.isfinite(nearest_veh) and nearest_veh < cfg.unsafe_veh_distance_m)
        unsafe_any_flag = bool(unsafe_ped_flag or unsafe_vehicle_flag)
        hard_brake_flag = bool(next_ego.accel_mps2 <= -(cfg.hard_brake_limit_mps2 - cfg.hard_brake_epsilon_mps2))
        ego_in_crossing_zone = bool(abs(ego.s_progress_m - crossing_prior.s_crossing_m) <= cfg.ego_crossing_zone_s_window_m)

        common_ids = {
            "osm_error_case": osm_case.osm_error_case,
            "offset_direction": osm_case.offset_direction,
            "offset_m": osm_case.offset_m,
            "injected_offset_signed_m": osm_case.injected_offset_signed_m,
            "injected_offset_abs_m": osm_case.injected_offset_abs_m,
            "osm_error_signed_m": osm_case.osm_error_signed_m,
            "osm_error_abs_m": osm_case.osm_error_abs_m,
            "normalized_osm_error": osm_case.normalized_osm_error,
            "degradation_mode": degradation_mode,
            "degradation_level_pct": int(level),
            "seed": int(seed),
        }

        feat_rows.append({
            **common_ids,
            "frame_index": frame,
            "timestamp_s": ts,
            "ego_s_progress_m": float(ego.s_progress_m),
            "num_objects": int(len(run_df)),
            "num_dynamic_objects": num_dyn,
            "nearest_ped_m": nearest_ped,
            "nearest_vehicle_m": nearest_veh,
            "nearest_dynamic_m": nearest_dyn,
            "nearest_relevant_ped_m": nearest_relevant_ped,
            "num_predicted_conflict_pedestrians": num_predicted_conflict_peds,
            "num_predicted_crossing_pedestrians": num_predicted_crossing_peds,
            "active_lead_type": constraint["lead_type"],
            "active_lead_token": constraint["lead_token"],
            "active_lead_gap_m": float(constraint["lead_gap_m"]),
            "active_lead_speed_mps": float(constraint["lead_speed_mps"]),
            "ped_conflict_present": bool(constraint["ped_conflict_present"]),
            "num_relevant_conflict_pedestrians_raw": int(constraint["num_relevant_conflict_pedestrians_raw"]),
            "num_relevant_conflict_pedestrians_used": int(constraint["num_relevant_conflict_pedestrians_used"]),
            "num_relevant_conflict_pedestrians": int(constraint["num_relevant_conflict_pedestrians"]),
            "min_ttc_ped_s": min_ttc_ped,
            "critical_ped_ttc_flag": critical_ttc_flag,
            "conflict_gate_s": conflict_gate_s,
            "ego_has_passed_conflict_gate": ego_has_passed_conflict_gate,
            "ped_conflict_zone_violation": ped_violation,
            "hard_brake_flag": hard_brake_flag,
            "unsafe_ped_flag": unsafe_ped_flag,
            "unsafe_vehicle_flag": unsafe_vehicle_flag,
            "unsafe_any_flag": unsafe_any_flag,
            "perturbed_crossing_s_m": float(osm_case.perturbed_crossing_s_m),
            "perturbed_crossing_x": float(osm_case.perturbed_crossing_x),
            "perturbed_crossing_y": float(osm_case.perturbed_crossing_y),
            "osm_crossing_ahead_flag": bool(crossing_ctx["crossing_ahead"]),
            "osm_crossing_distance_along_path_m": float(ds) if np.isfinite(ds) else np.nan,
            "osm_crossing_core_flag": bool(crossing_ctx["crossing_core"]),
            "osm_nearest_crossing_id": str(crossing_ctx["nearest_crossing_id"]),
            "osm_crossing_has_signal": bool(crossing_ctx["nearest_has_signal"]),
            "osm_speed_limit_v0_mps_used": float(v0_used),
            "osm_caution_active_flag": bool(caution_active),
            "osm_context_weight_used": float(context_w),
            "adaptive_ped_prediction_horizon_s": float(ped_horizon_s),
            "adaptive_vehicle_prediction_horizon_s": float(veh_horizon_s),
            "crossing_proximity_factor": route_based_proximity,
            "route_gap_to_osm_crossing_m": route_gap_to_crossing_m,
            "route_based_crossing_proximity_factor": route_based_proximity,
            "degradation_fraction": float(horizon_info["degradation_fraction"]),
            "actual_camera_drop_fraction_used": float(horizon_info.get("actual_camera_drop_fraction_used", np.nan)),
            "visible_relevant_ped_count": int(visible_relevant_ped_count),
            "expected_ped_missing_flag": bool(expected_ped_missing_flag),
            "combined_uncertainty_score": float(combined_unc),
            "osm_gate_active_flag": bool(osm_virtual_gate_active),
            "osm_virtual_crossing_lead_active_flag": bool(constraint.get("osm_virtual_crossing_lead_active_flag", False)),
            "adaptive_ped_horizon_rule_fallback_s": float(horizon_info.get("adaptive_ped_horizon_rule_fallback_s", np.nan)),
            "adaptive_vehicle_horizon_rule_fallback_s": float(horizon_info.get("adaptive_vehicle_horizon_rule_fallback_s", np.nan)),
            "adaptive_horizon_policy_type": str(horizon_info.get("adaptive_horizon_policy_type", "rule_based_bounded_linear")),
            "learned_horizon_policy_used_flag": bool(horizon_info.get("learned_horizon_policy_used_flag", False)),
            "learned_policy_latent_score_ped": float(horizon_info.get("learned_policy_latent_score_ped", np.nan)),
            "learned_policy_latent_score_vehicle": float(horizon_info.get("learned_policy_latent_score_vehicle", np.nan)),
            # Degraded-perception pedestrian occupancy: useful diagnostic, not used for final PET if true path is available.
            "num_peds_in_osm_crossing_zone": int(num_peds_in_zone),
            "any_ped_in_osm_crossing_zone_flag": bool(any_ped_in_zone),
            # True/refined pedestrian occupancy at the same shifted OSM crossing: used for PET evaluation.
            "num_true_peds_in_osm_crossing_zone": int(num_true_peds_in_zone),
            "any_true_ped_in_osm_crossing_zone_flag": bool(any_true_ped_in_zone),
            "pet_uses_true_ped_path_flag": bool(objects_true_df is not None and len(objects_true_df) > 0),
            "ego_in_osm_crossing_zone_flag": bool(ego_in_crossing_zone),
        })

        action_rows.append({
            **common_ids,
            "frame_index": frame,
            "timestamp_s": ts,
            "idm_accel_cmd_mps2": acc_cmd,
            "lead_type": constraint["lead_type"],
            "lead_gap_m": float(constraint["lead_gap_m"]),
            "lead_speed_mps": float(constraint["lead_speed_mps"]),
            "osm_speed_limit_v0_mps_used": float(v0_used),
            "osm_caution_active_flag": bool(caution_active),
            "osm_context_weight_used": float(context_w),
            "adaptive_ped_prediction_horizon_s": float(ped_horizon_s),
            "adaptive_vehicle_prediction_horizon_s": float(veh_horizon_s),
            "adaptive_horizon_policy_type": str(horizon_info.get("adaptive_horizon_policy_type", "rule_based_bounded_linear")),
            "learned_horizon_policy_used_flag": bool(horizon_info.get("learned_horizon_policy_used_flag", False)),
        })

        constraint_rows.append({
            **common_ids,
            "frame_index": frame,
            "timestamp_s": ts,
            "lead_type": constraint["lead_type"],
            "lead_token": constraint["lead_token"],
            "lead_gap_m": float(constraint["lead_gap_m"]),
            "ped_conflict_present": bool(constraint["ped_conflict_present"]),
            "num_relevant_conflict_pedestrians_raw": int(constraint["num_relevant_conflict_pedestrians_raw"]),
            "num_relevant_conflict_pedestrians_used": int(constraint["num_relevant_conflict_pedestrians_used"]),
            "conflict_gate_s": float(constraint["conflict_gate_s"]) if np.isfinite(constraint["conflict_gate_s"]) else np.nan,
            "osm_crossing_related": bool(constraint["osm_crossing_related"]),
            "osm_crossing_related_stop_buffer_m": float(constraint["osm_crossing_related_stop_buffer_m"]),
            "osm_context_weight_used": float(context_w),
            "adaptive_ped_prediction_horizon_s": float(ped_horizon_s),
            "adaptive_vehicle_prediction_horizon_s": float(veh_horizon_s),
            "adaptive_horizon_policy_type": str(horizon_info.get("adaptive_horizon_policy_type", "rule_based_bounded_linear")),
            "learned_horizon_policy_used_flag": bool(horizon_info.get("learned_horizon_policy_used_flag", False)),
        })

        ego_rows.append({
            **common_ids,
            "frame_index": frame,
            "timestamp_s": ts,
            "delta_t_s": dt,
            "s_progress_m": next_ego.s_progress_m,
            "speed_mps": next_ego.speed_mps,
            "accel_mps2": next_ego.accel_mps2,
            "ego_x": next_ego.x,
            "ego_y": next_ego.y,
            "ego_yaw_rad": next_ego.yaw_rad,
            "adaptive_ped_prediction_horizon_s": float(ped_horizon_s),
            "adaptive_vehicle_prediction_horizon_s": float(veh_horizon_s),
            "adaptive_horizon_policy_type": str(horizon_info.get("adaptive_horizon_policy_type", "rule_based_bounded_linear")),
            "learned_horizon_policy_used_flag": bool(horizon_info.get("learned_horizon_policy_used_flag", False)),
        })

        ego = next_ego
        if ego.s_progress_m >= pref.s_max - 1e-3:
            break

    ego_roll = pd.DataFrame(ego_rows)
    actions = pd.DataFrame(action_rows)
    feats = pd.DataFrame(feat_rows)
    constraints = pd.DataFrame(constraint_rows)

    metrics = compute_run_metrics(ego_roll, feats, cfg)
    metrics.update({
        "osm_error_case": osm_case.osm_error_case,
        "offset_direction": osm_case.offset_direction,
        "offset_m": osm_case.offset_m,
        "injected_offset_signed_m": osm_case.injected_offset_signed_m,
        "injected_offset_abs_m": osm_case.injected_offset_abs_m,
        "osm_error_signed_m": osm_case.osm_error_signed_m,
        "osm_error_abs_m": osm_case.osm_error_abs_m,
        "normalized_osm_error": osm_case.normalized_osm_error,
        "perturbed_crossing_s_m": osm_case.perturbed_crossing_s_m,
        "perturbed_crossing_x": osm_case.perturbed_crossing_x,
        "perturbed_crossing_y": osm_case.perturbed_crossing_y,
        "crossing_length_m": osm_case.crossing_length_m,
        "crossing_width_m": osm_case.crossing_width_m,
        "clean_osm_feature_id": osm_case.clean_osm_feature_id,
        "clean_osm_feature_type": osm_case.clean_osm_feature_type,
        "clean_osm_crossing_s_m": osm_case.clean_osm_crossing_s_m,
        "clean_osm_distance_to_reference_xy_m": osm_case.clean_osm_distance_to_reference_xy_m,
        "clean_osm_distance_to_route_m": osm_case.clean_osm_distance_to_route_m,
        "degradation_mode": degradation_mode,
        "degradation_level_pct": int(level),
        "seed": int(seed),
    })

    if degradation_audit:
        for k in (
            "actual_camera_drop_fraction",
            "camera_eligible_rows_before",
            "camera_eligible_rows_after",
            "frames_with_any_camera_eligible_rows_before",
            "frames_with_all_camera_eligible_rows_lost",
            "mean_frame_camera_drop_fraction",
            "mean_drop_probability_across_frames",
            "crossing_start_frame",
            "crossing_end_frame",
            "burst_start_frame",
            "burst_end_frame",
        ):
            if k in degradation_audit:
                metrics[k] = degradation_audit[k]

    ego_roll.to_csv(out_dir / "ego_rollout.csv", index=False)
    actions.to_csv(out_dir / "planner_actions.csv", index=False)
    feats.to_csv(out_dir / "per_frame_features.csv", index=False)
    constraints.to_csv(out_dir / "active_constraints.csv", index=False)
    (out_dir / "safety_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    return metrics


def aggregate_metrics(per_run_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "min_ped_distance_m",
        "min_relevant_ped_distance_m",
        "min_vehicle_distance_m",
        "min_dynamic_distance_m",
        "min_active_constraint_distance_m",
        "unsafe_ped_frames",
        "unsafe_vehicle_frames",
        "unsafe_frames_total",
        "unsafe_ped_episodes",
        "unsafe_vehicle_episodes",
        "unsafe_total_episodes",
        "min_ped_ttc_s",
        "critical_ped_ttc_frames",
        "critical_ped_ttc_episodes",
        "post_encroachment_time_s",
        "pet_available_flag",
        "pet_critical_flag",
        "pet_uses_true_ped_path_flag",
        "num_frames_true_ped_in_osm_zone",
        "num_frames_degraded_ped_in_osm_zone",
        "ped_conflict_zone_violation_frames",
        "ped_conflict_zone_violation_episodes",
        "travel_time_s",
        "stop_time_s",
        "average_speed_mps",
        "max_braking_mps2",
        "mean_braking_mps2_during_brake",
        "hard_brake_frames",
        "hard_brake_episodes",
        "comfort_jerk_rms_mps3",
        "active_ped_conflict_frames",
        "active_ped_conflict_episodes",
        "mean_osm_context_weight",
        "mean_osm_speed_limit_v0_mps_used",
        "mean_predicted_conflict_pedestrians",
        "mean_predicted_crossing_pedestrians",
        "mean_adaptive_ped_horizon_s",
        "min_adaptive_ped_horizon_s",
        "max_adaptive_ped_horizon_s",
        "mean_adaptive_vehicle_horizon_s",
        "max_adaptive_vehicle_horizon_s",
        "mean_crossing_proximity_factor",
        "actual_camera_drop_fraction",
        "frames_with_all_camera_eligible_rows_lost",
        "mean_frame_camera_drop_fraction",
        "mean_drop_probability_across_frames",
    ]
    metrics = [m for m in metrics if m in per_run_df.columns]
    group_cols = ["osm_error_case", "offset_direction", "osm_error_abs_m", "normalized_osm_error", "degradation_mode", "degradation_level_pct"]
    agg = per_run_df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "min", "max"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]).rstrip("_") for col in agg.columns]
    return agg


def plot_mean_std_for_case_and_mode(agg_case_mode_df: pd.DataFrame, metric_prefix: str, ylabel: str, title: str, out_path: Path, color: str) -> None:
    y_mean_col = f"{metric_prefix}_mean"
    y_std_col = f"{metric_prefix}_std"
    if y_mean_col not in agg_case_mode_df.columns:
        return
    x = agg_case_mode_df["degradation_level_pct"].to_numpy(dtype=float)
    y = agg_case_mode_df[y_mean_col].to_numpy(dtype=float)
    ys = agg_case_mode_df[y_std_col].fillna(0.0).to_numpy(dtype=float) if y_std_col in agg_case_mode_df.columns else np.zeros_like(y)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    ys = ys[order]
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o", color=color, label="Mean")
    ax.fill_between(x, y - ys, y + ys, color=color, alpha=0.2, label="Mean ± std")
    ax.set_xlabel("Camera degradation (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_case_mode_bundle(agg_df: pd.DataFrame, scene_root: Path, osm_error_case: str, mode: str) -> None:
    case_mode_df = agg_df[(agg_df["osm_error_case"] == osm_error_case) & (agg_df["degradation_mode"] == mode)].copy()
    if len(case_mode_df) == 0:
        return
    plot_dir = scene_root / "plots" / osm_error_case
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_mean_std_for_case_and_mode(case_mode_df, "min_ped_distance_m", "Distance (m)", f"Min pedestrian distance vs camera degradation ({osm_error_case}, {mode})", plot_dir / f"{mode}_min_ped_distance_vs_degradation.png", "#dc2626")
    plot_mean_std_for_case_and_mode(case_mode_df, "unsafe_frames_total", "Frame count", f"Unsafe frames total vs camera degradation ({osm_error_case}, {mode})", plot_dir / f"{mode}_unsafe_frames_total_vs_degradation.png", "#7c3aed")
    plot_mean_std_for_case_and_mode(case_mode_df, "min_ped_ttc_s", "Time (s)", f"Minimum pedestrian TTC vs camera degradation ({osm_error_case}, {mode})", plot_dir / f"{mode}_min_ttc_ped_vs_degradation.png", "#2563eb")
    plot_mean_std_for_case_and_mode(case_mode_df, "post_encroachment_time_s", "Time (s)", f"PET proxy vs camera degradation ({osm_error_case}, {mode})", plot_dir / f"{mode}_pet_vs_degradation.png", "#16a34a")
    plot_mean_std_for_case_and_mode(case_mode_df, "comfort_jerk_rms_mps3", "Jerk RMS (m/s³)", f"Comfort jerk vs camera degradation ({osm_error_case}, {mode})", plot_dir / f"{mode}_comfort_jerk_vs_degradation.png", "#ea580c")
    plot_mean_std_for_case_and_mode(case_mode_df, "mean_adaptive_ped_horizon_s", "Horizon (s)", f"Adaptive pedestrian horizon ({osm_error_case}, {mode})", plot_dir / f"{mode}_adaptive_ped_horizon_vs_degradation.png", "#0891b2")
    plot_mean_std_for_case_and_mode(case_mode_df, "travel_time_s", "Time (s)", f"Travel time vs camera degradation ({osm_error_case}, {mode})", plot_dir / f"{mode}_travel_time_vs_degradation.png", "#6d28d9")


DEFAULT_BATCH_SCENES = ["scene-1019", "scene-0061", "scene-0383"]


# ---------------------------------------------------------------------------
# Geometry-specific planner profiles
# ---------------------------------------------------------------------------
# The method remains one unified OSM-aware IDM planner. These profiles only tune
# interpretable sensitivity gates for each geometry so the same algorithm is not
# accidentally biased toward the original moderate-curvature case.
GEOMETRY_PLANNER_PROFILES: Dict[str, Dict[str, Any]] = {
    "scene-1019": {
        "geometry_case": "straight_approach",
        "rationale": "Straight approach: earlier and stronger crossing-prior activation is appropriate because the ego path is simpler and the crossing should be easier to anticipate.",
        "overrides": {
            "adaptive_horizon_proximity_distance_m": 58.0,
            "osm_crossing_approach_distance_m": 62.0,
            "osm_crossing_core_distance_m": 15.0,
            "osm_ped_relevance_distance_m": 15.0,
            "osm_ped_relevance_bonus": 12.0,
            "osm_ped_stop_buffer_extra_m": 2.3,
            "osm_crossing_rel_long_bonus_m": 4.0,
            "predicted_path_conflict_threshold_m": 2.6,
            "predicted_crossing_zone_threshold_m": 3.2,
            "predicted_conflict_bonus": 12.0,
            "predicted_crossing_bonus": 9.0,
            "predicted_conflict_time_bonus": 9.0,
            "prediction_longitudinal_activation_m": 36.0,
            "ped_conflict_longitudinal_max_m": 28.0,
            "ped_conflict_lateral_max_m": 3.3,
            "ped_conflict_future_path_max_m": 3.2,
            "max_relevant_pedestrians_for_planning": 2,
            "ped_horizon_base_s": 2.25,
            "ped_horizon_degradation_gain_s": 1.15,
            "ped_horizon_osm_error_gain_s": 1.70,
            "ped_horizon_proximity_gain_s": 1.25,
        },
    },
    "scene-0061": {
        "geometry_case": "moderate_curvature_approach",
        "rationale": "Moderate curvature: keep the original strong setting because it is the reference case where OSM context already activates correctly.",
        "overrides": {
            "adaptive_horizon_proximity_distance_m": 42.0,
            "osm_crossing_approach_distance_m": 42.0,
            "osm_crossing_core_distance_m": 11.0,
            "osm_ped_relevance_distance_m": 10.0,
            "osm_ped_relevance_bonus": 7.0,
            "osm_ped_stop_buffer_extra_m": 1.5,
            "predicted_path_conflict_threshold_m": 2.2,
            "predicted_crossing_zone_threshold_m": 2.7,
            "predicted_conflict_bonus": 9.0,
            "predicted_crossing_bonus": 6.5,
            "ped_horizon_base_s": 2.05,
            "ped_horizon_degradation_gain_s": 1.00,
            "ped_horizon_osm_error_gain_s": 1.50,
            "ped_horizon_proximity_gain_s": 1.05,
        },
    },
    "scene-0383": {
        "geometry_case": "high_curvature_circular_approach",
        "rationale": "High curvature: rely on route-progress crossing proximity and uncertainty-gated OSM caution so missing camera pedestrians can still trigger crossing-aware slowing near the conflict zone.",
        "overrides": {
            "adaptive_horizon_proximity_distance_m": 30.0,
            "osm_crossing_approach_distance_m": 33.0,
            "osm_crossing_core_distance_m": 9.0,
            "osm_caution_speed_mps": 3.4,
            "osm_low_speed_floor_mps": 2.0,
            "osm_ped_relevance_distance_m": 7.0,
            "osm_ped_relevance_bonus": 3.0,
            "osm_ped_stop_buffer_extra_m": 1.5,
            "osm_crossing_rel_long_bonus_m": 1.0,
            "predicted_path_conflict_threshold_m": 2.4,
            "predicted_crossing_zone_threshold_m": 3.0,
            "predicted_conflict_bonus": 5.0,
            "predicted_crossing_bonus": 3.0,
            "predicted_conflict_time_bonus": 4.5,
            "prediction_longitudinal_activation_m": 24.0,
            "ped_conflict_longitudinal_max_m": 20.0,
            "ped_conflict_lateral_max_m": 2.7,
            "ped_conflict_future_path_max_m": 2.4,
            "hard_brake_limit_mps2": 4.0,
            "max_relevant_pedestrians_for_planning": 1,
            "ped_horizon_min_s": 1.1,
            "ped_horizon_base_s": 2.1,
            "ped_horizon_max_s": 5.0,
            "ped_horizon_degradation_gain_s": 1.2,
            "ped_horizon_osm_error_gain_s": 1.2,
            "ped_horizon_proximity_gain_s": 1.4,
        },
    },
}


def apply_geometry_planner_profile(scene_name: str, cfg: IDMConfig, disabled: bool = False) -> Dict[str, Any]:
    """Apply interpretable scene-geometry sensitivity settings to the unified planner."""
    scene = normalize_scene_name_token(scene_name)
    if disabled:
        return {"scene_name": scene, "profile_used": False, "reason": "disabled_by_cli", "overrides": {}}
    profile = GEOMETRY_PLANNER_PROFILES.get(scene)
    if not profile:
        return {"scene_name": scene, "profile_used": False, "reason": "no_profile_for_scene", "overrides": {}}
    applied: Dict[str, Any] = {}
    for key, value in dict(profile.get("overrides", {})).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            applied[key] = value
    return {
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
    """Resolve batch scenes while keeping backward compatibility with --scene-name."""
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
    p = argparse.ArgumentParser(description="Run OSM-error-aware adaptive-horizon IDM planner on degraded camera inputs.")
    p.add_argument(
        "--scene-name",
        default="all",
        help="Scene to run, short scene number, comma-separated scenes, or 'all'. Default: all current project scenes.",
    )
    p.add_argument(
        "--scene-names",
        default=",".join(DEFAULT_BATCH_SCENES),
        help="Comma-separated scene list used when --scene-name all. Default: scene-1019,scene-0061,scene-0383.",
    )
    p.add_argument("--refined-root", default="data/refined")
    p.add_argument("--idm-calibration-root", default="data/idm_calibration")
    p.add_argument("--perception-results-root", default="data/planner_perception_only")
    p.add_argument("--osm-root", default="data/osm")
    p.add_argument("--results-root", default="data/planner_osm_aware")
    p.add_argument(
        "--true-objects-file",
        default="",
        help=(
            "Optional full/refined object trajectory CSV used only for PET evaluation. "
            "Can be absolute or relative to data/refined/<scene>. If omitted, the script scans common names. "
            "For batch runs, leave this empty unless the same relative file name exists for every scene."
        ),
    )
    p.add_argument("--degradation-modes", default="random_dropout,burst_dropout,distance_based_dropout")
    p.add_argument("--degradation-levels", default="10,30,50,70,90")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--osm-error-cases", default="", help="Optional comma-separated subset of osm_error_case names. Default runs all cases from Script 08.")
    p.add_argument(
        "--horizon-policy-root",
        default="data/learned_horizon_policy",
        help="Root folder containing learned adaptive_horizon_policy.json files.",
    )
    p.add_argument(
        "--horizon-policy-path",
        default="",
        help="Optional explicit path to a learned adaptive horizon policy JSON. Overrides --horizon-policy-root. Use only for a single-scene run.",
    )
    p.add_argument(
        "--disable-learned-horizon-policy",
        action="store_true",
        help="Force the original rule-based adaptive horizon even if a learned policy file exists.",
    )
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="For batch runs, skip scenes/runs with missing inputs instead of stopping immediately.",
    )
    p.add_argument(
        "--disable-geometry-profiles",
        action="store_true",
        help="Disable the three-scene geometry sensitivity profiles and use the raw shared planner settings only.",
    )
    p.add_argument(
        "--print-geometry-profile",
        action="store_true",
        help="Print the applied geometry profile overrides for each scene.",
    )
    return p.parse_args()


def run_scene(
    scene: str,
    args: argparse.Namespace,
    modes: List[str],
    levels: List[int],
    seeds: List[int],
    requested_cases: Optional[List[str]],
) -> Dict[str, Any]:
    scene = normalize_scene_name_token(scene)
    print("\nosm_aware_planner_scene_start")
    print(f"  scene: {scene}")

    cfg, idm_calibration_payload = load_calibrated_idm_config(scene, Path(args.idm_calibration_root))
    geometry_profile_info = apply_geometry_planner_profile(
        scene, cfg, disabled=bool(getattr(args, "disable_geometry_profiles", False))
    )
    if bool(getattr(args, "print_geometry_profile", False)) or geometry_profile_info.get("profile_used", False):
        print(f"[geometry-profile] {scene}: {geometry_profile_info.get('geometry_case', geometry_profile_info.get('reason', 'none'))}")
        if bool(getattr(args, "print_geometry_profile", False)):
            print(json.dumps(geometry_profile_info, indent=2))

    learned_horizon_policy: Optional[Dict[str, Any]] = None
    horizon_policy_path: Optional[Path] = None
    if not bool(args.disable_learned_horizon_policy):
        if str(args.horizon_policy_path).strip():
            horizon_policy_path = Path(args.horizon_policy_path)
        else:
            horizon_policy_path = Path(args.horizon_policy_root) / scene / "adaptive_horizon_policy.json"
        learned_horizon_policy = load_learned_horizon_policy(horizon_policy_path)
        if learned_horizon_policy is not None:
            print(f"[ok] learned adaptive horizon policy loaded from: {horizon_policy_path}")
        else:
            print("[info] no learned adaptive horizon policy found; using rule-based fallback horizon policy")
    else:
        print("[info] learned adaptive horizon policy disabled; using rule-based fallback horizon policy")

    refined_scene_dir = Path(args.refined_root) / scene
    result_scene_dir = Path(args.results_root) / scene
    degraded_root = Path(args.perception_results_root) / scene / "degraded_inputs"
    osm_scene_dir = Path(args.osm_root) / scene
    runs_root = result_scene_dir / "runs"
    priors_root = result_scene_dir / "derived_osm_priors"
    runs_root.mkdir(parents=True, exist_ok=True)
    result_scene_dir.mkdir(parents=True, exist_ok=True)

    required_inputs = [
        refined_scene_dir / "sample_timeline.csv",
        refined_scene_dir / "ego_timeline.csv",
        osm_scene_dir / "converted" / "osm_features_local.csv",
        osm_scene_dir / "error_cases" / "osm_error_cases.csv",
    ]
    missing_inputs = [str(x) for x in required_inputs if not x.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required inputs for {scene}: {missing_inputs}")

    sample_df = pd.read_csv(refined_scene_dir / "sample_timeline.csv").sort_values("frame_index").reset_index(drop=True)
    ego_ref_df = pd.read_csv(refined_scene_dir / "ego_timeline.csv").sort_values("frame_index").reset_index(drop=True)

    # Full/refined object trajectories are used only for PET evaluation.
    # The planner itself still consumes objects_timeline_degraded.csv.
    objects_true_df, true_objects_path = load_true_objects_for_pet(refined_scene_dir, requested_file=args.true_objects_file)
    print(f"[ok] true pedestrian trajectory for PET loaded from: {true_objects_path}")

    osm_local_path = osm_scene_dir / "converted" / "osm_features_local.csv"
    if not osm_local_path.is_file():
        raise FileNotFoundError(f"Missing OSM local features: {osm_local_path}. Run Script 07 first.")
    osm_local_df = pd.read_csv(osm_local_path)

    osm_cases = load_osm_error_cases(osm_scene_dir, requested_cases=requested_cases)
    degradation_formula_summary = load_optional_json(degraded_root / "degradation_formula_summary.json")

    prior_rows: List[Dict[str, Any]] = []
    case_to_prior: Dict[str, CrossingPrior] = {}
    for case in osm_cases:
        prior = build_crossing_prior_from_error_case(case, osm_local_df, cfg)
        case_to_prior[case.osm_error_case] = prior
        prior_rows.append({**asdict(prior), **asdict(case)})
    save_crossing_priors(prior_rows, priors_root)

    metrics_rows: List[Dict[str, Any]] = []
    for osm_case in osm_cases:
        crossing_prior = case_to_prior[osm_case.osm_error_case]
        for mode in modes:
            for level in levels:
                for seed in seeds:
                    deg_run_dir = degraded_root / mode / f"degradation_{level:02d}" / f"seed_{seed:03d}"
                    deg_path = deg_run_dir / "objects_timeline_degraded.csv"
                    if not deg_path.is_file():
                        raise FileNotFoundError(f"Missing degraded input: {deg_path}. Run Script 04 first for {scene}.")
                    degradation_audit = load_optional_json(deg_run_dir / "degradation_audit.json")
                    objects_deg_df = pd.read_csv(deg_path)
                    run_dir = runs_root / osm_case.osm_error_case / mode / f"degradation_{level:02d}" / f"seed_{seed:03d}"
                    metrics = run_single(
                        sample_df=sample_df,
                        ego_ref_df=ego_ref_df,
                        objects_deg_df=objects_deg_df,
                        objects_true_df=objects_true_df,
                        osm_case=osm_case,
                        degradation_mode=mode,
                        level=level,
                        seed=seed,
                        out_dir=run_dir,
                        cfg=cfg,
                        crossing_prior=crossing_prior,
                        degradation_audit=degradation_audit,
                        learned_horizon_policy=learned_horizon_policy,
                    )
                    metrics["scene_name"] = scene
                    metrics_rows.append(metrics)
                    print(f"[ok] scene={scene} osm_error_case={osm_case.osm_error_case} mode={mode} level={level:02d}% seed={seed:03d}")

    per_run = pd.DataFrame(metrics_rows).sort_values(["osm_error_case", "degradation_mode", "degradation_level_pct", "seed"]).reset_index(drop=True)
    per_run.to_csv(result_scene_dir / "per_run_metrics.csv", index=False)

    agg = aggregate_metrics(per_run)
    agg.to_csv(result_scene_dir / "aggregated_metrics.csv", index=False)

    payload = {
        "scene_name": scene,
        "planner_type": "osm_error_adaptive_horizon_idm",
        "planner_notes": [
            "Uses degraded camera detections from Script 04 as the dynamic perception input.",
            "Loads shared calibrated IDM parameters from Script 05 when available.",
            "Uses OSM crossing positional-error cases from Script 08 instead of reliability scores.",
            "Computes adaptive pedestrian and vehicle prediction horizons per frame from camera degradation, normalized OSM error, and crossing proximity.",
            "If a learned policy JSON is available, horizon selection uses the learned bounded policy; otherwise the script falls back to the original rule-based bounded linear policy.",
            "Uses lightweight constant-velocity prediction for surrounding pedestrians and vehicles.",
            "Perception remains the dynamic source of truth; OSM contributes static crossing priors with explicit positional error.",
            "PET is evaluated at each shifted OSM crossing case using the full/refined true pedestrian trajectory, not degraded camera detections.",
        ],
        "geometry_profile": geometry_profile_info,
        "adaptive_horizon_model": {
            "policy_family": "learned_bounded_logistic" if learned_horizon_policy is not None else "rule_based_bounded_linear",
            "learned_policy_loaded": bool(learned_horizon_policy is not None),
            "learned_policy_path": str(horizon_policy_path) if horizon_policy_path is not None else "",
            "concept": "uncertainty-conditioned adaptive prediction horizon",
            "rule_based_fallback_pedestrian": "clip(H_base + alpha*degradation_fraction + beta*normalized_osm_error + gamma*crossing_proximity, H_min, H_max)",
            "rule_based_fallback_vehicle": "clip(H_base + alpha*degradation_fraction + beta*normalized_osm_error + gamma*crossing_proximity, H_min, H_max)",
            "learned_policy_payload": learned_horizon_policy,
            "parameters": {
                "ped_horizon_min_s": cfg.ped_horizon_min_s,
                "ped_horizon_base_s": cfg.ped_horizon_base_s,
                "ped_horizon_max_s": cfg.ped_horizon_max_s,
                "ped_horizon_degradation_gain_s": cfg.ped_horizon_degradation_gain_s,
                "ped_horizon_osm_error_gain_s": cfg.ped_horizon_osm_error_gain_s,
                "ped_horizon_proximity_gain_s": cfg.ped_horizon_proximity_gain_s,
                "vehicle_horizon_min_s": cfg.vehicle_horizon_min_s,
                "vehicle_horizon_base_s": cfg.vehicle_horizon_base_s,
                "vehicle_horizon_max_s": cfg.vehicle_horizon_max_s,
                "vehicle_horizon_degradation_gain_s": cfg.vehicle_horizon_degradation_gain_s,
                "vehicle_horizon_osm_error_gain_s": cfg.vehicle_horizon_osm_error_gain_s,
                "vehicle_horizon_proximity_gain_s": cfg.vehicle_horizon_proximity_gain_s,
                "adaptive_horizon_proximity_distance_m": cfg.adaptive_horizon_proximity_distance_m,
            },
        },
        "idm_parameters": asdict(cfg),
        "idm_calibration_payload": idm_calibration_payload,
        "num_osm_error_cases": int(len(osm_cases)),
        "osm_error_cases": [asdict(c) for c in osm_cases],
        "degradation_modes": modes,
        "degradation_levels": levels,
        "seeds": seeds,
        "degradation_formula_summary": degradation_formula_summary,
        "true_objects_file_for_pet": str(true_objects_path),
        "derived_osm_priors_file": str(priors_root / "derived_osm_error_priors.csv"),
        "per_run_metrics": per_run.to_dict(orient="records"),
        "aggregated_metrics": agg.to_dict(orient="records"),
    }
    (result_scene_dir / "aggregated_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    for osm_case in osm_cases:
        for mode in modes:
            plot_case_mode_bundle(agg, result_scene_dir, osm_case.osm_error_case, mode)

    print("planner_summary:")
    print(f"  scene = {scene}")
    print("  planner = osm_error_adaptive_horizon_idm")
    print("  input = degraded camera-perception object table + explicit OSM crossing positional-error cases")
    print(f"  shared_idm_params_loaded = {idm_calibration_payload is not None}")
    print("  prediction = lightweight constant-velocity prediction for pedestrians and vehicles")
    print("  horizon_model = adaptive per-frame horizon based on degradation, normalized OSM error, and crossing proximity")
    print(f"  learned_horizon_policy_loaded = {learned_horizon_policy is not None}")
    print(f"  num_osm_error_cases = {len(osm_cases)}")
    print(f"[done] OSM-error-aware adaptive-horizon IDM planner outputs written to {result_scene_dir}")

    return {
        "scene_name": scene,
        "status": "completed",
        "result_scene_dir": str(result_scene_dir),
        "num_runs": int(len(per_run)),
        "num_osm_error_cases": int(len(osm_cases)),
        "learned_horizon_policy_loaded": bool(learned_horizon_policy is not None),
        "horizon_policy_path": str(horizon_policy_path) if horizon_policy_path is not None else "",
        "idm_calibration_loaded": bool(idm_calibration_payload is not None),
    }


def main() -> None:
    apply_plot_style()
    args = parse_args()
    modes = parse_str_list_csv(args.degradation_modes)
    levels = parse_int_list_csv(args.degradation_levels)
    seeds = parse_int_list_csv(args.seeds)
    requested_cases = parse_str_list_csv(args.osm_error_cases) if str(args.osm_error_cases).strip() else None
    scenes = parse_scene_name_list(args.scene_name, args.scene_names)

    bad_modes = [m for m in modes if m not in SUPPORTED_MODES]
    if bad_modes:
        raise ValueError(f"Unsupported degradation modes: {bad_modes}. Supported: {sorted(SUPPORTED_MODES)}")
    if len(scenes) > 1 and str(args.horizon_policy_path).strip():
        raise ValueError("--horizon-policy-path is only safe for a single scene. For batch runs, use --horizon-policy-root.")

    print("batch_osm_aware_planner_start")
    print(f"  scenes: {scenes}")
    print(f"  modes: {modes}")
    print(f"  levels: {levels}")
    print(f"  seeds: {seeds}")
    print(f"  results_root: {args.results_root}")
    print(f"  learned_policy_disabled: {bool(args.disable_learned_horizon_policy)}")

    scene_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for scene in scenes:
        try:
            scene_summaries.append(run_scene(scene, args, modes, levels, seeds, requested_cases))
        except Exception as exc:
            failures.append({"scene_name": str(scene), "error": str(exc)})
            print(f"[failed] {scene}: {exc}")
            if not args.skip_missing:
                raise

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    combined_frames: List[pd.DataFrame] = []
    for summary in scene_summaries:
        scene = str(summary["scene_name"])
        per_run_path = results_root / scene / "per_run_metrics.csv"
        if per_run_path.is_file():
            df = pd.read_csv(per_run_path)
            if "scene_name" not in df.columns:
                df["scene_name"] = scene
            combined_frames.append(df)
    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined.to_csv(results_root / "all_scenes_osm_aware_per_run_metrics.csv", index=False)
    else:
        combined = pd.DataFrame()

    batch_payload = {
        "planner_type": "osm_error_adaptive_horizon_idm_batch_three_scenes",
        "scenes_requested": scenes,
        "scenes_completed": [s["scene_name"] for s in scene_summaries],
        "failures": failures,
        "degradation_modes": modes,
        "degradation_levels": levels,
        "seeds": seeds,
        "results_root": str(results_root),
        "learned_horizon_policy_disabled": bool(args.disable_learned_horizon_policy),
        "per_scene_outputs_preserved": True,
        "combined_per_run_metrics_csv": str(results_root / "all_scenes_osm_aware_per_run_metrics.csv"),
        "scene_summaries": scene_summaries,
    }
    (results_root / "all_scenes_osm_aware_batch_summary.json").write_text(json.dumps(batch_payload, indent=2, allow_nan=True), encoding="utf-8")

    print("\nbatch_osm_aware_planner_done")
    print(f"  completed: {[s['scene_name'] for s in scene_summaries]}")
    print(f"  failures: {failures}")
    print(f"  batch_summary: {results_root / 'all_scenes_osm_aware_batch_summary.json'}")
    if failures and not args.skip_missing:
        raise RuntimeError(f"OSM-aware planner failed for scenes: {[f['scene_name'] for f in failures]}")


if __name__ == "__main__":
    main()
