#!/usr/bin/env python3
"""
Calibrate scene-1019 IDM parameters from the refined straight-approach scene package.

Purpose
-------
This script estimates one scene-conditioned IDM parameter set from the refined scene so that both:
- perception-only IDM
- OSM-aware IDM

can use the same base IDM parameters later.

Inputs (default root: data/refined/<scene_name>/)
-------------------------------------------------
- sample_timeline.csv
- ego_timeline.csv
- objects_timeline.csv
- scene_summary.json
- optional: planner_observations.csv

Outputs (default root: data/idm_calibration/<scene_name>/)
----------------------------------------------------------
- idm_parameters.json
- idm_parameter_calibration_summary.json
- idm_fit_trace.csv
- idm_fit_rollout.csv
- idm_fit_comparison.csv
- plots/observed_vs_fitted_speed.png
- plots/observed_vs_fitted_progress.png
- plots/observed_vs_fitted_acceleration.png

Method
------
1) Build generic scene-statistic initial guesses from the refined scenario.
2) Simulate a lightweight active-constraint IDM rollout (virtual crossing gate vs closer vehicle lead).
3) Run two-stage bounded local random search around realistic parameter ranges.
4) Save the best shared IDM parameter set and diagnostics.

Notes
-----
- This is a scene-conditioned calibration, not a claim of learning a universal human driver model.
- Scene-1019 is a synthetically refined straight-approach case, so the objective emphasizes the approach, stop, and realistic restart while down-weighting any forced terminal catch-up tail.
- We calibrate the main IDM parameters:
    v0_mps, s0_m, T_s, a_max_mps2, b_comfort_mps2
- We keep delta fixed at 4.0 by default.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------

def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (9.2, 5.8),
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


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class IDMParams:
    v0_mps: float
    s0_m: float
    T_s: float
    a_max_mps2: float
    b_comfort_mps2: float
    delta: float = 4.0


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
class Bounds:
    v0_mps: Tuple[float, float] = (3.0, 10.5)
    s0_m: Tuple[float, float] = (1.0, 8.0)
    T_s: Tuple[float, float] = (0.8, 3.2)
    a_max_mps2: Tuple[float, float] = (0.35, 2.2)
    b_comfort_mps2: Tuple[float, float] = (0.6, 3.8)
    delta: Tuple[float, float] = (4.0, 4.0)


@dataclass
class CalibrationConfig:
    scene_name: str
    refined_root: Path
    output_root: Path
    delta_fixed: float
    num_random_samples: int
    seed: int
    hold_buffer_extra_m: float
    near_stop_speed_threshold_mps: float
    post_crossing_buffer_frames: int
    pre_crossing_buffer_frames: int
    objective_weight_speed: float
    objective_weight_progress: float
    objective_weight_accel: float
    objective_weight_stop_timing: float
    objective_weight_restart_timing: float
    objective_weight_transition_speed_shape: float
    objective_accel_smoothing_window: int
    objective_accel_residual_clip_mps2: float
    prefer_scene_summary_crossing: bool = True


# ---------------------------------------------------------------------
# Path interpolation
# ---------------------------------------------------------------------

class PathReference:
    def __init__(self, ego_df: pd.DataFrame) -> None:
        df = ego_df.sort_values("frame_index").reset_index(drop=True)
        self.s = df["cumulative_distance_m"].to_numpy(dtype=float)
        self.x = df["ego_x"].to_numpy(dtype=float)
        self.y = df["ego_y"].to_numpy(dtype=float)
        self.yaw = np.unwrap(df["ego_yaw_rad"].to_numpy(dtype=float))
        if len(self.s) < 2:
            raise ValueError("ego_timeline.csv must contain at least 2 rows.")
        self.s_min = float(self.s[0])
        self.s_max = float(self.s[-1])

    def interp_pose(self, s_query: float) -> Tuple[float, float, float]:
        s_q = float(np.clip(s_query, self.s_min, self.s_max))
        x = float(np.interp(s_q, self.s, self.x))
        y = float(np.interp(s_q, self.s, self.y))
        yaw = float(np.interp(s_q, self.s, self.yaw))
        return x, y, yaw


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate scene-1019 IDM parameters from the refined straight-approach scene.")
    p.add_argument("--scene-name", default="scene-1019")
    p.add_argument("--refined-root", default="data/refined")
    p.add_argument("--output-root", default="data/idm_calibration")
    p.add_argument("--delta-fixed", type=float, default=4.0)
    p.add_argument("--num-random-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hold-buffer-extra-m", type=float, default=3.0)
    p.add_argument("--near-stop-speed-threshold-mps", type=float, default=0.25)
    p.add_argument("--post-crossing-buffer-frames", type=int, default=2)
    p.add_argument("--pre-crossing-buffer-frames", type=int, default=2)
    p.add_argument("--objective-weight-speed", type=float, default=1.15)
    p.add_argument("--objective-weight-progress", type=float, default=0.35)
    p.add_argument("--objective-weight-accel", type=float, default=0.25)
    p.add_argument("--objective-weight-stop-timing", type=float, default=0.65)
    p.add_argument("--objective-weight-restart-timing", type=float, default=0.55)
    p.add_argument("--objective-weight-transition-speed-shape", type=float, default=0.65)
    p.add_argument("--objective-accel-smoothing-window", type=int, default=5)
    p.add_argument("--objective-accel-residual-clip-mps2", type=float, default=3.5)
    return p.parse_args()


def safe_percentile(arr: np.ndarray, q: float, default: float) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(default)
    return float(np.percentile(arr, q))


def safe_median(arr: np.ndarray, default: float) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(default)
    return float(np.median(arr))


def safe_mean(arr: np.ndarray, default: float) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(default)
    return float(np.mean(arr))


def clip_to_bounds(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if int(window) <= 1 or len(arr) == 0:
        return arr.copy()
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    smoothed = pd.Series(arr).rolling(window=w, min_periods=1, center=True).median()
    return smoothed.to_numpy(dtype=float)


def weighted_rmse(obs: np.ndarray, sim: np.ndarray, weights: np.ndarray, residual_clip: Optional[float] = None) -> float:
    obs_arr = np.asarray(obs, dtype=float)
    sim_arr = np.asarray(sim, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    valid = np.isfinite(obs_arr) & np.isfinite(sim_arr) & np.isfinite(w_arr) & (w_arr > 0.0)
    if not np.any(valid):
        return float("inf")
    residual = obs_arr[valid] - sim_arr[valid]
    if residual_clip is not None and float(residual_clip) > 0.0:
        residual = np.clip(residual, -float(residual_clip), float(residual_clip))
    wv = w_arr[valid]
    denom = float(np.sum(wv))
    if denom <= 1e-8:
        return float("inf")
    return float(np.sqrt(np.sum(wv * residual * residual) / denom))


def load_refined_scene(scene_dir: Path) -> Dict[str, Any]:
    sample_df = pd.read_csv(scene_dir / "sample_timeline.csv").sort_values("frame_index").reset_index(drop=True)
    ego_df = pd.read_csv(scene_dir / "ego_timeline.csv").sort_values("frame_index").reset_index(drop=True)
    objects_df = pd.read_csv(scene_dir / "objects_timeline.csv")
    scene_summary = json.loads((scene_dir / "scene_summary.json").read_text(encoding="utf-8"))
    planner_obs_path = scene_dir / "planner_observations.csv"
    planner_obs_df = pd.read_csv(planner_obs_path) if planner_obs_path.is_file() else None
    return {
        "sample_df": sample_df,
        "ego_df": ego_df,
        "objects_df": objects_df,
        "scene_summary": scene_summary,
        "planner_obs_df": planner_obs_df,
    }


# ---------------------------------------------------------------------
# Virtual gate / crossing logic
# ---------------------------------------------------------------------

def infer_conflict_gate_s(
    scene_summary: Dict[str, Any],
    ego_df: pd.DataFrame,
    cfg: CalibrationConfig,
) -> Dict[str, Any]:
    s_vals = ego_df["cumulative_distance_m"].to_numpy(dtype=float)
    n = len(ego_df)

    crossing_start = scene_summary.get("crossing_start_frame")
    crossing_end = scene_summary.get("crossing_end_frame")
    hold_arc = scene_summary.get("ego_hold_line_arc_length_m")
    crossing_arc = scene_summary.get("crossing_zone_arc_length_m")

    crossing_start_frame = int(crossing_start) if crossing_start is not None else max(1, int(round(0.45 * (n - 1))))
    crossing_end_frame = int(crossing_end) if crossing_end is not None else min(n - 2, crossing_start_frame + 6)

    if hold_arc is not None and np.isfinite(float(hold_arc)):
        gate_s = float(hold_arc)
        gate_source = "scene_summary.ego_hold_line_arc_length_m"
    elif crossing_arc is not None and np.isfinite(float(crossing_arc)):
        gate_s = float(crossing_arc) - float(cfg.hold_buffer_extra_m)
        gate_source = "scene_summary.crossing_zone_arc_length_m_minus_buffer"
    else:
        gate_s = float(np.interp(crossing_start_frame, np.arange(n, dtype=float), s_vals)) - float(cfg.hold_buffer_extra_m)
        gate_source = "crossing_start_frame_progress_minus_buffer"

    ped_constraint_start_frame = max(0, crossing_start_frame - max(2, int(cfg.pre_crossing_buffer_frames) + 1))
    gate_s = float(np.clip(gate_s, float(s_vals[0]), float(s_vals[-1]) - 0.5))
    return {
        "gate_s_progress_m": gate_s,
        "gate_source": gate_source,
        "ped_constraint_start_frame": int(ped_constraint_start_frame),
        "crossing_start_frame": crossing_start_frame,
        "crossing_end_frame": crossing_end_frame,
    }


def build_phase_masks(
    ego_df: pd.DataFrame,
    crossing_start_frame: int,
    crossing_end_frame: int,
    cfg: CalibrationConfig,
) -> Dict[str, np.ndarray]:
    n = len(ego_df)
    idx = np.arange(n, dtype=int)

    pre_cross_start = max(0, crossing_start_frame - max(1, cfg.pre_crossing_buffer_frames) - 6)
    pre_cross_end = max(0, crossing_start_frame - 1)
    post_cross_start = min(n - 1, crossing_end_frame + max(1, cfg.post_crossing_buffer_frames))
    post_cross_end = n - 1

    pre_mask = (idx >= pre_cross_start) & (idx <= pre_cross_end)
    cross_mask = (idx >= crossing_start_frame) & (idx <= crossing_end_frame)
    post_mask = (idx >= post_cross_start) & (idx <= post_cross_end)

    return {
        "pre_cross_mask": pre_mask,
        "cross_mask": cross_mask,
        "post_cross_mask": post_mask,
        "pre_crossing_frame_range": [int(pre_cross_start), int(pre_cross_end)],
        "crossing_frame_range": [int(crossing_start_frame), int(crossing_end_frame)],
        "post_crossing_frame_range": [int(post_cross_start), int(post_cross_end)],
    }


def scene1019_tail_mask(n: int, crossing_end_frame: int) -> np.ndarray:
    """Down-weight the forced end-of-clip catch-up frames in the straight scene.

    The refined scene-1019 crossing location is intentionally preserved, which can leave only a
    few post-crossing frames. The last frames may therefore contain artificial endpoint catch-up
    speed/acceleration. For IDM calibration, we fit the meaningful behavior: approach, stopping,
    waiting, and early restart, while keeping final route progress weakly weighted.
    """
    idx = np.arange(n, dtype=int)
    tail_start = max(0, min(n, int(crossing_end_frame) + 3))
    if n - tail_start < 2:
        tail_start = max(0, n - 3)
    return idx >= tail_start


def scene1019_speed_cap(values: np.ndarray, cap_mps: float = 10.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(arr)
    arr[valid] = np.minimum(arr[valid], float(cap_mps))
    return arr


def build_vehicle_lead_timeline(
    sample_df: pd.DataFrame,
    ego_ref_df: pd.DataFrame,
    objects_df: pd.DataFrame,
) -> pd.DataFrame:
    frame_base = sample_df[["frame_index", "timestamp_s"]].copy()
    frame_base["frame_index"] = pd.to_numeric(frame_base["frame_index"], errors="coerce").astype("Int64")
    frame_base["timestamp_s"] = pd.to_numeric(frame_base["timestamp_s"], errors="coerce")
    frame_base = frame_base.dropna(subset=["frame_index"]).copy()
    frame_base["frame_index"] = frame_base["frame_index"].astype(int)

    if len(frame_base) == 0:
        return pd.DataFrame(
            {
                "frame_index": [],
                "vehicle_lead_available_flag": [],
                "vehicle_lead_instance_token": [],
                "vehicle_lead_s_progress_m": [],
                "vehicle_lead_speed_mps": [],
                "vehicle_lead_gap_ref_m": [],
            }
        )

    if len(objects_df) == 0:
        out = frame_base[["frame_index"]].copy()
        out["vehicle_lead_available_flag"] = False
        out["vehicle_lead_instance_token"] = ""
        out["vehicle_lead_s_progress_m"] = np.nan
        out["vehicle_lead_speed_mps"] = np.nan
        out["vehicle_lead_gap_ref_m"] = np.nan
        return out

    ego_small = ego_ref_df[["frame_index", "cumulative_distance_m"]].copy()
    ego_small["frame_index"] = pd.to_numeric(ego_small["frame_index"], errors="coerce").astype("Int64")
    ego_small["cumulative_distance_m"] = pd.to_numeric(ego_small["cumulative_distance_m"], errors="coerce")
    ego_small = ego_small.dropna(subset=["frame_index", "cumulative_distance_m"]).copy()
    ego_small["frame_index"] = ego_small["frame_index"].astype(int)

    obj = objects_df.copy()
    obj["frame_index"] = pd.to_numeric(obj.get("frame_index"), errors="coerce")
    obj["timestamp_s"] = pd.to_numeric(obj.get("timestamp_s"), errors="coerce")
    obj["relative_longitudinal_m"] = pd.to_numeric(obj.get("relative_longitudinal_m"), errors="coerce")
    obj["obj_length"] = pd.to_numeric(obj.get("obj_length"), errors="coerce")
    if "is_vehicle" in obj.columns:
        obj["is_vehicle"] = obj["is_vehicle"].fillna(False).astype(bool)
    elif "coarse_class" in obj.columns:
        obj["is_vehicle"] = obj["coarse_class"].astype(str).str.lower().eq("vehicle")
    else:
        obj["is_vehicle"] = False
    if "forward_corridor_flag" in obj.columns:
        forward_flag = obj["forward_corridor_flag"].fillna(False).astype(bool)
    else:
        forward_flag = pd.Series(True, index=obj.index)
    if "near_future_path_flag" in obj.columns:
        near_path_flag = obj["near_future_path_flag"].fillna(False).astype(bool)
    else:
        near_path_flag = pd.Series(True, index=obj.index)
    obj = obj[
        obj["is_vehicle"]
        & np.isfinite(obj["frame_index"])
        & np.isfinite(obj["relative_longitudinal_m"])
        & (obj["relative_longitudinal_m"] > 0.8)
        & (forward_flag | near_path_flag)
    ].copy()
    if len(obj) == 0:
        out = frame_base[["frame_index"]].copy()
        out["vehicle_lead_available_flag"] = False
        out["vehicle_lead_instance_token"] = ""
        out["vehicle_lead_s_progress_m"] = np.nan
        out["vehicle_lead_speed_mps"] = np.nan
        out["vehicle_lead_gap_ref_m"] = np.nan
        return out

    obj["frame_index"] = obj["frame_index"].astype(int)
    obj = obj.merge(ego_small, on="frame_index", how="left")
    obj = obj[np.isfinite(obj["cumulative_distance_m"])].copy()
    if len(obj) == 0:
        out = frame_base[["frame_index"]].copy()
        out["vehicle_lead_available_flag"] = False
        out["vehicle_lead_instance_token"] = ""
        out["vehicle_lead_s_progress_m"] = np.nan
        out["vehicle_lead_speed_mps"] = np.nan
        out["vehicle_lead_gap_ref_m"] = np.nan
        return out

    obj["lead_s_progress_m"] = obj["cumulative_distance_m"] + obj["relative_longitudinal_m"]
    obj["lead_gap_ref_m"] = np.maximum(
        0.15,
        obj["relative_longitudinal_m"].to_numpy(dtype=float) - 0.5 * np.nan_to_num(obj["obj_length"].to_numpy(dtype=float), nan=4.2),
    )
    if "instance_token" not in obj.columns:
        obj["instance_token"] = "unknown_vehicle"
    obj = obj.sort_values(["instance_token", "frame_index"]).reset_index(drop=True)

    g = obj.groupby("instance_token", sort=False)
    dt_prev = g["timestamp_s"].diff()
    ds_prev = g["lead_s_progress_m"].diff()
    v_prev = ds_prev / dt_prev
    dt_next = g["timestamp_s"].shift(-1) - obj["timestamp_s"]
    ds_next = g["lead_s_progress_m"].shift(-1) - obj["lead_s_progress_m"]
    v_next = ds_next / dt_next
    v_prev_arr = v_prev.to_numpy(dtype=float)
    v_next_arr = v_next.to_numpy(dtype=float)
    obj["lead_speed_mps"] = np.where(
        np.isfinite(v_prev_arr) & np.isfinite(v_next_arr),
        0.5 * (v_prev_arr + v_next_arr),
        np.where(np.isfinite(v_prev_arr), v_prev_arr, np.where(np.isfinite(v_next_arr), v_next_arr, np.nan)),
    )
    obj["lead_speed_mps"] = np.clip(obj["lead_speed_mps"], 0.0, 20.0)
    obj["lead_speed_mps"] = obj["lead_speed_mps"].fillna(0.0)

    obj = obj.sort_values(["frame_index", "lead_gap_ref_m"]).drop_duplicates(subset=["frame_index"], keep="first")
    lead_cols = obj[
        ["frame_index", "instance_token", "lead_s_progress_m", "lead_speed_mps", "lead_gap_ref_m"]
    ].rename(columns={"instance_token": "vehicle_lead_instance_token"})
    out = frame_base[["frame_index"]].merge(lead_cols, on="frame_index", how="left")
    out["vehicle_lead_available_flag"] = np.isfinite(pd.to_numeric(out["lead_s_progress_m"], errors="coerce"))
    out["vehicle_lead_instance_token"] = out["vehicle_lead_instance_token"].fillna("").astype(str)
    out = out.rename(
        columns={
            "lead_s_progress_m": "vehicle_lead_s_progress_m",
            "lead_speed_mps": "vehicle_lead_speed_mps",
            "lead_gap_ref_m": "vehicle_lead_gap_ref_m",
        }
    )
    return out


# ---------------------------------------------------------------------
# Stage A: scene-statistic initialization
# ---------------------------------------------------------------------

def estimate_initial_parameters(
    ego_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    gate_info: Dict[str, Any],
    bounds: Bounds,
    cfg: CalibrationConfig,
) -> Tuple[IDMParams, Dict[str, Any]]:
    speed = pd.to_numeric(ego_df["ego_speed_mps"], errors="coerce").to_numpy(dtype=float)
    accel = pd.to_numeric(ego_df["ego_accel_est_mps2"], errors="coerce").to_numpy(dtype=float)
    s_prog = pd.to_numeric(ego_df["cumulative_distance_m"], errors="coerce").to_numpy(dtype=float)
    idx = np.arange(len(ego_df), dtype=int)

    masks = build_phase_masks(
        ego_df,
        gate_info["crossing_start_frame"],
        gate_info["crossing_end_frame"],
        cfg,
    )
    pre_mask = masks["pre_cross_mask"]
    post_mask = masks["post_cross_mask"]
    cross_mask = masks["cross_mask"]

    gate_s = float(gate_info["gate_s_progress_m"])
    gap_to_gate = gate_s - s_prog

    # v0 for scene-1019 should come from the realistic straight approach before braking,
    # not from the forced end-of-clip catch-up frames after the preserved crossing.
    tail_mask = scene1019_tail_mask(len(idx), gate_info["crossing_end_frame"])
    pre_free_mask = (
        (idx <= max(0, gate_info["crossing_start_frame"] - 9))
        & np.isfinite(speed)
        & np.isfinite(accel)
        & (speed > 0.8)
        & (accel > -0.8)
        & (~tail_mask)
    )
    free_flow_speed_samples = speed[pre_free_mask]
    if len(free_flow_speed_samples) < 4:
        post_not_tail_mask = post_mask & (~tail_mask)
        post_stable_mask = post_not_tail_mask & np.isfinite(speed) & (accel > -0.35)
        free_flow_speed_samples = speed[post_stable_mask]
    if len(free_flow_speed_samples) < 4:
        free_flow_speed_samples = speed[np.isfinite(speed) & (~tail_mask)]
    v0_init = safe_percentile(
        scene1019_speed_cap(free_flow_speed_samples, 10.0),
        80.0,
        default=safe_percentile(scene1019_speed_cap(speed[np.isfinite(speed) & (~tail_mask)], 10.0), 85.0, default=7.5),
    )
    v0_init = clip_to_bounds(v0_init, *bounds.v0_mps)

    # s0 from near-stop distance to gate, with a robust floor to avoid frequent lower-bound collapse.
    near_stop_mask = (
        (speed <= max(cfg.near_stop_speed_threshold_mps, 0.35))
        & np.isfinite(speed)
        & np.isfinite(gap_to_gate)
        & (gap_to_gate >= 0.0)
        & (pre_mask | cross_mask)
    )
    near_stop_gap_samples = gap_to_gate[near_stop_mask]
    s0_init = safe_percentile(near_stop_gap_samples, 60.0, default=2.2)
    s0_init = max(s0_init, bounds.s0_m[0] + 0.35)
    s0_init = clip_to_bounds(s0_init, *bounds.s0_m)

    # T from valid approach phase: finite, approaching gate, and before full stop.
    approach_mask = (
        pre_mask
        & np.isfinite(speed)
        & np.isfinite(gap_to_gate)
        & np.isfinite(accel)
        & (speed > 0.8)
        & (speed < max(v0_init + 1.0, 3.0))
        & (gap_to_gate > s0_init + 0.6)
        & (gap_to_gate < 45.0)
        & (accel <= 0.6)
    )
    T_samples = (gap_to_gate[approach_mask] - s0_init) / np.maximum(speed[approach_mask], 1e-3)
    T_init = safe_median(T_samples, default=1.5)
    T_init = clip_to_bounds(T_init, *bounds.T_s)

    # a from early restart positive acceleration, excluding the terminal catch-up tail.
    post_not_tail_mask = post_mask & (~tail_mask)
    post_stable_mask = post_not_tail_mask & np.isfinite(speed) & np.isfinite(accel) & (accel > -0.35)
    pos_acc_samples = accel[post_stable_mask & np.isfinite(accel) & (accel > 0.05)]
    a_init = safe_percentile(pos_acc_samples, 75.0, default=1.8)
    a_init = clip_to_bounds(a_init, *bounds.a_max_mps2)

    # b from approach-to-stop braking magnitude, protected against unrealistically low values.
    braking_mask = (
        (pre_mask | cross_mask)
        & np.isfinite(accel)
        & np.isfinite(speed)
        & np.isfinite(gap_to_gate)
        & (accel < -0.05)
        & (speed > 0.6)
        & (gap_to_gate < 35.0)
    )
    neg_acc_samples = -accel[braking_mask]
    b_init = max(
        safe_percentile(neg_acc_samples, 80.0, default=2.2),
        safe_percentile(neg_acc_samples, 60.0, default=1.3),
    )
    b_init = max(b_init, 1.1)
    b_init = clip_to_bounds(b_init, *bounds.b_comfort_mps2)

    init = IDMParams(
        v0_mps=v0_init,
        s0_m=s0_init,
        T_s=T_init,
        a_max_mps2=a_init,
        b_comfort_mps2=b_init,
        delta=cfg.delta_fixed,
    )

    stats = {
        "free_flow_speed_samples_count": int(np.isfinite(free_flow_speed_samples).sum()),
        "free_flow_speed_80th_percentile_mps": float(v0_init),
        "near_stop_gap_samples_count": int(np.isfinite(near_stop_gap_samples).sum()),
        "near_stop_gap_60th_percentile_m": float(s0_init),
        "headway_samples_count": int(np.isfinite(T_samples).sum()),
        "headway_median_s": float(T_init),
        "positive_acc_samples_count": int(np.isfinite(pos_acc_samples).sum()),
        "positive_acc_75th_percentile_mps2": float(a_init),
        "braking_samples_count": int(np.isfinite(neg_acc_samples).sum()),
        "braking_80th_percentile_mps2": float(b_init),
        "gate_s_progress_m": float(gate_s),
        "gate_source": str(gate_info["gate_source"]),
        "ped_constraint_start_frame": int(gate_info["ped_constraint_start_frame"]),
        "crossing_start_frame": int(gate_info["crossing_start_frame"]),
        "crossing_end_frame": int(gate_info["crossing_end_frame"]),
        "pre_crossing_frame_range": masks["pre_crossing_frame_range"],
        "crossing_frame_range": masks["crossing_frame_range"],
        "post_crossing_frame_range": masks["post_crossing_frame_range"],
    }
    return init, stats


# ---------------------------------------------------------------------
# IDM simulation during calibration
# ---------------------------------------------------------------------

def idm_acceleration(v_ego: float, gap_m: float, v_lead: float, p: IDMParams) -> float:
    v = max(0.0, float(v_ego))
    s = max(0.1, float(gap_m))
    dv = v - float(v_lead)
    a = max(1e-4, float(p.a_max_mps2))
    b = max(1e-4, float(p.b_comfort_mps2))
    s_star = float(p.s0_m) + max(0.0, v * float(p.T_s) + (v * dv) / (2.0 * math.sqrt(a * b)))
    free_term = 1.0 - (v / max(float(p.v0_mps), 0.1)) ** float(p.delta)
    interact_term = (s_star / s) ** 2
    acc = a * (free_term - interact_term)
    return float(np.clip(acc, -6.0, float(p.a_max_mps2)))


def step_ego(
    ego: EgoState,
    frame_index: int,
    timestamp_s: float,
    dt: float,
    acc_cmd: float,
    pref: PathReference,
) -> EgoState:
    dt_eff = max(1e-3, float(dt))
    a = float(acc_cmd)
    v_next = float(np.clip(ego.speed_mps + a * dt_eff, 0.0, 20.0))
    s_next = float(np.clip(ego.s_progress_m + ego.speed_mps * dt_eff + 0.5 * a * dt_eff * dt_eff, pref.s_min, pref.s_max))
    x, y, yaw = pref.interp_pose(s_next)
    return EgoState(int(frame_index), float(timestamp_s), s_next, v_next, a, x, y, yaw)


def simulate_idm_against_active_constraints(
    sample_df: pd.DataFrame,
    ego_ref_df: pd.DataFrame,
    gate_info: Dict[str, Any],
    vehicle_lead_df: pd.DataFrame,
    params: IDMParams,
) -> pd.DataFrame:
    pref = PathReference(ego_ref_df)
    s0 = float(pd.to_numeric(ego_ref_df.iloc[0]["cumulative_distance_m"], errors="coerce"))
    t0 = float(pd.to_numeric(sample_df.iloc[0]["timestamp_s"], errors="coerce"))
    v_init_series = pd.to_numeric(ego_ref_df["ego_speed_mps"], errors="coerce").dropna()
    v0 = float(v_init_series.iloc[0]) if len(v_init_series) else 0.0
    x0, y0, yaw0 = pref.interp_pose(s0)
    ego = EgoState(0, t0, s0, v0, 0.0, x0, y0, yaw0)

    dt_series = pd.to_numeric(sample_df["delta_t_s"], errors="coerce")
    dt_candidates = dt_series[(dt_series > 1e-4) & np.isfinite(dt_series)]
    dt_fallback = float(dt_candidates.median()) if len(dt_candidates) else 0.5

    lead_by_frame: Dict[int, Dict[str, Any]] = {}
    if len(vehicle_lead_df):
        for _, lr in vehicle_lead_df.iterrows():
            f = int(lr["frame_index"])
            lead_by_frame[f] = {
                "vehicle_lead_available_flag": bool(lr.get("vehicle_lead_available_flag", False)),
                "vehicle_lead_instance_token": str(lr.get("vehicle_lead_instance_token", "")),
                "vehicle_lead_s_progress_m": float(lr.get("vehicle_lead_s_progress_m")) if np.isfinite(lr.get("vehicle_lead_s_progress_m")) else np.nan,
                "vehicle_lead_speed_mps": float(lr.get("vehicle_lead_speed_mps")) if np.isfinite(lr.get("vehicle_lead_speed_mps")) else np.nan,
                "vehicle_lead_gap_ref_m": float(lr.get("vehicle_lead_gap_ref_m")) if np.isfinite(lr.get("vehicle_lead_gap_ref_m")) else np.nan,
            }

    gate_s_progress_m = float(gate_info["gate_s_progress_m"])
    ped_constraint_start_frame = int(gate_info["ped_constraint_start_frame"])
    crossing_end_frame = int(gate_info["crossing_end_frame"])

    rows: List[Dict[str, Any]] = []
    for _, sr in sample_df.iterrows():
        frame = int(sr["frame_index"])
        ts = float(sr["timestamp_s"])
        dt = float(sr["delta_t_s"]) if np.isfinite(sr["delta_t_s"]) and float(sr["delta_t_s"]) > 1e-4 else dt_fallback

        ped_gate_active = ped_constraint_start_frame <= frame <= crossing_end_frame
        lead_ctx = lead_by_frame.get(frame, {})
        vehicle_lead_available = bool(lead_ctx.get("vehicle_lead_available_flag", False))
        vehicle_lead_s = float(lead_ctx.get("vehicle_lead_s_progress_m", np.nan))
        vehicle_lead_speed = float(lead_ctx.get("vehicle_lead_speed_mps", np.nan))
        vehicle_lead_gap = (
            max(0.15, float(vehicle_lead_s - ego.s_progress_m))
            if vehicle_lead_available and np.isfinite(vehicle_lead_s)
            else float("inf")
        )
        gate_gap = max(0.15, float(gate_s_progress_m - ego.s_progress_m))

        if ped_gate_active:
            if vehicle_lead_available and vehicle_lead_gap < gate_gap:
                acc_cmd = idm_acceleration(ego.speed_mps, gap_m=vehicle_lead_gap, v_lead=max(0.0, vehicle_lead_speed), p=params)
                lead_type = "vehicle"
                lead_gap_m = vehicle_lead_gap
                lead_speed_mps = max(0.0, vehicle_lead_speed) if np.isfinite(vehicle_lead_speed) else 0.0
            else:
                acc_cmd = idm_acceleration(ego.speed_mps, gap_m=gate_gap, v_lead=0.0, p=params)
                lead_type = "virtual_crossing_gate"
                lead_gap_m = gate_gap
                lead_speed_mps = 0.0
        else:
            acc_cmd = idm_acceleration(ego.speed_mps, gap_m=1e6, v_lead=float(params.v0_mps), p=params)
            lead_type = "free_flow"
            lead_gap_m = float("inf")
            lead_speed_mps = float(params.v0_mps)

        next_ego = step_ego(ego, frame, ts, dt, acc_cmd, pref)
        rows.append(
            {
                "frame_index": frame,
                "timestamp_s": ts,
                "delta_t_s": dt,
                "s_progress_m": next_ego.s_progress_m,
                "speed_mps": next_ego.speed_mps,
                "accel_mps2": next_ego.accel_mps2,
                "ego_x": next_ego.x,
                "ego_y": next_ego.y,
                "ego_yaw_rad": next_ego.yaw_rad,
                "lead_type": lead_type,
                "active_constraint_type": lead_type,
                "lead_gap_m": lead_gap_m,
                "lead_speed_mps": lead_speed_mps,
                "ped_crossing_active_flag": bool(ped_gate_active),
                "virtual_gate_active_flag": bool(ped_gate_active),
                "vehicle_lead_available_flag": bool(vehicle_lead_available),
                "vehicle_lead_used_flag": bool(lead_type == "vehicle"),
                "vehicle_lead_instance_token": str(lead_ctx.get("vehicle_lead_instance_token", "")),
                "vehicle_lead_s_progress_m": float(vehicle_lead_s) if np.isfinite(vehicle_lead_s) else np.nan,
                "vehicle_lead_speed_mps": float(vehicle_lead_speed) if np.isfinite(vehicle_lead_speed) else np.nan,
                "virtual_gate_s_progress_m": gate_s_progress_m,
            }
        )
        ego = next_ego

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Objective / fitting
# ---------------------------------------------------------------------

def detect_stop_restart_frames(
    speed_obs: np.ndarray,
    crossing_start_frame: int,
    crossing_end_frame: int,
    near_stop_speed_threshold_mps: float,
) -> Dict[str, Optional[int]]:
    n = len(speed_obs)
    stop_candidates = np.where(
        (np.arange(n) >= max(0, crossing_start_frame - 3))
        & (np.arange(n) <= min(n - 1, crossing_end_frame + 2))
        & np.isfinite(speed_obs)
        & (speed_obs <= near_stop_speed_threshold_mps)
    )[0]
    stop_frame = int(stop_candidates[0]) if len(stop_candidates) else None

    restart_candidates = np.where(
        (np.arange(n) >= min(n - 1, crossing_end_frame))
        & np.isfinite(speed_obs)
        & (speed_obs >= max(0.6, 2.5 * near_stop_speed_threshold_mps))
    )[0]
    restart_frame = int(restart_candidates[0]) if len(restart_candidates) else None

    return {
        "stop_frame": stop_frame,
        "restart_frame": restart_frame,
    }


def compute_fit_objective(
    sample_df: pd.DataFrame,
    ego_obs_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    crossing_start_frame: int,
    crossing_end_frame: int,
    cfg: CalibrationConfig,
) -> Tuple[float, Dict[str, Any]]:
    v_obs = pd.to_numeric(ego_obs_df["ego_speed_mps"], errors="coerce").to_numpy(dtype=float)
    a_obs = pd.to_numeric(ego_obs_df["ego_accel_est_mps2"], errors="coerce").to_numpy(dtype=float)
    s_obs = pd.to_numeric(ego_obs_df["cumulative_distance_m"], errors="coerce").to_numpy(dtype=float)

    v_sim = pd.to_numeric(sim_df["speed_mps"], errors="coerce").to_numpy(dtype=float)
    a_sim = pd.to_numeric(sim_df["accel_mps2"], errors="coerce").to_numpy(dtype=float)
    s_sim = pd.to_numeric(sim_df["s_progress_m"], errors="coerce").to_numpy(dtype=float)

    timing_obs = detect_stop_restart_frames(v_obs, crossing_start_frame, crossing_end_frame, cfg.near_stop_speed_threshold_mps)
    timing_sim = detect_stop_restart_frames(v_sim, crossing_start_frame, crossing_end_frame, cfg.near_stop_speed_threshold_mps)

    n = min(len(v_obs), len(v_sim), len(a_obs), len(a_sim), len(s_obs), len(s_sim))
    idx = np.arange(n, dtype=int)
    v_obs = scene1019_speed_cap(v_obs[:n], 10.0)
    v_sim = v_sim[:n]
    a_obs = a_obs[:n]
    a_sim = a_sim[:n]
    s_obs = s_obs[:n]
    s_sim = s_sim[:n]

    tail_mask = scene1019_tail_mask(n, crossing_end_frame)

    # Emphasize approach, stop, and restart dynamics where calibration quality is most critical.
    phase_weights = np.ones(n, dtype=float)
    approach_mask = (idx >= max(0, crossing_start_frame - 8)) & (idx <= min(n - 1, crossing_start_frame + 1))
    crossing_mask = (idx >= max(0, crossing_start_frame - 1)) & (idx <= min(n - 1, crossing_end_frame + 3))
    phase_weights[approach_mask] += 0.8
    phase_weights[crossing_mask] += 1.2
    if timing_obs["restart_frame"] is not None:
        rf = int(timing_obs["restart_frame"])
        restart_mask = (idx >= max(0, rf - 4)) & (idx <= min(n - 1, rf + 6))
        phase_weights[restart_mask] += 0.9

    speed_weights = phase_weights
    progress_weights = 0.55 * np.ones(n, dtype=float) + 0.45 * phase_weights
    accel_weights = 0.4 * np.ones(n, dtype=float) + 0.6 * phase_weights

    # The final few frames of scene-1019 can be dominated by synthetic endpoint catch-up.
    # Keep them in the output comparison, but make them weak in the objective so the
    # fitted IDM remains physically reasonable.
    speed_weights[tail_mask] *= 0.12
    progress_weights[tail_mask] *= 0.08
    accel_weights[tail_mask] *= 0.03

    a_obs_smooth = smooth_series(a_obs, window=cfg.objective_accel_smoothing_window)

    speed_rmse = weighted_rmse(v_obs, v_sim, speed_weights)
    progress_rmse = weighted_rmse(s_obs, s_sim, progress_weights)
    accel_rmse = weighted_rmse(
        a_obs_smooth,
        a_sim,
        accel_weights,
        residual_clip=cfg.objective_accel_residual_clip_mps2,
    )

    stop_timing_error = 0.0
    if timing_obs["stop_frame"] is not None and timing_sim["stop_frame"] is not None:
        stop_timing_error = abs(int(timing_obs["stop_frame"]) - int(timing_sim["stop_frame"]))
    elif timing_obs["stop_frame"] is not None or timing_sim["stop_frame"] is not None:
        stop_timing_error = 6.0

    restart_timing_error = 0.0
    if timing_obs["restart_frame"] is not None and timing_sim["restart_frame"] is not None:
        restart_timing_error = abs(int(timing_obs["restart_frame"]) - int(timing_sim["restart_frame"]))
    elif timing_obs["restart_frame"] is not None or timing_sim["restart_frame"] is not None:
        restart_timing_error = 6.0

    transition_mask = np.zeros(n, dtype=bool)
    if timing_obs["stop_frame"] is not None:
        sf = int(timing_obs["stop_frame"])
        transition_mask |= (idx >= max(0, sf - 3)) & (idx <= min(n - 1, sf + 6))
    if timing_obs["restart_frame"] is not None:
        rf = int(timing_obs["restart_frame"])
        transition_mask |= (idx >= max(0, rf - 3)) & (idx <= min(n - 1, rf + 6))
    if not transition_mask.any():
        transition_mask = (idx >= max(0, crossing_start_frame - 2)) & (idx <= min(n - 1, crossing_end_frame + 4))
    transition_speed_shape_rmse = weighted_rmse(
        v_obs[transition_mask],
        v_sim[transition_mask],
        np.ones(int(np.sum(transition_mask)), dtype=float),
    )

    objective = (
        cfg.objective_weight_speed * speed_rmse
        + cfg.objective_weight_progress * progress_rmse
        + cfg.objective_weight_accel * accel_rmse
        + cfg.objective_weight_stop_timing * stop_timing_error
        + cfg.objective_weight_restart_timing * restart_timing_error
        + cfg.objective_weight_transition_speed_shape * transition_speed_shape_rmse
    )

    detail = {
        "speed_rmse": float(speed_rmse),
        "progress_rmse": float(progress_rmse),
        "accel_rmse": float(accel_rmse),
        "transition_speed_shape_rmse": float(transition_speed_shape_rmse),
        "stop_timing_error_frames": float(stop_timing_error),
        "restart_timing_error_frames": float(restart_timing_error),
        "objective_total": float(objective),
        "observed_stop_frame": timing_obs["stop_frame"] if timing_obs["stop_frame"] is not None else None,
        "simulated_stop_frame": timing_sim["stop_frame"] if timing_sim["stop_frame"] is not None else None,
        "observed_restart_frame": timing_obs["restart_frame"] if timing_obs["restart_frame"] is not None else None,
        "simulated_restart_frame": timing_sim["restart_frame"] if timing_sim["restart_frame"] is not None else None,
        "approach_frames_used": [int(max(0, crossing_start_frame - 8)), int(min(n - 1, crossing_start_frame + 1))],
        "crossing_frames_used": [int(max(0, crossing_start_frame - 1)), int(min(n - 1, crossing_end_frame + 3))],
        "accel_smoothing_window": int(cfg.objective_accel_smoothing_window),
        "accel_residual_clip_mps2": float(cfg.objective_accel_residual_clip_mps2),
        "scene1019_tail_downweighted_frames": [int(i) for i in np.where(tail_mask)[0].tolist()],
        "scene1019_observed_speed_cap_mps": 10.0,
    }
    return float(objective), detail


def sample_random_params(rng: np.random.Generator, bounds: Bounds, delta_fixed: float) -> IDMParams:
    return IDMParams(
        v0_mps=float(rng.uniform(*bounds.v0_mps)),
        s0_m=float(rng.uniform(*bounds.s0_m)),
        T_s=float(rng.uniform(*bounds.T_s)),
        a_max_mps2=float(rng.uniform(*bounds.a_max_mps2)),
        b_comfort_mps2=float(rng.uniform(*bounds.b_comfort_mps2)),
        delta=float(delta_fixed),
    )


def sample_local_params_around_init(
    rng: np.random.Generator,
    init: IDMParams,
    bounds: Bounds,
    delta_fixed: float,
    shrink: float,
) -> IDMParams:
    def around(v: float, lo: float, hi: float) -> float:
        span = (hi - lo) * float(shrink)
        a = max(lo, v - 0.5 * span)
        b = min(hi, v + 0.5 * span)
        return float(rng.uniform(a, b))

    return IDMParams(
        v0_mps=around(init.v0_mps, *bounds.v0_mps),
        s0_m=around(init.s0_m, *bounds.s0_m),
        T_s=around(init.T_s, *bounds.T_s),
        a_max_mps2=around(init.a_max_mps2, *bounds.a_max_mps2),
        b_comfort_mps2=around(init.b_comfort_mps2, *bounds.b_comfort_mps2),
        delta=float(delta_fixed),
    )


def run_calibration_search(
    sample_df: pd.DataFrame,
    ego_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    init_params: IDMParams,
    gate_info: Dict[str, Any],
    bounds: Bounds,
    cfg: CalibrationConfig,
) -> Tuple[IDMParams, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)

    tested_rows: List[Dict[str, Any]] = []
    vehicle_lead_df = build_vehicle_lead_timeline(sample_df=sample_df, ego_ref_df=ego_df, objects_df=objects_df)

    best_params = init_params
    best_sim_df = simulate_idm_against_active_constraints(
        sample_df=sample_df,
        ego_ref_df=ego_df,
        gate_info=gate_info,
        vehicle_lead_df=vehicle_lead_df,
        params=init_params,
    )
    best_obj, best_detail = compute_fit_objective(
        sample_df, ego_df, best_sim_df,
        crossing_start_frame=int(gate_info["crossing_start_frame"]),
        crossing_end_frame=int(gate_info["crossing_end_frame"]),
        cfg=cfg,
    )

    tested_rows.append(
        {
            "candidate_source": "init",
            **asdict(init_params),
            **best_detail,
        }
    )

    n_total = max(1, int(cfg.num_random_samples))
    n_b1 = max(1, int(round(0.70 * n_total)))
    n_b2 = max(1, n_total - n_b1)

    # Stage B1: local random search around robust scene-based initialization.
    for i in range(n_b1):
        shrink = 0.48 if (i % 2 == 0) else 0.30
        p = sample_local_params_around_init(rng, init_params, bounds, cfg.delta_fixed, shrink=shrink)
        source = "stage_b1_local"
        sim_df = simulate_idm_against_active_constraints(
            sample_df=sample_df,
            ego_ref_df=ego_df,
            gate_info=gate_info,
            vehicle_lead_df=vehicle_lead_df,
            params=p,
        )
        obj, detail = compute_fit_objective(
            sample_df, ego_df, sim_df,
            crossing_start_frame=int(gate_info["crossing_start_frame"]),
            crossing_end_frame=int(gate_info["crossing_end_frame"]),
            cfg=cfg,
        )
        tested_rows.append(
            {
                "candidate_source": source,
                **asdict(p),
                **detail,
            }
        )
        if obj < best_obj:
            best_obj = obj
            best_params = p
            best_sim_df = sim_df
            best_detail = detail

    # Stage B2: narrower refinement around best candidate from stage B1.
    best_after_b1 = best_params
    for i in range(n_b2):
        shrink = 0.16 if (i % 2 == 0) else 0.08
        p = sample_local_params_around_init(rng, best_after_b1, bounds, cfg.delta_fixed, shrink=shrink)
        source = "stage_b2_refine"
        sim_df = simulate_idm_against_active_constraints(
            sample_df=sample_df,
            ego_ref_df=ego_df,
            gate_info=gate_info,
            vehicle_lead_df=vehicle_lead_df,
            params=p,
        )
        obj, detail = compute_fit_objective(
            sample_df, ego_df, sim_df,
            crossing_start_frame=int(gate_info["crossing_start_frame"]),
            crossing_end_frame=int(gate_info["crossing_end_frame"]),
            cfg=cfg,
        )
        tested_rows.append(
            {
                "candidate_source": source,
                **asdict(p),
                **detail,
            }
        )
        if obj < best_obj:
            best_obj = obj
            best_params = p
            best_sim_df = sim_df
            best_detail = detail

    trace_df = pd.DataFrame(tested_rows).sort_values("objective_total").reset_index(drop=True)
    vehicle_lead_used_in_best = bool(best_sim_df["vehicle_lead_used_flag"].fillna(False).astype(bool).any()) if "vehicle_lead_used_flag" in best_sim_df.columns else False
    summary = {
        "best_objective": float(best_obj),
        "best_detail": best_detail,
        "num_candidates_tested": int(len(trace_df)),
        "search_stage_counts": {
            "stage_b1_local": int(n_b1),
            "stage_b2_refine": int(n_b2),
        },
        "vehicle_lead_ever_available_during_fit": bool(vehicle_lead_df["vehicle_lead_available_flag"].fillna(False).astype(bool).any()),
        "vehicle_lead_ever_used_in_best_rollout": bool(vehicle_lead_used_in_best),
    }
    return best_params, best_sim_df, trace_df, summary


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def build_fit_comparison_df(
    sample_df: pd.DataFrame,
    ego_obs_df: pd.DataFrame,
    sim_df: pd.DataFrame,
) -> pd.DataFrame:
    obs = pd.DataFrame(
        {
            "frame_index": ego_obs_df["frame_index"],
            "timestamp_s": ego_obs_df["timestamp_s"],
            "observed_speed_mps": ego_obs_df["ego_speed_mps"],
            "observed_accel_mps2": ego_obs_df["ego_accel_est_mps2"],
            "observed_progress_m": ego_obs_df["cumulative_distance_m"],
        }
    )
    sim = pd.DataFrame(
        {
            "frame_index": sim_df["frame_index"],
            "simulated_speed_mps": sim_df["speed_mps"],
            "simulated_accel_mps2": sim_df["accel_mps2"],
            "simulated_progress_m": sim_df["s_progress_m"],
        }
    )
    cmp_df = obs.merge(sim, on="frame_index", how="inner")
    cmp_df["speed_error_mps"] = pd.to_numeric(cmp_df["simulated_speed_mps"], errors="coerce") - pd.to_numeric(cmp_df["observed_speed_mps"], errors="coerce")
    cmp_df["accel_error_mps2"] = pd.to_numeric(cmp_df["simulated_accel_mps2"], errors="coerce") - pd.to_numeric(cmp_df["observed_accel_mps2"], errors="coerce")
    cmp_df["progress_error_m"] = pd.to_numeric(cmp_df["simulated_progress_m"], errors="coerce") - pd.to_numeric(cmp_df["observed_progress_m"], errors="coerce")
    return cmp_df


def make_plots(
    cmp_df: pd.DataFrame,
    plot_dir: Path,
    scene_name: str,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    t = pd.to_numeric(cmp_df["timestamp_s"], errors="coerce").to_numpy(dtype=float)
    if len(t):
        t = t - t[0]

    # Speed
    fig, ax = plt.subplots()
    ax.plot(t, cmp_df["observed_speed_mps"], label="Observed")
    ax.plot(t, cmp_df["simulated_speed_mps"], label="Fitted IDM")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(f"Observed vs fitted speed ({scene_name})")
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(plot_dir / "observed_vs_fitted_speed.png", dpi=180)
    plt.close(fig)

    # Progress
    fig, ax = plt.subplots()
    ax.plot(t, cmp_df["observed_progress_m"], label="Observed")
    ax.plot(t, cmp_df["simulated_progress_m"], label="Fitted IDM")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Progress along path (m)")
    ax.set_title(f"Observed vs fitted progress ({scene_name})")
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(plot_dir / "observed_vs_fitted_progress.png", dpi=180)
    plt.close(fig)

    # Acceleration
    fig, ax = plt.subplots()
    ax.plot(t, cmp_df["observed_accel_mps2"], label="Observed")
    ax.plot(t, cmp_df["simulated_accel_mps2"], label="Fitted IDM")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title(f"Observed vs fitted acceleration ({scene_name})")
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(plot_dir / "observed_vs_fitted_acceleration.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    apply_plot_style()
    args = parse_args()
    if str(args.scene_name).strip() != "scene-1019":
        raise ValueError(
            f"This scene-specific IDM calibration script is locked to scene-1019, but got {args.scene_name!r}. "
            "Use the other scene-specific tuning scripts for scene-0061 or scene-0221."
        )

    cfg = CalibrationConfig(
        scene_name=args.scene_name,
        refined_root=Path(args.refined_root),
        output_root=Path(args.output_root),
        delta_fixed=float(args.delta_fixed),
        num_random_samples=int(args.num_random_samples),
        seed=int(args.seed),
        hold_buffer_extra_m=float(args.hold_buffer_extra_m),
        near_stop_speed_threshold_mps=float(args.near_stop_speed_threshold_mps),
        post_crossing_buffer_frames=int(args.post_crossing_buffer_frames),
        pre_crossing_buffer_frames=int(args.pre_crossing_buffer_frames),
        objective_weight_speed=float(args.objective_weight_speed),
        objective_weight_progress=float(args.objective_weight_progress),
        objective_weight_accel=float(args.objective_weight_accel),
        objective_weight_stop_timing=float(args.objective_weight_stop_timing),
        objective_weight_restart_timing=float(args.objective_weight_restart_timing),
        objective_weight_transition_speed_shape=float(args.objective_weight_transition_speed_shape),
        objective_accel_smoothing_window=int(args.objective_accel_smoothing_window),
        objective_accel_residual_clip_mps2=float(args.objective_accel_residual_clip_mps2),
    )

    scene_dir = cfg.refined_root / cfg.scene_name
    out_dir = cfg.output_root / cfg.scene_name
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    data = load_refined_scene(scene_dir)
    sample_df = data["sample_df"]
    ego_df = data["ego_df"]
    objects_df = data["objects_df"]
    scene_summary = data["scene_summary"]

    bounds = Bounds(delta=(cfg.delta_fixed, cfg.delta_fixed))
    gate_info = infer_conflict_gate_s(scene_summary=scene_summary, ego_df=ego_df, cfg=cfg)
    init_params, init_stats = estimate_initial_parameters(
        ego_df=ego_df,
        sample_df=sample_df,
        gate_info=gate_info,
        bounds=bounds,
        cfg=cfg,
    )

    best_params, best_sim_df, trace_df, fit_summary = run_calibration_search(
        sample_df=sample_df,
        ego_df=ego_df,
        objects_df=objects_df,
        init_params=init_params,
        gate_info=gate_info,
        bounds=bounds,
        cfg=cfg,
    )

    cmp_df = build_fit_comparison_df(sample_df=sample_df, ego_obs_df=ego_df, sim_df=best_sim_df)
    make_plots(cmp_df, plot_dir, cfg.scene_name)

    params_payload = {
        "scene_name": cfg.scene_name,
        "calibration_timestamp_utc": utc_now_iso(),
        "parameter_source": "scene_1019_straight_approach_calibration",
        "parameter_notes": [
            "Parameters were estimated from the refined straight-approach scene using robust trajectory-statistic initialization followed by two-stage bounded local search.",
            "Scene-1019 objective down-weights the final synthetic endpoint catch-up frames so the fitted IDM parameters stay physically realistic.",
            "This parameter set is intended to be shared by both the perception-only and OSM-aware planners.",
            "IDM delta was kept fixed during calibration.",
        ],
        "fitted_parameters": asdict(best_params),
        "fixed_parameters": {"delta": float(cfg.delta_fixed)},
        "bounds": {
            "v0_mps": list(bounds.v0_mps),
            "s0_m": list(bounds.s0_m),
            "T_s": list(bounds.T_s),
            "a_max_mps2": list(bounds.a_max_mps2),
            "b_comfort_mps2": list(bounds.b_comfort_mps2),
            "delta": list(bounds.delta),
        },
        "method": {
            "stage_a": "scene-statistic initialization",
            "stage_b1": "local random search around initialization",
            "stage_b2": "local refinement around current best",
            "num_random_samples": int(cfg.num_random_samples),
            "seed": int(cfg.seed),
        },
        "fit_objective": fit_summary["best_detail"],
        "virtual_gate": gate_info,
    }
    (out_dir / "idm_parameters.json").write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    summary_payload = {
        "scene_name": cfg.scene_name,
        "calibration_timestamp_utc": utc_now_iso(),
        "initial_parameters": asdict(init_params),
        "final_parameters": asdict(best_params),
        "initial_scene_statistics": init_stats,
        "best_fit_summary": fit_summary,
        "phase_frames": {
            "ped_constraint_start_frame": int(gate_info["ped_constraint_start_frame"]),
            "crossing_start_frame": int(gate_info["crossing_start_frame"]),
            "crossing_end_frame": int(gate_info["crossing_end_frame"]),
            "pre_crossing_frame_range": init_stats.get("pre_crossing_frame_range"),
            "crossing_frame_range": init_stats.get("crossing_frame_range"),
            "post_crossing_frame_range": init_stats.get("post_crossing_frame_range"),
        },
        "config": {
            "delta_fixed": cfg.delta_fixed,
            "num_random_samples": cfg.num_random_samples,
            "seed": cfg.seed,
            "hold_buffer_extra_m": cfg.hold_buffer_extra_m,
            "near_stop_speed_threshold_mps": cfg.near_stop_speed_threshold_mps,
            "post_crossing_buffer_frames": cfg.post_crossing_buffer_frames,
            "pre_crossing_buffer_frames": cfg.pre_crossing_buffer_frames,
            "objective_weight_speed": cfg.objective_weight_speed,
            "objective_weight_progress": cfg.objective_weight_progress,
            "objective_weight_accel": cfg.objective_weight_accel,
            "objective_weight_stop_timing": cfg.objective_weight_stop_timing,
            "objective_weight_restart_timing": cfg.objective_weight_restart_timing,
            "objective_weight_transition_speed_shape": cfg.objective_weight_transition_speed_shape,
            "objective_accel_smoothing_window": cfg.objective_accel_smoothing_window,
            "objective_accel_residual_clip_mps2": cfg.objective_accel_residual_clip_mps2,
        },
    }
    (out_dir / "idm_parameter_calibration_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    trace_df.to_csv(out_dir / "idm_fit_trace.csv", index=False)
    best_sim_df.to_csv(out_dir / "idm_fit_rollout.csv", index=False)
    cmp_df.to_csv(out_dir / "idm_fit_comparison.csv", index=False)

    print("idm_calibration_summary_scene_1019:")
    print(f"  scene_name: {cfg.scene_name}")
    print(f"  output_dir: {out_dir}")
    print(f"  fitted_v0_mps: {best_params.v0_mps:.4f}")
    print(f"  fitted_s0_m: {best_params.s0_m:.4f}")
    print(f"  fitted_T_s: {best_params.T_s:.4f}")
    print(f"  fitted_a_max_mps2: {best_params.a_max_mps2:.4f}")
    print(f"  fitted_b_comfort_mps2: {best_params.b_comfort_mps2:.4f}")
    print(f"  fixed_delta: {best_params.delta:.4f}")
    print(f"  best_objective: {float(fit_summary['best_objective']):.6f}")


if __name__ == "__main__":
    main()
