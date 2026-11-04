#!/usr/bin/env python3
"""
Run perception-only closed-loop IDM longitudinal planning over degraded camera-based inputs.

Batch version: by default this script runs the same planner over the three final
project scenes (scene-1019, scene-0061, scene-0383) while keeping the original
per-scene output structure unchanged.

Reads:
- data/refined/<scene>/sample_timeline.csv
- data/refined/<scene>/ego_timeline.csv
- data/planner_perception_only/<scene>/degraded_inputs/<mode>/degradation_XX/seed_YYY/objects_timeline_degraded.csv
- optional: data/planner_perception_only/<scene>/degraded_inputs/<mode>/degradation_XX/seed_YYY/degradation_audit.json
- optional: data/planner_perception_only/<scene>/degraded_inputs/degradation_formula_summary.json

Writes (default root: data/planner_perception_only):
- data/planner_perception_only/<scene>/runs/<mode>/degradation_XX/seed_YYY/
  - planner_actions.csv
  - ego_rollout.csv
  - per_frame_features.csv
  - active_constraints.csv
  - safety_metrics.json
- data/planner_perception_only/<scene>/per_run_metrics.csv
- data/planner_perception_only/<scene>/aggregated_metrics.csv
- data/planner_perception_only/<scene>/aggregated_metrics.json
- report-ready PNG plots in data/planner_perception_only/<scene>/

Planner:
- IDM (Intelligent Driver Model), perception-only baseline.
- It consumes the already-degraded camera-perception object table directly.
- No OSM priors are used here.
- PET is computed only as an evaluation metric at the refined/reference crossing zone.
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


def parse_int_list_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


DEFAULT_BATCH_SCENES = ["scene-1019", "scene-0061", "scene-0383"]


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
    """Resolve --scene-name / --scene-names robustly.

    Important: --scene-name all should run the default three project scenes,
    not look for a folder named data/refined/all.
    """
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
    v0_mps: float = 8.5
    s0_m: float = 2.0
    T_s: float = 1.5
    a_max_mps2: float = 2.0
    b_comfort_mps2: float = 2.4
    delta: float = 4.0
    hard_brake_limit_mps2: float = 6.0
    max_speed_mps: float = 12.0
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
    unsafe_ped_distance_m: float = 4.0
    unsafe_veh_distance_m: float = 5.0
    stopped_speed_threshold_mps: float = 0.2
    critical_ped_ttc_threshold_s: float = 1.5
    critical_pet_threshold_s: float = 1.5
    hard_brake_epsilon_mps2: float = 0.05

    # PET / true crossing-zone evaluation parameters.
    # These are evaluation-only fields; they do not affect the baseline planner.
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
class ReferenceCrossing:
    """Ground-truth/refined crossing used only for PET evaluation."""

    crossing_x: float
    crossing_y: float
    crossing_s_m: float
    crossing_length_m: float
    crossing_width_m: float


class PathReference:
    def __init__(self, ego_df: pd.DataFrame) -> None:
        df = ego_df.sort_values("frame_index").reset_index(drop=True)
        self.s = df["cumulative_distance_m"].to_numpy(dtype=float)
        self.x = df["ego_x"].to_numpy(dtype=float)
        self.y = df["ego_y"].to_numpy(dtype=float)
        self.yaw = np.unwrap(df["ego_yaw_rad"].to_numpy(dtype=float))
        if len(self.s) < 2:
            raise ValueError("ego_timeline.csv must contain at least 2 points.")
        self.s_min = float(self.s[0])
        self.s_max = float(self.s[-1])

    def interp_pose(self, s_query: float) -> Tuple[float, float, float]:
        s_q = float(np.clip(s_query, self.s_min, self.s_max))
        x = float(np.interp(s_q, self.s, self.x))
        y = float(np.interp(s_q, self.s, self.y))
        yaw = float(np.interp(s_q, self.s, self.yaw))
        return x, y, yaw

    def future_polyline(self, s_start: float, speed_mps: float, horizon_s: float, step_m: float) -> np.ndarray:
        horizon_m = max(10.0, speed_mps * horizon_s)
        n = max(2, int(horizon_m / max(step_m, 0.5)))
        ss = np.linspace(s_start, min(self.s_max, s_start + horizon_m), n)
        return np.column_stack([np.interp(ss, self.s, self.x), np.interp(ss, self.s, self.y)])


def _safe_float_from_summary(scene_summary: Dict[str, Any], key: str, default: float) -> float:
    try:
        value = scene_summary.get(key, default)
        if value is None:
            return float(default)
        value = float(value)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def build_reference_crossing(scene_summary: Dict[str, Any], pref: PathReference) -> ReferenceCrossing:
    """
    Build the evaluation crossing from the refined scene.

    For the refined synthetic scenes (1019 and 0061), the crossing comes from
    scene_summary.json. For pass-through scenes such as 0221, the raw summary may
    not contain crossing_center_x/y. In that case, PET still needs a stable
    reference zone, so we infer a fallback crossing-like evaluation point from
    the ego trajectory itself: first near-stop / slowest point, otherwise the
    middle of the path. This fallback affects PET evaluation only; it does not
    add OSM or change the planner behavior.
    """
    cx = _safe_float_from_summary(scene_summary, "crossing_center_x", float("nan"))
    cy = _safe_float_from_summary(scene_summary, "crossing_center_y", float("nan"))
    s_cross = scene_summary.get("crossing_zone_arc_length_m", None)

    if np.isfinite(cx) and np.isfinite(cy):
        if s_cross is None or not np.isfinite(float(s_cross)):
            # Fallback: nearest path progress to crossing center.
            dx = pref.x - cx
            dy = pref.y - cy
            idx = int(np.argmin(dx * dx + dy * dy))
            s_cross = float(pref.s[idx])
        else:
            s_cross = float(s_cross)
    else:
        # Pass-through/no-refinement scenes may not have an explicit crossing.
        # Use the first strong slow/stop point if present; otherwise use 60% of
        # the route so PET remains computable without changing planner behavior.
        try:
            # A repeated or almost repeated progress point usually indicates a
            # natural stop/hold in the raw scene.
            ds = np.diff(np.asarray(pref.s, dtype=float), prepend=np.asarray(pref.s, dtype=float)[0])
            slow_candidates = np.where(ds <= 0.15)[0]
            slow_candidates = slow_candidates[slow_candidates >= max(2, int(0.15 * len(pref.s)))]
            if len(slow_candidates):
                idx = int(slow_candidates[0])
            else:
                idx = int(np.argmin(np.abs(pref.s - (pref.s_min + 0.60 * (pref.s_max - pref.s_min)))))
        except Exception:
            idx = int(len(pref.s) // 2)
        s_cross = float(pref.s[idx])
        cx, cy, _yaw = pref.interp_pose(s_cross)

    return ReferenceCrossing(
        crossing_x=float(cx),
        crossing_y=float(cy),
        crossing_s_m=float(np.clip(float(s_cross), pref.s_min, pref.s_max)),
        crossing_length_m=_safe_float_from_summary(scene_summary, "crossing_length_m", 10.0),
        crossing_width_m=_safe_float_from_summary(scene_summary, "crossing_width_m", 6.5),
    )

def init_ego(sample_df: pd.DataFrame, ego_ref_df: pd.DataFrame, pref: PathReference) -> EgoState:
    t0 = float(sample_df.iloc[0]["timestamp_s"])
    s0 = float(ego_ref_df.iloc[0]["cumulative_distance_m"])
    vs = ego_ref_df["ego_speed_mps"].dropna().astype(float)
    v0 = float(vs.iloc[0]) if len(vs) else 0.0
    x0, y0, yaw0 = pref.interp_pose(s0)
    return EgoState(0, t0, s0, v0, 0.0, x0, y0, yaw0)


def compute_object_velocity(df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    ts_map = sample_df.set_index("frame_index")["timestamp_s"].to_dict()
    work = df.copy().sort_values(["instance_token", "frame_index"]).reset_index(drop=True)
    work["vx_est_mps"] = 0.0
    work["vy_est_mps"] = 0.0
    for _, g in work.groupby("instance_token"):
        idx = g.index.to_numpy()
        gx = g["obj_x"].to_numpy(dtype=float)
        gy = g["obj_y"].to_numpy(dtype=float)
        gf = g["frame_index"].to_numpy(dtype=int)
        t = np.array([float(ts_map.get(int(fi), np.nan)) for fi in gf], dtype=float)
        vx = np.zeros(len(g), dtype=float)
        vy = np.zeros(len(g), dtype=float)
        for i in range(len(g)):
            if i == 0 and len(g) > 1:
                dt = max(1e-3, t[i + 1] - t[i])
                vx[i] = (gx[i + 1] - gx[i]) / dt
                vy[i] = (gy[i + 1] - gy[i]) / dt
            elif i == len(g) - 1 and len(g) > 1:
                dt = max(1e-3, t[i] - t[i - 1])
                vx[i] = (gx[i] - gx[i - 1]) / dt
                vy[i] = (gy[i] - gy[i - 1]) / dt
            elif len(g) > 2:
                dt = max(1e-3, t[i + 1] - t[i - 1])
                vx[i] = (gx[i + 1] - gx[i - 1]) / dt
                vy[i] = (gy[i + 1] - gy[i - 1]) / dt
        work.loc[idx, "vx_est_mps"] = vx
        work.loc[idx, "vy_est_mps"] = vy
    return work


def point_to_polyline_distance(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
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


def compute_runtime_features(frame_df: pd.DataFrame, ego: EgoState, pref: PathReference, cfg: IDMConfig) -> pd.DataFrame:
    out = frame_df.copy()
    if len(out) == 0:
        for col in (
            "runtime_distance_to_ego_m",
            "runtime_rel_long_m",
            "runtime_rel_lat_m",
            "runtime_ahead_flag",
            "runtime_future_path_distance_m",
            "runtime_near_future_path_flag",
            "runtime_v_long_mps",
        ):
            out[col] = []
        return out

    dx = out["obj_x"].to_numpy(dtype=float) - ego.x
    dy = out["obj_y"].to_numpy(dtype=float) - ego.y
    c = float(math.cos(ego.yaw_rad))
    s = float(math.sin(ego.yaw_rad))
    rel_long = dx * c + dy * s
    rel_lat = -dx * s + dy * c
    dist = np.sqrt(dx * dx + dy * dy)
    ahead = rel_long > 0.0
    v_long = out["vx_est_mps"].to_numpy(dtype=float) * c + out["vy_est_mps"].to_numpy(dtype=float) * s
    future = pref.future_polyline(
        s_start=ego.s_progress_m,
        speed_mps=ego.speed_mps,
        horizon_s=cfg.lookahead_horizon_s,
        step_m=cfg.lookahead_step_m,
    )
    fdist = point_to_polyline_distance(out["obj_x"].to_numpy(dtype=float), out["obj_y"].to_numpy(dtype=float), future)
    near_path = fdist <= cfg.ped_future_path_threshold_m

    out["runtime_distance_to_ego_m"] = dist
    out["runtime_rel_long_m"] = rel_long
    out["runtime_rel_lat_m"] = rel_lat
    out["runtime_ahead_flag"] = ahead
    out["runtime_future_path_distance_m"] = fdist
    out["runtime_near_future_path_flag"] = near_path
    out["runtime_v_long_mps"] = v_long
    return out


def get_relevant_conflict_pedestrians(runtime_df: pd.DataFrame, cfg: IDMConfig) -> pd.DataFrame:
    if len(runtime_df) == 0:
        return runtime_df.iloc[0:0].copy()
    ped = runtime_df[runtime_df["coarse_class"].eq("pedestrian")].copy()
    if len(ped) == 0:
        return ped

    ped = ped[
        ped["runtime_ahead_flag"].astype(bool)
        & ped["runtime_near_future_path_flag"].astype(bool)
        & (ped["runtime_future_path_distance_m"] <= cfg.ped_conflict_future_path_max_m)
        & (ped["runtime_rel_long_m"] > 0.0)
        & (ped["runtime_rel_long_m"] <= cfg.ped_conflict_longitudinal_max_m)
        & (ped["runtime_rel_lat_m"].abs() <= cfg.ped_conflict_lateral_max_m)
    ].copy()
    if len(ped) == 0:
        return ped

    ped["ped_conflict_score_used"] = (
        4.0 * ped["runtime_future_path_distance_m"]
        + 2.0 * ped["runtime_rel_lat_m"].abs()
        + 1.0 * ped["runtime_rel_long_m"]
    )
    ped = ped.sort_values(
        [
            "ped_conflict_score_used",
            "runtime_future_path_distance_m",
            "runtime_rel_lat_m",
            "runtime_rel_long_m",
        ],
        key=lambda c: c.abs() if c.name == "runtime_rel_lat_m" else c,
    ).reset_index(drop=True)
    return ped


def select_active_constraint(runtime_df: pd.DataFrame, relevant_peds_df: pd.DataFrame, ego_s_progress_m: float, cfg: IDMConfig) -> Dict[str, Any]:
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
        "ped_conflict_score_used": np.nan,
        "conflict_gate_s": np.nan,
    }
    if len(runtime_df) == 0:
        return default

    is_veh = runtime_df["coarse_class"].eq("vehicle")
    ahead = runtime_df["runtime_ahead_flag"].astype(bool)
    ped_candidates_raw = relevant_peds_df.copy()
    n_raw = int(len(ped_candidates_raw))
    ped_candidates_planning = ped_candidates_raw.head(max(1, int(cfg.max_relevant_pedestrians_for_planning))).copy()
    ped_conflict = len(ped_candidates_planning) > 0
    ped_choice: Optional[pd.Series] = None
    ped_gap = np.inf
    conflict_gate_s = np.nan
    nearest_relevant_ped_m = float(ped_candidates_raw["runtime_distance_to_ego_m"].min()) if len(ped_candidates_raw) else np.inf

    if ped_conflict:
        ped_candidates_planning["virtual_stop_gap_m"] = ped_candidates_planning["runtime_rel_long_m"] - cfg.ped_stop_buffer_m
        ped_candidates_planning = ped_candidates_planning.sort_values("virtual_stop_gap_m")
        ped_choice = ped_candidates_planning.iloc[0]
        ped_gap = max(0.15, float(ped_choice["virtual_stop_gap_m"]))
        conflict_gate_s = float(ego_s_progress_m + float(ped_choice["runtime_rel_long_m"]) - cfg.ped_stop_buffer_m)

    veh_candidates = runtime_df[
        is_veh & ahead & (np.abs(runtime_df["runtime_rel_lat_m"]) <= cfg.lead_lateral_threshold_m)
    ].copy()
    veh_choice: Optional[pd.Series] = None
    veh_gap = np.inf
    if len(veh_candidates):
        veh_candidates = veh_candidates.sort_values("runtime_rel_long_m")
        veh_choice = veh_candidates.iloc[0]
        veh_gap = max(0.15, float(veh_choice["runtime_rel_long_m"]))

    if ped_conflict and ped_gap <= veh_gap:
        return {
            "lead_type": "ped_virtual_stop",
            "lead_token": str(ped_choice["instance_token"]) if ped_choice is not None else "",
            "lead_gap_m": float(ped_gap),
            "lead_speed_mps": 0.0,
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
            "ped_conflict_score_used": float(ped_choice["ped_conflict_score_used"]) if ped_choice is not None else np.nan,
            "conflict_gate_s": conflict_gate_s,
        }
    if veh_choice is not None:
        return {
            "lead_type": "vehicle",
            "lead_token": str(veh_choice["instance_token"]),
            "lead_gap_m": float(veh_gap),
            "lead_speed_mps": float(veh_choice["runtime_v_long_mps"]),
            "ped_conflict_present": ped_conflict,
            "num_relevant_conflict_pedestrians_raw": n_raw,
            "num_relevant_conflict_pedestrians_used": int(len(ped_candidates_planning)),
            "num_relevant_conflict_pedestrians": int(len(ped_candidates_planning)),
            "closest_relevant_ped_distance_m": nearest_relevant_ped_m,
            "strongest_relevant_ped_token": str(ped_choice["instance_token"]) if ped_choice is not None else "",
            "strongest_relevant_ped_rel_long_m": float(ped_choice["runtime_rel_long_m"]) if ped_choice is not None else np.nan,
            "strongest_relevant_ped_rel_lat_m": float(ped_choice["runtime_rel_lat_m"]) if ped_choice is not None else np.nan,
            "strongest_relevant_ped_future_path_distance_m": float(ped_choice["runtime_future_path_distance_m"]) if ped_choice is not None else np.nan,
            "strongest_relevant_ped_distance_to_ego_m": float(ped_choice["runtime_distance_to_ego_m"]) if ped_choice is not None else np.nan,
            "ped_conflict_score_used": float(ped_choice["ped_conflict_score_used"]) if ped_choice is not None else np.nan,
            "conflict_gate_s": conflict_gate_s,
        }
    return default | {
        "ped_conflict_present": ped_conflict,
        "num_relevant_conflict_pedestrians_raw": n_raw,
        "num_relevant_conflict_pedestrians_used": int(len(ped_candidates_planning)),
        "num_relevant_conflict_pedestrians": int(len(ped_candidates_planning)),
        "closest_relevant_ped_distance_m": nearest_relevant_ped_m,
        "strongest_relevant_ped_token": str(ped_choice["instance_token"]) if ped_choice is not None else "",
        "strongest_relevant_ped_rel_long_m": float(ped_choice["runtime_rel_long_m"]) if ped_choice is not None else np.nan,
        "strongest_relevant_ped_rel_lat_m": float(ped_choice["runtime_rel_lat_m"]) if ped_choice is not None else np.nan,
        "strongest_relevant_ped_future_path_distance_m": float(ped_choice["runtime_future_path_distance_m"]) if ped_choice is not None else np.nan,
        "strongest_relevant_ped_distance_to_ego_m": float(ped_choice["runtime_distance_to_ego_m"]) if ped_choice is not None else np.nan,
        "ped_conflict_score_used": float(ped_choice["ped_conflict_score_used"]) if ped_choice is not None else np.nan,
        "conflict_gate_s": conflict_gate_s,
    }


def idm_acceleration(v_ego: float, gap_m: float, v_lead: float, cfg: IDMConfig) -> float:
    v = max(0.0, float(v_ego))
    s = max(0.1, float(gap_m))
    dv = v - float(v_lead)
    a = cfg.a_max_mps2
    b = max(1e-3, cfg.b_comfort_mps2)
    s_star = cfg.s0_m + max(0.0, v * cfg.T_s + (v * dv) / (2.0 * math.sqrt(a * b)))
    free_term = 1.0 - (v / max(cfg.v0_mps, 0.1)) ** cfg.delta
    interact_term = (s_star / s) ** 2
    acc = a * (free_term - interact_term)
    return float(np.clip(acc, -cfg.hard_brake_limit_mps2, cfg.a_max_mps2))


def step_ego(ego: EgoState, frame_index: int, timestamp_s: float, dt: float, acc_cmd: float, pref: PathReference, cfg: IDMConfig) -> EgoState:
    """Advance ego state with non-reversing longitudinal kinematics."""
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


def pedestrian_zone_counts(run_df: pd.DataFrame, ref_crossing: ReferenceCrossing, cfg: IDMConfig) -> Tuple[int, bool]:
    """Count degraded-perception pedestrians inside the refined/reference crossing zone."""
    if len(run_df) == 0 or "coarse_class" not in run_df.columns:
        return 0, False
    ped = run_df[run_df["coarse_class"].eq("pedestrian")].copy()
    if len(ped) == 0:
        return 0, False
    px = pd.to_numeric(ped["obj_x"], errors="coerce").to_numpy(dtype=float)
    py = pd.to_numeric(ped["obj_y"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(px) & np.isfinite(py)
    if not valid.any():
        return 0, False
    d = np.sqrt((px[valid] - ref_crossing.crossing_x) ** 2 + (py[valid] - ref_crossing.crossing_y) ** 2)
    # Use a compact crossing-zone disk, scaled slightly by crossing width.
    radius = max(float(cfg.pet_crossing_zone_radius_m), 0.5 * float(ref_crossing.crossing_width_m))
    count = int(np.sum(d <= radius))
    return count, bool(count > 0)


def compute_pet_metrics(feat_df: pd.DataFrame, cfg: IDMConfig) -> Dict[str, Any]:
    """
    Compute post-encroachment time (PET) from per-frame baseline features.

    PET is evaluated at the refined/reference crossing zone. It is not used by
    the planner; it is an output metric for fair comparison in Script 10.
    """
    empty = {
        "post_encroachment_time_s": float("nan"),
        "pet_available_flag": False,
        "pet_critical_flag": False,
        "ego_crossing_time_s": float("nan"),
        "last_ped_zone_time_before_ego_s": float("nan"),
        "first_ped_zone_time_after_ego_s": float("nan"),
    }
    if len(feat_df) == 0 or "timestamp_s" not in feat_df.columns:
        return empty

    d = feat_df.copy().sort_values("timestamp_s").reset_index(drop=True)
    t = pd.to_numeric(d["timestamp_s"], errors="coerce").to_numpy(dtype=float)

    ego_zone = d.get("ego_in_reference_crossing_zone_flag", pd.Series(False, index=d.index)).fillna(False).astype(bool).to_numpy()
    ped_zone = d.get("any_ped_in_reference_crossing_zone_flag", pd.Series(False, index=d.index)).fillna(False).astype(bool).to_numpy()

    if not ego_zone.any():
        if "ego_s_progress_m" in d.columns and "reference_crossing_s_m" in d.columns:
            es = pd.to_numeric(d["ego_s_progress_m"], errors="coerce").to_numpy(dtype=float)
            cs = pd.to_numeric(d["reference_crossing_s_m"], errors="coerce").to_numpy(dtype=float)
            passed = np.isfinite(es) & np.isfinite(cs) & (es >= cs)
            if passed.any():
                ego_idx = int(np.argmax(passed))
            else:
                return empty
        else:
            return empty
    else:
        ego_idx = int(np.argmax(ego_zone))

    ego_time = float(t[ego_idx]) if np.isfinite(t[ego_idx]) else float("nan")
    if not np.isfinite(ego_time):
        return empty

    if ped_zone[ego_idx]:
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

    dt = ego_df["delta_t_s"].fillna(0.0).to_numpy(dtype=float)
    speed = ego_df["speed_mps"].to_numpy(dtype=float)
    acc = ego_df["accel_mps2"].to_numpy(dtype=float)

    travel = float(np.sum(dt))
    stop_time = float(np.sum(dt[speed <= cfg.stopped_speed_threshold_mps]))
    avg_speed = float(np.mean(speed)) if len(speed) else 0.0
    max_braking = float(np.max(np.maximum(0.0, -acc))) if len(acc) else 0.0
    braking_magnitudes = np.maximum(0.0, -acc)
    mean_braking_during_brake = float(np.mean(braking_magnitudes[acc < 0.0])) if np.any(acc < 0.0) else 0.0
    jerk = np.diff(acc) / np.maximum(dt[1:], 1e-3) if len(acc) > 1 else np.array([0.0])
    comfort = float(np.sqrt(np.mean(jerk * jerk))) if len(jerk) else 0.0

    pet = compute_pet_metrics(feat_df, cfg)

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
    }
    out.update(pet)
    return out


def load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def run_single(
    sample_df: pd.DataFrame,
    ego_ref_df: pd.DataFrame,
    objects_deg_df: pd.DataFrame,
    degradation_mode: str,
    level: int,
    seed: int,
    out_dir: Path,
    cfg: IDMConfig,
    degradation_audit: Optional[Dict[str, Any]],
    ref_crossing: ReferenceCrossing,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pref = PathReference(ego_ref_df)
    ego = init_ego(sample_df, ego_ref_df, pref)
    dt_candidates = sample_df["delta_t_s"][sample_df["delta_t_s"] > 1e-4]
    dt_fallback = float(dt_candidates.median()) if len(dt_candidates) else 0.5
    obj_vel_df = compute_object_velocity(objects_deg_df, sample_df)

    ego_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    feat_rows: List[Dict[str, Any]] = []
    constraint_rows: List[Dict[str, Any]] = []

    for _, sr in sample_df.iterrows():
        frame = int(sr["frame_index"])
        ts = float(sr["timestamp_s"])
        dt = float(sr["delta_t_s"]) if float(sr["delta_t_s"]) > 1e-4 else dt_fallback
        frame_df = obj_vel_df[obj_vel_df["frame_index"] == frame].copy()
        run_df = compute_runtime_features(frame_df, ego, pref, cfg)
        relevant_peds_df = get_relevant_conflict_pedestrians(run_df, cfg)
        constraint = select_active_constraint(run_df, relevant_peds_df, ego.s_progress_m, cfg)

        if constraint["lead_type"] == "none":
            acc_cmd = idm_acceleration(ego.speed_mps, gap_m=1e6, v_lead=cfg.v0_mps, cfg=cfg)
        else:
            acc_cmd = idm_acceleration(
                ego.speed_mps,
                gap_m=float(constraint["lead_gap_m"]),
                v_lead=float(constraint["lead_speed_mps"]),
                cfg=cfg,
            )

        next_ego = step_ego(ego, frame, ts, dt, acc_cmd, pref, cfg)
        min_ttc_ped = compute_ttc_ped(relevant_peds_df, ego.speed_mps)

        nearest_dyn = float(run_df[run_df["coarse_class"].isin(["pedestrian", "vehicle"])]["runtime_distance_to_ego_m"].min()) if int(run_df["coarse_class"].isin(["pedestrian", "vehicle"]).sum()) else float("inf")
        nearest_ped = float(run_df[run_df["coarse_class"] == "pedestrian"]["runtime_distance_to_ego_m"].min()) if int((run_df["coarse_class"] == "pedestrian").sum()) else float("inf")
        nearest_veh = float(run_df[run_df["coarse_class"] == "vehicle"]["runtime_distance_to_ego_m"].min()) if int((run_df["coarse_class"] == "vehicle").sum()) else float("inf")
        nearest_relevant_ped = float(relevant_peds_df["runtime_distance_to_ego_m"].min()) if len(relevant_peds_df) else float("inf")
        num_peds_in_ref_zone, any_ped_in_ref_zone = pedestrian_zone_counts(run_df, ref_crossing, cfg)
        ego_in_ref_crossing_zone = bool(abs(ego.s_progress_m - ref_crossing.crossing_s_m) <= cfg.ego_crossing_zone_s_window_m)

        conflict_gate_s = float(constraint["conflict_gate_s"]) if np.isfinite(constraint["conflict_gate_s"]) else float("nan")
        ego_has_passed_conflict_gate = bool(np.isfinite(conflict_gate_s) and ego.s_progress_m >= conflict_gate_s)
        ped_violation = bool(ego_has_passed_conflict_gate and len(relevant_peds_df) > 0)
        critical_ttc_flag = bool(np.isfinite(min_ttc_ped) and min_ttc_ped < cfg.critical_ped_ttc_threshold_s)
        unsafe_ped_flag = bool(np.isfinite(nearest_ped) and nearest_ped < cfg.unsafe_ped_distance_m)
        unsafe_vehicle_flag = bool(np.isfinite(nearest_veh) and nearest_veh < cfg.unsafe_veh_distance_m)
        unsafe_any_flag = bool(unsafe_ped_flag or unsafe_vehicle_flag)
        hard_brake_flag = bool(next_ego.accel_mps2 <= -(cfg.hard_brake_limit_mps2 - cfg.hard_brake_epsilon_mps2))

        feat_rows.append(
            {
                "degradation_mode": degradation_mode,
                "degradation_level_pct": int(level),
                "seed": int(seed),
                "frame_index": frame,
                "timestamp_s": ts,
                "ego_s_progress_m": float(ego.s_progress_m),
                "reference_crossing_s_m": float(ref_crossing.crossing_s_m),
                "reference_crossing_x": float(ref_crossing.crossing_x),
                "reference_crossing_y": float(ref_crossing.crossing_y),
                "num_peds_in_reference_crossing_zone": int(num_peds_in_ref_zone),
                "any_ped_in_reference_crossing_zone_flag": bool(any_ped_in_ref_zone),
                "ego_in_reference_crossing_zone_flag": bool(ego_in_ref_crossing_zone),
                "num_objects": int(len(run_df)),
                "num_dynamic_objects": int(run_df["coarse_class"].isin(["pedestrian", "vehicle"]).sum()),
                "nearest_ped_m": nearest_ped,
                "nearest_vehicle_m": nearest_veh,
                "nearest_dynamic_m": nearest_dyn,
                "nearest_relevant_ped_m": nearest_relevant_ped,
                "active_lead_type": constraint["lead_type"],
                "active_lead_token": constraint["lead_token"],
                "active_lead_gap_m": float(constraint["lead_gap_m"]),
                "active_lead_speed_mps": float(constraint["lead_speed_mps"]),
                "ped_conflict_present": bool(constraint["ped_conflict_present"]),
                "num_relevant_conflict_pedestrians_raw": int(constraint["num_relevant_conflict_pedestrians_raw"]),
                "num_relevant_conflict_pedestrians_used": int(constraint["num_relevant_conflict_pedestrians_used"]),
                "num_relevant_conflict_pedestrians": int(constraint["num_relevant_conflict_pedestrians"]),
                "strongest_relevant_ped_token": str(constraint["strongest_relevant_ped_token"]),
                "strongest_relevant_ped_rel_long_m": float(constraint["strongest_relevant_ped_rel_long_m"]) if np.isfinite(constraint["strongest_relevant_ped_rel_long_m"]) else np.nan,
                "strongest_relevant_ped_rel_lat_m": float(constraint["strongest_relevant_ped_rel_lat_m"]) if np.isfinite(constraint["strongest_relevant_ped_rel_lat_m"]) else np.nan,
                "strongest_relevant_ped_future_path_distance_m": float(constraint["strongest_relevant_ped_future_path_distance_m"]) if np.isfinite(constraint["strongest_relevant_ped_future_path_distance_m"]) else np.nan,
                "strongest_relevant_ped_distance_to_ego_m": float(constraint["strongest_relevant_ped_distance_to_ego_m"]) if np.isfinite(constraint["strongest_relevant_ped_distance_to_ego_m"]) else np.nan,
                "ped_conflict_score_used": float(constraint["ped_conflict_score_used"]) if np.isfinite(constraint["ped_conflict_score_used"]) else np.nan,
                "min_ttc_ped_s": min_ttc_ped,
                "critical_ped_ttc_flag": critical_ttc_flag,
                "conflict_gate_s": conflict_gate_s,
                "ego_has_passed_conflict_gate": ego_has_passed_conflict_gate,
                "ped_conflict_zone_violation": ped_violation,
                "hard_brake_flag": hard_brake_flag,
                "unsafe_ped_flag": unsafe_ped_flag,
                "unsafe_vehicle_flag": unsafe_vehicle_flag,
                "unsafe_any_flag": unsafe_any_flag,
            }
        )
        action_rows.append(
            {
                "degradation_mode": degradation_mode,
                "degradation_level_pct": int(level),
                "seed": int(seed),
                "frame_index": frame,
                "timestamp_s": ts,
                "idm_accel_cmd_mps2": acc_cmd,
                "lead_type": constraint["lead_type"],
                "lead_gap_m": float(constraint["lead_gap_m"]),
                "lead_speed_mps": float(constraint["lead_speed_mps"]),
            }
        )
        constraint_rows.append(
            {
                "degradation_mode": degradation_mode,
                "degradation_level_pct": int(level),
                "seed": int(seed),
                "frame_index": frame,
                "timestamp_s": ts,
                "lead_type": constraint["lead_type"],
                "lead_token": constraint["lead_token"],
                "lead_gap_m": float(constraint["lead_gap_m"]),
                "ped_conflict_present": bool(constraint["ped_conflict_present"]),
                "num_relevant_conflict_pedestrians_raw": int(constraint["num_relevant_conflict_pedestrians_raw"]),
                "num_relevant_conflict_pedestrians_used": int(constraint["num_relevant_conflict_pedestrians_used"]),
                "strongest_relevant_ped_token": str(constraint["strongest_relevant_ped_token"]),
                "conflict_gate_s": float(constraint["conflict_gate_s"]) if np.isfinite(constraint["conflict_gate_s"]) else np.nan,
                "ped_conflict_score_used": float(constraint["ped_conflict_score_used"]) if np.isfinite(constraint["ped_conflict_score_used"]) else np.nan,
            }
        )
        ego_rows.append(
            {
                "degradation_mode": degradation_mode,
                "degradation_level_pct": int(level),
                "seed": int(seed),
                "frame_index": frame,
                "timestamp_s": ts,
                "delta_t_s": dt,
                "s_progress_m": next_ego.s_progress_m,
                "speed_mps": next_ego.speed_mps,
                "accel_mps2": next_ego.accel_mps2,
                "ego_x": next_ego.x,
                "ego_y": next_ego.y,
                "ego_yaw_rad": next_ego.yaw_rad,
            }
        )
        ego = next_ego
        if ego.s_progress_m >= pref.s_max - 1e-3:
            break

    ego_roll = pd.DataFrame(ego_rows)
    actions = pd.DataFrame(action_rows)
    feats = pd.DataFrame(feat_rows)
    constraints = pd.DataFrame(constraint_rows)
    metrics = compute_run_metrics(ego_roll, feats, cfg)
    metrics["degradation_mode"] = degradation_mode
    metrics["degradation_level_pct"] = int(level)
    metrics["seed"] = int(seed)
    metrics["reference_crossing_s_m"] = float(ref_crossing.crossing_s_m)
    metrics["reference_crossing_x"] = float(ref_crossing.crossing_x)
    metrics["reference_crossing_y"] = float(ref_crossing.crossing_y)
    metrics["crossing_length_m"] = float(ref_crossing.crossing_length_m)
    metrics["crossing_width_m"] = float(ref_crossing.crossing_width_m)

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
        "actual_camera_drop_fraction",
        "frames_with_all_camera_eligible_rows_lost",
        "mean_frame_camera_drop_fraction",
        "mean_drop_probability_across_frames",
    ]
    metrics = [m for m in metrics if m in per_run_df.columns]
    agg = per_run_df.groupby(["degradation_mode", "degradation_level_pct"])[metrics].agg(["mean", "std", "min", "max"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]).rstrip("_") for col in agg.columns]
    return agg


def plot_mean_std_for_mode(
    agg_mode_df: pd.DataFrame,
    metric_prefix: str,
    ylabel: str,
    title: str,
    out_path: Path,
    color: str,
) -> None:
    y_mean_col = f"{metric_prefix}_mean"
    y_std_col = f"{metric_prefix}_std"
    if y_mean_col not in agg_mode_df.columns:
        return
    x = agg_mode_df["degradation_level_pct"].to_numpy(dtype=float)
    y = agg_mode_df[y_mean_col].to_numpy(dtype=float)
    ys = agg_mode_df[y_std_col].fillna(0.0).to_numpy(dtype=float) if y_std_col in agg_mode_df.columns else np.zeros_like(y)
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


def plot_mode_bundle(agg_df: pd.DataFrame, scene_root: Path, mode: str) -> None:
    mode_df = agg_df[agg_df["degradation_mode"] == mode].copy()
    if len(mode_df) == 0:
        return

    plot_mean_std_for_mode(mode_df, "min_ped_distance_m", "Distance (m)", f"Minimum pedestrian distance vs camera degradation ({mode})", scene_root / f"{mode}_min_ped_distance_vs_degradation.png", "#dc2626")
    plot_mean_std_for_mode(mode_df, "min_relevant_ped_distance_m", "Distance (m)", f"Minimum relevant pedestrian distance vs camera degradation ({mode})", scene_root / f"{mode}_min_relevant_ped_distance_vs_degradation.png", "#ef4444")
    plot_mean_std_for_mode(mode_df, "unsafe_frames_total", "Frame count", f"Unsafe frames total vs camera degradation ({mode})", scene_root / f"{mode}_unsafe_frames_total_vs_degradation.png", "#7c3aed")
    plot_mean_std_for_mode(mode_df, "unsafe_total_episodes", "Episode count", f"Unsafe total episodes vs camera degradation ({mode})", scene_root / f"{mode}_unsafe_total_episodes_vs_degradation.png", "#7c3aed")
    plot_mean_std_for_mode(mode_df, "min_ped_ttc_s", "Time (s)", f"Minimum pedestrian TTC vs camera degradation ({mode})", scene_root / f"{mode}_min_ttc_ped_vs_degradation.png", "#2563eb")
    plot_mean_std_for_mode(mode_df, "critical_ped_ttc_frames", "Frame count", f"Critical pedestrian TTC frames vs camera degradation ({mode})", scene_root / f"{mode}_critical_ped_ttc_frames_vs_degradation.png", "#1d4ed8")
    plot_mean_std_for_mode(mode_df, "post_encroachment_time_s", "Time (s)", f"Post-encroachment time vs camera degradation ({mode})", scene_root / f"{mode}_pet_vs_degradation.png", "#16a34a")
    plot_mean_std_for_mode(mode_df, "ped_conflict_zone_violation_frames", "Frame count", f"Pedestrian conflict-zone violation frames vs camera degradation ({mode})", scene_root / f"{mode}_ped_conflict_zone_violation_frames_vs_degradation.png", "#be123c")
    plot_mean_std_for_mode(mode_df, "travel_time_s", "Time (s)", f"Travel time vs camera degradation ({mode})", scene_root / f"{mode}_travel_time_vs_degradation.png", "#0891b2")
    plot_mean_std_for_mode(mode_df, "average_speed_mps", "Speed (m/s)", f"Average speed vs camera degradation ({mode})", scene_root / f"{mode}_average_speed_vs_degradation.png", "#16a34a")
    plot_mean_std_for_mode(mode_df, "max_braking_mps2", "Braking magnitude (m/s²)", f"Maximum braking vs camera degradation ({mode})", scene_root / f"{mode}_max_braking_vs_degradation.png", "#b45309")
    plot_mean_std_for_mode(mode_df, "hard_brake_frames", "Frame count", f"Hard brake frames vs camera degradation ({mode})", scene_root / f"{mode}_hard_brake_frames_vs_degradation.png", "#92400e")
    plot_mean_std_for_mode(mode_df, "comfort_jerk_rms_mps3", "Jerk RMS (m/s³)", f"Comfort vs camera degradation ({mode})", scene_root / f"{mode}_comfort_vs_degradation.png", "#475569")
    plot_mean_std_for_mode(mode_df, "actual_camera_drop_fraction", "Fraction", f"Actual camera drop fraction vs camera degradation ({mode})", scene_root / f"{mode}_actual_camera_drop_fraction_vs_degradation.png", "#6b7280")


def plot_diagnostics(run_dir: Path, scene_root: Path, mode: str, level: int, seed: int) -> None:
    ego = pd.read_csv(run_dir / "ego_rollout.csv")
    acts = pd.read_csv(run_dir / "planner_actions.csv")
    feats = pd.read_csv(run_dir / "per_frame_features.csv")

    out_dir = scene_root / "diagnostics" / mode / f"degradation_{level:02d}" / f"seed_{seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t = ego["timestamp_s"].to_numpy(dtype=float)
    t = t - t[0] if len(t) else t

    fig, ax = plt.subplots()
    ax.plot(t, ego["speed_mps"], color="#1d4ed8")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(f"Ego speed vs time ({mode}, deg={level}%, seed={seed})")
    fig.tight_layout()
    fig.savefig(out_dir / "ego_speed_vs_time.png", dpi=180)
    plt.close(fig)

    gap = feats["active_lead_gap_m"].replace(np.inf, np.nan)
    fig, ax = plt.subplots()
    ax.plot(t, gap, color="#9333ea")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Active constraint gap (m)")
    ax.set_title(f"Active constraint distance vs time ({mode}, deg={level}%, seed={seed})")
    fig.tight_layout()
    fig.savefig(out_dir / "active_constraint_distance_vs_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, acts["idm_accel_cmd_mps2"], color="#b91c1c")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration command (m/s²)")
    ax.set_title(f"IDM acceleration vs time ({mode}, deg={level}%, seed={seed})")
    fig.tight_layout()
    fig.savefig(out_dir / "planner_acceleration_vs_time.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run perception-only IDM planner on degraded camera inputs for one or more scenes.")
    p.add_argument(
        "--scene-names",
        default="scene-1019,scene-0061,scene-0383",
        help="Comma-separated scenes to run. Default: the three final project scenes.",
    )
    p.add_argument(
        "--scene-name",
        default="",
        help="Optional single-scene override, e.g. scene-0061. If provided, this overrides --scene-names.",
    )
    p.add_argument("--refined-root", default="data/refined")
    p.add_argument("--idm-calibration-root", default="data/idm_calibration")
    p.add_argument(
        "--results-root",
        default="data/planner_perception_only",
        help="Planner output root under data/; also where degraded_inputs/ is read.",
    )
    p.add_argument("--degradation-modes", default="random_dropout,burst_dropout,distance_based_dropout")
    p.add_argument("--degradation-levels", default="10,30,50,70,90")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip a scene/run if its refined scene or degraded inputs are missing instead of stopping.",
    )
    return p.parse_args()


def run_scene(
    scene: str,
    args: argparse.Namespace,
    modes: List[str],
    levels: List[int],
    seeds: List[int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    scene = str(scene).strip()
    cfg, idm_calibration_payload = load_calibrated_idm_config(scene, Path(args.idm_calibration_root))

    refined_scene_dir = Path(args.refined_root) / scene
    result_scene_dir = Path(args.results_root) / scene
    degraded_root = result_scene_dir / "degraded_inputs"
    runs_root = result_scene_dir / "runs"

    required_refined = [
        refined_scene_dir / "sample_timeline.csv",
        refined_scene_dir / "ego_timeline.csv",
        refined_scene_dir / "scene_summary.json",
    ]
    missing_refined = [str(p) for p in required_refined if not p.is_file()]
    if missing_refined:
        raise FileNotFoundError(f"Missing refined scene inputs for {scene}: {missing_refined}")

    runs_root.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(refined_scene_dir / "sample_timeline.csv").sort_values("frame_index").reset_index(drop=True)
    ego_ref_df = pd.read_csv(refined_scene_dir / "ego_timeline.csv").sort_values("frame_index").reset_index(drop=True)
    scene_summary = json.loads((refined_scene_dir / "scene_summary.json").read_text(encoding="utf-8"))
    ref_crossing_for_eval = build_reference_crossing(scene_summary, PathReference(ego_ref_df))

    formula_summary = load_optional_json(degraded_root / "degradation_formula_summary.json")
    metrics_rows: List[Dict[str, Any]] = []

    print(f"\n[scene] {scene}")
    print(f"  refined_input: {refined_scene_dir}")
    print(f"  degraded_input_root: {degraded_root}")
    print(f"  calibrated_idm_loaded: {idm_calibration_payload is not None}")

    for mode in modes:
        for level in levels:
            for seed in seeds:
                deg_run_dir = degraded_root / mode / f"degradation_{level:02d}" / f"seed_{seed:03d}"
                deg_path = deg_run_dir / "objects_timeline_degraded.csv"
                if not deg_path.is_file():
                    raise FileNotFoundError(
                        f"Missing degraded input: {deg_path}. Run 04_perception_degradation.py first for {scene}."
                    )

                degradation_audit = load_optional_json(deg_run_dir / "degradation_audit.json")
                objects_deg_df = pd.read_csv(deg_path)
                run_dir = runs_root / mode / f"degradation_{level:02d}" / f"seed_{seed:03d}"

                metrics = run_single(
                    sample_df=sample_df,
                    ego_ref_df=ego_ref_df,
                    objects_deg_df=objects_deg_df,
                    degradation_mode=mode,
                    level=level,
                    seed=seed,
                    out_dir=run_dir,
                    cfg=cfg,
                    degradation_audit=degradation_audit,
                    ref_crossing=ref_crossing_for_eval,
                )
                metrics["scene_name"] = scene
                metrics_rows.append(metrics)
                print(f"  [ok] {scene} mode={mode} level={level:02d}% seed={seed:03d}")

    per_run = pd.DataFrame(metrics_rows).sort_values(["degradation_mode", "degradation_level_pct", "seed"]).reset_index(drop=True)
    per_run.to_csv(result_scene_dir / "per_run_metrics.csv", index=False)

    agg = aggregate_metrics(per_run)
    agg.to_csv(result_scene_dir / "aggregated_metrics.csv", index=False)

    payload = {
        "scene_name": scene,
        "planner_type": "perception_only_idm",
        "planner_notes": [
            "Uses camera-based degraded perceived objects only.",
            "Consumes pre-generated degraded camera detections from Script 04.",
            "Uses refined ego reference path for rollout geometry.",
            "Loads scene-specific IDM parameters from Script 05 when available.",
            "Does not use OSM priors or map confidence logic.",
            "Computes PET as an evaluation-only metric at the refined/reference crossing zone.",
        ],
        "idm_parameters": asdict(cfg),
        "idm_calibration_payload": idm_calibration_payload,
        "degradation_modes": modes,
        "degradation_levels": levels,
        "seeds": seeds,
        "degradation_formula_summary": formula_summary,
        "reference_crossing_for_pet_evaluation": asdict(ref_crossing_for_eval),
        "per_run_metrics": per_run.to_dict(orient="records"),
        "aggregated_metrics": agg.to_dict(orient="records"),
    }
    (result_scene_dir / "aggregated_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    for mode in modes:
        plot_mode_bundle(agg, result_scene_dir, mode)

    if modes and levels and seeds:
        diag_run = runs_root / modes[0] / f"degradation_{levels[0]:02d}" / f"seed_{seeds[0]:03d}"
        if diag_run.is_dir():
            plot_diagnostics(diag_run, result_scene_dir, modes[0], levels[0], seeds[0])

    print("baseline_summary:")
    print(f"  scene = {scene}")
    print("  planner = perception_only_idm")
    print("  input = degraded camera-perception object table")
    print(f"  scene_specific_idm_params_loaded = {idm_calibration_payload is not None}")
    print("  no OSM priors used")
    print("  outputs separated by degradation_mode + degradation_level_pct + seed")
    print(f"[done] IDM planner outputs written to {result_scene_dir}")

    scene_summary_payload = {
        "scene_name": scene,
        "status": "completed",
        "result_scene_dir": str(result_scene_dir),
        "num_runs": int(len(per_run)),
        "idm_calibration_loaded": bool(idm_calibration_payload is not None),
        "reference_crossing_for_pet_evaluation": asdict(ref_crossing_for_eval),
    }
    return per_run, scene_summary_payload


def aggregate_combined_metrics(per_run_df: pd.DataFrame) -> pd.DataFrame:
    if len(per_run_df) == 0:
        return pd.DataFrame()
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
        "actual_camera_drop_fraction",
        "frames_with_all_camera_eligible_rows_lost",
        "mean_frame_camera_drop_fraction",
        "mean_drop_probability_across_frames",
    ]
    metrics = [m for m in metrics if m in per_run_df.columns]
    agg = per_run_df.groupby(["scene_name", "degradation_mode", "degradation_level_pct"])[metrics].agg(["mean", "std", "min", "max"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]).rstrip("_") for col in agg.columns]
    return agg


def main() -> None:
    apply_plot_style()
    args = parse_args()

    modes = parse_str_list_csv(args.degradation_modes)
    bad_modes = [m for m in modes if m not in SUPPORTED_MODES]
    if bad_modes:
        raise ValueError(f"Unsupported degradation modes: {bad_modes}. Supported: {sorted(SUPPORTED_MODES)}")

    levels = parse_int_list_csv(args.degradation_levels)
    seeds = parse_int_list_csv(args.seeds)
    scenes = parse_scene_name_list(args.scene_name, args.scene_names)

    if not scenes:
        raise ValueError("No scenes specified. Use --scene-names scene-1019,scene-0061,scene-0383 or --scene-name scene-0383.")

    all_per_run: List[pd.DataFrame] = []
    scene_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    print("batch_idm_planner_start")
    print(f"  scenes: {scenes}")
    print(f"  modes: {modes}")
    print(f"  levels: {levels}")
    print(f"  seeds: {seeds}")

    for scene in scenes:
        try:
            per_run, scene_summary_payload = run_scene(scene, args, modes, levels, seeds)
            all_per_run.append(per_run)
            scene_summaries.append(scene_summary_payload)
        except Exception as exc:
            failures.append({"scene_name": str(scene), "error": str(exc)})
            if args.skip_missing:
                print(f"[skip] {scene}: {exc}")
                continue
            raise

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    if all_per_run:
        combined = pd.concat(all_per_run, ignore_index=True)
        combined.to_csv(results_root / "all_scenes_per_run_metrics.csv", index=False)
        combined_agg = aggregate_combined_metrics(combined)
        combined_agg.to_csv(results_root / "all_scenes_aggregated_metrics.csv", index=False)
    else:
        combined = pd.DataFrame()
        combined_agg = pd.DataFrame()

    batch_payload = {
        "planner_type": "perception_only_idm_batch_three_scenes",
        "scenes_requested": scenes,
        "scenes_completed": [s["scene_name"] for s in scene_summaries],
        "failures": failures,
        "degradation_modes": modes,
        "degradation_levels": levels,
        "seeds": seeds,
        "per_scene_outputs_preserved": True,
        "combined_outputs": {
            "all_scenes_per_run_metrics_csv": str(results_root / "all_scenes_per_run_metrics.csv"),
            "all_scenes_aggregated_metrics_csv": str(results_root / "all_scenes_aggregated_metrics.csv"),
        },
        "scene_summaries": scene_summaries,
    }
    (results_root / "all_scenes_batch_summary.json").write_text(json.dumps(batch_payload, indent=2, allow_nan=True), encoding="utf-8")

    print("\nbatch_idm_planner_done")
    print(f"  completed: {[s['scene_name'] for s in scene_summaries]}")
    print(f"  failures: {failures}")
    print(f"  combined outputs: {results_root / 'all_scenes_per_run_metrics.csv'}")



if __name__ == "__main__":
    main()
