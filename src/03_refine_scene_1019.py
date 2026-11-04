#!/usr/bin/env python3
"""
Refine scene-1019 as the straight ego-approach pedestrian-crossing case.

Methodology for this scene
--------------------------
This scene is treated independently from the moderate-curvature scene-0061.
The intended refined behavior is:

    straight ego approach -> clean crossing -> full-scene smooth ego motion -> smooth stop -> pedestrians cross -> smooth reacceleration

Input:
  data/raw/scene-1019/

Output:
  data/refined/scene-1019/

This script keeps the same output contract used by the rest of the pipeline:
- scene_metadata.json
- scene_summary.json
- sample_timeline.csv
- ego_timeline.csv
- objects_timeline.csv
- agent_trajectories.csv
- planner_observations.csv
- sensor_metadata.json
- class_counts.json
- camera_detection_summary.json
- extraction_log.txt
- animations/scene_replay.gif

Implementation note
-------------------
To avoid duplicating thousands of lines of shared table/GIF/summary logic, this
script imports the already-tested base refinement module and patches only the
scene-specific methodology functions:
  1) straight-scene crossing-zone choice,
  2) target pedestrian selection,
  3) synthetic pedestrian crossing trajectories,
  4) ego stop-and-go profile.

The base module is loaded from 03_refine_scene_0061.py by default, because that
file already contains the shared output-writing logic. If it is not present, the
script falls back to 03_refine_scene_synthetic.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Scene-specific refinement profile
# =============================================================================

@dataclass
class RefinementProfile:
    # Scene identity.
    scene_name: str = "scene-1019"
    geometry_case: str = "straight_approach"
    selection_reason: str = (
        "Scene-specific straight-approach refinement: clean crossing, smooth stop, "
        "pedestrians cross, then smooth reacceleration."
    )

    # Crossing-zone design.
    # The straight case should not place the crossing too late, otherwise there
    # are too few remaining frames for a realistic reacceleration.
    default_crossing_phase: float = 0.55
    preferred_crossing_frame_fraction: float = 0.56  # kept for compatibility; crossing zone is not timing-shifted
    min_crossing_center_frame: int = 8
    min_post_crossing_frames: int = 10
    crossing_width_m: float = 6.5
    crossing_length_m: float = 11.0
    hold_buffer_m: float = 8.5
    zone_search_radius_m: float = 8.0
    heading_window_m: float = 18.0

    # Pedestrian design.
    crossing_duration_frames: int = 8
    max_target_pedestrians: int = 3
    ped_min_frames: int = 3
    add_tangent_drift: bool = False

    # Straight-road preference for selecting the crossing zone.
    prefer_low_heading_change: bool = True
    high_curvature_weight: float = 0.0

    # Ego smooth stop/restart design.
    stop_smooth_frames: int = 6
    reaccel_delay_frames: int = 0
    # Gentler than the previous actual-crossing version: preserve the same
    # crossing position and output contract, but reduce the early speed jump
    # after crossing_end_frame.
    reaccel_shape_power: float = 1.45
    reaccel_linear_blend: float = 0.08

    # Full-scene ego smoothing. These control the approach before the stop
    # and the restart after the pedestrians leave. The crossing-zone and
    # pedestrian placement are not moved by these parameters.
    approach_shape_power: float = 1.0
    approach_linear_blend: float = 0.0


def default_base_script() -> Path:
    """Prefer the scene-0061 shared refinement file, then fallback to the generic one."""
    p1 = SCRIPT_DIR / "03_refine_scene_0061.py"
    if p1.is_file():
        return p1
    return SCRIPT_DIR / "03_refine_scene_synthetic.py"


# =============================================================================
# Base module loading
# =============================================================================


def load_base_module(base_script: Path) -> Any:
    if not base_script.is_file():
        raise FileNotFoundError(
            f"Could not find base refinement script: {base_script}. "
            "Place this file beside 03_refine_scene_0061.py or pass --base-script."
        )
    spec = importlib.util.spec_from_file_location("base_refine_shared", str(base_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base refinement script from {base_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# =============================================================================
# Small geometry helpers
# =============================================================================


def safe_unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return vec / n


def smootherstep01(t: float) -> float:
    """Minimum-jerk-style easing: zero slope at start and end."""
    t = max(0.0, min(1.0, float(t)))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def local_heading_change_deg(geom: Any, arc_center: float, window_m: float) -> float:
    arc = np.asarray(geom.arc, dtype=float)
    tan = np.asarray(geom.tangent_xy, dtype=float)
    mask = (arc >= arc_center - window_m) & (arc <= arc_center + window_m)
    if int(mask.sum()) < 2:
        return 0.0
    yaw = np.unwrap(np.arctan2(tan[mask, 1], tan[mask, 0]))
    return float(np.degrees(np.nanmax(yaw) - np.nanmin(yaw)))


def local_tortuosity(geom: Any, arc_center: float, window_m: float) -> float:
    arc = np.asarray(geom.arc, dtype=float)
    xy = np.asarray(geom.xy, dtype=float)
    mask = (arc >= arc_center - window_m) & (arc <= arc_center + window_m)
    pts = xy[mask]
    if len(pts) < 2:
        return 1.0
    path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    if chord < 1e-6:
        return 1.0
    return float(path_len / chord)


def frame_from_arc(geom: Any, arc_m: float) -> int:
    return int(np.argmin(np.abs(np.asarray(geom.arc, dtype=float) - float(arc_m))))


# =============================================================================
# Straight-scene crossing-zone methodology
# =============================================================================


def estimate_crossing_arc_from_peds(base: Any, objects_df: pd.DataFrame, geom: Any, profile: RefinementProfile) -> Dict[str, Any]:
    """
    Pick the crossing-zone arc from actual scene evidence, without moving it earlier.

    This version intentionally does NOT shift the crossing zone for timing convenience.
    The previous 1019 script moved the synthetic crossing earlier to leave more
    post-crossing frames, but that placed pedestrians on the stop-line area rather
    than the actual crossing. Here, the crossing center stays tied to the best
    pedestrian/future-path evidence, so the pedestrians cross at the real crossing
    location seen in the scene/map context.
    """
    ped_df = objects_df[objects_df["category_name"].map(base.is_pedestrian_category)].copy()

    fallback_arc = float(np.clip(profile.default_crossing_phase, 0.10, 0.90) * geom.total_arc_m)
    fallback = {
        "arc_m": fallback_arc,
        "source": "fallback_straight_default_phase_actual_crossing_preserved",
        "ped_token": "",
        "score": -1e9,
        "min_future_path_distance_m": None,
        "heading_change_deg": local_heading_change_deg(geom, fallback_arc, profile.heading_window_m),
        "tortuosity": local_tortuosity(geom, fallback_arc, profile.heading_window_m),
        "frame_index": frame_from_arc(geom, fallback_arc),
    }
    if ped_df.empty:
        return fallback

    ped_df = base.annotate_with_ego_projection(ped_df, geom)
    has_fpd = "future_path_distance_m" in ped_df.columns
    rows: List[Dict[str, Any]] = []

    for tok, grp in ped_df.groupby("instance_token"):
        grp = grp.sort_values("frame_index").copy()
        if len(grp) < profile.ped_min_frames:
            continue

        if has_fpd:
            d_series = pd.to_numeric(grp["future_path_distance_m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if d_series.notna().any():
                best_idx = int(d_series.idxmin())
                min_d = float(d_series.loc[best_idx])
            else:
                abs_lat = grp["ego_signed_lateral_m"].abs()
                best_idx = int(abs_lat.idxmin())
                min_d = float(abs_lat.loc[best_idx])
        else:
            abs_lat = grp["ego_signed_lateral_m"].abs()
            best_idx = int(abs_lat.idxmin())
            min_d = float(abs_lat.loc[best_idx])

        best = grp.loc[best_idx]
        arc_m = float(best["ego_nearest_arc_m"])
        if not (0.10 * geom.total_arc_m <= arc_m <= 0.95 * geom.total_arc_m):
            continue

        hchg = local_heading_change_deg(geom, arc_m, profile.heading_window_m)
        tort = local_tortuosity(geom, arc_m, profile.heading_window_m)

        # Straight scene: prefer pedestrian evidence close to the ego path while
        # still favoring a locally straight segment. No timing penalty is used,
        # because timing should not move the physical crossing location.
        geometry_score = -1.8 * hchg - 25.0 * max(0.0, tort - 1.0)
        score = -22.0 * min_d + 0.45 * len(grp) + geometry_score

        rows.append(
            {
                "arc_m": arc_m,
                "source": "pedestrian_future_path_proxy_actual_crossing_preserved",
                "ped_token": str(tok),
                "score": float(score),
                "min_future_path_distance_m": min_d,
                "heading_change_deg": hchg,
                "tortuosity": tort,
                "frame_index": int(best.get("frame_index", frame_from_arc(geom, arc_m))),
            }
        )

    if not rows:
        return fallback
    return sorted(rows, key=lambda r: r["score"], reverse=True)[0]

def make_identify_main_crossing_zone(base: Any, profile: RefinementProfile):
    def identify_main_crossing_zone(objects_df: pd.DataFrame, geom: Any, n_frames: int) -> Dict[str, Any]:
        arc_pick = estimate_crossing_arc_from_peds(base, objects_df, geom, profile)

        # Preserve the actual crossing-zone arc. Do NOT move it earlier just to
        # create more post-crossing frames, because that can place the synthetic
        # pedestrians on a stop line instead of the real crossing.
        center_arc = float(np.clip(arc_pick["arc_m"], 1.0, max(1.0, geom.total_arc_m - 1.0)))
        center_frame = frame_from_arc(geom, center_arc)

        cx, cy, _ = base.sample_path_at_arc(geom, np.array([center_arc], dtype=float))
        tidx = int(np.argmin(np.abs(geom.arc - center_arc)))
        t = safe_unit(np.asarray(geom.tangent_xy[tidx], dtype=float))
        n = np.array([-t[1], t[0]], dtype=float)
        hold_arc = float(np.clip(center_arc - profile.hold_buffer_m, 0.5, max(1.0, geom.total_arc_m - 2.0)))

        return {
            "crossing_center_x": float(cx[0]),
            "crossing_center_y": float(cy[0]),
            "crossing_center_frame": int(np.clip(center_frame, 0, n_frames - 1)),
            "crossing_zone_arc_length_m": float(center_arc),
            "ego_path_tangent": [float(t[0]), float(t[1])],
            "crossing_direction": [float(n[0]), float(n[1])],
            "crossing_width_m": float(profile.crossing_width_m),
            "crossing_length_m": float(profile.crossing_length_m),
            "ego_hold_line_arc_length_m": hold_arc,
            "event_candidates": [arc_pick],
            "scenario_geometry_case": profile.geometry_case,
            "zone_selection_source": arc_pick.get("source", "unknown"),
            "zone_selection_ped_token": arc_pick.get("ped_token", ""),
            "zone_raw_center_frame": int(center_frame),
            "zone_timing_adjusted_for_post_crossing_motion": False,
            "zone_note": "Actual crossing zone preserved; ego motion is smoothed without shifting pedestrians to the stop line.",
            "zone_heading_change_deg": float(local_heading_change_deg(geom, center_arc, profile.heading_window_m)),
            "zone_tortuosity": float(local_tortuosity(geom, center_arc, profile.heading_window_m)),
        }

    return identify_main_crossing_zone


# =============================================================================
# Straight-scene pedestrian methodology
# =============================================================================


def make_select_target_crossing_group(base: Any, profile: RefinementProfile):
    def select_target_crossing_group(objects_df: pd.DataFrame, geom: Any, zone: Dict[str, Any]) -> Dict[str, Any]:
        ped_df = objects_df[objects_df["category_name"].map(base.is_pedestrian_category)].copy()
        if ped_df.empty:
            return {
                "target_crossing_pedestrian_tokens": [],
                "ignored_pedestrian_tokens": [],
                "target_crossing_group_selection_reason": "No pedestrian instances available in raw scene.",
            }

        ped_df = base.annotate_with_ego_projection(ped_df, geom)
        center_arc = float(zone["crossing_zone_arc_length_m"])
        length = float(zone["crossing_length_m"])
        center_xy = np.array([zone["crossing_center_x"], zone["crossing_center_y"]], dtype=float)
        rows: List[Tuple[str, float, int]] = []

        for tok, grp in ped_df.groupby("instance_token"):
            grp = grp.sort_values("frame_index")
            if len(grp) < profile.ped_min_frames:
                continue
            arc = grp["ego_nearest_arc_m"].to_numpy(dtype=float)
            lat = grp["ego_signed_lateral_m"].to_numpy(dtype=float)
            xy = grp[["obj_x", "obj_y"]].to_numpy(dtype=float)
            frames = grp["frame_index"].to_numpy(dtype=int)

            near_zone = np.abs(arc - center_arc) <= (0.85 * length + profile.zone_search_radius_m)
            if not np.any(near_zone):
                near_zone = np.ones(len(grp), dtype=bool)
            local_idx = np.where(near_zone)[0]

            d_center = np.linalg.norm(xy - center_xy[None, :], axis=1)
            abs_lat = np.abs(lat)
            best_local = local_idx[int(np.argmin(1.7 * abs_lat[local_idx] + 0.55 * d_center[local_idx]))]

            sign_change = False
            if len(lat) >= 2:
                finite = lat[np.isfinite(lat)]
                if len(finite) >= 2:
                    sign_change = bool(np.nanmin(finite) < -0.35 and np.nanmax(finite) > 0.35)

            # Lower is better.
            score = float(
                1.7 * abs_lat[best_local]
                + 0.55 * d_center[best_local]
                + 0.20 * abs(arc[best_local] - center_arc)
                - (3.0 if sign_change else 0.0)
                - 0.08 * len(grp)
            )
            rows.append((str(tok), score, int(frames[best_local])))

        rows.sort(key=lambda x: (x[1], x[2]))
        targets = [r[0] for r in rows[: profile.max_target_pedestrians]]
        all_peds = sorted(ped_df["instance_token"].astype(str).unique().tolist())
        ignored = [t for t in all_peds if t not in targets]

        return {
            "target_crossing_pedestrian_tokens": targets,
            "ignored_pedestrian_tokens": ignored,
            "target_crossing_group_selection_reason": (
                profile.selection_reason
                + " Target pedestrians were selected by proximity to the straight crossing zone, "
                + "local lateral offset, and future-path interaction evidence."
            ),
        }

    return select_target_crossing_group


def make_build_target_crossing_trajectories(base: Any, profile: RefinementProfile):
    def build_target_crossing_trajectories(
        objects_df: pd.DataFrame,
        target_tokens: List[str],
        zone: Dict[str, Any],
        n_frames: int,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        out = objects_df.copy()
        center = np.array([zone["crossing_center_x"], zone["crossing_center_y"]], dtype=float)
        t = safe_unit(np.array(zone["ego_path_tangent"], dtype=float))
        n = safe_unit(np.array(zone["crossing_direction"], dtype=float))
        width = float(zone["crossing_width_m"])
        intervals: Dict[str, Dict[str, int]] = {}
        base_center = int(zone["crossing_center_frame"])

        for i, tok in enumerate(target_tokens):
            mask = out["instance_token"].astype(str).eq(str(tok))
            grp = out[mask].sort_values("frame_index")
            if grp.empty:
                continue

            min_f = int(grp["frame_index"].min())
            max_f = int(grp["frame_index"].max())
            duration = max(5, int(profile.crossing_duration_frames + (i % 2)))

            # Keep the crossing centered around the selected zone and leave the
            # end of the clip for ego reacceleration.
            enter = int(np.clip(base_center - duration // 2 + i, min_f, max(min_f, max_f - 1)))
            exit_ = int(np.clip(enter + duration, enter + 1, max_f))

            tangent_offset = (i - 0.5 * max(0, len(target_tokens) - 1)) * 0.90
            side_padding = 0.5 * width + 1.7 + 0.25 * i
            p0 = center + tangent_offset * t - side_padding * n
            p1 = center + tangent_offset * t + (side_padding + 0.35) * n

            drift_vec = np.zeros(2, dtype=float)
            if profile.add_tangent_drift:
                drift_vec = 1.4 * t
            move = p1 - p0 + drift_vec
            ped_yaw = float(math.atan2(move[1], move[0]))

            frames = grp["frame_index"].to_numpy(dtype=int)
            idx = grp.index.to_numpy()
            new_xy: List[np.ndarray] = []
            for f in frames:
                if f <= enter:
                    alpha = 0.0
                elif f >= exit_:
                    alpha = 1.0
                else:
                    alpha = base.smoothstep01((f - enter) / max(1e-6, (exit_ - enter)))
                mid_drift = (4.0 * alpha * (1.0 - alpha)) * drift_vec if profile.add_tangent_drift else 0.0
                xy = (1.0 - alpha) * p0 + alpha * p1 + mid_drift
                new_xy.append(np.asarray(xy, dtype=float))

            new_xy_arr = np.vstack(new_xy)
            out.loc[idx, "obj_x"] = new_xy_arr[:, 0]
            out.loc[idx, "obj_y"] = new_xy_arr[:, 1]
            out.loc[idx, "obj_yaw_rad"] = ped_yaw
            out.loc[idx, "attribute_names"] = "pedestrian.moving"
            intervals[str(tok)] = {"ped_enter_frame": int(enter), "ped_exit_frame": int(exit_)}

        return out, intervals

    return build_target_crossing_trajectories


# =============================================================================
# Straight-scene ego stop-and-go methodology
# =============================================================================


def make_build_refined_ego_profile(base: Any, profile: RefinementProfile):
    def build_refined_ego_profile(
        ego_df: pd.DataFrame,
        geom: Any,
        crossing_start_frame: int,
        crossing_end_frame: int,
        hold_arc_m: float,
    ) -> pd.DataFrame:
        """
        Build a full-scene smooth ego trajectory for scene-1019.

        Important: this function only changes the ego motion along the existing
        route. It does NOT move the crossing zone and does NOT move the
        pedestrian crossing location.

        Design:
          1) From the first frame to the hold line, progress along the same ego
             path using a smooth minimum-jerk curve. This removes sudden speed
             changes before the stop.
          2) Stay stopped while pedestrians occupy the crossing.
          3) After the crossing, restart with a smooth minimum-jerk curve toward
             the original end of the scene. This removes the aggressive
             reacceleration jump while keeping the scene length unchanged.
        """
        out = ego_df.copy()
        n_frames = len(out)
        if n_frames < 2:
            return base.build_refined_ego_profile(ego_df, geom, crossing_start_frame, crossing_end_frame, hold_arc_m)

        orig_arc = out["cumulative_distance_m"].fillna(0.0).to_numpy(dtype=float)
        total_arc = float(max(orig_arc[-1], geom.total_arc_m))
        start_arc = float(max(0.0, orig_arc[0]))
        hold_arc = float(np.clip(hold_arc_m, start_arc + 1.0, total_arc - 2.0))
        s_wait = float(max(start_arc, hold_arc - 0.15))

        # Safety bounds: do not change crossing location, only the ego motion.
        crossing_start_frame = int(np.clip(crossing_start_frame, 1, n_frames - 2))
        crossing_end_frame = int(np.clip(crossing_end_frame, crossing_start_frame, n_frames - 2))
        restart_start = int(np.clip(crossing_end_frame + max(0, int(profile.reaccel_delay_frames)), crossing_end_frame, n_frames - 2))

        new_arc = np.zeros(n_frames, dtype=float)

        # -------------------------
        # 1) Smooth approach to stop
        # -------------------------
        # Use the whole available approach interval, rather than following noisy
        # raw frame-to-frame motion and then suddenly clamping near the stop.
        # The final derivative is zero, so the car naturally reaches the stop.
        approach_denom = max(1, crossing_start_frame)
        approach_power = max(0.60, float(profile.approach_shape_power))
        approach_blend = float(np.clip(profile.approach_linear_blend, 0.0, 0.35))

        for i in range(0, crossing_start_frame):
            tau = i / approach_denom
            mj = smootherstep01(tau) ** approach_power
            alpha = (1.0 - approach_blend) * mj + approach_blend * tau
            new_arc[i] = start_arc + alpha * (s_wait - start_arc)

        # -------------------------
        # 2) Stop during crossing
        # -------------------------
        for i in range(crossing_start_frame, crossing_end_frame + 1):
            new_arc[i] = s_wait

        # Optional extra restart delay if the user sets it from CLI.
        for i in range(crossing_end_frame + 1, restart_start + 1):
            if i < n_frames:
                new_arc[i] = s_wait

        # -------------------------
        # 3) Smooth reacceleration
        # -------------------------
        # Minimum-jerk at the start avoids the sharp speed jump after stopping.
        # A tiny linear blend can be used to prevent the car from looking frozen
        # too long in very short clips, but default stays very smooth.
        post_denom = max(1, n_frames - 1 - restart_start)
        reaccel_power = max(0.60, float(profile.reaccel_shape_power))
        reaccel_blend = float(np.clip(profile.reaccel_linear_blend, 0.0, 0.50))

        for i in range(restart_start + 1, n_frames):
            tau = (i - restart_start) / post_denom
            mj = smootherstep01(tau) ** reaccel_power
            alpha = (1.0 - reaccel_blend) * mj + reaccel_blend * tau
            new_arc[i] = s_wait + alpha * (total_arc - s_wait)

        # Numerical protection: monotonic route progress and exact original end.
        new_arc = np.maximum.accumulate(new_arc)
        new_arc[0] = start_arc
        new_arc[-1] = total_arc

        x, y, yaw = base.sample_path_at_arc(geom, new_arc)
        out["ego_x"] = x
        out["ego_y"] = y
        out["ego_yaw_rad"] = yaw
        out["ego_yaw_deg"] = np.degrees(yaw)

        step = np.concatenate([[0.0], np.diff(new_arc)])
        out["ego_step_distance_m"] = step
        out["cumulative_distance_m"] = new_arc

        dt = out["delta_t_s"].fillna(0.0).to_numpy(dtype=float)
        speed = np.divide(step, np.where(dt > 1e-6, dt, np.nan))
        accel = np.concatenate([[np.nan], np.divide(np.diff(speed), np.where(dt[1:] > 1e-6, dt[1:], np.nan))])
        yawdiff = np.array([np.nan] + [base.wrap_angle(float(yaw[i] - yaw[i - 1])) for i in range(1, n_frames)], dtype=float)
        yawrate = np.divide(yawdiff, np.where(dt > 1e-6, dt, np.nan))

        out["ego_speed_mps"] = speed
        out["ego_accel_est_mps2"] = accel
        out["ego_angular_rate_est_rps"] = yawrate

        if {"ego_qw", "ego_qx", "ego_qy", "ego_qz"}.issubset(out.columns):
            out["ego_qw"] = np.cos(yaw / 2.0)
            out["ego_qx"] = 0.0
            out["ego_qy"] = 0.0
            out["ego_qz"] = np.sin(yaw / 2.0)

        return out

    return build_refined_ego_profile


# =============================================================================
# Patching and metadata
# =============================================================================


def patch_base_module(base: Any, profile: RefinementProfile) -> None:
    base.identify_main_crossing_zone = make_identify_main_crossing_zone(base, profile)
    base.select_target_crossing_group = make_select_target_crossing_group(base, profile)
    base.build_target_crossing_trajectories = make_build_target_crossing_trajectories(base, profile)
    base.build_refined_ego_profile = make_build_refined_ego_profile(base, profile)


def add_profile_metadata(refined_root: Path, scene_name: str, profile: RefinementProfile, base_script: Path) -> None:
    scene_dir = refined_root / scene_name
    profile_payload = asdict(profile)
    profile_payload["base_refinement_script_used"] = str(base_script.name)
    profile_payload["methodology"] = (
        "straight road -> actual crossing preserved -> full-scene smooth ego motion -> smooth stop -> pedestrians cross -> smooth reacceleration"
    )
    (scene_dir / "scenario_refinement_profile.json").write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")

    summary_path = scene_dir / "scene_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["scenario_geometry_case"] = profile.geometry_case
        summary["scenario_refinement_methodology"] = profile_payload["methodology"]
        summary["refinement_profile"] = profile_payload
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    meta_path = scene_dir / "scene_metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["scenario_geometry_case"] = profile.geometry_case
        meta["scene_specific_refinement_script"] = Path(__file__).name
        meta["base_refinement_script_used"] = str(base_script.name)
        meta_path.write_text(json.dumps(meta, indent=2, allow_nan=True), encoding="utf-8")

    log_path = scene_dir / "extraction_log.txt"
    if log_path.is_file():
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\nscene_specific_profile:\n")
            for k, v in profile_payload.items():
                f.write(f"  {k}: {v}\n")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refine scene-1019 into the straight-approach crossing scenario."
    )
    p.add_argument("--scene-name", default="scene-1019", help="Scene name. This script is locked to scene-1019.")
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"), help="Raw input root.")
    p.add_argument("--refined-root", type=Path, default=Path("data/refined"), help="Refined output root.")
    p.add_argument("--map-dataroot", type=Path, default=SCRIPT_DIR / "v1.0-mini", help="nuScenes map dataroot, maps folder, or maps/expansion folder.")
    p.add_argument("--base-script", type=Path, default=default_base_script(), help="Shared refinement script to import for common output logic.")

    # Optional tuning without touching the code.
    p.add_argument("--crossing-width-m", type=float, default=None)
    p.add_argument("--crossing-length-m", type=float, default=None)
    p.add_argument("--hold-buffer-m", type=float, default=None)
    p.add_argument("--crossing-duration-frames", type=int, default=None)
    p.add_argument("--min-post-crossing-frames", type=int, default=None)
    p.add_argument("--preferred-crossing-frame-fraction", type=float, default=None)
    p.add_argument("--stop-smooth-frames", type=int, default=None)
    p.add_argument("--reaccel-delay-frames", type=int, default=None)
    p.add_argument("--reaccel-shape-power", type=float, default=None)
    p.add_argument("--reaccel-linear-blend", type=float, default=None)
    p.add_argument("--approach-shape-power", type=float, default=None)
    p.add_argument("--approach-linear-blend", type=float, default=None)
    return p.parse_args()


def apply_overrides(profile: RefinementProfile, args: argparse.Namespace) -> None:
    if args.crossing_width_m is not None:
        profile.crossing_width_m = float(args.crossing_width_m)
    if args.crossing_length_m is not None:
        profile.crossing_length_m = float(args.crossing_length_m)
    if args.hold_buffer_m is not None:
        profile.hold_buffer_m = float(args.hold_buffer_m)
    if args.crossing_duration_frames is not None:
        profile.crossing_duration_frames = int(args.crossing_duration_frames)
    if args.min_post_crossing_frames is not None:
        profile.min_post_crossing_frames = int(args.min_post_crossing_frames)
    if args.preferred_crossing_frame_fraction is not None:
        profile.preferred_crossing_frame_fraction = float(args.preferred_crossing_frame_fraction)
    if args.stop_smooth_frames is not None:
        profile.stop_smooth_frames = int(args.stop_smooth_frames)
    if args.reaccel_delay_frames is not None:
        profile.reaccel_delay_frames = int(args.reaccel_delay_frames)
    if args.reaccel_shape_power is not None:
        profile.reaccel_shape_power = float(args.reaccel_shape_power)
    if args.reaccel_linear_blend is not None:
        profile.reaccel_linear_blend = float(args.reaccel_linear_blend)
    if args.approach_shape_power is not None:
        profile.approach_shape_power = float(args.approach_shape_power)
    if args.approach_linear_blend is not None:
        profile.approach_linear_blend = float(args.approach_linear_blend)


def main() -> None:
    args = parse_args()
    scene_name = str(args.scene_name).strip()
    if scene_name != "scene-1019":
        raise ValueError(
            f"This scene-specific refinement script is locked to scene-1019, but got {scene_name!r}. "
            "Use the other scene-specific refinement scripts for scene-0061 or scene-0185."
        )

    profile = RefinementProfile(scene_name=scene_name)
    apply_overrides(profile, args)

    base_script = args.base_script.resolve()
    base = load_base_module(base_script)
    patch_base_module(base, profile)

    raw_root = args.raw_root.resolve()
    refined_root = args.refined_root.resolve()
    base.process_scene(raw_root=raw_root, refined_root=refined_root, scene_name=scene_name, map_dataroot=args.map_dataroot)
    add_profile_metadata(refined_root, scene_name, profile, base_script)

    print(f"[ok] refined {scene_name} using {profile.geometry_case} profile")
    print(f"     methodology: straight road -> actual crossing preserved -> full-scene smooth ego motion -> smooth stop -> pedestrians cross -> smooth reacceleration")
    print(f"     output: {(refined_root / scene_name).resolve()}")


if __name__ == "__main__":
    main()
