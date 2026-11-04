#!/usr/bin/env python3
"""
Refine scene-0383 by adding an interactive pedestrian-crossing event.

Purpose
-------
This scene-specific refinement keeps the same lean output contract used by the
existing refinement scripts. It does not create extra plot folders or custom
artifacts. The only behavioral change is:

    two pedestrians are simulated so they cross the selected crossing zone during
    the ego approach, with overlap near the critical arrival window.

Input:
  data/raw/scene-0383/

Output:
  data/refined/scene-0383/

Output contract:
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
To preserve the same architecture and output fields as the previous refinement
work, this script imports the shared table/GIF/summary logic from
03_refine_scene_0061.py and patches only the scene-specific functions:
  1) crossing-zone selection for scene-0383,
  2) selecting/simulating exactly two target pedestrians,
  3) moving those pedestrians slowly across the crossing before ego arrival,
  4) preserving the original ego trajectory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class RefinementProfile:
    scene_name: str = "scene-0383"
    geometry_case: str = "interactive_high_curvature_two_pedestrian_crossing"
    methodology: str = (
        "Preserve raw ego trajectory; simulate two realistic pedestrian crossing tracks "
        "so they occupy the crossing during ego approach and create a meaningful conflict window."
    )

    # Crossing-zone location along the ego route. For scene-0383 this places the
    # crossing on the upper-left crossing/turn area, not near the final vertical
    # end of the trajectory.
    default_crossing_phase: float = 0.38
    crossing_width_m: float = 6.8
    crossing_length_m: float = 11.0
    hold_buffer_m: float = 8.0

    # Pedestrian design: keep it intentionally small and simple.
    # Slower default: 12 frames at nuScenes keyframe spacing gives a smoother,
    # more realistic crossing than the earlier 7-frame motion.
    target_pedestrian_count: int = 2
    crossing_duration_frames: int = 14
    lead_frames_before_ego_arrival: int = 2
    pedestrian_side_padding_m: float = 0.9
    pedestrian_tangent_spacing_m: float = 1.1
    min_ped_frames: int = 5

    # Search / scoring.
    zone_search_radius_m: float = 18.0
    prefer_existing_pedestrians_near_zone: bool = True


def default_base_script() -> Path:
    p = SCRIPT_DIR / "03_refine_scene_0061.py"
    if p.is_file():
        return p
    return SCRIPT_DIR / "03_refine_scene_synthetic.py"


def load_base_module(base_script: Path) -> Any:
    if not base_script.is_file():
        raise FileNotFoundError(
            f"Could not find base refinement script: {base_script}. "
            "Place this file beside 03_refine_scene_0061.py or pass --base-script."
        )
    spec = importlib.util.spec_from_file_location("base_refine_shared_0383", str(base_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base refinement script from {base_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def safe_unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return vec / n


def smootherstep01(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def frame_from_arc(geom: Any, arc_m: float) -> int:
    return int(np.argmin(np.abs(np.asarray(geom.arc, dtype=float) - float(arc_m))))


def local_heading_change_deg(geom: Any, arc_center: float, window_m: float = 16.0) -> float:
    arc = np.asarray(geom.arc, dtype=float)
    tan = np.asarray(geom.tangent_xy, dtype=float)
    mask = (arc >= float(arc_center) - window_m) & (arc <= float(arc_center) + window_m)
    if int(mask.sum()) < 2:
        return 0.0
    yaw = np.unwrap(np.arctan2(tan[mask, 1], tan[mask, 0]))
    return float(np.degrees(np.nanmax(yaw) - np.nanmin(yaw)))


def estimate_crossing_arc_from_scene(base: Any, objects_df: pd.DataFrame, geom: Any, profile: RefinementProfile) -> Dict[str, Any]:
    """Pick a route point near existing pedestrians around the intended crossing zone."""
    fallback_arc = float(np.clip(profile.default_crossing_phase, 0.15, 0.85) * geom.total_arc_m)
    fallback = {
        "arc_m": fallback_arc,
        "source": "default_scene0383_route_phase",
        "ped_token": "",
        "score": 0.0,
        "heading_change_deg": local_heading_change_deg(geom, fallback_arc),
        "frame_index": frame_from_arc(geom, fallback_arc),
    }

    if not profile.prefer_existing_pedestrians_near_zone:
        return fallback

    ped_df = objects_df[objects_df["category_name"].map(base.is_pedestrian_category)].copy()
    if ped_df.empty:
        return fallback

    ped_df = base.annotate_with_ego_projection(ped_df, geom)
    rows: List[Dict[str, Any]] = []
    desired = fallback_arc
    for tok, grp in ped_df.groupby("instance_token"):
        grp = grp.sort_values("frame_index")
        if len(grp) < profile.min_ped_frames:
            continue
        arc = pd.to_numeric(grp["ego_nearest_arc_m"], errors="coerce").to_numpy(dtype=float)
        lat = pd.to_numeric(grp["ego_signed_lateral_m"], errors="coerce").to_numpy(dtype=float)
        frames = pd.to_numeric(grp["frame_index"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(arc) & np.isfinite(lat)
        if not finite.any():
            continue
        # Prefer pedestrians already near the upper-left crossing zone, but keep
        # the zone anchored to the route rather than to a pedestrian outlier.
        j = int(np.nanargmin(np.abs(arc - desired) + 0.20 * np.abs(lat)))
        candidate_arc = float(np.clip(0.70 * desired + 0.30 * arc[j], 0.15 * geom.total_arc_m, 0.85 * geom.total_arc_m))
        score = -abs(candidate_arc - desired) - 0.25 * abs(float(lat[j])) + 0.05 * len(grp)
        rows.append(
            {
                "arc_m": candidate_arc,
                "source": "pedestrian_supported_scene0383_crossing_zone",
                "ped_token": str(tok),
                "score": float(score),
                "heading_change_deg": local_heading_change_deg(geom, candidate_arc),
                "frame_index": int(frames[j]) if np.isfinite(frames[j]) else frame_from_arc(geom, candidate_arc),
            }
        )

    if not rows:
        return fallback
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[0]


def make_identify_main_crossing_zone(base: Any, profile: RefinementProfile):
    def identify_main_crossing_zone(objects_df: pd.DataFrame, geom: Any, n_frames: int) -> Dict[str, Any]:
        arc_pick = estimate_crossing_arc_from_scene(base, objects_df, geom, profile)
        center_arc = float(np.clip(arc_pick["arc_m"], 1.0, max(1.0, geom.total_arc_m - 1.0)))
        center_frame = frame_from_arc(geom, center_arc)
        cx, cy, _yaw = base.sample_path_at_arc(geom, np.array([center_arc], dtype=float))
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
            "ego_hold_line_arc_length_m": float(hold_arc),
            "event_candidates": [arc_pick],
            "scenario_geometry_case": profile.geometry_case,
            "zone_selection_source": arc_pick.get("source", "unknown"),
            "zone_selection_ped_token": arc_pick.get("ped_token", ""),
            "zone_note": "Scene-0383 crossing zone used only to stage two pedestrians before ego arrival; ego motion is preserved.",
            "zone_heading_change_deg": float(local_heading_change_deg(geom, center_arc)),
        }

    return identify_main_crossing_zone


def crossing_duration_for_scene(profile: RefinementProfile, n_frames: int) -> int:
    desired = int(profile.crossing_duration_frames)
    max_allowed = max(8, min(desired, max(8, n_frames - 2)))
    return int(max(8, max_allowed))


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
        n_frames = int(objects_df["frame_index"].max()) + 1 if len(objects_df) else 0
        ego_crossing_frame = int(zone["crossing_center_frame"])
        crossing_duration = crossing_duration_for_scene(profile, n_frames)
        enter = max(0, ego_crossing_frame - crossing_duration + 1)
        exit_ = min(max(0, n_frames - 1), ego_crossing_frame + 1)
        center_arc = float(zone["crossing_zone_arc_length_m"])
        center = np.array([zone["crossing_center_x"], zone["crossing_center_y"]], dtype=float)
        rows: List[Tuple[str, float, int, int]] = []

        for tok, grp in ped_df.groupby("instance_token"):
            grp = grp.sort_values("frame_index")
            if len(grp) < profile.min_ped_frames:
                continue
            min_f = int(grp["frame_index"].min())
            max_f = int(grp["frame_index"].max())
            if min_f > enter or max_f < exit_:
                # We need the pedestrian to exist during the planned crossing.
                continue
            arc = pd.to_numeric(grp["ego_nearest_arc_m"], errors="coerce").to_numpy(dtype=float)
            lat = pd.to_numeric(grp["ego_signed_lateral_m"], errors="coerce").to_numpy(dtype=float)
            xy = grp[["obj_x", "obj_y"]].to_numpy(dtype=float)
            d_center = np.linalg.norm(xy - center[None, :], axis=1)
            valid = np.isfinite(arc) & np.isfinite(lat) & np.isfinite(d_center)
            if not valid.any():
                continue
            score_arr = 0.85 * np.abs(arc - center_arc) + 0.35 * np.abs(lat) + 0.30 * d_center
            score = float(np.nanmin(score_arr[valid])) - 0.04 * len(grp)
            rows.append((str(tok), score, min_f, max_f))

        # Fallback: if the exact interval is too strict, choose the longest peds.
        if len(rows) < profile.target_pedestrian_count:
            existing = {r[0] for r in rows}
            for tok, grp in ped_df.groupby("instance_token"):
                if str(tok) in existing or len(grp) < profile.min_ped_frames:
                    continue
                min_f = int(grp["frame_index"].min())
                max_f = int(grp["frame_index"].max())
                overlap = max(0, min(max_f, exit_) - max(min_f, enter) + 1)
                if overlap <= 0:
                    continue
                score = -2.0 * overlap - 0.03 * len(grp)
                rows.append((str(tok), score, min_f, max_f))

        rows.sort(key=lambda r: (r[1], r[2]))
        targets = [r[0] for r in rows[: int(profile.target_pedestrian_count)]]
        if len(targets) != int(profile.target_pedestrian_count):
            raise RuntimeError(
                f"Scene-0383 refinement requires exactly {profile.target_pedestrian_count} target pedestrians, "
                f"but found {len(targets)} eligible tokens."
            )
        all_peds = sorted(ped_df["instance_token"].astype(str).unique().tolist())
        ignored = [t for t in all_peds if t not in targets]

        return {
            "target_crossing_pedestrian_tokens": targets,
            "ignored_pedestrian_tokens": ignored,
            "target_crossing_group_selection_reason": (
                "Selected two pedestrian instances available during the ego-approach crossing window. "
                "They are synthetically moved across the crossing zone with overlap near ego arrival."
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
        ego_crossing_frame = int(zone["crossing_center_frame"])
        crossing_duration = crossing_duration_for_scene(profile, n_frames)

        intervals: Dict[str, Dict[str, int]] = {}
        targets = list(target_tokens)[: int(profile.target_pedestrian_count)]
        for i, tok in enumerate(targets):
            mask = out["instance_token"].astype(str).eq(str(tok))
            grp = out[mask].sort_values("frame_index")
            if grp.empty:
                continue

            min_f = int(grp["frame_index"].min())
            max_f = int(grp["frame_index"].max())
            # Pedestrian 0 clears slightly before ego crossing; pedestrian 1 is still
            # near/inside the crossing as ego arrives.
            target_end = ego_crossing_frame - 1 if i == 0 else ego_crossing_frame + 1
            exit_ = int(np.clip(target_end, min_f + 2, max_f))
            enter = int(exit_ - crossing_duration + 1)
            if enter < min_f:
                enter = min_f
                exit_ = min(max_f, enter + crossing_duration - 1)
            if exit_ - enter + 1 < 8:
                exit_ = min(max_f, enter + 7)
                if exit_ - enter + 1 < 8:
                    enter = max(min_f, exit_ - 7)

            tangent_offset = (i - 0.5 * max(0, len(targets) - 1)) * float(profile.pedestrian_tangent_spacing_m)
            side_padding = 0.5 * width + float(profile.pedestrian_side_padding_m) + 0.15 * i

            # Pedestrian 0 crosses from negative to positive side; pedestrian 1
            # crosses in the opposite direction. The crossing interval is now
            # longer and the side padding is smaller, so the motion looks like
            # walking instead of running.
            direction_sign = -1.0 if i % 2 == 0 else 1.0
            p0 = center + tangent_offset * t + direction_sign * side_padding * n
            p1 = center + tangent_offset * t - direction_sign * (side_padding + 0.35) * n
            move = p1 - p0
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
                    alpha = smootherstep01((f - enter) / max(1e-6, (exit_ - enter)))
                xy = (1.0 - alpha) * p0 + alpha * p1
                new_xy.append(np.asarray(xy, dtype=float))

            arr = np.vstack(new_xy)
            out.loc[idx, "obj_x"] = arr[:, 0]
            out.loc[idx, "obj_y"] = arr[:, 1]
            out.loc[idx, "obj_yaw_rad"] = ped_yaw
            out.loc[idx, "attribute_names"] = "pedestrian.moving"
            out.loc[idx, "category_name"] = "human.pedestrian.adult"
            if "coarse_class" in out.columns:
                out.loc[idx, "coarse_class"] = "pedestrian"
            if "is_pedestrian" in out.columns:
                out.loc[idx, "is_pedestrian"] = True
            if "camera_degradable_flag" in out.columns:
                out.loc[idx, "camera_degradable_flag"] = True
            if "degradation_include_flag" in out.columns:
                out.loc[idx, "degradation_include_flag"] = True
            intervals[str(tok)] = {"ped_enter_frame": int(enter), "ped_exit_frame": int(exit_)}

        return out, intervals

    return build_target_crossing_trajectories


def make_build_refined_ego_profile(base: Any, profile: RefinementProfile):
    def build_refined_ego_profile(
        ego_df: pd.DataFrame,
        geom: Any,
        crossing_start_frame: int,
        crossing_end_frame: int,
        hold_arc_m: float,
    ) -> pd.DataFrame:
        """Preserve the raw ego trajectory; only pedestrian motion is refined."""
        out = ego_df.copy()
        # Ensure derived ego kinematic fields remain internally consistent even
        # if the raw input has minor formatting differences.
        if "cumulative_distance_m" not in out.columns or out["cumulative_distance_m"].isna().all():
            xy = out[["ego_x", "ego_y"]].to_numpy(dtype=float)
            step = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1)])
            out["ego_step_distance_m"] = step
            out["cumulative_distance_m"] = np.cumsum(step)
        return out

    return build_refined_ego_profile


def patch_base_module(base: Any, profile: RefinementProfile) -> None:
    base.identify_main_crossing_zone = make_identify_main_crossing_zone(base, profile)
    base.select_target_crossing_group = make_select_target_crossing_group(base, profile)
    base.build_target_crossing_trajectories = make_build_target_crossing_trajectories(base, profile)
    base.build_refined_ego_profile = make_build_refined_ego_profile(base, profile)


def add_profile_to_existing_outputs(refined_root: Path, scene_name: str, profile: RefinementProfile, base_script: Path) -> None:
    scene_dir = refined_root / scene_name
    profile_payload = asdict(profile)
    profile_payload["base_refinement_script_used"] = str(base_script.name)

    summary_path = scene_dir / "scene_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["scenario_geometry_case"] = profile.geometry_case
        summary["scenario_refinement_methodology"] = profile.methodology
        summary["refinement_profile"] = profile_payload
        summary["ego_trajectory_preserved"] = True
        summary["pedestrian_refinement_only"] = True
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    meta_path = scene_dir / "scene_metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["scenario_geometry_case"] = profile.geometry_case
        meta["scene_specific_refinement_script"] = Path(__file__).name
        meta["base_refinement_script_used"] = str(base_script.name)
        meta["ego_trajectory_preserved"] = True
        meta_path.write_text(json.dumps(meta, indent=2, allow_nan=True), encoding="utf-8")

    log_path = scene_dir / "extraction_log.txt"
    if log_path.is_file():
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\nscene_specific_profile:\n")
            for k, v in profile_payload.items():
                f.write(f"  {k}: {v}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refine scene-0383 by moving two pedestrians to cross during ego approach."
    )
    p.add_argument("--scene-name", default="scene-0383", help="Scene name. This script is locked to scene-0383.")
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"), help="Raw input root.")
    p.add_argument("--refined-root", type=Path, default=Path("data/refined"), help="Refined output root.")
    p.add_argument("--map-dataroot", type=Path, default=SCRIPT_DIR / "v1.0-mini", help="nuScenes map dataroot, maps folder, or maps/expansion folder.")
    p.add_argument("--base-script", type=Path, default=default_base_script(), help="Shared refinement script to import for common output logic.")

    # Tuning knobs, but defaults should be enough for scene-0383.
    p.add_argument("--crossing-phase", type=float, default=None, help="Route fraction for crossing zone, default 0.38.")
    p.add_argument("--target-pedestrian-count", type=int, default=None, help="Default 2.")
    p.add_argument("--crossing-duration-frames", type=int, default=None, help="Default 14. Larger values make the synthetic pedestrian crossing slower.")
    p.add_argument("--lead-frames-before-ego-arrival", type=int, default=None, help="Reserved compatibility knob; crossing now anchors to ego crossing frame.")
    p.add_argument("--crossing-width-m", type=float, default=None)
    p.add_argument("--crossing-length-m", type=float, default=None)
    return p.parse_args()


def apply_overrides(profile: RefinementProfile, args: argparse.Namespace) -> None:
    if args.crossing_phase is not None:
        profile.default_crossing_phase = float(args.crossing_phase)
    if args.target_pedestrian_count is not None:
        profile.target_pedestrian_count = int(args.target_pedestrian_count)
    if args.crossing_duration_frames is not None:
        profile.crossing_duration_frames = int(args.crossing_duration_frames)
    if args.lead_frames_before_ego_arrival is not None:
        profile.lead_frames_before_ego_arrival = int(args.lead_frames_before_ego_arrival)
    if args.crossing_width_m is not None:
        profile.crossing_width_m = float(args.crossing_width_m)
    if args.crossing_length_m is not None:
        profile.crossing_length_m = float(args.crossing_length_m)


def main() -> None:
    args = parse_args()
    scene_name = str(args.scene_name).strip()
    if scene_name != "scene-0383":
        raise ValueError(
            f"This scene-specific refinement script is locked to scene-0383, but got {scene_name!r}."
        )

    profile = RefinementProfile(scene_name=scene_name)
    apply_overrides(profile, args)

    base_script = args.base_script.resolve()
    base = load_base_module(base_script)
    patch_base_module(base, profile)

    raw_root = args.raw_root.resolve()
    refined_root = args.refined_root.resolve()
    base.process_scene(
        raw_root=raw_root,
        refined_root=refined_root,
        scene_name=scene_name,
        map_dataroot=args.map_dataroot,
    )
    add_profile_to_existing_outputs(refined_root, scene_name, profile, base_script)

    scene_dir = refined_root / scene_name
    summary = json.loads((scene_dir / "scene_summary.json").read_text(encoding="utf-8"))
    objects = pd.read_csv(scene_dir / "objects_timeline.csv")
    target_tokens = [str(t) for t in summary.get("target_crossing_pedestrian_tokens", [])]
    synthetic = objects[objects["instance_token"].astype(str).isin(target_tokens)].copy()
    ego_crossing_frame = int(summary.get("crossing_center_frame", -1))
    crossing_start = int(summary.get("crossing_start_frame", -1))
    crossing_end = int(summary.get("crossing_end_frame", -1))
    approach_start = max(0, ego_crossing_frame - 3)
    approach_end = ego_crossing_frame + 2
    per_ped_ranges: List[str] = []
    overlap_any = False
    for tok in target_tokens:
        g = synthetic[synthetic["instance_token"].astype(str).eq(tok)]
        if len(g):
            fmin = int(g["frame_index"].min())
            fmax = int(g["frame_index"].max())
            per_ped_ranges.append(f"{tok}: {fmin}-{fmax}")
            overlap_any = overlap_any or not (fmax < approach_start or fmin > approach_end)
        else:
            per_ped_ranges.append(f"{tok}: missing")

    print(f"[ok] refined {scene_name} using {profile.geometry_case} profile")
    print("     ego trajectory preserved; two pedestrians cross during ego approach")
    print(f"     output: {(refined_root / scene_name).resolve()}")
    print("validation:")
    print(f"  ego_crossing_frame: {ego_crossing_frame}")
    print(f"  crossing_start_frame: {crossing_start}")
    print(f"  crossing_end_frame: {crossing_end}")
    print(f"  synthetic_pedestrian_rows: {len(synthetic)}")
    print("  synthetic_pedestrian_frame_ranges:")
    for line in per_ped_ranges:
        print(f"    {line}")
    print(f"  pedestrians_overlap_ego_approach_window: {bool(overlap_any)}")


if __name__ == "__main__":
    main()
