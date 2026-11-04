#!/usr/bin/env python3
"""
Refine scene-0061 raw nuScenes package into the synthetic crossing scenario.

Input:  data/raw/scene-0061/
Output: data/refined/scene-0061/

This refined stage keeps the same lean output structure as the updated raw extraction stage:
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

Per-frame JSON files and saved frame/plot PNGs are intentionally not generated here.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

FORWARD_CORRIDOR_LATERAL_M = 3.0
FUTURE_PATH_DISTANCE_THRESHOLD_M = 2.5
FUTURE_PATH_LOOKAHEAD_FRAMES = 8
MAX_FILTER_DISTANCE_M = 50.0
MAX_PEDESTRIAN_DISTANCE_M = 40.0
MAX_VEHICLE_DISTANCE_M = 60.0
MAX_STATIC_DISTANCE_M = 35.0
CONFLICT_PED_SCORE_THRESHOLD = 35.0
IN_PATH_LATERAL_THRESHOLD_M = 3.0

LOW_SUPPORT_THRESHOLD = 2
ALLOW_ZERO_POINT_VISIBLE_PEDESTRIANS = True


def normalize_nuscenes_dataroot_for_maps(p: Path) -> Path:
    p = p.resolve()
    if (p / "maps" / "expansion").is_dir():
        return p
    if p.name == "maps" and (p / "expansion").is_dir():
        return p.parent
    if p.name == "expansion" and p.parent.name == "maps":
        return p.parent.parent
    return p


def resolve_map_dataroot(explicit: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(normalize_nuscenes_dataroot_for_maps(explicit))
    env = os.environ.get("NUSCENES_DATAROOT")
    if env:
        candidates.append(normalize_nuscenes_dataroot_for_maps(Path(env)))
    candidates.append(normalize_nuscenes_dataroot_for_maps(SCRIPT_DIR / "v1.0-mini"))
    seen: set = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if (p / "maps" / "expansion").is_dir():
            return p
    return None


def patch_box_from_xy_limits(xlim: Tuple[float, float], ylim: Tuple[float, float]) -> Tuple[float, float, float, float]:
    xmin, xmax = xlim
    ymin, ymax = ylim
    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)
    width = xmax - xmin
    height = ymax - ymin
    return (xc, yc, height, width)


def _patch_matplotlib_style_for_nuscenes_map_import() -> None:
    import matplotlib.pyplot as plt
    if getattr(plt, "_nuscenes_map_style_patched", False):
        return
    _orig_use = plt.style.use

    def _use(style: Any) -> None:
        try:
            _orig_use(style)
        except OSError:
            if style == "seaborn-whitegrid":
                for alt in ("seaborn-v0_8-whitegrid", "seaborn-v0_8", "default"):
                    try:
                        _orig_use(alt)
                        return
                    except OSError:
                        continue
            raise

    plt.style.use = _use  # type: ignore[method-assign]
    plt._nuscenes_map_style_patched = True  # type: ignore[attr-defined]


def load_nuscenes_map_or_none(map_dataroot: Optional[Path], map_name: Optional[str]) -> Any:
    if map_dataroot is None or not map_name:
        return None
    try:
        _patch_matplotlib_style_for_nuscenes_map_import()
        from nuscenes.map_expansion.map_api import NuScenesMap
        return NuScenesMap(str(map_dataroot), map_name=map_name)
    except Exception as exc:
        warnings.warn(f"NuScenesMap could not be loaded (dataroot={map_dataroot}, map_name={map_name}): {exc}")
        return None


def draw_map_expansion_layers(
    ax: Any,
    nmap: Any,
    patch_box: Tuple[float, float, float, float],
    patch_angle_deg: float = 0.0,
    zorder: float = 0.0,
) -> List[Any]:
    from matplotlib.patches import Patch, Polygon as MplPolygon

    explorer = nmap.explorer
    colors = explorer.color_map
    patch = explorer.get_patch_coord(patch_box, patch_angle_deg)

    def add_polygon_like(geom: Any, layer_name: str) -> None:
        if geom.is_empty:
            return
        alpha = 0.52 if layer_name == "ped_crossing" else 0.62
        fc = colors[layer_name]

        def patch_one(poly: Any) -> None:
            if poly.is_empty or poly.area <= 1e-12:
                return
            try:
                x, y = poly.exterior.xy
                xy = np.column_stack([x, y])
                if xy.shape[0] < 3:
                    return
                ax.add_patch(
                    MplPolygon(xy, closed=True, facecolor=fc, edgecolor="0.2", lw=0.65, alpha=alpha, zorder=zorder)
                )
            except Exception:
                return

        if geom.geom_type == "Polygon":
            patch_one(geom)
        elif geom.geom_type == "MultiPolygon":
            for p in geom.geoms:
                patch_one(p)
        elif geom.geom_type == "GeometryCollection":
            for g in geom.geoms:
                if g.geom_type in ("Polygon", "MultiPolygon"):
                    add_polygon_like(g, layer_name)

    for layer_name in ("ped_crossing", "stop_line"):
        for record in getattr(nmap, layer_name):
            poly = nmap.extract_polygon(record["polygon_token"])
            if not poly.is_valid:
                continue
            inter = poly.intersection(patch)
            if inter.is_empty:
                continue
            if inter.geom_type not in ("Polygon", "MultiPolygon", "GeometryCollection"):
                continue
            add_polygon_like(inter, layer_name)

    return [
        Patch(facecolor=colors["ped_crossing"], edgecolor="0.25", linewidth=0.5, alpha=0.52, label="Map: ped crossing"),
        Patch(facecolor=colors["stop_line"], edgecolor="0.25", linewidth=0.5, alpha=0.62, label="Map: stop line"),
    ]


def wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def smoothstep01(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def safe_unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return vec / n


@dataclass
class EgoPathGeometry:
    xy: np.ndarray
    arc: np.ndarray
    tangent_xy: np.ndarray
    total_arc_m: float


def build_ego_path_geometry(ego_df: pd.DataFrame) -> EgoPathGeometry:
    xy = ego_df[["ego_x", "ego_y"]].to_numpy(dtype=float)
    if len(xy) < 2:
        raise ValueError("Need at least 2 ego frames.")
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    tangent = np.zeros_like(xy)
    tangent[0] = safe_unit(xy[1] - xy[0])
    tangent[-1] = safe_unit(xy[-1] - xy[-2])
    for i in range(1, len(xy) - 1):
        tangent[i] = safe_unit(xy[i + 1] - xy[i - 1])
    return EgoPathGeometry(xy=xy, arc=arc, tangent_xy=tangent, total_arc_m=float(arc[-1]))


def sample_path_at_arc(geom: EgoPathGeometry, arc_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arc_q = np.clip(arc_q, geom.arc[0], geom.arc[-1])
    x = np.interp(arc_q, geom.arc, geom.xy[:, 0])
    y = np.interp(arc_q, geom.arc, geom.xy[:, 1])
    tx = np.interp(arc_q, geom.arc, geom.tangent_xy[:, 0])
    ty = np.interp(arc_q, geom.arc, geom.tangent_xy[:, 1])
    yaw = np.arctan2(ty, tx)
    return x, y, yaw


def annotate_with_ego_projection(df: pd.DataFrame, geom: EgoPathGeometry) -> pd.DataFrame:
    out = df.copy()
    obj_xy = out[["obj_x", "obj_y"]].to_numpy(dtype=float)
    ego_xy = geom.xy
    nearest_idx = np.argmin(((obj_xy[:, None, :] - ego_xy[None, :, :]) ** 2).sum(axis=2), axis=1)
    t = geom.tangent_xy[nearest_idx]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    rel = obj_xy - ego_xy[nearest_idx]
    signed_lat = np.einsum("ij,ij->i", rel, n)
    along = np.einsum("ij,ij->i", rel, t)
    out["ego_nearest_index"] = nearest_idx
    out["ego_nearest_arc_m"] = geom.arc[nearest_idx]
    out["ego_signed_lateral_m"] = signed_lat
    out["ego_along_offset_m"] = along
    return out


def is_pedestrian_category(cat: str) -> bool:
    return isinstance(cat, str) and cat.startswith("human.pedestrian")


def identify_main_crossing_zone(objects_df: pd.DataFrame, geom: EgoPathGeometry, n_frames: int) -> Dict[str, Any]:
    ped_df = objects_df[objects_df["category_name"].map(is_pedestrian_category)].copy()
    ped_df = annotate_with_ego_projection(ped_df, geom)

    crossing_events: List[Dict[str, Any]] = []
    min_arc = 0.2 * geom.total_arc_m
    max_arc = 0.95 * geom.total_arc_m
    for token, grp in ped_df.groupby("instance_token"):
        grp = grp.sort_values("frame_index")
        lat = grp["ego_signed_lateral_m"].to_numpy(dtype=float)
        arc = grp["ego_nearest_arc_m"].to_numpy(dtype=float)
        frames = grp["frame_index"].to_numpy(dtype=int)
        xs = grp["obj_x"].to_numpy(dtype=float)
        ys = grp["obj_y"].to_numpy(dtype=float)
        for i in range(1, len(grp)):
            if not (min_arc <= arc[i] <= max_arc):
                continue
            crossed = (lat[i - 1] == 0.0) or (lat[i] == 0.0) or ((lat[i - 1] < 0.0) != (lat[i] < 0.0))
            near = min(abs(lat[i - 1]), abs(lat[i])) <= 3.5
            if crossed and near:
                j = i if abs(lat[i]) <= abs(lat[i - 1]) else i - 1
                crossing_events.append(
                    {
                        "instance_token": token,
                        "frame_index": int(frames[j]),
                        "arc_m": float(arc[j]),
                        "obj_x": float(xs[j]),
                        "obj_y": float(ys[j]),
                    }
                )
                break

    if not crossing_events:
        mid_arc = 0.62 * geom.total_arc_m
        cx, cy, _ = sample_path_at_arc(geom, np.array([mid_arc], dtype=float))
        tidx = int(np.argmin(np.abs(geom.arc - mid_arc)))
        t = safe_unit(geom.tangent_xy[tidx])
        n = np.array([-t[1], t[0]], dtype=float)
        return {
            "crossing_center_x": float(cx[0]),
            "crossing_center_y": float(cy[0]),
            "crossing_center_frame": int(np.clip(round(0.62 * (n_frames - 1)), 0, n_frames - 1)),
            "crossing_zone_arc_length_m": float(mid_arc),
            "ego_path_tangent": [float(t[0]), float(t[1])],
            "crossing_direction": [float(n[0]), float(n[1])],
            "crossing_width_m": 6.5,
            "crossing_length_m": 10.0,
            "ego_hold_line_arc_length_m": float(max(1.0, mid_arc - 8.0)),
            "event_candidates": [],
        }

    crossing_events = sorted(crossing_events, key=lambda x: x["arc_m"])
    clusters: List[List[Dict[str, Any]]] = []
    for e in crossing_events:
        if not clusters or abs(e["arc_m"] - clusters[-1][-1]["arc_m"]) > 6.0:
            clusters.append([e])
        else:
            clusters[-1].append(e)

    best = None
    best_score = -1e9
    for c in clusters:
        cnt = len({x["instance_token"] for x in c})
        mean_arc = float(np.mean([x["arc_m"] for x in c]))
        phase = mean_arc / max(1e-6, geom.total_arc_m)
        score = cnt + 0.2 * phase - abs(phase - 0.62)
        if score > best_score:
            best_score = score
            best = c
    assert best is not None

    center_arc = float(np.mean([e["arc_m"] for e in best]))
    center_frame = int(np.argmin(np.abs(geom.arc - center_arc)))
    cx, cy, _ = sample_path_at_arc(geom, np.array([center_arc], dtype=float))
    tidx = int(np.argmin(np.abs(geom.arc - center_arc)))
    t = safe_unit(geom.tangent_xy[tidx])
    n = np.array([-t[1], t[0]], dtype=float)

    ped_pts = np.array([[e["obj_x"], e["obj_y"]] for e in best], dtype=float)
    rel = ped_pts - np.array([cx[0], cy[0]], dtype=float)
    lat_proj = rel @ n
    along_proj = rel @ t
    width = float(np.clip(np.ptp(lat_proj) + 3.0, 5.0, 10.0))
    length = float(np.clip(np.ptp(along_proj) + 8.0, 8.0, 16.0))
    hold_arc = float(np.clip(center_arc - 0.5 * length - 3.0, 1.0, geom.total_arc_m - 2.0))

    return {
        "crossing_center_x": float(cx[0]),
        "crossing_center_y": float(cy[0]),
        "crossing_center_frame": int(np.clip(center_frame, 0, n_frames - 1)),
        "crossing_zone_arc_length_m": center_arc,
        "ego_path_tangent": [float(t[0]), float(t[1])],
        "crossing_direction": [float(n[0]), float(n[1])],
        "crossing_width_m": width,
        "crossing_length_m": length,
        "ego_hold_line_arc_length_m": hold_arc,
        "event_candidates": best,
    }


def select_target_crossing_group(objects_df: pd.DataFrame, geom: EgoPathGeometry, zone: Dict[str, Any]) -> Dict[str, Any]:
    ped_df = objects_df[objects_df["category_name"].map(is_pedestrian_category)].copy()
    ped_df = annotate_with_ego_projection(ped_df, geom)
    center_arc = float(zone["crossing_zone_arc_length_m"])
    width = float(zone["crossing_width_m"])
    length = float(zone["crossing_length_m"])

    candidates: List[Tuple[str, float, int]] = []
    for token, grp in ped_df.groupby("instance_token"):
        grp = grp.sort_values("frame_index")
        arc = grp["ego_nearest_arc_m"].to_numpy(dtype=float)
        lat = grp["ego_signed_lateral_m"].to_numpy(dtype=float)
        frames = grp["frame_index"].to_numpy(dtype=int)
        sign_change_idx = []
        for i in range(1, len(grp)):
            crossed = (lat[i - 1] == 0.0) or (lat[i] == 0.0) or ((lat[i - 1] < 0.0) != (lat[i] < 0.0))
            if crossed:
                sign_change_idx.append(i)
        if not sign_change_idx:
            continue
        best_i = min(sign_change_idx, key=lambda i: abs(arc[i] - center_arc) + abs(lat[i]))
        good_arc = abs(arc[best_i] - center_arc) <= (0.7 * length + 4.0)
        good_lat = min(abs(lat[max(0, best_i - 1)]), abs(lat[best_i])) <= (0.6 * width + 2.0)
        if good_arc and good_lat:
            score = abs(arc[best_i] - center_arc) + 0.5 * abs(lat[best_i])
            candidates.append((token, score, int(frames[best_i])))

    candidates.sort(key=lambda x: (x[1], x[2]))
    target_tokens = [x[0] for x in candidates[:6]]
    all_peds = sorted(ped_df["instance_token"].unique().tolist())
    ignored = [t for t in all_peds if t not in target_tokens]
    reason = (
        "Selected pedestrians that truly cross ego path at main crossing zone using "
        "sign-change in lateral side and local arc proximity to zone center; ignored other groups."
    )
    return {
        "target_crossing_pedestrian_tokens": target_tokens,
        "ignored_pedestrian_tokens": ignored,
        "target_crossing_group_selection_reason": reason,
    }


def build_target_crossing_trajectories(
    objects_df: pd.DataFrame,
    target_tokens: List[str],
    zone: Dict[str, Any],
    n_frames: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    out = objects_df.copy()
    center = np.array([zone["crossing_center_x"], zone["crossing_center_y"]], dtype=float)
    t = np.array(zone["ego_path_tangent"], dtype=float)
    n = np.array(zone["crossing_direction"], dtype=float)
    width = float(zone["crossing_width_m"])

    per_ped_interval: Dict[str, Dict[str, int]] = {}
    base_center = int(zone["crossing_center_frame"])
    for i, tok in enumerate(target_tokens):
        mask = out["instance_token"] == tok
        grp = out[mask].sort_values("frame_index")
        if grp.empty:
            continue
        min_f = int(grp["frame_index"].min())
        max_f = int(grp["frame_index"].max())
        enter = int(np.clip(base_center - 4 + i, min_f + 1, max_f - 2))
        exit_ = int(np.clip(enter + 7 + (i % 3), enter + 1, max_f - 1))
        if exit_ <= enter:
            enter = int(np.clip(min_f + 1, 1, n_frames - 3))
            exit_ = int(np.clip(enter + 4, enter + 1, max_f))
        tangent_shift = (i - 0.5 * (len(target_tokens) - 1)) * 0.8
        p0 = center + tangent_shift * t - (0.5 * width + 1.5) * n
        p1 = center + tangent_shift * t + (0.5 * width + 1.8) * n
        move_vec = p1 - p0
        ped_yaw = float(math.atan2(move_vec[1], move_vec[0]))

        idx = grp.index.tolist()
        frames = grp["frame_index"].to_numpy(dtype=int)
        new_xy = []
        for f in frames:
            if f <= enter:
                alpha = 0.0
            elif f >= exit_:
                alpha = 1.0
            else:
                alpha = smoothstep01((f - enter) / max(1e-6, (exit_ - enter)))
            new_xy.append(p0 + alpha * move_vec)
        new_xy_arr = np.array(new_xy, dtype=float)
        out.loc[idx, "obj_x"] = new_xy_arr[:, 0]
        out.loc[idx, "obj_y"] = new_xy_arr[:, 1]
        out.loc[idx, "obj_yaw_rad"] = ped_yaw
        out.loc[idx, "attribute_names"] = "pedestrian.moving"
        per_ped_interval[tok] = {"ped_enter_frame": enter, "ped_exit_frame": exit_}

    return out, per_ped_interval


def find_pedestrian_crossing_interval(
    objects_df: pd.DataFrame,
    geom: EgoPathGeometry,
    zone: Dict[str, Any],
    token: str,
) -> Dict[str, Any]:
    grp = objects_df[objects_df["instance_token"] == token].sort_values("frame_index").copy()
    if grp.empty:
        return {"ped_enter_frame": None, "ped_exit_frame": None}
    grp = annotate_with_ego_projection(grp, geom)
    arc = grp["ego_nearest_arc_m"].to_numpy(dtype=float)
    lat = grp["ego_signed_lateral_m"].to_numpy(dtype=float)
    fr = grp["frame_index"].to_numpy(dtype=int)
    center_arc = float(zone["crossing_zone_arc_length_m"])
    in_zone = (np.abs(lat) <= 0.5 * float(zone["crossing_width_m"])) & (
        np.abs(arc - center_arc) <= 0.5 * float(zone["crossing_length_m"])
    )
    if not np.any(in_zone):
        return {"ped_enter_frame": None, "ped_exit_frame": None}
    return {"ped_enter_frame": int(fr[np.argmax(in_zone)]), "ped_exit_frame": int(fr[len(fr) - 1 - np.argmax(in_zone[::-1])])}


def find_global_crossing_period(per_ped: Dict[str, Dict[str, Any]], n_frames: int) -> Tuple[int, int]:
    starts = [v["ped_enter_frame"] for v in per_ped.values() if v.get("ped_enter_frame") is not None]
    ends = [v["ped_exit_frame"] for v in per_ped.values() if v.get("ped_exit_frame") is not None]
    if not starts or not ends:
        s = int(np.clip(round(0.45 * (n_frames - 1)), 1, n_frames - 3))
        e = int(np.clip(s + 6, s + 1, n_frames - 2))
        return s, e
    return int(min(starts)), int(max(ends))


def build_refined_ego_profile(
    ego_df: pd.DataFrame,
    geom: EgoPathGeometry,
    crossing_start_frame: int,
    crossing_end_frame: int,
    hold_arc_m: float,
) -> pd.DataFrame:
    out = ego_df.copy()
    n = len(out)
    orig_arc = out["cumulative_distance_m"].fillna(0.0).to_numpy(dtype=float)
    total_arc = float(max(orig_arc[-1], geom.total_arc_m))
    hold_arc = float(np.clip(hold_arc_m, 1.0, total_arc - 2.0))
    s_wait = hold_arc - 0.15
    new_arc = orig_arc.copy()
    for i in range(n):
        if i < crossing_start_frame:
            new_arc[i] = min(orig_arc[i], s_wait)
        elif i <= crossing_end_frame:
            new_arc[i] = s_wait
        else:
            denom = max(1, (n - 1 - crossing_end_frame))
            a = smoothstep01((i - crossing_end_frame) / denom)
            new_arc[i] = s_wait + a * (total_arc - s_wait)
    new_arc = np.maximum.accumulate(new_arc)
    new_arc[-1] = total_arc
    x, y, yaw = sample_path_at_arc(geom, new_arc)
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
    yawdiff = np.array([np.nan] + [wrap_angle(yaw[i] - yaw[i - 1]) for i in range(1, n)], dtype=float)
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


def lidar_radar_sum(row: Any) -> int:
    nl = row["num_lidar_pts"] if "num_lidar_pts" in row.index else np.nan
    nr = row["num_radar_pts"] if "num_radar_pts" in row.index else np.nan
    if pd.isna(nl):
        nl = 0.0
    if pd.isna(nr):
        nr = 0.0
    return int(max(0, round(float(nl) + float(nr))))


def max_distance_for_coarse_class(cc: str) -> float:
    if cc == "pedestrian":
        return MAX_PEDESTRIAN_DISTANCE_M
    if cc == "vehicle":
        return MAX_VEHICLE_DISTANCE_M
    if cc == "static_object":
        return MAX_STATIC_DISTANCE_M
    return MAX_FILTER_DISTANCE_M


def support_minimum_for_row(cc: str) -> int:
    return 1


def passes_support(row: pd.Series, cc: str) -> bool:
    pts = lidar_radar_sum(row)
    need = support_minimum_for_row(cc)
    if cc == "pedestrian" and ALLOW_ZERO_POINT_VISIBLE_PEDESTRIANS and pts == 0:
        return True
    return pts >= need


def camera_visibility_quality_from_row(row: pd.Series) -> str:
    vt = str(row.get("visibility_token", "") or "")
    pts = lidar_radar_sum(row)
    if vt in ("4", "3"):
        return "good"
    if vt == "2":
        return "fair"
    if vt == "1":
        return "poor"
    if pts > 0:
        return "fair"
    return "unknown"


def add_camera_awareness_layer(objects_df: pd.DataFrame) -> pd.DataFrame:
    if objects_df.empty:
        return objects_df.assign(
            camera_candidate_flag=pd.Series(dtype=bool),
            camera_dynamic_target_flag=pd.Series(dtype=bool),
            camera_visibility_quality=pd.Series(dtype=str),
            camera_distance_ok_flag=pd.Series(dtype=bool),
            camera_visibility_ok_flag=pd.Series(dtype=bool),
            camera_support_ok_flag=pd.Series(dtype=bool),
            camera_degradable_flag=pd.Series(dtype=bool),
            camera_dropout_priority=pd.Series(dtype=float),
            degradation_include_flag=pd.Series(dtype=bool),
            camera_source_assumption=pd.Series(dtype=str),
            degradation_class=pd.Series(dtype=str),
        )

    out = objects_df.copy()
    vis_quality, candidate, distance_ok, visibility_ok, support_ok, degradable, priorities, degradation_class = ([] for _ in range(8))
    rel_vals = pd.to_numeric(out.get("relevance_score", pd.Series(np.zeros(len(out)))), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ped_vals = pd.to_numeric(out.get("ped_conflict_score", pd.Series(np.zeros(len(out)))), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    for j, (_, row) in enumerate(out.iterrows()):
        cc = str(row.get("coarse_class", ""))
        q = camera_visibility_quality_from_row(row)
        is_dyn = cc in ("pedestrian", "vehicle")
        d_ok = float(row.get("distance_to_ego_m", np.inf)) <= max_distance_for_coarse_class(cc)
        v_ok = q in ("good", "fair") or (cc == "pedestrian" and q in ("poor", "unknown"))
        s_ok = passes_support(row, cc)
        cam_deg = bool(is_dyn and d_ok and v_ok and s_ok)
        dcls = "pedestrian" if cc == "pedestrian" else "vehicle" if cc == "vehicle" else "non_degradable"
        priority = 0.0
        if is_dyn:
            priority = 0.65 * float(rel_vals[j])
            if cc == "pedestrian":
                priority += 0.35 * float(ped_vals[j])
            if bool(row.get("near_future_path_flag", False)):
                priority += 8.0
            if bool(row.get("forward_corridor_flag", False)):
                priority += 5.0
            if q == "good":
                priority += 3.0
            elif q == "poor":
                priority -= 4.0
        vis_quality.append(q)
        candidate.append(is_dyn)
        distance_ok.append(d_ok)
        visibility_ok.append(v_ok)
        support_ok.append(s_ok)
        degradable.append(cam_deg)
        priorities.append(float(max(0.0, priority)))
        degradation_class.append(dcls)

    out["camera_candidate_flag"] = candidate
    out["camera_dynamic_target_flag"] = candidate
    out["camera_visibility_quality"] = vis_quality
    out["camera_distance_ok_flag"] = distance_ok
    out["camera_visibility_ok_flag"] = visibility_ok
    out["camera_support_ok_flag"] = support_ok
    out["camera_degradable_flag"] = degradable
    out["camera_dropout_priority"] = priorities
    out["degradation_include_flag"] = np.asarray(degradable, dtype=bool)
    out["camera_source_assumption"] = np.where(out["camera_candidate_flag"], "camera_only_experiment", "not_camera_degraded")
    out["degradation_class"] = degradation_class
    return out


def build_camera_detection_summary(objects_df: pd.DataFrame, n_frames: int) -> Dict[str, Any]:
    if objects_df.empty:
        return {
            "total_annotation_rows": 0,
            "camera_candidate_rows": 0,
            "camera_degradable_rows": 0,
            "camera_candidate_pedestrian_rows": 0,
            "camera_candidate_vehicle_rows": 0,
            "camera_degradable_pedestrian_rows": 0,
            "camera_degradable_vehicle_rows": 0,
            "frames_with_any_camera_candidate": 0,
            "frames_with_any_camera_degradable_object": 0,
            "peak_camera_candidate_rows_in_frame": 0,
            "peak_camera_degradable_rows_in_frame": 0,
            "mean_camera_candidate_rows_per_frame": 0.0,
            "mean_camera_degradable_rows_per_frame": 0.0,
        }
    cand = objects_df["camera_candidate_flag"].fillna(False).astype(bool)
    deg = objects_df["camera_degradable_flag"].fillna(False).astype(bool)
    by_frame = objects_df.groupby("frame_index")
    cand_pf = by_frame["camera_candidate_flag"].sum()
    deg_pf = by_frame["camera_degradable_flag"].sum()
    return {
        "total_annotation_rows": int(len(objects_df)),
        "camera_candidate_rows": int(cand.sum()),
        "camera_degradable_rows": int(deg.sum()),
        "camera_candidate_pedestrian_rows": int((cand & objects_df["coarse_class"].eq("pedestrian")).sum()),
        "camera_candidate_vehicle_rows": int((cand & objects_df["coarse_class"].eq("vehicle")).sum()),
        "camera_degradable_pedestrian_rows": int((deg & objects_df["coarse_class"].eq("pedestrian")).sum()),
        "camera_degradable_vehicle_rows": int((deg & objects_df["coarse_class"].eq("vehicle")).sum()),
        "frames_with_any_camera_candidate": int((cand_pf > 0).sum()),
        "frames_with_any_camera_degradable_object": int((deg_pf > 0).sum()),
        "peak_camera_candidate_rows_in_frame": int(cand_pf.max()) if len(cand_pf) else 0,
        "peak_camera_degradable_rows_in_frame": int(deg_pf.max()) if len(deg_pf) else 0,
        "mean_camera_candidate_rows_per_frame": float(cand_pf.mean()) if len(cand_pf) else 0.0,
        "mean_camera_degradable_rows_per_frame": float(deg_pf.mean()) if len(deg_pf) else 0.0,
        "camera_degradation_assumption": "Dynamic pedestrian/vehicle rows are tagged as camera-degradable candidates for downstream controlled dropout.",
        "n_frames": int(n_frames),
    }


def recompute_object_metrics(objects_df: pd.DataFrame, ego_df: pd.DataFrame) -> pd.DataFrame:
    out = objects_df.copy()
    ego_lut = ego_df.set_index("frame_index")[["ego_x", "ego_y", "ego_yaw_rad"]]
    out = out.merge(ego_lut, left_on="frame_index", right_index=True, how="left")
    dx = out["obj_x"] - out["ego_x"]
    dy = out["obj_y"] - out["ego_y"]
    out["relative_x"] = dx
    out["relative_y"] = dy
    cy = np.cos(out["ego_yaw_rad"])
    sy = np.sin(out["ego_yaw_rad"])
    out["relative_longitudinal_m"] = dx * cy + dy * sy
    out["relative_lateral_m"] = -dx * sy + dy * cy
    out["distance_to_ego_m"] = np.sqrt(dx * dx + dy * dy)
    out["relative_heading_rad"] = (out["obj_yaw_rad"] - out["ego_yaw_rad"]).map(wrap_angle)

    coarse = []
    for c in out["category_name"].tolist():
        if is_pedestrian_category(c):
            coarse.append("pedestrian")
        elif isinstance(c, str) and c.startswith("vehicle."):
            coarse.append("vehicle")
        elif isinstance(c, str) and (c.startswith("movable_object.") or c.startswith("static_object.")):
            coarse.append("static_object")
        else:
            coarse.append("other")
    out["coarse_class"] = coarse
    out["is_pedestrian"] = out["coarse_class"] == "pedestrian"
    out["is_vehicle"] = out["coarse_class"] == "vehicle"
    out["is_static_object"] = out["coarse_class"] == "static_object"

    future_dist = np.full(len(out), np.nan, dtype=float)
    for frame_i, grp in out.groupby("frame_index"):
        f = int(frame_i)
        e = ego_df[(ego_df["frame_index"] >= f) & (ego_df["frame_index"] <= f + FUTURE_PATH_LOOKAHEAD_FRAMES)][["ego_x", "ego_y"]].to_numpy(dtype=float)
        if len(e) == 0:
            continue
        idx = grp.index.to_numpy()
        p = out.loc[idx, ["obj_x", "obj_y"]].to_numpy(dtype=float)
        d = np.sqrt(((p[:, None, :] - e[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        future_dist[idx] = d
    out["future_path_distance_m"] = future_dist
    out["forward_corridor_flag"] = (out["relative_longitudinal_m"] > 0.0) & (out["relative_longitudinal_m"] < 35.0) & (out["relative_lateral_m"].abs() <= FORWARD_CORRIDOR_LATERAL_M)
    out["near_future_path_flag"] = out["future_path_distance_m"] <= FUTURE_PATH_DISTANCE_THRESHOLD_M

    max_dist = np.where(
        out["is_pedestrian"],
        MAX_PEDESTRIAN_DISTANCE_M,
        np.where(out["is_vehicle"], MAX_VEHICLE_DISTANCE_M, np.where(out["is_static_object"], MAX_STATIC_DISTANCE_M, MAX_FILTER_DISTANCE_M)),
    )
    out["filtered_planner_object_flag"] = (
        (out["distance_to_ego_m"] <= max_dist)
        & (out["distance_to_ego_m"] <= MAX_FILTER_DISTANCE_M)
        & ((out["num_lidar_pts"].fillna(0) + out["num_radar_pts"].fillna(0)) >= 0)
    )

    base = np.clip(60.0 - out["distance_to_ego_m"], 0.0, 60.0)
    out["relevance_score"] = (
        base
        + out["forward_corridor_flag"].astype(float) * 15.0
        + out["near_future_path_flag"].astype(float) * 25.0
        + out["is_pedestrian"].astype(float) * 8.0
        + out["is_vehicle"].astype(float) * 4.0
    )
    ped_conf = np.clip(
        70.0
        - 2.0 * out["relative_lateral_m"].abs()
        - 1.1 * np.clip(out["relative_longitudinal_m"], 0.0, None)
        - 0.8 * out["future_path_distance_m"].fillna(99.0),
        0.0,
        110.0,
    )
    ped_conf[out["relative_longitudinal_m"] < -2.0] *= 0.4
    out["ped_conflict_score"] = np.where(out["is_pedestrian"], ped_conf, 0.0)
    out = out.drop(columns=["ego_x", "ego_y", "ego_yaw_rad"])
    return out


def build_agent_trajectories(objects_df: pd.DataFrame, ego_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ego_lookup = ego_df.set_index("frame_index")[["timestamp_s"]]
    for tok, grp in objects_df.sort_values(["instance_token", "frame_index"]).groupby("instance_token"):
        grp = grp.sort_values("frame_index")
        fr = grp["frame_index"].to_numpy(dtype=int)
        x = grp["obj_x"].to_numpy(dtype=float)
        y = grp["obj_y"].to_numpy(dtype=float)
        dt = np.diff(grp["timestamp_s"].to_numpy(dtype=float), prepend=np.nan)
        vx = np.concatenate([[np.nan], np.divide(np.diff(x), np.where(dt[1:] > 1e-6, dt[1:], np.nan))])
        vy = np.concatenate([[np.nan], np.divide(np.diff(y), np.where(dt[1:] > 1e-6, dt[1:], np.nan))])
        sp = np.sqrt(vx * vx + vy * vy)
        ax = np.concatenate([[np.nan], np.diff(vx)])
        ay = np.concatenate([[np.nan], np.diff(vy)])
        for i in range(len(grp)):
            r = grp.iloc[i]
            rows.append(
                {
                    "instance_token": tok,
                    "category_name": r["category_name"],
                    "coarse_class": r["coarse_class"],
                    "frame_index": int(fr[i]),
                    "timestamp_s": float(ego_lookup.loc[int(fr[i]), "timestamp_s"]),
                    "x": float(x[i]),
                    "y": float(y[i]),
                    "z": float(r["obj_z"]),
                    "yaw_rad": float(r["obj_yaw_rad"]),
                    "vx_est_mps": float(vx[i]) if not np.isnan(vx[i]) else np.nan,
                    "vy_est_mps": float(vy[i]) if not np.isnan(vy[i]) else np.nan,
                    "speed_est_mps": float(sp[i]) if not np.isnan(sp[i]) else np.nan,
                    "ax_est_mps2": float(ax[i]) if not np.isnan(ax[i]) else np.nan,
                    "ay_est_mps2": float(ay[i]) if not np.isnan(ay[i]) else np.nan,
                    "distance_to_ego_m": float(r["distance_to_ego_m"]),
                    "motion_toward_ego_path_flag": bool(r["near_future_path_flag"]),
                    "motion_relevance_note": "pedestrian velocity reduces distance to future ego path"
                    if bool(r["is_pedestrian"]) and bool(r["near_future_path_flag"]) else "",
                }
            )
    return pd.DataFrame(rows)


def planner_attention(nearest_ped: float, nearest_veh: float, n_conflict: int) -> Tuple[str, str, str]:
    ped = "high" if n_conflict > 0 or nearest_ped < 7.5 else "medium" if nearest_ped < 15.0 else "low"
    veh = "high" if nearest_veh < 7.5 else "medium" if nearest_veh < 18.0 else "low"
    overall = "high" if ("high" in (ped, veh)) else "medium" if ("medium" in (ped, veh)) else "low"
    return ped, veh, overall


def build_planner_observations(ego_df: pd.DataFrame, objects_df: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    for _, e in ego_df.sort_values("frame_index").iterrows():
        f = int(e["frame_index"])
        g = objects_df[objects_df["frame_index"] == f].copy()
        ped = g[g["is_pedestrian"]]
        veh = g[g["is_vehicle"]]
        sta = g[g["is_static_object"]]
        gf = g[g["filtered_planner_object_flag"]]
        gfp = gf[gf["is_pedestrian"]]
        gfv = gf[gf["is_vehicle"]]
        gfs = gf[gf["is_static_object"]]
        nearest_any = float(g["distance_to_ego_m"].min()) if len(g) else np.nan
        nearest_ped = float(ped["distance_to_ego_m"].min()) if len(ped) else np.nan
        nearest_veh = float(veh["distance_to_ego_m"].min()) if len(veh) else np.nan
        nearest_any_f = float(gf["distance_to_ego_m"].min()) if len(gf) else np.nan
        nearest_ped_f = float(gfp["distance_to_ego_m"].min()) if len(gfp) else np.nan
        nearest_veh_f = float(gfv["distance_to_ego_m"].min()) if len(gfv) else np.nan
        corr = g[g["forward_corridor_flag"]]
        nearp = g[g["near_future_path_flag"]]
        conflict = ped[ped["ped_conflict_score"] >= CONFLICT_PED_SCORE_THRESHOLD]
        top_rel = g.sort_values("relevance_score", ascending=False).head(1)
        top = top_rel.iloc[0] if len(top_rel) else None
        in_path = g[(g["relative_longitudinal_m"] > 0.0) & (g["relative_lateral_m"].abs() <= IN_PATH_LATERAL_THRESHOLD_M)]
        ped_alert, veh_alert, overall_alert = planner_attention(
            nearest_ped if not np.isnan(nearest_ped) else 999.0,
            nearest_veh if not np.isnan(nearest_veh) else 999.0,
            len(conflict),
        )
        out_rows.append(
            {
                "frame_index": f,
                "sample_token": e["sample_token"],
                "timestamp_us": int(e["timestamp_us"]),
                "timestamp_s": float(e["timestamp_s"]),
                "delta_t_s": float(e["delta_t_s"]),
                "ego_x": float(e["ego_x"]),
                "ego_y": float(e["ego_y"]),
                "ego_yaw_rad": float(e["ego_yaw_rad"]),
                "ego_speed_mps": float(e["ego_speed_mps"]) if not pd.isna(e["ego_speed_mps"]) else np.nan,
                "ego_accel_est_mps2": float(e["ego_accel_est_mps2"]) if not pd.isna(e["ego_accel_est_mps2"]) else np.nan,
                "raw_num_objects_total": int(len(g)),
                "raw_num_pedestrians": int(len(ped)),
                "raw_num_vehicles": int(len(veh)),
                "raw_num_static_objects": int(len(sta)),
                "filtered_num_objects_total": int(len(gf)),
                "filtered_num_pedestrians": int(len(gfp)),
                "filtered_num_vehicles": int(len(gfv)),
                "filtered_num_static_objects": int(len(gfs)),
                "num_objects_total": int(len(g)),
                "num_pedestrians": int(len(ped)),
                "num_vehicles": int(len(veh)),
                "num_static_objects": int(len(sta)),
                "nearest_any_raw_m": nearest_any,
                "nearest_ped_raw_m": nearest_ped,
                "nearest_vehicle_raw_m": nearest_veh,
                "nearest_any_filtered_m": nearest_any_f,
                "nearest_ped_filtered_m": nearest_ped_f,
                "nearest_vehicle_filtered_m": nearest_veh_f,
                "nearest_object_distance_m": nearest_any,
                "nearest_object_category": g.loc[g["distance_to_ego_m"].idxmin(), "category_name"] if len(g) else "",
                "nearest_object_instance_token": g.loc[g["distance_to_ego_m"].idxmin(), "instance_token"] if len(g) else "",
                "nearest_ped_distance_m": nearest_ped,
                "nearest_vehicle_distance_m": nearest_veh,
                "num_objects_within_5m": int((g["distance_to_ego_m"] <= 5.0).sum()),
                "num_pedestrians_within_5m": int((ped["distance_to_ego_m"] <= 5.0).sum()),
                "num_objects_within_10m": int((g["distance_to_ego_m"] <= 10.0).sum()),
                "num_pedestrians_within_10m": int((ped["distance_to_ego_m"] <= 10.0).sum()),
                "num_objects_within_20m": int((g["distance_to_ego_m"] <= 20.0).sum()),
                "num_pedestrians_within_20m": int((ped["distance_to_ego_m"] <= 20.0).sum()),
                "num_vehicles_within_10m": int((veh["distance_to_ego_m"] <= 10.0).sum()),
                "num_vehicles_within_20m": int((veh["distance_to_ego_m"] <= 20.0).sum()),
                "num_objects_in_forward_corridor": int(len(corr)),
                "num_pedestrians_in_forward_corridor": int(corr["is_pedestrian"].sum()),
                "num_vehicles_in_forward_corridor": int(corr["is_vehicle"].sum()),
                "closest_forward_corridor_object_distance_m": float(corr["distance_to_ego_m"].min()) if len(corr) else np.nan,
                "closest_forward_corridor_ped_distance_m": float(corr[corr["is_pedestrian"]]["distance_to_ego_m"].min()) if int(corr["is_pedestrian"].sum()) else np.nan,
                "closest_forward_corridor_vehicle_distance_m": float(corr[corr["is_vehicle"]]["distance_to_ego_m"].min()) if int(corr["is_vehicle"].sum()) else np.nan,
                "num_objects_near_future_ego_path": int(len(nearp)),
                "num_pedestrians_near_future_ego_path": int(nearp["is_pedestrian"].sum()),
                "num_vehicles_near_future_ego_path": int(nearp["is_vehicle"].sum()),
                "closest_future_path_object_distance_m": float(nearp["future_path_distance_m"].min()) if len(nearp) else np.nan,
                "closest_future_path_ped_distance_m": float(nearp[nearp["is_pedestrian"]]["future_path_distance_m"].min()) if int(nearp["is_pedestrian"].sum()) else np.nan,
                "closest_future_path_vehicle_distance_m": float(nearp[nearp["is_vehicle"]]["future_path_distance_m"].min()) if int(nearp["is_vehicle"].sum()) else np.nan,
                "num_candidate_conflict_pedestrians": int(len(conflict)),
                "closest_conflict_ped_distance_m": float(conflict["distance_to_ego_m"].min()) if len(conflict) else np.nan,
                "highest_conflict_ped_instance_token": conflict.sort_values("ped_conflict_score", ascending=False).iloc[0]["instance_token"] if len(conflict) else "",
                "highest_conflict_ped_score": float(conflict["ped_conflict_score"].max()) if len(conflict) else np.nan,
                "conflict_exists_binary": int(len(conflict) > 0),
                "top_relevant_object_instance_token": top["instance_token"] if top is not None else "",
                "top_relevant_object_category": top["category_name"] if top is not None else "",
                "top_relevant_object_coarse_class": top["coarse_class"] if top is not None else "",
                "top_relevant_object_distance_m": float(top["distance_to_ego_m"]) if top is not None else np.nan,
                "top_relevant_object_relevance_score": float(top["relevance_score"]) if top is not None else np.nan,
                "pedestrian_alert_level": ped_alert,
                "vehicle_alert_level": veh_alert,
                "overall_planner_attention_level": overall_alert,
                "num_objects_zero_points": int(((g["num_lidar_pts"].fillna(0) + g["num_radar_pts"].fillna(0)) == 0).sum()),
                "num_objects_low_support": int(((g["num_lidar_pts"].fillna(0) + g["num_radar_pts"].fillna(0)) <= LOW_SUPPORT_THRESHOLD).sum()),
                "num_objects_filtered_out_by_distance": int(len(g) - len(gf)),
                "num_objects_filtered_out_by_support": 0,
                "num_objects_filtered_out_by_visibility": 0,
                "closest_in_path_object_distance_m": float(in_path["distance_to_ego_m"].min()) if len(in_path) else np.nan,
                "closest_in_path_object_instance": in_path.sort_values("distance_to_ego_m").iloc[0]["instance_token"] if len(in_path) else "",
                "closest_in_path_object_category": in_path.sort_values("distance_to_ego_m").iloc[0]["category_name"] if len(in_path) else "",
                "closest_in_path_ped_distance_m": float(in_path[in_path["is_pedestrian"]]["distance_to_ego_m"].min()) if int(in_path["is_pedestrian"].sum()) else np.nan,
                "closest_in_path_vehicle_distance_m": float(in_path[in_path["is_vehicle"]]["distance_to_ego_m"].min()) if int(in_path["is_vehicle"].sum()) else np.nan,
            }
        )
    return pd.DataFrame(out_rows)


def build_class_counts(objects_df: pd.DataFrame, planner_df: pd.DataFrame, n_frames: int) -> Dict[str, Any]:
    by_cat = objects_df["category_name"].value_counts().to_dict()
    by_cat_u = objects_df.groupby("category_name")["instance_token"].nunique().to_dict()
    by_coarse = objects_df["coarse_class"].value_counts().to_dict()
    per_frame = []
    for f in range(n_frames):
        g = objects_df[objects_df["frame_index"] == f]
        per_frame.append(
            {
                "frame_index": int(f),
                "total_objects": int(len(g)),
                "by_category": {k: int(v) for k, v in g["category_name"].value_counts().to_dict().items()},
                "by_coarse": {k: int(v) for k, v in g["coarse_class"].value_counts().to_dict().items()},
            }
        )
    top_dyn = [
        {"category": k, "annotation_rows": int(v)}
        for k, v in objects_df[objects_df["coarse_class"] != "static_object"]["category_name"].value_counts().head(8).to_dict().items()
    ]
    planner_summary = {
        "annotations_in_forward_corridor": int(objects_df["forward_corridor_flag"].sum()),
        "annotations_near_future_ego_path": int(objects_df["near_future_path_flag"].sum()),
        "annotations_passing_planner_filter": int(objects_df["filtered_planner_object_flag"].sum()),
        "mean_relevance_score": float(objects_df["relevance_score"].mean()) if len(objects_df) else 0.0,
        "max_ped_conflict_score": float(objects_df.loc[objects_df["coarse_class"] == "pedestrian", "ped_conflict_score"].max())
        if (objects_df["coarse_class"] == "pedestrian").any() else None,
        "camera_candidate_rows": int(objects_df["camera_candidate_flag"].sum()) if "camera_candidate_flag" in objects_df.columns else 0,
        "camera_degradable_rows": int(objects_df["camera_degradable_flag"].sum()) if "camera_degradable_flag" in objects_df.columns else 0,
    }
    return {
        "annotation_counts_by_category": {k: int(v) for k, v in by_cat.items()},
        "unique_instance_counts_by_category": {k: int(v) for k, v in by_cat_u.items()},
        "aggregated_counts_by_coarse_class": {k: int(v) for k, v in by_coarse.items()},
        "per_frame_count_summary": per_frame,
        "top_dynamic_categories": top_dyn,
        "planner_annotation_summary": planner_summary,
    }


def compute_xy_bounds(ego_df: pd.DataFrame, objects_df: pd.DataFrame, margin_m: float = 12.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = [float(v) for v in ego_df["ego_x"].values if not pd.isna(v)]
    ys = [float(v) for v in ego_df["ego_y"].values if not pd.isna(v)]
    if len(objects_df):
        xs.extend(objects_df["obj_x"].astype(float).tolist())
        ys.extend(objects_df["obj_y"].astype(float).tolist())
    if not xs:
        return (-50.0, 50.0), (-50.0, 50.0)
    return (min(xs) - margin_m, max(xs) + margin_m), (min(ys) - margin_m, max(ys) + margin_m)


def make_animation_from_tables(
    out_gif: Path,
    sample_df: pd.DataFrame,
    ego_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    nmap: Any,
    patch_box: Tuple[float, float, float, float],
) -> Tuple[bool, str]:
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    PED_COLOR = "#ff2b7a"
    VEH_COLOR = "#00a7ff"
    PED_DARK = "#c40058"
    VEH_DARK = "#0076b6"
    cmap = {"pedestrian": PED_COLOR, "vehicle": VEH_COLOR, "static_object": "tab:brown", "other": "0.45"}

    ego_xy = ego_df[["ego_x", "ego_y"]].to_numpy(dtype=float)
    frames = []
    try:
        for i in sorted(sample_df["frame_index"].astype(int).tolist()):
            er = ego_df[ego_df["frame_index"] == i].iloc[0]
            sub = objects_df[objects_df["frame_index"] == i]
            fig, ax = plt.subplots(figsize=(9, 9))
            map_handles = []
            if nmap is not None:
                map_handles = draw_map_expansion_layers(ax, nmap, patch_box, 0.0, zorder=0.0)
            ex, ey = float(er["ego_x"]), float(er["ego_y"])
            yaw = float(er["ego_yaw_rad"])
            hist = ego_xy[: i + 1]
            hist = hist[np.isfinite(hist[:, 0]) & np.isfinite(hist[:, 1])]
            if len(hist) >= 2:
                ax.plot(hist[:, 0], hist[:, 1], "-", color="navy", lw=2.2, alpha=0.95, label="ego history", zorder=3)
                ax.plot(hist[-1, 0], hist[-1, 1], "o", color="navy", markersize=5, zorder=4)
            elif len(hist) == 1:
                ax.plot(hist[0, 0], hist[0, 1], "o", color="navy", markersize=5, zorder=4)
            if np.isfinite(ex) and np.isfinite(ey):
                ax.plot(ex, ey, "s", color="navy", markersize=10, label="ego current", zorder=5)
                if np.isfinite(yaw):
                    ax.arrow(ex, ey, 3 * math.cos(yaw), 3 * math.sin(yaw), head_width=1.0, head_length=1.2,
                             fc="navy", ec="navy", length_includes_head=True)
            seen_cc: set = set()
            for _, r in sub.iterrows():
                col = cmap.get(str(r["coarse_class"]), "gray")
                cc = str(r["coarse_class"])
                lbl = None
                if cc not in seen_cc:
                    lbl = {"pedestrian": "Pedestrian", "vehicle": "Vehicle", "static_object": "Static object"}.get(cc, "Other")
                    seen_cc.add(cc)
                ax.plot(
                    r["obj_x"], r["obj_y"], "o", color=col,
                    markersize=5.6 if cc in ("pedestrian", "vehicle") else 4.0,
                    markeredgecolor=(PED_DARK if cc == "pedestrian" else VEH_DARK if cc == "vehicle" else "none"),
                    markeredgewidth=0.35 if cc in ("pedestrian", "vehicle") else 0.0,
                    alpha=0.92 if cc in ("pedestrian", "vehicle") else 0.82,
                    label=lbl, zorder=4.2 if cc in ("pedestrian", "vehicle") else 4.0
                )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            ts = float(sample_df[sample_df["frame_index"] == i]["timestamp_s"].iloc[0])
            ax.set_title(f"Frame {i:03d}  |  t = {ts:.3f} s")
            ax.set_xlabel("X global (m)")
            ax.set_ylabel("Y global (m)")
            ax.grid(True, alpha=0.25)
            h2, _ = ax.get_legend_handles_labels()
            ax.legend(handles=map_handles + list(h2), loc="upper left", fontsize=7, ncol=2, framealpha=0.92)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out_gif), frames, duration=0.2)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def process_scene(raw_root: Path, refined_root: Path, scene_name: str, map_dataroot: Optional[Path] = None) -> None:
    raw_scene_dir = raw_root / scene_name
    refined_scene_dir = refined_root / scene_name
    if not raw_scene_dir.is_dir():
        raise FileNotFoundError(f"Raw scene not found: {raw_scene_dir}")
    refined_scene_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(raw_scene_dir, refined_scene_dir, dirs_exist_ok=True)

    # Remove outputs we no longer keep.
    shutil.rmtree(refined_scene_dir / "per_frame", ignore_errors=True)
    shutil.rmtree(refined_scene_dir / "plots", ignore_errors=True)

    sample_df = pd.read_csv(raw_scene_dir / "sample_timeline.csv")
    ego_df = pd.read_csv(raw_scene_dir / "ego_timeline.csv")
    objects_df = pd.read_csv(raw_scene_dir / "objects_timeline.csv")
    raw_scene_summary = json.loads((raw_scene_dir / "scene_summary.json").read_text(encoding="utf-8"))
    scene_metadata = json.loads((raw_scene_dir / "scene_metadata.json").read_text(encoding="utf-8"))

    n_frames = int(sample_df["frame_index"].max()) + 1
    geom = build_ego_path_geometry(ego_df)
    zone = identify_main_crossing_zone(objects_df, geom, n_frames)
    selection = select_target_crossing_group(objects_df, geom, zone)
    target_tokens = selection["target_crossing_pedestrian_tokens"]

    objects_df, designed_intervals = build_target_crossing_trajectories(objects_df, target_tokens, zone, n_frames)

    per_ped_intervals: Dict[str, Dict[str, Any]] = {}
    for tok in target_tokens:
        measured = find_pedestrian_crossing_interval(objects_df, geom, zone, tok)
        if measured["ped_enter_frame"] is None or measured["ped_exit_frame"] is None:
            measured = designed_intervals.get(tok, {"ped_enter_frame": None, "ped_exit_frame": None})
        per_ped_intervals[tok] = measured

    crossing_start, crossing_end = find_global_crossing_period(per_ped_intervals, n_frames)
    last_exit = crossing_end
    ego_df = build_refined_ego_profile(
        ego_df=ego_df,
        geom=geom,
        crossing_start_frame=crossing_start,
        crossing_end_frame=crossing_end,
        hold_arc_m=float(zone["ego_hold_line_arc_length_m"]),
    )
    hold_arc = float(zone["ego_hold_line_arc_length_m"])
    ego_passed = bool((ego_df[ego_df["frame_index"].between(crossing_start, crossing_end)]["cumulative_distance_m"] > hold_arc).any())
    ego_resumed = bool(
        ego_df[ego_df["frame_index"] > crossing_end]["cumulative_distance_m"].max()
        > ego_df[ego_df["frame_index"] == crossing_end]["cumulative_distance_m"].iloc[0] + 0.5
    )

    objects_df = recompute_object_metrics(objects_df, ego_df)
    objects_df = add_camera_awareness_layer(objects_df)
    agent_df = build_agent_trajectories(objects_df, ego_df)
    planner_df = build_planner_observations(ego_df, objects_df)
    class_counts = build_class_counts(objects_df, planner_df, n_frames)
    camera_summary = build_camera_detection_summary(objects_df, n_frames)

    # Core CSV/JSON outputs
    sample_df.to_csv(refined_scene_dir / "sample_timeline.csv", index=False)
    ego_df.to_csv(refined_scene_dir / "ego_timeline.csv", index=False)
    objects_df.to_csv(refined_scene_dir / "objects_timeline.csv", index=False)
    agent_df.to_csv(refined_scene_dir / "agent_trajectories.csv", index=False)
    planner_df.to_csv(refined_scene_dir / "planner_observations.csv", index=False)
    (refined_scene_dir / "class_counts.json").write_text(json.dumps(class_counts, indent=2, allow_nan=True), encoding="utf-8")
    (refined_scene_dir / "camera_detection_summary.json").write_text(json.dumps(camera_summary, indent=2, allow_nan=True), encoding="utf-8")

    # Metadata
    sensor_meta_path = raw_scene_dir / "sensor_metadata.json"
    if sensor_meta_path.is_file():
        sensor_meta = json.loads(sensor_meta_path.read_text(encoding="utf-8"))
    else:
        sensor_meta = {}
    sensor_meta["refinement_stage"] = "02_refine_scene_synthetic.py"
    sensor_meta["refinement_mode"] = "synthetic_main_crossing_zone"
    sensor_meta["camera_degradation_protocol"] = {
        "dynamic_classes_tagged": ["pedestrian", "vehicle"],
        "static_objects_degraded": False,
        "description": "Camera-awareness flags were recomputed after synthetic refinement so downstream degradation uses refined geometry and relevance."
    }
    (refined_scene_dir / "sensor_metadata.json").write_text(json.dumps(sensor_meta, indent=2, allow_nan=True), encoding="utf-8")

    scene_metadata["extraction_timestamp"] = datetime.now(timezone.utc).isoformat()
    scene_metadata["refinement_script"] = "02_refine_scene_synthetic.py"
    scene_metadata["refinement_mode"] = "synthetic_main_crossing_zone"
    (refined_scene_dir / "scene_metadata.json").write_text(json.dumps(scene_metadata, indent=2, allow_nan=True), encoding="utf-8")

    before_mask = ego_df["frame_index"] < crossing_start
    during_mask = ego_df["frame_index"].between(crossing_start, crossing_end)
    after_mask = ego_df["frame_index"] > crossing_end

    summary = dict(raw_scene_summary)
    summary.update(
        {
            "scene_name": scene_name,
            "total_frames": n_frames,
            "total_duration_seconds": float(ego_df["timestamp_s"].iloc[-1] - ego_df["timestamp_s"].iloc[0]),
            "ego_total_distance_m": float(ego_df["cumulative_distance_m"].iloc[-1]),
            "ego_average_speed_mps": float(ego_df["ego_speed_mps"].dropna().mean()),
            "ego_max_speed_mps": float(ego_df["ego_speed_mps"].dropna().max()),
            "target_crossing_pedestrian_tokens": target_tokens,
            "ignored_pedestrian_tokens": selection["ignored_pedestrian_tokens"],
            "target_crossing_group_selection_reason": selection["target_crossing_group_selection_reason"],
            "crossing_center_x": float(zone["crossing_center_x"]),
            "crossing_center_y": float(zone["crossing_center_y"]),
            "crossing_center_frame": int(zone["crossing_center_frame"]),
            "crossing_zone_arc_length_m": float(zone["crossing_zone_arc_length_m"]),
            "ego_path_tangent": zone["ego_path_tangent"],
            "crossing_direction": zone["crossing_direction"],
            "crossing_width_m": float(zone["crossing_width_m"]),
            "crossing_length_m": float(zone["crossing_length_m"]),
            "ego_hold_line_arc_length_m": hold_arc,
            "crossing_start_frame": int(crossing_start),
            "crossing_end_frame": int(crossing_end),
            "last_pedestrian_exit_frame": int(last_exit),
            "crossing_period_definition": "crossing_start_frame=min(target ped enter), crossing_end_frame=max(target ped exit)",
            "ego_did_not_pass_during_crossing_period": bool(not ego_passed),
            "ego_resumed_after_last_pedestrian_exit": bool(ego_resumed),
            "average_ego_speed_before_crossing_period": float(ego_df.loc[before_mask, "ego_speed_mps"].dropna().mean()) if before_mask.any() else np.nan,
            "average_ego_speed_during_crossing_period": float(ego_df.loc[during_mask, "ego_speed_mps"].dropna().mean()) if during_mask.any() else np.nan,
            "average_ego_speed_after_crossing_period": float(ego_df.loc[after_mask, "ego_speed_mps"].dropna().mean()) if after_mask.any() else np.nan,
            "camera_awareness_summary": {
                "camera_candidate_rows": int(camera_summary.get("camera_candidate_rows", 0)),
                "camera_degradable_rows": int(camera_summary.get("camera_degradable_rows", 0)),
                "camera_degradation_design_note": "Only tagged dynamic rows are intended for later camera dropout experiments."
            },
            "validation_summary": {
                "all_target_pedestrian_crossing_intervals_found": bool(
                    all(v.get("ped_enter_frame") is not None and v.get("ped_exit_frame") is not None for v in per_ped_intervals.values())
                ),
                "crossing_period_from_target_pedestrian_activity": True,
                "ego_did_not_pass_during_crossing_period": bool(not ego_passed),
                "ego_resumed_after_last_pedestrian_exit": bool(ego_resumed),
                "target_crossing_group_only_used": True,
                "irrelevant_upper_group_ignored": True,
                "camera_degradable_rows_nonzero": bool(objects_df["camera_degradable_flag"].any()) if "camera_degradable_flag" in objects_df.columns else False,
                "camera_awareness_fields_present": bool(
                    {"camera_candidate_flag", "camera_degradable_flag", "camera_dropout_priority"}.issubset(objects_df.columns)
                ),
            },
            "target_crossing_pedestrian_intervals": per_ped_intervals,
            "run_statistics": {
                "camera_candidate_rows": int(camera_summary.get("camera_candidate_rows", 0)),
                "camera_degradable_rows": int(camera_summary.get("camera_degradable_rows", 0)),
                "annotations_in_forward_corridor": int(objects_df["forward_corridor_flag"].sum()),
                "annotations_near_future_ego_path": int(objects_df["near_future_path_flag"].sum()),
                "annotations_passing_planner_filter": int(objects_df["filtered_planner_object_flag"].sum()),
            },
        }
    )
    (refined_scene_dir / "scene_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    # Animation only, directly from tables.
    animations_dir = refined_scene_dir / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)
    xlim, ylim = compute_xy_bounds(ego_df, objects_df)
    patch_box = patch_box_from_xy_limits(xlim, ylim)
    map_name = scene_metadata.get("map_name") or scene_metadata.get("location")
    nmap = load_nuscenes_map_or_none(resolve_map_dataroot(map_dataroot), str(map_name) if map_name else None)
    gif_ok, gif_error = make_animation_from_tables(
        animations_dir / "scene_replay.gif", sample_df, ego_df, objects_df, xlim, ylim, nmap, patch_box
    )

    # Log
    log_lines = [
        f"start_time: {datetime.now(timezone.utc).isoformat()}",
        f"scene: {scene_name}",
        f"input_raw_dir: {raw_scene_dir}",
        f"output_refined_dir: {refined_scene_dir}",
        "",
        "crossing_zone:",
        f"  crossing_center_frame: {int(zone['crossing_center_frame'])}",
        f"  crossing_start_frame: {int(crossing_start)}",
        f"  crossing_end_frame: {int(crossing_end)}",
        f"  last_pedestrian_exit_frame: {int(last_exit)}",
        f"  ego_hold_line_arc_length_m: {hold_arc}",
        "",
        "selection:",
        f"  target_crossing_pedestrian_tokens: {target_tokens}",
        f"  ignored_pedestrian_tokens: {selection['ignored_pedestrian_tokens']}",
        f"  target_crossing_group_selection_reason: {selection['target_crossing_group_selection_reason']}",
        "",
        "camera_awareness:",
        f"  camera_candidate_rows: {camera_summary.get('camera_candidate_rows', 0)}",
        f"  camera_degradable_rows: {camera_summary.get('camera_degradable_rows', 0)}",
        "",
        "animation:",
        f"  gif_ok: {gif_ok}",
        f"  gif_error: {gif_error}",
        "",
        "validations:",
        f"  all_target_pedestrian_crossing_intervals_found: {summary['validation_summary']['all_target_pedestrian_crossing_intervals_found']}",
        f"  ego_did_not_pass_during_crossing_period: {summary['validation_summary']['ego_did_not_pass_during_crossing_period']}",
        f"  ego_resumed_after_last_pedestrian_exit: {summary['validation_summary']['ego_resumed_after_last_pedestrian_exit']}",
        f"  camera_degradable_rows_nonzero: {summary['validation_summary']['camera_degradable_rows_nonzero']}",
        f"  camera_awareness_fields_present: {summary['validation_summary']['camera_awareness_fields_present']}",
    ]
    (refined_scene_dir / "extraction_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def discover_scene_names(raw_root: Path) -> List[str]:
    return sorted([p.name for p in raw_root.glob("scene-*") if p.is_dir()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine only scene-0061 into the synthetic crossing scenario."
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default="scene-0061",
        help="Scene name to refine. This script is locked to scene-0061 by default.",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"), help="Raw root directory.")
    parser.add_argument("--refined-root", type=Path, default=Path("data/refined"), help="Refined output root directory.")
    parser.add_argument(
        "--map-dataroot",
        type=Path,
        default=SCRIPT_DIR / "v1.0-mini",
        help="Dataset root with maps/expansion JSONs, or the maps/maps-expansion folder under it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    refined_root = args.refined_root.resolve()

    scene_name = args.scene_name.strip()
    if scene_name != "scene-0061":
        raise ValueError(
            f"This scene-specific refinement script is locked to scene-0061, but got {scene_name!r}. "
            "Use the straight or high-curvature scene-specific scripts for other scenes."
        )

    process_scene(raw_root=raw_root, refined_root=refined_root, scene_name=scene_name, map_dataroot=args.map_dataroot)
    print(f"[ok] refined {scene_name}")


if __name__ == "__main__":
    main()
