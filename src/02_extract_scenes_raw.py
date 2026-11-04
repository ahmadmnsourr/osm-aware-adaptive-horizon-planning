#!/usr/bin/env python3
"""
Extract one nuScenes scene into planner-oriented raw artifacts for downstream camera-aware
degradation and IDM planning experiments.

Reads the selected scene from config/best_candidate_scene.json (never hardcodes scene id).
Writes under data/raw/<scene_name>/: timelines, summaries, compact plots, and replay animation.

This script preserves raw scene annotations (all categories, boxes, timestamps) in CSV/JSON
outputs, and enriches those same files with planner-oriented geometry plus an explicit
camera-aware layer. The camera-aware layer highlights dynamic objects that are plausible
camera detections and can later be degraded before being fed to the planner. These heuristics
approximate planning relevance and camera degradation eligibility; they are not a perception
model and are meant to be tuned via the constants below.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe PNG generation

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Planner-tunable thresholds (geometry, filtering, scoring)
#
# Coordinate frame: nuScenes global / map coordinates (meters). Ego pose and object
# centers use ego_pose.translation and sample_annotation.translation in this same
# XY plane for distances, plots, and longitudinal/lateral decomposition about ego yaw.
#
# Legacy “in_path” (longitudinal > 0, |lateral| <= threshold) is kept for JSON/CSV
# backward compatibility; forward corridor uses FORWARD_CORRIDOR_LATERAL_M.
# ---------------------------------------------------------------------------
IN_PATH_LATERAL_THRESHOLD_M: float = 3.0
FORWARD_CORRIDOR_LATERAL_M: float = 3.0
FUTURE_PATH_DISTANCE_THRESHOLD_M: float = 2.5
FUTURE_PATH_LOOKAHEAD_FRAMES: int = 8

MAX_FILTER_DISTANCE_M: float = 50.0
MAX_PEDESTRIAN_DISTANCE_M: float = 40.0
MAX_VEHICLE_DISTANCE_M: float = 60.0
MAX_STATIC_DISTANCE_M: float = 35.0
MIN_LIDAR_RADAR_POINTS_DYNAMIC: int = 1
MIN_LIDAR_RADAR_POINTS_STATIC: int = 1
ALLOW_ZERO_POINT_VISIBLE_PEDESTRIANS: bool = True
LOW_SUPPORT_THRESHOLD: int = 2

# Conflict / relevance tuning (interpretable, not learned)
CONFLICT_PED_SCORE_THRESHOLD: float = 35.0
RELEVANCE_NEAR_EGO_SCALE_M: float = 8.0
RELEVANCE_PATH_SCALE_M: float = 4.0
MOTION_SPEED_THRESHOLD_MPS: float = 0.35

LIDAR_CHANNEL: str = "LIDAR_TOP"
DEFAULT_NUSCENES_VERSION: str = "v1.0-mini"
CONFIG_REL_DEFAULT: str = "config/best_candidate_scene.json"
RAW_ROOT_REL: str = "data/raw"

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))


def normalize_nuscenes_dataroot_for_maps(p: Path) -> Path:
    """
    NuScenesMap expects `<dataroot>/maps/expansion/<map_name>.json`.
    Accept dataset root, `.../maps`, or `.../maps/expansion`.
    """
    p = p.resolve()
    if (p / "maps" / "expansion").is_dir():
        return p
    if p.name == "maps" and (p / "expansion").is_dir():
        return p.parent
    if p.name == "expansion" and p.parent.name == "maps":
        return p.parent.parent
    return p


def resolve_map_dataroot(explicit: Optional[str], nusc_dataroot: str, version: str) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(normalize_nuscenes_dataroot_for_maps(Path(explicit)))
    if nusc_dataroot:
        candidates.append(normalize_nuscenes_dataroot_for_maps(Path(nusc_dataroot)))
        candidates.append(normalize_nuscenes_dataroot_for_maps(Path(nusc_dataroot) / version))
    candidates.append(normalize_nuscenes_dataroot_for_maps(Path(SCRIPT_DIR) / "v1.0-mini"))
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
    return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), ymax - ymin, xmax - xmin)


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
                ax.add_patch(MplPolygon(xy, closed=True, facecolor=fc, edgecolor="0.2", lw=0.65, alpha=alpha, zorder=zorder))
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
            if poly.is_empty:
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


# =============================================================================
# Logging accumulator
# =============================================================================


@dataclass
class ExtractionLog:
    start_wall: str = ""
    end_wall: str = ""
    warnings: List[str] = field(default_factory=list)
    config_path: str = ""
    scene_name: str = ""
    scene_token: str = ""
    dataroot: str = ""
    version: str = ""
    files_written: List[str] = field(default_factory=list)
    gif_ok: bool = False
    gif_error: str = ""
    run_stats: Dict[str, Any] = field(default_factory=dict)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_file(self, p: str) -> None:
        self.files_written.append(p)


def _table_root_ok(dataroot: str, version: str) -> bool:
    root = os.path.join(os.path.abspath(dataroot), version)
    return os.path.isfile(os.path.join(root, "scene.json"))


def install_mapmask_dummy_for_missing_png() -> None:
    """Avoid NuScenes init failure when map PNGs are missing (partial installs)."""
    import nuscenes.utils.map_mask as mm
    import nuscenes.nuscenes as ns_mod

    _Base = mm.MapMask

    class MapMaskOrDummy(_Base):
        def __init__(self, img_file: str, resolution: float = 0.1) -> None:
            import os.path as osp

            if osp.isfile(img_file):
                _Base.__init__(self, img_file, resolution)
                self._dummy_map = False
            else:
                self.img_file = img_file
                self.resolution = max(0.1, float(resolution))
                self.foreground = 255
                self.background = 0
                self._dummy_map = True

        @property
        def _base_mask(self):
            import numpy as np

            if getattr(self, "_dummy_map", False):
                return np.zeros((256, 256), dtype=np.uint8)
            return super()._base_mask

    mm.MapMask = MapMaskOrDummy
    ns_mod.MapMask = MapMaskOrDummy


# =============================================================================
# Config & nuScenes init
# =============================================================================


def load_best_scene_config(config_path: str) -> Dict[str, Any]:
    """Load JSON from config/best_candidate_scene.json; validate structure."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object.")
    bc = data.get("best_candidate")
    if not isinstance(bc, dict):
        raise ValueError("Config must contain object 'best_candidate'.")
    name = bc.get("scene_name")
    token = bc.get("scene_token")
    if not name or not isinstance(name, str):
        raise ValueError("best_candidate.scene_name is missing or invalid.")
    if not token or not isinstance(token, str):
        raise ValueError("best_candidate.scene_token is missing or invalid.")
    return data


def resolve_dataroot(version: str, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Resolve nuScenes dataroot (parent folder of <version>/scene.json).
    Order: metadata.resolved_dataroot, NUSCENES_DATAROOT, SCRIPT_DIR, cwd, ./data, /data/sets/nuscenes.
    """
    candidates: List[str] = []
    if metadata and isinstance(metadata.get("resolved_dataroot"), str):
        candidates.append(metadata["resolved_dataroot"].strip())
    env = os.environ.get("NUSCENES_DATAROOT", "").strip()
    if env:
        candidates.append(os.path.expanduser(env))
    candidates.extend(
        [
            SCRIPT_DIR,
            os.getcwd(),
            os.path.join(os.getcwd(), "data"),
            os.path.join(SCRIPT_DIR, "data"),
            os.path.join(os.getcwd(), "nuscenes"),
            "/data/sets/nuscenes",
        ]
    )
    seen = set()
    for raw in candidates:
        if not raw:
            continue
        path = os.path.abspath(os.path.expanduser(raw))
        if path in seen:
            continue
        seen.add(path)
        if _table_root_ok(path, version):
            return path
        # User pointed at .../v1.0-mini
        if os.path.basename(path.rstrip(os.sep)) == version and os.path.isfile(
            os.path.join(path, "scene.json")
        ):
            parent = os.path.dirname(path)
            if parent and _table_root_ok(parent, version):
                return parent
    return None


def init_nuscenes(dataroot: str, version: str, log: ExtractionLog) -> Any:
    from nuscenes.nuscenes import NuScenes

    install_mapmask_dummy_for_missing_png()
    try:
        return NuScenes(version=version, dataroot=dataroot, verbose=False)
    except Exception as exc:
        log.warn(f"NuScenes init failed: {exc}")
        raise


def get_scene_record(nusc: Any, scene_token: str) -> Dict[str, Any]:
    for s in nusc.scene:
        if s["token"] == scene_token:
            return s
    raise KeyError(f"No scene with token {scene_token} in loaded database.")


def get_scene_record_by_name(nusc: Any, scene_name: str) -> Dict[str, Any]:
    """Find a scene record by nuScenes scene name, e.g., scene-0061."""
    scene_name = str(scene_name).strip()
    for s in nusc.scene:
        if s.get("name") == scene_name:
            return s
    raise KeyError(f"No scene named {scene_name!r} in loaded database.")


def get_ordered_scene_samples(nusc: Any, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Traverse sample chain first -> next in timestamp order (scene order)."""
    out: List[Dict[str, Any]] = []
    cur: Optional[str] = scene.get("first_sample_token") or None
    while cur:
        sample = nusc.get("sample", cur)
        out.append(sample)
        nxt = sample.get("next") or ""
        if not nxt:
            break
        cur = nxt
    return out


# =============================================================================
# Geometry & category helpers
# =============================================================================


def quat_wxyz_to_yaw_rad(rot: Sequence[float]) -> float:
    """Yaw (rad) about +Z from nuScenes quaternion [w,x,y,z] in global frame."""
    from pyquaternion import Quaternion

    q = Quaternion(rot[0], rot[1], rot[2], rot[3])
    yaw, _pitch, _roll = q.yaw_pitch_roll
    return float(yaw)


def wrap_pi(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def coarse_class_from_category(category_name: str) -> str:
    if not category_name or not isinstance(category_name, str):
        return "other"
    c = category_name
    if c.startswith("human.pedestrian"):
        return "pedestrian"
    if c.startswith("vehicle."):
        return "vehicle"
    if c.startswith("movable_object.") or c.startswith("static_object."):
        return "static_object"
    return "other"


def global_delta_to_long_lat(dx: float, dy: float, ego_yaw_rad: float) -> Tuple[float, float]:
    """
    Ego-centric Frenet-like coordinates in horizontal plane (global frame delta dx,dy).
    Forward = ego heading; lateral = left-positive (nuScenes vehicle/box convention compatible).
    longitudinal = dx * cos(yaw) + dy * sin(yaw)
    lateral = -dx * sin(yaw) + dy * cos(yaw)  (left of heading)
    """
    c, s = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
    longitudinal = dx * c + dy * s
    lateral = -dx * s + dy * c
    return float(longitudinal), float(lateral)


def resolve_attribute_names(nusc: Any, attribute_tokens: Any) -> str:
    if not attribute_tokens:
        return ""
    names: List[str] = []
    for t in attribute_tokens:
        try:
            ar = nusc.get("attribute", t)
            names.append(ar.get("name", t))
        except Exception:
            names.append(str(t))
    return "|".join(names)


# =============================================================================
# Polyline distance & planner geometry (future ego path)
# =============================================================================


def point_to_segment_distance_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom < 1e-18:
        dx, dy = px - ax, py - ay
        return dx * dx + dy * dy
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    qx, qy = ax + t * abx, ay + t * aby
    dx, dy = px - qx, py - qy
    return dx * dx + dy * dy


def distance_point_to_polyline_xy(px: float, py: float, poly_xy: np.ndarray) -> float:
    """
    Minimum distance from point to a polyline in 2D (meters).
    poly_xy: (N, 2); N==1 is point distance; N==0 returns +inf.
    """
    if poly_xy.shape[0] == 0:
        return float("inf")
    if poly_xy.shape[0] == 1:
        return float(math.hypot(px - poly_xy[0, 0], py - poly_xy[0, 1]))
    best = float("inf")
    for j in range(poly_xy.shape[0] - 1):
        d2 = point_to_segment_distance_sq(
            px, py, poly_xy[j, 0], poly_xy[j, 1], poly_xy[j + 1, 0], poly_xy[j + 1, 1]
        )
        if d2 < best:
            best = d2
    return math.sqrt(max(0.0, best))


def build_future_ego_polyline_xy(ego_df: pd.DataFrame, frame_index: int, lookahead_frames: int) -> np.ndarray:
    """Ego XY points from frame_index through frame_index+lookahead (inclusive start)."""
    n = len(ego_df)
    end = min(frame_index + lookahead_frames + 1, n)
    xs = ego_df.loc[frame_index : end - 1, "ego_x"].values.astype(np.float64)
    ys = ego_df.loc[frame_index : end - 1, "ego_y"].values.astype(np.float64)
    m = np.isfinite(xs) & np.isfinite(ys)
    pts = np.column_stack([xs[m], ys[m]])
    return pts


def precompute_future_polylines(ego_df: pd.DataFrame, lookahead_frames: int) -> List[np.ndarray]:
    """One polyline per keyframe — shared by all objects at that frame (major speedup)."""
    return [build_future_ego_polyline_xy(ego_df, fi, lookahead_frames) for fi in range(len(ego_df))]


def max_distance_for_coarse_class(cc: str) -> float:
    if cc == "pedestrian":
        return MAX_PEDESTRIAN_DISTANCE_M
    if cc == "vehicle":
        return MAX_VEHICLE_DISTANCE_M
    if cc == "static_object":
        return MAX_STATIC_DISTANCE_M
    return MAX_FILTER_DISTANCE_M


def lidar_radar_sum(row: Any) -> int:
    """Robust lidar+radar point count (missing -> 0)."""
    nl = row["num_lidar_pts"] if "num_lidar_pts" in row.index else np.nan
    nr = row["num_radar_pts"] if "num_radar_pts" in row.index else np.nan
    if pd.isna(nl):
        nl = 0.0
    if pd.isna(nr):
        nr = 0.0
    return int(max(0, round(float(nl) + float(nr))))


def support_minimum_for_row(cc: str) -> int:
    return MIN_LIDAR_RADAR_POINTS_STATIC if cc == "static_object" else MIN_LIDAR_RADAR_POINTS_DYNAMIC


def passes_support(row: pd.Series, cc: str) -> bool:
    pts = lidar_radar_sum(row)
    need = support_minimum_for_row(cc)
    if cc == "pedestrian" and ALLOW_ZERO_POINT_VISIBLE_PEDESTRIANS:
        if pts == 0:
            return True
    return pts >= need


def low_support_flag(row: pd.Series, cc: str) -> bool:
    pts = lidar_radar_sum(row)
    return pts < LOW_SUPPORT_THRESHOLD


def filtered_by_visibility(row: pd.Series, cc: str) -> bool:
    """Hard-exclude only worst visibility for non-pedestrian dynamics (heuristic)."""
    vt = str(row.get("visibility_token", "") or "")
    if cc == "pedestrian":
        return False
    if cc in ("vehicle", "other") and vt == "1":
        return True
    return False


def compute_object_relevance_score(
    cc: str,
    distance_ego_m: float,
    future_path_dist_m: float,
    in_forward_corridor: bool,
    near_future_path: bool,
    low_support: bool,
    filtered_by_vis: bool,
) -> float:
    """
    Higher = more important for planner attention (roughly 0–100+).
    Tune weights via module-level constants.
    """
    inv_ego = 100.0 / (1.0 + max(0.0, distance_ego_m) / max(RELEVANCE_NEAR_EGO_SCALE_M, 1e-6))
    inv_path = 0.0
    if math.isfinite(future_path_dist_m):
        inv_path = 100.0 / (1.0 + max(0.0, future_path_dist_m) / max(RELEVANCE_PATH_SCALE_M, 1e-6))
    class_weight = 1.0
    if cc == "pedestrian":
        class_weight = 1.35
    elif cc == "vehicle":
        class_weight = 1.1
    elif cc == "static_object":
        class_weight = 0.75
    else:
        class_weight = 0.85
    bonus_fc = 18.0 if in_forward_corridor else 0.0
    bonus_fp = 25.0 if near_future_path else 0.0
    penalty = 0.0
    if low_support:
        penalty += 8.0
    if filtered_by_vis:
        penalty += 12.0
    if distance_ego_m > max_distance_for_coarse_class(cc):
        penalty += 25.0
    score = (0.55 * inv_ego + 0.45 * inv_path) * class_weight + bonus_fc + bonus_fp - penalty
    return float(max(0.0, score))


def compute_ped_conflict_score(
    distance_ego_m: float,
    future_path_dist_m: float,
    in_forward_corridor: bool,
    near_future_path: bool,
    coarse: str,
    speed_mps: float,
    motion_toward_path: bool,
) -> float:
    """Pedestrian-only conflict score (0–100+); geometric + motion hints."""
    if coarse != "pedestrian":
        return 0.0
    inv_ego = 100.0 / (1.0 + max(0.0, distance_ego_m) / 6.0)
    inv_path = 100.0 / (1.0 + max(0.0, future_path_dist_m) / 3.0) if math.isfinite(future_path_dist_m) else 0.0
    s = 0.5 * inv_ego + 0.5 * inv_path
    if in_forward_corridor:
        s += 15.0
    if near_future_path:
        s += 22.0
    if not math.isnan(speed_mps) and speed_mps > MOTION_SPEED_THRESHOLD_MPS:
        s += 8.0
    if motion_toward_path:
        s += 12.0
    return float(max(0.0, s))


def motion_toward_future_path(
    vx: float, vy: float, px: float, py: float, poly_xy: np.ndarray, lookahead: bool = True
) -> bool:
    """Heuristic: one-step velocity reduces distance to polyline."""
    if not math.isfinite(vx) or not math.isfinite(vy):
        return False
    if poly_xy.shape[0] == 0:
        return False
    spd = math.hypot(vx, vy)
    if spd < MOTION_SPEED_THRESHOLD_MPS:
        return False
    dt = 0.25
    nx, ny = px + vx * dt, py + vy * dt
    d0 = distance_point_to_polyline_xy(px, py, poly_xy)
    d1 = distance_point_to_polyline_xy(nx, ny, poly_xy)
    return d1 < d0 - 0.05


def planner_filtered_row(row: pd.Series) -> Tuple[bool, Dict[str, bool]]:
    """Returns (passes_filter, reason_flags)."""
    cc = row["coarse_class"]
    d = float(row["distance_to_ego_m"])
    reasons = {
        "distance_ok": d <= max_distance_for_coarse_class(cc),
        "support_ok": passes_support(row, cc),
        "visibility_ok": not filtered_by_visibility(row, cc),
    }
    ok = reasons["distance_ok"] and reasons["support_ok"] and reasons["visibility_ok"]
    return ok, reasons

def camera_visibility_quality_from_row(row: pd.Series) -> str:
    """Coarse camera-visibility label from nuScenes visibility token + support."""
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
    """
    Add explicit camera-oriented extraction flags so later degradation is tied to a
    clear subset of planner-consumed dynamic objects rather than all raw annotations.
    """
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
    vis_quality = []
    candidate = []
    distance_ok = []
    visibility_ok = []
    support_ok = []
    degradable = []
    priorities = []
    degradation_class = []

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
        if cc == "pedestrian":
            dcls = "pedestrian"
        elif cc == "vehicle":
            dcls = "vehicle"
        else:
            dcls = "non_degradable"

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
    out["camera_source_assumption"] = np.where(
        out["camera_candidate_flag"], "camera_only_experiment", "not_camera_degraded"
    )
    out["degradation_class"] = degradation_class
    return out


def build_camera_detection_summary(objects_df: pd.DataFrame, n_frames: int) -> Dict[str, Any]:
    """Compact summary of dynamic rows that are eligible for later camera degradation."""
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
    cand_pf = by_frame["camera_candidate_flag"].sum() if "camera_candidate_flag" in objects_df.columns else pd.Series(dtype=float)
    deg_pf = by_frame["camera_degradable_flag"].sum() if "camera_degradable_flag" in objects_df.columns else pd.Series(dtype=float)

    return {
        "total_annotation_rows": int(len(objects_df)),
        "camera_candidate_rows": int(cand.sum()),
        "camera_degradable_rows": int(deg.sum()),
        "camera_candidate_pedestrian_rows": int((cand & objects_df["coarse_class"].eq("pedestrian")).sum()),
        "camera_candidate_vehicle_rows": int((cand & objects_df["coarse_class"].eq("vehicle")).sum()),
        "camera_degradable_pedestrian_rows": int((deg & objects_df["coarse_class"].eq("pedestrian")).sum()),
        "camera_degradable_vehicle_rows": int((deg & objects_df["coarse_class"].eq("vehicle")).sum()),
        "frames_with_any_camera_candidate": int((cand_pf > 0).sum()) if len(cand_pf) else 0,
        "frames_with_any_camera_degradable_object": int((deg_pf > 0).sum()) if len(deg_pf) else 0,
        "peak_camera_candidate_rows_in_frame": int(cand_pf.max()) if len(cand_pf) else 0,
        "peak_camera_degradable_rows_in_frame": int(deg_pf.max()) if len(deg_pf) else 0,
        "mean_camera_candidate_rows_per_frame": float(cand_pf.mean()) if len(cand_pf) else 0.0,
        "mean_camera_degradable_rows_per_frame": float(deg_pf.mean()) if len(deg_pf) else 0.0,
        "camera_degradation_assumption": "Dynamic pedestrian/vehicle rows are tagged as camera-degradable candidates for downstream controlled dropout.",
        "n_frames": int(n_frames),
    }



def enrich_objects_timeline_planner(
    objects_df: pd.DataFrame,
    ego_df: pd.DataFrame,
    log: ExtractionLog,
    future_polylines: List[np.ndarray],
) -> pd.DataFrame:
    """Add forward corridor, future-path distance, scores, and filter flags per annotation row."""
    if objects_df.empty:
        return objects_df.assign(
            future_path_distance_m=pd.Series(dtype=float),
            forward_corridor_flag=pd.Series(dtype=bool),
            near_future_path_flag=pd.Series(dtype=bool),
            relevance_score=pd.Series(dtype=float),
            ped_conflict_score=pd.Series(dtype=float),
            filtered_planner_object_flag=pd.Series(dtype=bool),
        )

    out = objects_df.copy()
    out["future_path_distance_m"] = np.nan
    out["forward_corridor_flag"] = False
    out["near_future_path_flag"] = False
    out["relevance_score"] = np.nan
    out["ped_conflict_score"] = np.nan
    out["filtered_planner_object_flag"] = False

    # One polyline per frame; process all annotations at that frame together.
    for fi, sub in objects_df.groupby("frame_index", sort=True):
        poly = future_polylines[int(fi)]
        ox = sub["obj_x"].to_numpy(dtype=np.float64)
        oy = sub["obj_y"].to_numpy(dtype=np.float64)
        if poly.shape[0] == 0:
            d_paths = np.full(len(sub), np.inf)
        else:
            d_paths = np.array(
                [distance_point_to_polyline_xy(float(ox[k]), float(oy[k]), poly) for k in range(len(sub))],
                dtype=np.float64,
            )
        d_paths[~np.isfinite(d_paths)] = np.nan
        out.loc[sub.index, "future_path_distance_m"] = d_paths

        lon = sub["relative_longitudinal_m"].to_numpy(dtype=np.float64)
        lat = sub["relative_lateral_m"].to_numpy(dtype=np.float64)
        fc_arr = np.isfinite(lon) & (lon > 0) & (np.abs(lat) <= FORWARD_CORRIDOR_LATERAL_M)
        out.loc[sub.index, "forward_corridor_flag"] = fc_arr

        dp = out.loc[sub.index, "future_path_distance_m"].to_numpy()
        nf_arr = np.isfinite(dp) & (dp <= FUTURE_PATH_DISTANCE_THRESHOLD_M)
        out.loc[sub.index, "near_future_path_flag"] = nf_arr

        for k, idx in enumerate(sub.index):
            row = sub.loc[idx]
            cc = str(row["coarse_class"])
            dist_ego = float(row["distance_to_ego_m"])
            d_path_v = float(d_paths[k]) if np.isfinite(d_paths[k]) else float("inf")
            fc = bool(fc_arr[k])
            nf = bool(nf_arr[k])
            low_sup = low_support_flag(row, cc)
            ok_filt, rsn = planner_filtered_row(row)
            vis_filt = not rsn["visibility_ok"]
            rs = compute_object_relevance_score(cc, dist_ego, d_path_v, fc, nf, low_sup, vis_filt)
            pcs = compute_ped_conflict_score(dist_ego, d_path_v, fc, nf, cc, float("nan"), False)
            out.at[idx, "relevance_score"] = rs
            out.at[idx, "ped_conflict_score"] = pcs
            out.at[idx, "filtered_planner_object_flag"] = ok_filt

    return out


def merge_agent_motion_into_objects(
    objects_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    future_polylines: List[np.ndarray],
) -> pd.DataFrame:
    """Refresh ped_conflict_score using speed and motion-toward-path heuristic."""
    if objects_df.empty or agent_df.empty:
        return objects_df
    key_cols = ["instance_token", "frame_index"]
    ag = agent_df[key_cols + ["vx_est_mps", "vy_est_mps", "speed_est_mps"]].copy()
    merged = objects_df.merge(ag, on=key_cols, how="left")
    new_pcs = np.full(len(merged), np.nan, dtype=np.float64)
    fi_arr = merged["frame_index"].to_numpy(dtype=np.int64)
    for j in range(len(merged)):
        fi = int(fi_arr[j])
        poly = future_polylines[fi]
        cc = str(merged["coarse_class"].iloc[j])
        vx_f = float(merged["vx_est_mps"].iloc[j]) if pd.notna(merged["vx_est_mps"].iloc[j]) else float("nan")
        vy_f = float(merged["vy_est_mps"].iloc[j]) if pd.notna(merged["vy_est_mps"].iloc[j]) else float("nan")
        sp_f = float(merged["speed_est_mps"].iloc[j]) if pd.notna(merged["speed_est_mps"].iloc[j]) else float("nan")
        mtp = motion_toward_future_path(
            vx_f, vy_f, float(merged["obj_x"].iloc[j]), float(merged["obj_y"].iloc[j]), poly
        )
        dp = merged["future_path_distance_m"].iloc[j]
        d_path = float(dp) if pd.notna(dp) else float("inf")
        new_pcs[j] = compute_ped_conflict_score(
            float(merged["distance_to_ego_m"].iloc[j]),
            d_path,
            bool(merged["forward_corridor_flag"].iloc[j]),
            bool(merged["near_future_path_flag"].iloc[j]),
            cc,
            sp_f,
            mtp,
        )
    merged["ped_conflict_score"] = new_pcs
    merged.drop(columns=["vx_est_mps", "vy_est_mps", "speed_est_mps"], inplace=True, errors="ignore")
    return merged


# =============================================================================
# Ego pose from sample_data / LIDAR_TOP
# =============================================================================


def pick_lidar_sample_data_token(sample: Dict[str, Any], log: ExtractionLog) -> Optional[str]:
    data = sample.get("data") or {}
    tok = data.get(LIDAR_CHANNEL)
    if tok:
        return tok
    log.warn(f"No {LIDAR_CHANNEL} in sample {sample.get('token')}; trying fallback channel.")
    for ch in sorted(data.keys()):
        if "LIDAR" in ch.upper():
            return data[ch]
    if data:
        return next(iter(data.values()))
    return None


def ego_pose_from_sample(nusc: Any, sample: Dict[str, Any], log: ExtractionLog) -> Optional[Dict[str, Any]]:
    """Ego pose record (global) aligned with keyframe; prefer LIDAR_TOP sample_data chain."""
    try:
        sd_token = pick_lidar_sample_data_token(sample, log)
        if not sd_token:
            return None
        sd = nusc.get("sample_data", sd_token)
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        return pose
    except Exception as exc:
        log.warn(f"ego_pose_from_sample failed for {sample.get('token')}: {exc}")
        return None


def extract_scene_metadata(
    nusc: Any,
    scene: Dict[str, Any],
    samples: List[Dict[str, Any]],
    version: str,
    dataroot: str,
    channel_set: List[str],
) -> Dict[str, Any]:
    ts0 = samples[0]["timestamp"] if samples else 0
    ts1 = samples[-1]["timestamp"] if samples else 0
    dt_us = max(0, int(ts1) - int(ts0))
    n = len(samples)
    avg_dt = (dt_us / 1e6 / max(n - 1, 1)) if n > 1 else 0.0

    log_token = scene.get("log_token")
    location = None
    map_name = None
    if log_token:
        try:
            lr = nusc.get("log", log_token)
            location = lr.get("location")
            map_name = location
        except Exception:
            pass

    return {
        "scene_name": scene["name"],
        "scene_token": scene["token"],
        "scene_description": scene.get("description"),
        "location": location,
        "map_name": map_name,
        "nbr_samples": scene.get("nbr_samples"),
        "first_sample_token": scene.get("first_sample_token"),
        "last_sample_token": scene.get("last_sample_token"),
        "dataset_version": version,
        "dataroot_used": os.path.abspath(dataroot),
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_duration_seconds": dt_us / 1e6,
        "average_timestep_seconds": avg_dt,
        "available_sensor_channels": channel_set,
    }


def build_sample_timeline(samples: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    prev_ts: Optional[int] = None
    prev_tok: str = ""
    for i, s in enumerate(samples):
        ts = int(s["timestamp"])
        delta_t_s = 0.0 if prev_ts is None else (ts - prev_ts) / 1e6
        rows.append(
            {
                "frame_index": i,
                "sample_token": s["token"],
                "timestamp_us": ts,
                "timestamp_s": ts / 1e6,
                "delta_t_s": delta_t_s,
                "prev_sample_token": prev_tok or "",
                "next_sample_token": s.get("next") or "",
            }
        )
        prev_ts = ts
        prev_tok = s["token"]
    return pd.DataFrame(rows)


def collect_sensor_channels(samples: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for s in samples:
        for k in (s.get("data") or {}).keys():
            keys.add(k)
    return sorted(keys)


def per_channel_frame_counts(samples: List[Dict[str, Any]], channels: List[str]) -> Dict[str, Dict[str, int]]:
    n = len(samples)
    out: Dict[str, Dict[str, int]] = {}
    for ch in channels:
        present = 0
        for s in samples:
            if ch in (s.get("data") or {}):
                present += 1
        out[ch] = {"frames_present": present, "total_frames": n}
    return out


# =============================================================================
# Main extraction pipelines
# =============================================================================


def extract_ego_timeline(
    nusc: Any, samples: List[Dict[str, Any]], log: ExtractionLog
) -> Tuple[pd.DataFrame, List[Optional[Dict[str, Any]]]]:
    """One row per frame; speed/accel from finite differences in global XY."""
    poses_raw: List[Optional[Dict[str, Any]]] = []
    rows: List[Dict[str, Any]] = []

    for i, s in enumerate(samples):
        pose = ego_pose_from_sample(nusc, s, log)
        poses_raw.append(pose)
        ts = int(s["timestamp"])
        if pose is None:
            rows.append(
                {
                    "frame_index": i,
                    "sample_token": s["token"],
                    "timestamp_us": ts,
                    "timestamp_s": ts / 1e6,
                    "ego_x": np.nan,
                    "ego_y": np.nan,
                    "ego_z": np.nan,
                    "ego_yaw_rad": np.nan,
                    "ego_yaw_deg": np.nan,
                    "ego_qw": np.nan,
                    "ego_qx": np.nan,
                    "ego_qy": np.nan,
                    "ego_qz": np.nan,
                    "ego_speed_mps": np.nan,
                    "ego_step_distance_m": np.nan,
                    "cumulative_distance_m": np.nan,
                    "ego_accel_est_mps2": np.nan,
                    "ego_angular_rate_est_rps": np.nan,
                }
            )
            continue

        t = pose["translation"]
        r = pose["rotation"]
        yaw = quat_wxyz_to_yaw_rad(r)
        rows.append(
            {
                "frame_index": i,
                "sample_token": s["token"],
                "timestamp_us": ts,
                "timestamp_s": ts / 1e6,
                "ego_x": float(t[0]),
                "ego_y": float(t[1]),
                "ego_z": float(t[2]),
                "ego_yaw_rad": yaw,
                "ego_yaw_deg": math.degrees(yaw),
                "ego_qw": float(r[0]),
                "ego_qx": float(r[1]),
                "ego_qy": float(r[2]),
                "ego_qz": float(r[3]),
                "ego_speed_mps": np.nan,
                "ego_step_distance_m": np.nan,
                "cumulative_distance_m": np.nan,
                "ego_accel_est_mps2": np.nan,
                "ego_angular_rate_est_rps": np.nan,
            }
        )

    df = pd.DataFrame(rows)
    # Finite differences
    cum = 0.0
    for i in range(len(df)):
        if i == 0 or math.isnan(df.at[i, "ego_x"]):
            df.at[i, "ego_step_distance_m"] = 0.0 if i == 0 else np.nan
            df.at[i, "cumulative_distance_m"] = cum
            continue
        if math.isnan(df.at[i - 1, "ego_x"]):
            df.at[i, "ego_step_distance_m"] = np.nan
            df.at[i, "cumulative_distance_m"] = cum
            continue
        p0 = np.array([df.at[i - 1, "ego_x"], df.at[i - 1, "ego_y"]], dtype=np.float64)
        p1 = np.array([df.at[i, "ego_x"], df.at[i, "ego_y"]], dtype=np.float64)
        step = float(np.linalg.norm(p1 - p0))
        cum += step
        df.at[i, "ego_step_distance_m"] = step
        df.at[i, "cumulative_distance_m"] = cum

        dt = df.at[i, "timestamp_s"] - df.at[i - 1, "timestamp_s"]
        if dt > 1e-9:
            spd = step / dt
            df.at[i, "ego_speed_mps"] = spd
            df.at[i, "ego_angular_rate_est_rps"] = wrap_pi(df.at[i, "ego_yaw_rad"] - df.at[i - 1, "ego_yaw_rad"]) / dt
            prev_spd = df.at[i - 1, "ego_speed_mps"]
            if not math.isnan(prev_spd):
                df.at[i, "ego_accel_est_mps2"] = (spd - prev_spd) / dt
        else:
            df.at[i, "ego_speed_mps"] = 0.0

    # First keyframe: no prior sample — speed/accel undefined by one-sided differences.
    if len(df) and not math.isnan(df.at[0, "ego_x"]):
        df.at[0, "ego_speed_mps"] = np.nan
        df.at[0, "ego_accel_est_mps2"] = np.nan
        df.at[0, "ego_angular_rate_est_rps"] = np.nan

    return df, poses_raw


def extract_objects_timeline(
    nusc: Any,
    samples: List[Dict[str, Any]],
    ego_df: pd.DataFrame,
    log: ExtractionLog,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        ego_row = ego_df[ego_df["frame_index"] == i].iloc[0]
        ex, ey = ego_row["ego_x"], ego_row["ego_y"]
        eyaw = ego_row["ego_yaw_rad"]
        ts = int(s["timestamp"])
        if math.isnan(ex):
            continue
        for ann_t in s.get("anns") or []:
            try:
                ann = nusc.get("sample_annotation", ann_t)
            except Exception as exc:
                log.warn(f"Missing annotation {ann_t}: {exc}")
                continue
            ox, oy, oz = ann["translation"]
            dist = math.hypot(ox - ex, oy - ey)
            dx, dy = ox - ex, oy - ey
            lon, lat = global_delta_to_long_lat(dx, dy, eyaw if not math.isnan(eyaw) else 0.0)
            oyaw = quat_wxyz_to_yaw_rad(ann["rotation"])
            rel_h = wrap_pi(oyaw - (eyaw if not math.isnan(eyaw) else 0.0))
            w, l, h = ann["size"]
            cat = ann.get("category_name", "")
            cc = coarse_class_from_category(cat)
            rows.append(
                {
                    "frame_index": i,
                    "sample_token": s["token"],
                    "timestamp_us": ts,
                    "timestamp_s": ts / 1e6,
                    "annotation_token": ann_t,
                    "instance_token": ann.get("instance_token", ""),
                    "category_name": cat,
                    "attribute_names": resolve_attribute_names(nusc, ann.get("attribute_tokens")),
                    "visibility_token": ann.get("visibility_token", ""),
                    "num_lidar_pts": ann.get("num_lidar_pts", np.nan),
                    "num_radar_pts": ann.get("num_radar_pts", np.nan),
                    "obj_x": float(ox),
                    "obj_y": float(oy),
                    "obj_z": float(oz),
                    "obj_yaw_rad": oyaw,
                    "obj_length": float(l),
                    "obj_width": float(w),
                    "obj_height": float(h),
                    "distance_to_ego_m": dist,
                    "relative_x": float(dx),
                    "relative_y": float(dy),
                    "relative_longitudinal_m": lon,
                    "relative_lateral_m": lat,
                    "relative_heading_rad": rel_h,
                    "coarse_class": cc,
                    "is_pedestrian": cc == "pedestrian",
                    "is_vehicle": cc == "vehicle",
                    "is_static_object": cc == "static_object",
                }
            )
    return pd.DataFrame(rows)


def build_agent_trajectories(objects_df: pd.DataFrame, ego_df: pd.DataFrame) -> pd.DataFrame:
    if objects_df.empty:
        return pd.DataFrame(
            columns=[
                "instance_token",
                "category_name",
                "coarse_class",
                "frame_index",
                "timestamp_s",
                "x",
                "y",
                "z",
                "yaw_rad",
                "vx_est_mps",
                "vy_est_mps",
                "speed_est_mps",
                "ax_est_mps2",
                "ay_est_mps2",
                "distance_to_ego_m",
                "motion_toward_ego_path_flag",
                "motion_relevance_note",
            ]
        )
    parts: List[pd.DataFrame] = []
    for inst, g in objects_df.sort_values(["instance_token", "frame_index"]).groupby("instance_token"):
        g = g.copy().sort_values("frame_index")
        x = g["obj_x"].values
        y = g["obj_y"].values
        z = g["obj_z"].values
        yaw = g["obj_yaw_rad"].values
        ts = g["timestamp_s"].values
        d_ego = g["distance_to_ego_m"].values
        n = len(g)
        vx = np.full(n, np.nan)
        vy = np.full(n, np.nan)
        spd = np.full(n, np.nan)
        ax = np.full(n, np.nan)
        ay = np.full(n, np.nan)
        for j in range(n):
            if j == 0 and n > 1:
                dt = ts[j + 1] - ts[j]
                if dt > 1e-9:
                    vx[j] = (x[j + 1] - x[j]) / dt
                    vy[j] = (y[j + 1] - y[j]) / dt
            elif j == n - 1 and n > 1:
                dt = ts[j] - ts[j - 1]
                if dt > 1e-9:
                    vx[j] = (x[j] - x[j - 1]) / dt
                    vy[j] = (y[j] - y[j - 1]) / dt
            elif 0 < j < n - 1:
                dt = ts[j + 1] - ts[j - 1]
                if dt > 1e-9:
                    vx[j] = (x[j + 1] - x[j - 1]) / dt
                    vy[j] = (y[j + 1] - y[j - 1]) / dt
            if not math.isnan(vx[j]) and not math.isnan(vy[j]):
                spd[j] = math.hypot(vx[j], vy[j])
        for j in range(n):
            if 0 < j < n - 1:
                dt = ts[j + 1] - ts[j]
                if dt > 1e-9 and not math.isnan(vx[j + 1]) and not math.isnan(vx[j]):
                    ax[j] = (vx[j + 1] - vx[j]) / dt
                if dt > 1e-9 and not math.isnan(vy[j + 1]) and not math.isnan(vy[j]):
                    ay[j] = (vy[j + 1] - vy[j]) / dt
        mtp_flags: List[bool] = []
        mtp_notes: List[str] = []
        for j in range(n):
            fi = int(g["frame_index"].iloc[j])
            poly = build_future_ego_polyline_xy(ego_df, fi, FUTURE_PATH_LOOKAHEAD_FRAMES)
            vxj = vx[j] if not math.isnan(vx[j]) else 0.0
            vyj = vy[j] if not math.isnan(vy[j]) else 0.0
            mtp = motion_toward_future_path(vxj, vyj, x[j], y[j], poly)
            mtp_flags.append(mtp)
            cc = str(g["coarse_class"].iloc[j])
            if mtp and cc == "pedestrian":
                mtp_notes.append("pedestrian velocity reduces distance to future ego path")
            elif mtp:
                mtp_notes.append("velocity reduces distance to future ego path")
            else:
                mtp_notes.append("")
        out = pd.DataFrame(
            {
                "instance_token": inst,
                "category_name": g["category_name"].values,
                "coarse_class": g["coarse_class"].values,
                "frame_index": g["frame_index"].values,
                "timestamp_s": ts,
                "x": x,
                "y": y,
                "z": z,
                "yaw_rad": yaw,
                "vx_est_mps": vx,
                "vy_est_mps": vy,
                "speed_est_mps": spd,
                "ax_est_mps2": ax,
                "ay_est_mps2": ay,
                "distance_to_ego_m": d_ego,
                "motion_toward_ego_path_flag": mtp_flags,
                "motion_relevance_note": mtp_notes,
            }
        )
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def _alert_levels(
    nearest_ped_f: float,
    nearest_veh_f: float,
    top_rel: float,
    conflict_exists: bool,
    nearest_any_f: float,
) -> Tuple[str, str, str]:
    """Return (pedestrian_alert_level, vehicle_alert_level, overall_planner_attention_level)."""
    ped_lvl = "low"
    if not math.isnan(nearest_ped_f):
        if nearest_ped_f < 3.0:
            ped_lvl = "critical"
        elif nearest_ped_f < 7.0 or conflict_exists:
            ped_lvl = "high"
        elif nearest_ped_f < 18.0:
            ped_lvl = "medium"
    veh_lvl = "low"
    if not math.isnan(nearest_veh_f):
        if nearest_veh_f < 5.0:
            veh_lvl = "critical"
        elif nearest_veh_f < 12.0:
            veh_lvl = "high"
        elif nearest_veh_f < 35.0:
            veh_lvl = "medium"
    overall = "low"
    if ped_lvl in ("critical", "high") or (not math.isnan(nearest_any_f) and nearest_any_f < 4.0):
        overall = "critical" if (ped_lvl == "critical" or nearest_any_f < 3.0) else "high"
    elif ped_lvl == "medium" or veh_lvl in ("high", "medium") or top_rel > 55.0:
        overall = "medium"
    elif top_rel > 35.0:
        overall = "low"
    return ped_lvl, veh_lvl, overall


def why_relevant_for_object(row: pd.Series) -> str:
    """Short explanation string for JSON (tuning aid, not ground truth)."""
    bits: List[str] = []
    cc = str(row.get("coarse_class", ""))
    if bool(row.get("near_future_path_flag")):
        bits.append("near future ego path")
    if bool(row.get("forward_corridor_flag")):
        bits.append("ahead in forward corridor")
    d = float(row.get("distance_to_ego_m", 1e9))
    if d < 12.0:
        bits.append("close to ego")
    if cc == "pedestrian" and float(row.get("ped_conflict_score", 0)) >= CONFLICT_PED_SCORE_THRESHOLD:
        bits.append("elevated pedestrian conflict score")
    if cc == "vehicle" and bool(row.get("forward_corridor_flag")):
        bits.append("vehicle ahead in corridor")
    if cc == "static_object" and bool(row.get("near_future_path_flag")):
        bits.append("static object near path (lower urgency)")
    if not bits:
        bits.append("annotation in scene (baseline relevance)")
    return "; ".join(bits)


def planner_attention_sentence(pr: pd.Series) -> str:
    """One-line human summary for per-frame JSON."""
    lvl = str(pr.get("overall_planner_attention_level", "low"))
    nfp = int(pr.get("num_pedestrians_near_future_ego_path", 0) or 0)
    ncc = int(pr.get("num_candidate_conflict_pedestrians", 0) or 0)
    npf = pr.get("nearest_ped_filtered_m")
    if lvl == "critical" and ncc > 0:
        return f"High attention: conflict-oriented pedestrian context (overall={lvl})."
    if nfp > 0 and not pd.isna(npf) and float(npf) < 10.0:
        return f"High attention: pedestrian within ~10 m of future ego path ({nfp:.1f} m filtered nearest)."
    if lvl in ("high", "medium"):
        return f"Medium attention: vehicles or pedestrians in corridor / path neighborhood (overall={lvl})."
    return "Low attention: mostly distant or static context for planner heuristics."


def build_planner_observations(
    ego_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    sample_tl: pd.DataFrame,
    lateral_threshold_m: float,
) -> pd.DataFrame:
    """
    Per-frame planner table: raw counts, filtered subsets, corridor / future-path geometry,
    conflict and relevance summaries, diagnostics, and legacy distance columns.
    """
    rows: List[Dict[str, Any]] = []
    for i in range(len(ego_df)):
        er = ego_df[ego_df["frame_index"] == i].iloc[0]
        st = sample_tl[sample_tl["frame_index"] == i].iloc[0]
        sub = objects_df[objects_df["frame_index"] == i]
        ex, ey, eyaw, esp = er["ego_x"], er["ego_y"], er["ego_yaw_rad"], er["ego_speed_mps"]
        eacc = er["ego_accel_est_mps2"] if "ego_accel_est_mps2" in er.index else np.nan
        ts_us = int(st["timestamp_us"])
        dts = float(st["delta_t_s"]) if not pd.isna(st["delta_t_s"]) else np.nan

        raw_tot = len(sub)
        raw_ped = int(sub["is_pedestrian"].sum()) if raw_tot else 0
        raw_veh = int(sub["is_vehicle"].sum()) if raw_tot else 0
        raw_st = int(sub["is_static_object"].sum()) if raw_tot else 0

        filt = sub[sub["filtered_planner_object_flag"]] if "filtered_planner_object_flag" in sub.columns else sub.iloc[0:0]
        ft = len(filt)
        fped = int(filt["is_pedestrian"].sum()) if ft else 0
        fveh = int(filt["is_vehicle"].sum()) if ft else 0
        fst = int(filt["is_static_object"].sum()) if ft else 0

        row: Dict[str, Any] = {
            "frame_index": i,
            "sample_token": er["sample_token"],
            "timestamp_us": ts_us,
            "timestamp_s": er["timestamp_s"],
            "delta_t_s": dts,
            "ego_x": ex,
            "ego_y": ey,
            "ego_yaw_rad": eyaw,
            "ego_speed_mps": esp,
            "ego_accel_est_mps2": eacc,
            "raw_num_objects_total": raw_tot,
            "raw_num_pedestrians": raw_ped,
            "raw_num_vehicles": raw_veh,
            "raw_num_static_objects": raw_st,
            "filtered_num_objects_total": ft,
            "filtered_num_pedestrians": fped,
            "filtered_num_vehicles": fveh,
            "filtered_num_static_objects": fst,
            # legacy count names (same as raw)
            "num_objects_total": raw_tot,
            "num_pedestrians": raw_ped,
            "num_vehicles": raw_veh,
            "num_static_objects": raw_st,
        }

        def _nearest(df: pd.DataFrame, mask: Optional[pd.Series] = None) -> Tuple[float, str, str]:
            d = df[mask] if mask is not None else df
            if len(d) == 0:
                return float("nan"), "", ""
            j = d["distance_to_ego_m"].idxmin()
            r = d.loc[j]
            return float(r["distance_to_ego_m"]), str(r["category_name"]), str(r["instance_token"])

        na_raw, nc_raw, ni_raw = _nearest(sub)
        np_raw, _, _ = _nearest(sub, sub["coarse_class"] == "pedestrian")
        nv_raw, _, _ = _nearest(sub, sub["coarse_class"] == "vehicle")
        na_f, nc_f, ni_f = _nearest(filt)
        np_f, _, _ = _nearest(filt, filt["coarse_class"] == "pedestrian")
        nv_f, _, _ = _nearest(filt, filt["coarse_class"] == "vehicle")

        row["nearest_any_raw_m"] = na_raw
        row["nearest_ped_raw_m"] = np_raw
        row["nearest_vehicle_raw_m"] = nv_raw
        row["nearest_any_filtered_m"] = na_f
        row["nearest_ped_filtered_m"] = np_f
        row["nearest_vehicle_filtered_m"] = nv_f
        row["nearest_object_distance_m"] = na_raw
        row["nearest_object_category"] = nc_raw
        row["nearest_object_instance_token"] = ni_raw
        row["nearest_ped_distance_m"] = np_raw
        row["nearest_vehicle_distance_m"] = nv_raw

        # Radius counts (raw)
        for rad, label in [(5, "5m"), (10, "10m"), (20, "20m")]:
            ins = sub[sub["distance_to_ego_m"] <= rad] if len(sub) else sub
            row[f"num_objects_within_{label}"] = len(ins)
            row[f"num_pedestrians_within_{label}"] = int(ins["is_pedestrian"].sum()) if len(ins) else 0
        row["num_vehicles_within_10m"] = (
            int(sub[(sub["distance_to_ego_m"] <= 10) & (sub["is_vehicle"])].shape[0]) if len(sub) else 0
        )
        row["num_vehicles_within_20m"] = (
            int(sub[(sub["distance_to_ego_m"] <= 20) & (sub["is_vehicle"])].shape[0]) if len(sub) else 0
        )

        # Forward corridor (uses planner geometry columns)
        fc = sub[sub["forward_corridor_flag"]] if len(sub) and "forward_corridor_flag" in sub.columns else sub.iloc[0:0]
        row["num_objects_in_forward_corridor"] = len(fc)
        row["num_pedestrians_in_forward_corridor"] = int(fc["coarse_class"].eq("pedestrian").sum()) if len(fc) else 0
        row["num_vehicles_in_forward_corridor"] = int(fc["coarse_class"].eq("vehicle").sum()) if len(fc) else 0
        if len(fc):
            row["closest_forward_corridor_object_distance_m"] = float(fc["distance_to_ego_m"].min())
            fcp = fc[fc["coarse_class"] == "pedestrian"]
            fcv = fc[fc["coarse_class"] == "vehicle"]
            row["closest_forward_corridor_ped_distance_m"] = float(fcp["distance_to_ego_m"].min()) if len(fcp) else np.nan
            row["closest_forward_corridor_vehicle_distance_m"] = float(fcv["distance_to_ego_m"].min()) if len(fcv) else np.nan
        else:
            row["closest_forward_corridor_object_distance_m"] = np.nan
            row["closest_forward_corridor_ped_distance_m"] = np.nan
            row["closest_forward_corridor_vehicle_distance_m"] = np.nan

        # Future ego path proximity
        fp = sub[sub["near_future_path_flag"]] if len(sub) and "near_future_path_flag" in sub.columns else sub.iloc[0:0]
        row["num_objects_near_future_ego_path"] = len(fp)
        row["num_pedestrians_near_future_ego_path"] = int(fp["coarse_class"].eq("pedestrian").sum()) if len(fp) else 0
        row["num_vehicles_near_future_ego_path"] = int(fp["coarse_class"].eq("vehicle").sum()) if len(fp) else 0
        if len(fp):
            row["closest_future_path_object_distance_m"] = float(fp["distance_to_ego_m"].min())
            fpp = fp[fp["coarse_class"] == "pedestrian"]
            fpv = fp[fp["coarse_class"] == "vehicle"]
            row["closest_future_path_ped_distance_m"] = float(fpp["distance_to_ego_m"].min()) if len(fpp) else np.nan
            row["closest_future_path_vehicle_distance_m"] = float(fpv["distance_to_ego_m"].min()) if len(fpv) else np.nan
        else:
            row["closest_future_path_object_distance_m"] = np.nan
            row["closest_future_path_ped_distance_m"] = np.nan
            row["closest_future_path_vehicle_distance_m"] = np.nan

        # Pedestrian conflict
        peds = sub[sub["coarse_class"] == "pedestrian"] if len(sub) else sub
        cand = (
            peds[peds["ped_conflict_score"] >= CONFLICT_PED_SCORE_THRESHOLD]
            if len(peds) and "ped_conflict_score" in peds.columns
            else peds.iloc[0:0]
        )
        row["num_candidate_conflict_pedestrians"] = len(cand)
        if len(cand):
            j = cand["ped_conflict_score"].idxmax()
            best = cand.loc[j]
            row["closest_conflict_ped_distance_m"] = float(cand["distance_to_ego_m"].min())
            row["highest_conflict_ped_instance_token"] = str(best["instance_token"])
            row["highest_conflict_ped_score"] = float(best["ped_conflict_score"])
        else:
            row["closest_conflict_ped_distance_m"] = np.nan
            row["highest_conflict_ped_instance_token"] = ""
            row["highest_conflict_ped_score"] = np.nan
        row["conflict_exists_binary"] = int(len(cand) > 0)

        # Top relevant (raw list, rank by relevance_score)
        top_rel = np.nan
        if len(sub) and "relevance_score" in sub.columns:
            srt = sub.sort_values("relevance_score", ascending=False)
            top = srt.iloc[0]
            row["top_relevant_object_instance_token"] = str(top["instance_token"])
            row["top_relevant_object_category"] = str(top["category_name"])
            row["top_relevant_object_coarse_class"] = str(top["coarse_class"])
            row["top_relevant_object_distance_m"] = float(top["distance_to_ego_m"])
            row["top_relevant_object_relevance_score"] = float(top["relevance_score"])
            top_rel = float(top["relevance_score"])
        else:
            row["top_relevant_object_instance_token"] = ""
            row["top_relevant_object_category"] = ""
            row["top_relevant_object_coarse_class"] = ""
            row["top_relevant_object_distance_m"] = np.nan
            row["top_relevant_object_relevance_score"] = np.nan

        ped_a, veh_a, ov_a = _alert_levels(np_f, nv_f, top_rel if not math.isnan(top_rel) else 0.0, len(cand) > 0, na_f)
        row["pedestrian_alert_level"] = ped_a
        row["vehicle_alert_level"] = veh_a
        row["overall_planner_attention_level"] = ov_a

        # Filtering diagnostics (per frame)
        zero_pts = 0
        low_sup = 0
        f_dist = 0
        f_sup = 0
        f_vis = 0
        if len(sub):
            for _, rr in sub.iterrows():
                if lidar_radar_sum(rr) == 0:
                    zero_pts += 1
                if low_support_flag(rr, str(rr["coarse_class"])):
                    low_sup += 1
                cc = str(rr["coarse_class"])
                if float(rr["distance_to_ego_m"]) > max_distance_for_coarse_class(cc):
                    f_dist += 1
                if not passes_support(rr, cc):
                    f_sup += 1
                if filtered_by_visibility(rr, cc):
                    f_vis += 1
        row["num_objects_zero_points"] = zero_pts
        row["num_objects_low_support"] = low_sup
        row["num_objects_filtered_out_by_distance"] = f_dist
        row["num_objects_filtered_out_by_support"] = f_sup
        row["num_objects_filtered_out_by_visibility"] = f_vis

        # Legacy in-path (longitudinal > 0, |lateral| <= threshold)
        if len(sub) and not math.isnan(eyaw):
            ip = sub[(sub["relative_longitudinal_m"] > 0) & (sub["relative_lateral_m"].abs() <= lateral_threshold_m)]
        else:
            ip = sub.iloc[0:0]
        if len(ip):
            j = ip["distance_to_ego_m"].idxmin()
            best = ip.loc[j]
            row["closest_in_path_object_distance_m"] = float(best["distance_to_ego_m"])
            row["closest_in_path_object_instance"] = str(best["instance_token"])
            row["closest_in_path_object_category"] = str(best["category_name"])
            ipp = ip[ip["coarse_class"] == "pedestrian"]
            iv = ip[ip["coarse_class"] == "vehicle"]
            row["closest_in_path_ped_distance_m"] = float(ipp["distance_to_ego_m"].min()) if len(ipp) else np.nan
            row["closest_in_path_vehicle_distance_m"] = float(iv["distance_to_ego_m"].min()) if len(iv) else np.nan
        else:
            row["closest_in_path_object_distance_m"] = np.nan
            row["closest_in_path_ped_distance_m"] = np.nan
            row["closest_in_path_vehicle_distance_m"] = np.nan
            row["closest_in_path_object_instance"] = ""
            row["closest_in_path_object_category"] = ""

        rows.append(row)
    return pd.DataFrame(rows)


def save_per_frame_jsons(
    out_dir: Path,
    samples: List[Dict[str, Any]],
    ego_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    planner_df: pd.DataFrame,
    lateral_threshold_m: float,
    log: ExtractionLog,
) -> None:
    per_dir = out_dir / "per_frame"
    per_dir.mkdir(parents=True, exist_ok=True)
    fp_params = {
        "FORWARD_CORRIDOR_LATERAL_M": FORWARD_CORRIDOR_LATERAL_M,
        "FUTURE_PATH_DISTANCE_THRESHOLD_M": FUTURE_PATH_DISTANCE_THRESHOLD_M,
        "FUTURE_PATH_LOOKAHEAD_FRAMES": FUTURE_PATH_LOOKAHEAD_FRAMES,
        "MAX_FILTER_DISTANCE_M": MAX_FILTER_DISTANCE_M,
        "MAX_PEDESTRIAN_DISTANCE_M": MAX_PEDESTRIAN_DISTANCE_M,
        "MAX_VEHICLE_DISTANCE_M": MAX_VEHICLE_DISTANCE_M,
        "MAX_STATIC_DISTANCE_M": MAX_STATIC_DISTANCE_M,
        "CONFLICT_PED_SCORE_THRESHOLD": CONFLICT_PED_SCORE_THRESHOLD,
        "IN_PATH_LATERAL_THRESHOLD_M": lateral_threshold_m,
    }
    for i, s in enumerate(samples):
        er = ego_df[ego_df["frame_index"] == i].iloc[0]
        sub = objects_df[objects_df["frame_index"] == i]
        pr = planner_df[planner_df["frame_index"] == i].iloc[0]
        stl = sub.sort_values("distance_to_ego_m")
        nearby = stl.head(12)
        ip = sub[(sub["relative_longitudinal_m"] > 0) & (sub["relative_lateral_m"].abs() <= lateral_threshold_m)]
        cc = sub.groupby("category_name").size().to_dict() if len(sub) else {}
        filt = sub[sub["filtered_planner_object_flag"]] if "filtered_planner_object_flag" in sub.columns else sub.iloc[0:0]
        fc = sub[sub["forward_corridor_flag"]] if "forward_corridor_flag" in sub.columns else sub.iloc[0:0]
        fp = sub[sub["near_future_path_flag"]] if "near_future_path_flag" in sub.columns else sub.iloc[0:0]
        peds = sub[sub["coarse_class"] == "pedestrian"]
        cand = (
            peds[peds["ped_conflict_score"] >= CONFLICT_PED_SCORE_THRESHOLD]
            if len(peds) and "ped_conflict_score" in peds.columns
            else peds.iloc[0:0]
        )
        top5 = sub.sort_values("relevance_score", ascending=False).head(5) if len(sub) else sub
        top_objs: List[Dict[str, Any]] = []
        for _, rr in top5.iterrows():
            dpath = rr.get("future_path_distance_m")
            top_objs.append(
                {
                    "instance_token": str(rr["instance_token"]),
                    "category_name": str(rr["category_name"]),
                    "coarse_class": str(rr["coarse_class"]),
                    "distance_to_ego_m": float(rr["distance_to_ego_m"]),
                    "relative_longitudinal_m": float(rr["relative_longitudinal_m"]),
                    "relative_lateral_m": float(rr["relative_lateral_m"]),
                    "min_distance_to_future_path_m": float(dpath) if pd.notna(dpath) else None,
                    "relevance_score": float(rr["relevance_score"]) if "relevance_score" in rr.index else None,
                    "ped_conflict_score": float(rr["ped_conflict_score"])
                    if rr["coarse_class"] == "pedestrian"
                    else None,
                    "why_relevant": why_relevant_for_object(rr),
                }
            )
        val_notes: List[str] = []
        if len(sub) == 0:
            val_notes.append("no annotations this frame")
        if len(sub) and sub["relevance_score"].isna().all():
            val_notes.append("relevance scores unavailable")
        frame_obj: Dict[str, Any] = {
            "frame_index": i,
            "sample_token": s["token"],
            "timestamp_us": int(s["timestamp"]),
            "timestamp_s": int(s["timestamp"]) / 1e6,
            "delta_t_s": (float(er["delta_t_s"]) if not pd.isna(er["delta_t_s"]) else None)
            if "delta_t_s" in er.index
            else None,
            "prev_sample_token": s.get("prev") or "",
            "next_sample_token": s.get("next") or "",
            "ego_state": {
                "x": float(er["ego_x"]) if not math.isnan(er["ego_x"]) else None,
                "y": float(er["ego_y"]) if not math.isnan(er["ego_y"]) else None,
                "z": float(er["ego_z"]) if not math.isnan(er["ego_z"]) else None,
                "yaw_rad": float(er["ego_yaw_rad"]) if not math.isnan(er["ego_yaw_rad"]) else None,
                "speed_mps": float(er["ego_speed_mps"]) if not math.isnan(er["ego_speed_mps"]) else None,
                "accel_mps2": float(er["ego_accel_est_mps2"]) if "ego_accel_est_mps2" in er.index and not pd.isna(er.get("ego_accel_est_mps2")) else None,
                "cumulative_distance_m": float(er["cumulative_distance_m"])
                if not math.isnan(er["cumulative_distance_m"])
                else None,
            },
            "planner_summary": pr.to_dict(),
            "filtered_object_count": int(len(filt)),
            "filtered_pedestrian_count": int(filt["coarse_class"].eq("pedestrian").sum()) if len(filt) else 0,
            "filtered_vehicle_count": int(filt["coarse_class"].eq("vehicle").sum()) if len(filt) else 0,
            "raw_object_count": int(len(sub)),
            "nearby_objects": nearby.to_dict(orient="records"),
            "nearest_object_summaries": {
                "nearest_any_raw_m": float(pr["nearest_any_raw_m"]) if not pd.isna(pr.get("nearest_any_raw_m")) else None,
                "nearest_ped_raw_m": float(pr["nearest_ped_raw_m"]) if not pd.isna(pr.get("nearest_ped_raw_m")) else None,
                "nearest_vehicle_raw_m": float(pr["nearest_vehicle_raw_m"])
                if not pd.isna(pr.get("nearest_vehicle_raw_m"))
                else None,
                "nearest_any_filtered_m": float(pr["nearest_any_filtered_m"])
                if not pd.isna(pr.get("nearest_any_filtered_m"))
                else None,
                "nearest_ped_filtered_m": float(pr["nearest_ped_filtered_m"])
                if not pd.isna(pr.get("nearest_ped_filtered_m"))
                else None,
                "nearest_vehicle_filtered_m": float(pr["nearest_vehicle_filtered_m"])
                if not pd.isna(pr.get("nearest_vehicle_filtered_m"))
                else None,
            },
            "class_counts_this_frame": {k: int(v) for k, v in cc.items()},
            "in_path_objects": ip.sort_values("distance_to_ego_m").head(20).to_dict(orient="records"),
            "in_path_lateral_threshold_m": lateral_threshold_m,
            "forward_corridor_objects": fc.sort_values("distance_to_ego_m").head(25).to_dict(orient="records")
            if len(fc)
            else [],
            "future_path_objects": fp.sort_values("distance_to_ego_m").head(25).to_dict(orient="records")
            if len(fp)
            else [],
            "candidate_conflict_pedestrians": cand.sort_values("ped_conflict_score", ascending=False)
            .head(15)
            .to_dict(orient="records")
            if len(cand)
            else [],
            "top_relevant_objects": top_objs,
            "planner_attention_summary": planner_attention_sentence(pr),
            "filtering_parameters_used": fp_params,
            "validation_notes_this_frame": val_notes,
        }
        fn = per_dir / f"frame_{i:03d}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(frame_obj, f, indent=2, default=str)
        log.add_file(str(fn))


def build_class_counts(objects_df: pd.DataFrame, n_frames: int) -> Dict[str, Any]:
    ann_by_cat = (
        objects_df.groupby("category_name").size().to_dict() if len(objects_df) else {}
    )
    uniq_inst = (
        objects_df.groupby("category_name")["instance_token"].nunique().to_dict() if len(objects_df) else {}
    )
    coarse_agg = (
        objects_df.groupby("coarse_class").size().to_dict() if len(objects_df) else {}
    )
    per_frame: List[Dict[str, Any]] = []
    for fi in range(n_frames):
        sub = objects_df[objects_df["frame_index"] == fi]
        per_frame.append(
            {
                "frame_index": fi,
                "total_objects": len(sub),
                "by_category": sub.groupby("category_name").size().to_dict() if len(sub) else {},
                "by_coarse": sub.groupby("coarse_class").size().to_dict() if len(sub) else {},
            }
        )
    # Top dynamic: exclude static_object from frequency
    dyn = objects_df[objects_df["coarse_class"] != "static_object"] if len(objects_df) else objects_df
    top_dyn = (
        dyn.groupby("category_name").size().sort_values(ascending=False).head(12).to_dict() if len(dyn) else {}
    )
    return {
        "annotation_counts_by_category": {k: int(v) for k, v in ann_by_cat.items()},
        "unique_instance_counts_by_category": {k: int(v) for k, v in uniq_inst.items()},
        "aggregated_counts_by_coarse_class": {k: int(v) for k, v in coarse_agg.items()},
        "per_frame_count_summary": per_frame,
        "top_dynamic_categories": [{"category": k, "annotation_rows": int(v)} for k, v in top_dyn.items()],
    }


def make_global_plots(
    out_plots: Path,
    ego_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    planner_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    nmap: Any,
    patch_box: Tuple[float, float, float, float],
    log: ExtractionLog,
) -> None:
    import matplotlib.pyplot as plt

    out_plots.mkdir(parents=True, exist_ok=True)
    PED_COLOR = "#ff2b7a"
    VEH_COLOR = "#00a7ff"
    PED_DARK = "#c40058"

    def _legend_topdown(ax: Any, map_handles: List[Any]) -> None:
        h2, _ = ax.get_legend_handles_labels()
        ax.legend(handles=map_handles + list(h2), loc="upper right", fontsize=8, ncol=2, framealpha=0.92)

    # ego_trajectory
    fig, ax = plt.subplots(figsize=(8, 8))
    map_handles: List[Any] = []
    if nmap is not None:
        map_handles = draw_map_expansion_layers(ax, nmap, patch_box, 0.0, zorder=0.0)
    ex, ey = ego_df["ego_x"].values, ego_df["ego_y"].values
    ax.plot(ex, ey, "b-", lw=2, label="Ego path", zorder=3)
    ax.scatter([ex[0]], [ey[0]], c="green", s=80, zorder=5, label="Start")
    ax.scatter([ex[-1]], [ey[-1]], c="red", s=80, zorder=5, label="End")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X global (m)")
    ax.set_ylabel("Y global (m)")
    ax.set_title("Ego trajectory (global XY)")
    h2, _ = ax.get_legend_handles_labels()
    ax.legend(handles=map_handles + list(h2), loc="best", fontsize=9, ncol=2, framealpha=0.92)
    ax.grid(True, alpha=0.3)
    p = out_plots / "ego_trajectory.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.add_file(str(p))

    # ego_speed_profile
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ego_df["frame_index"], ego_df["ego_speed_mps"], "-", color="C0")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Ego speed (finite-difference estimate)")
    ax.grid(True, alpha=0.3)
    p = out_plots / "ego_speed_profile.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.add_file(str(p))

    def _plot_topdown(
        agents_only: bool,
        fname: str,
        title: str,
        emphasize_ped: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        map_handles: List[Any] = []
        if nmap is not None:
            map_handles = draw_map_expansion_layers(ax, nmap, patch_box, 0.0, zorder=0.0)
        ax.plot(ex, ey, "-", color="navy", lw=3, label="Ego path", zorder=3)
        ax.scatter([ex[0]], [ey[0]], c="green", s=100, zorder=6, marker="s", label="Start")
        ax.scatter([ex[-1]], [ey[-1]], c="red", s=100, zorder=6, marker="X", label="End")
        ped_labeled = False
        veh_labeled = False
        oth_labeled = False
        for _, g in agent_df.groupby("instance_token"):
            cc = g["coarse_class"].iloc[0]
            if agents_only and cc not in ("pedestrian", "vehicle"):
                continue
            gx, gy = g["x"].values, g["y"].values
            if cc == "pedestrian":
                lbl = "Pedestrian track" if not ped_labeled else None
                ped_labeled = True
                ax.plot(gx, gy, "-", color=PED_COLOR, alpha=0.82, lw=1.6, label=lbl, zorder=2.2)
            elif cc == "vehicle":
                lbl = "Vehicle track" if not veh_labeled else None
                veh_labeled = True
                ax.plot(gx, gy, "-", color=VEH_COLOR, alpha=0.72, lw=1.45, label=lbl, zorder=2.1)
            else:
                lbl = "Other / static track" if not oth_labeled else None
                oth_labeled = True
                ax.plot(gx, gy, "-", color="0.6", alpha=0.25, lw=0.8, label=lbl, zorder=1.5)
        if emphasize_ped:
            for _, g in agent_df[agent_df["coarse_class"] == "pedestrian"].groupby("instance_token"):
                ax.plot(g["x"].values, g["y"].values, "-", color=PED_DARK, alpha=0.95, lw=2.0, zorder=2.6)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X global (m)")
        ax.set_ylabel("Y global (m)")
        ax.set_title(title)
        _legend_topdown(ax, map_handles)
        ax.grid(True, alpha=0.25)
        path = out_plots / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.add_file(str(path))

    _plot_topdown(
        True,
        "agent_trajectories_topdown.png",
        "Dynamic agents (ped=pink, vehicle=cyan, ego=navy)",
    )
    _plot_topdown(
        False,
        "ego_and_agents_topdown.png",
        "Ego + all agents (pedestrians emphasized)",
        emphasize_ped=True,
    )
    _plot_topdown(
        False,
        "scene_overview.png",
        "Scene overview: ego + pedestrian + vehicle paths",
        emphasize_ped=False,
    )

    # distance_to_nearest_agents (raw distances)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(planner_df["frame_index"], planner_df["nearest_any_raw_m"], label="nearest any (raw)", color="k")
    ax.plot(planner_df["frame_index"], planner_df["nearest_ped_raw_m"], label="nearest ped (raw)", color=PED_COLOR)
    ax.plot(planner_df["frame_index"], planner_df["nearest_vehicle_raw_m"], label="nearest vehicle (raw)", color=VEH_COLOR)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Nearest-agent distances over time (raw annotations)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_plots / "distance_to_nearest_agents.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.add_file(str(p))

    # class_presence
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(planner_df["frame_index"], planner_df["raw_num_objects_total"], label="total (raw)", color="k")
    ax.plot(planner_df["frame_index"], planner_df["raw_num_pedestrians"], label="pedestrians (raw)", color=PED_COLOR)
    ax.plot(planner_df["frame_index"], planner_df["raw_num_vehicles"], label="vehicles (raw)", color=VEH_COLOR)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Count")
    ax.set_title("Object counts per frame (raw)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_plots / "class_presence_over_time.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.add_file(str(p))

    # relevance_over_time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(planner_df["frame_index"], planner_df["top_relevant_object_relevance_score"], color="purple", lw=1.5)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Score")
    ax.set_title("Top object relevance score over time")
    ax.grid(True, alpha=0.3)
    p = out_plots / "relevance_over_time.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.add_file(str(p))

    # conflict_pedestrians_over_time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(planner_df["frame_index"], planner_df["num_candidate_conflict_pedestrians"], color="orangered", alpha=0.75)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Count")
    ax.set_title("Candidate conflict pedestrians per frame (heuristic)")
    ax.grid(True, alpha=0.3, axis="y")
    p = out_plots / "conflict_pedestrians_over_time.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.add_file(str(p))

    # ego_path_conflict_overview (top-down ego + markers where conflict exists)
    fig, ax = plt.subplots(figsize=(10, 10))
    map_handles_cf: List[Any] = []
    if nmap is not None:
        map_handles_cf = draw_map_expansion_layers(ax, nmap, patch_box, 0.0, zorder=0.0)
    ax.plot(ex, ey, "-", color="navy", lw=3, label="Ego path")
    ax.scatter([ex[0]], [ey[0]], c="green", s=100, zorder=6, marker="s", label="Start")
    ax.scatter([ex[-1]], [ey[-1]], c="red", s=100, zorder=6, marker="X", label="End")
    cf = planner_df[planner_df["conflict_exists_binary"] == 1]
    if len(cf):
        xi = cf["frame_index"].astype(int).values
        ax.scatter(ex[xi], ey[xi], c="magenta", s=36, alpha=0.7, label="conflict flag frame", zorder=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X global (m)")
    ax.set_ylabel("Y global (m)")
    ax.set_title("Ego path with frames where conflict heuristic triggered")
    h2, _ = ax.get_legend_handles_labels()
    ax.legend(handles=map_handles_cf + list(h2), loc="upper right", fontsize=8, ncol=2, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    p = out_plots / "ego_path_conflict_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.add_file(str(p))


def make_frame_plots(
    out_plots: Path,
    ego_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    samples: List[Dict[str, Any]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    nmap: Any,
    patch_box: Tuple[float, float, float, float],
    log: ExtractionLog,
) -> None:
    import matplotlib.pyplot as plt

    PED_COLOR = "#ff2b7a"
    VEH_COLOR = "#00a7ff"
    PED_DARK = "#c40058"
    VEH_DARK = "#0076b6"
    cmap = {
        "pedestrian": PED_COLOR,
        "vehicle": VEH_COLOR,
        "static_object": "tab:brown",
        "other": "0.45",
    }
    ego_xy = ego_df[["ego_x", "ego_y"]].to_numpy(dtype=float)
    for i, s in enumerate(samples):
        er = ego_df[ego_df["frame_index"] == i].iloc[0]
        sub = objects_df[objects_df["frame_index"] == i]
        fig, ax = plt.subplots(figsize=(9, 9))
        map_handles_fr: List[Any] = []
        if nmap is not None:
            map_handles_fr = draw_map_expansion_layers(ax, nmap, patch_box, 0.0, zorder=0.0)
        ex, ey = er["ego_x"], er["ego_y"]
        yaw = er["ego_yaw_rad"]
        # Ego trajectory history up to current frame (used by frame PNGs and replay GIF).
        hist = ego_xy[: i + 1]
        hist_mask = np.isfinite(hist[:, 0]) & np.isfinite(hist[:, 1])
        hist = hist[hist_mask]
        if len(hist) >= 2:
            ax.plot(
                hist[:, 0],
                hist[:, 1],
                "-",
                color="navy",
                lw=2.2,
                alpha=0.95,
                label="ego history",
                zorder=3,
            )
            ax.plot(hist[-1, 0], hist[-1, 1], "o", color="navy", markersize=5, zorder=4)
        elif len(hist) == 1:
            ax.plot(hist[0, 0], hist[0, 1], "o", color="navy", markersize=5, zorder=4)
        if not math.isnan(ex):
            ax.plot(ex, ey, "s", color="navy", markersize=10, label="ego current", zorder=5)
            if not math.isnan(yaw):
                ax.arrow(
                    ex,
                    ey,
                    3 * math.cos(yaw),
                    3 * math.sin(yaw),
                    head_width=1.0,
                    head_length=1.2,
                    fc="navy",
                    ec="navy",
                    length_includes_head=True,
                )
        seen_cc: set = set()
        for _, r in sub.iterrows():
            col = cmap.get(r["coarse_class"], "gray")
            cc = str(r["coarse_class"])
            lbl = None
            if cc not in seen_cc:
                lbl = {"pedestrian": "Pedestrian", "vehicle": "Vehicle", "static_object": "Static object"}.get(cc, "Other")
                seen_cc.add(cc)
            ax.plot(
                r["obj_x"],
                r["obj_y"],
                "o",
                color=col,
                markersize=5.6 if cc in ("pedestrian", "vehicle") else 4.0,
                markeredgecolor=(PED_DARK if cc == "pedestrian" else VEH_DARK if cc == "vehicle" else "none"),
                markeredgewidth=0.35 if cc in ("pedestrian", "vehicle") else 0.0,
                alpha=0.92 if cc in ("pedestrian", "vehicle") else 0.82,
                label=lbl,
                zorder=4.2 if cc in ("pedestrian", "vehicle") else 4.0,
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ts = int(s["timestamp"]) / 1e6
        ax.set_title(f"Frame {i:03d}  |  t = {ts:.3f} s")
        ax.set_xlabel("X global (m)")
        ax.set_ylabel("Y global (m)")
        ax.grid(True, alpha=0.25)
        h2, _ = ax.get_legend_handles_labels()
        ax.legend(handles=map_handles_fr + list(h2), loc="upper left", fontsize=7, ncol=2, framealpha=0.92)
        p = out_plots / f"frame_{i:03d}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.add_file(str(p))




def make_animation_from_csv(
    out_gif: Path,
    ego_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    samples: List[Dict[str, Any]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    nmap: Any,
    patch_box: Tuple[float, float, float, float],
    log: ExtractionLog,
) -> None:
    """Build replay GIF directly from extracted CSV tables without saving per-frame PNG files."""
    import io
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    PED_COLOR = "#ff2b7a"
    VEH_COLOR = "#00a7ff"
    PED_DARK = "#c40058"
    VEH_DARK = "#0076b6"
    cmap = {
        "pedestrian": PED_COLOR,
        "vehicle": VEH_COLOR,
        "static_object": "tab:brown",
        "other": "0.45",
    }

    ego_xy = ego_df[["ego_x", "ego_y"]].to_numpy(dtype=float)
    frames = []
    try:
        for i, s in enumerate(samples):
            er = ego_df[ego_df["frame_index"] == i].iloc[0]
            sub = objects_df[objects_df["frame_index"] == i]
            fig, ax = plt.subplots(figsize=(9, 9))
            map_handles_fr: List[Any] = []
            if nmap is not None:
                map_handles_fr = draw_map_expansion_layers(ax, nmap, patch_box, 0.0, zorder=0.0)
            ex, ey = er["ego_x"], er["ego_y"]
            yaw = er["ego_yaw_rad"]
            hist = ego_xy[: i + 1]
            hist_mask = np.isfinite(hist[:, 0]) & np.isfinite(hist[:, 1])
            hist = hist[hist_mask]
            if len(hist) >= 2:
                ax.plot(hist[:, 0], hist[:, 1], "-", color="navy", lw=2.2, alpha=0.95, label="ego history", zorder=3)
                ax.plot(hist[-1, 0], hist[-1, 1], "o", color="navy", markersize=5, zorder=4)
            elif len(hist) == 1:
                ax.plot(hist[0, 0], hist[0, 1], "o", color="navy", markersize=5, zorder=4)
            if not math.isnan(ex):
                ax.plot(ex, ey, "s", color="navy", markersize=10, label="ego current", zorder=5)
                if not math.isnan(yaw):
                    ax.arrow(ex, ey, 3 * math.cos(yaw), 3 * math.sin(yaw), head_width=1.0, head_length=1.2,
                             fc="navy", ec="navy", length_includes_head=True)
            seen_cc: set = set()
            for _, r in sub.iterrows():
                col = cmap.get(r["coarse_class"], "gray")
                cc = str(r["coarse_class"])
                lbl = None
                if cc not in seen_cc:
                    lbl = {"pedestrian": "Pedestrian", "vehicle": "Vehicle", "static_object": "Static object"}.get(cc, "Other")
                    seen_cc.add(cc)
                ax.plot(r["obj_x"], r["obj_y"], "o", color=col,
                        markersize=5.6 if cc in ("pedestrian", "vehicle") else 4.0,
                        markeredgecolor=(PED_DARK if cc == "pedestrian" else VEH_DARK if cc == "vehicle" else "none"),
                        markeredgewidth=0.35 if cc in ("pedestrian", "vehicle") else 0.0,
                        alpha=0.92 if cc in ("pedestrian", "vehicle") else 0.82,
                        label=lbl, zorder=4.2 if cc in ("pedestrian", "vehicle") else 4.0)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            ts = int(s["timestamp"]) / 1e6
            ax.set_title(f"Frame {i:03d}  |  t = {ts:.3f} s")
            ax.set_xlabel("X global (m)")
            ax.set_ylabel("Y global (m)")
            ax.grid(True, alpha=0.25)
            h2, _ = ax.get_legend_handles_labels()
            ax.legend(handles=map_handles_fr + list(h2), loc="upper left", fontsize=7, ncol=2, framealpha=0.92)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out_gif), frames, duration=0.2)
        log.gif_ok = True
        log.add_file(str(out_gif))
    except Exception as exc:
        log.gif_ok = False
        log.gif_error = str(exc)
        log.warn(f"GIF creation failed: {exc}\n{traceback.format_exc()}")
def make_gif(frame_paths: List[Path], out_gif: Path, log: ExtractionLog) -> None:
    try:
        import imageio.v2 as imageio

        frames = [imageio.imread(str(p)) for p in frame_paths]
        imageio.mimwrite(str(out_gif), frames, duration=0.2)
        log.gif_ok = True
        log.add_file(str(out_gif))
    except Exception as exc:
        log.gif_ok = False
        log.gif_error = str(exc)
        log.warn(f"GIF creation failed: {exc}\n{traceback.format_exc()}")


def write_log(log: ExtractionLog, path: Path) -> None:
    lines = [
        f"start_time: {log.start_wall}",
        f"end_time: {log.end_wall}",
        f"config: {log.config_path}",
        f"scene: {log.scene_name} ({log.scene_token})",
        f"dataroot: {log.dataroot}",
        f"version: {log.version}",
        f"gif_ok: {log.gif_ok}",
        f"gif_error: {log.gif_error}",
        "",
    ]
    st = log.run_stats
    if st:
        lines.append("diagnostics:")
        for k in sorted(st.keys()):
            lines.append(f"  {k}: {st[k]}")
        lines.append("")
    lines.append("filtering_thresholds_used:")
    lines.append(f"  MAX_FILTER_DISTANCE_M={MAX_FILTER_DISTANCE_M}")
    lines.append(f"  MAX_PEDESTRIAN_DISTANCE_M={MAX_PEDESTRIAN_DISTANCE_M}")
    lines.append(f"  MAX_VEHICLE_DISTANCE_M={MAX_VEHICLE_DISTANCE_M}")
    lines.append(f"  MAX_STATIC_DISTANCE_M={MAX_STATIC_DISTANCE_M}")
    lines.append(f"  FORWARD_CORRIDOR_LATERAL_M={FORWARD_CORRIDOR_LATERAL_M}")
    lines.append(f"  FUTURE_PATH_DISTANCE_THRESHOLD_M={FUTURE_PATH_DISTANCE_THRESHOLD_M}")
    lines.append(f"  FUTURE_PATH_LOOKAHEAD_FRAMES={FUTURE_PATH_LOOKAHEAD_FRAMES}")
    lines.append(f"  CONFLICT_PED_SCORE_THRESHOLD={CONFLICT_PED_SCORE_THRESHOLD}")
    lines.append(f"  LOW_SUPPORT_THRESHOLD={LOW_SUPPORT_THRESHOLD}")
    lines.append("")
    lines.append("assumptions_used:")
    lines.append("  - Global XY from nuScenes ego_pose / annotation translations (meters).")
    lines.append("  - Future ego path polyline uses upcoming keyframe poses (not predicted dynamics).")
    lines.append("  - Relevance and conflict scores are heuristic tunables, not learned detectors.")
    lines.append("")
    if st.get("validation_summary_embedded"):
        lines.append("validation_summary:")
        for line in st["validation_summary_embedded"]:
            lines.append(f"  {line}")
        lines.append("")
    lines.append("relevance_scoring_summary:")
    lines.append(
        f"  relevance rows: {st.get('relevance_scoring_rows', 'n/a')}; "
        f"filtered planner True count: {st.get('total_filtered_planner_true', 'n/a')}"
    )
    lines.append("")
    lines.append("warnings:")
    lines.extend(f"  - {w}" for w in log.warnings)
    lines.append("")
    lines.append("files_written (partial list):")
    lines.extend(f"  - {f}" for f in log.files_written[:500])
    if len(log.files_written) > 500:
        lines.append(f"  ... ({len(log.files_written) - 500} more)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def compute_xy_bounds(
    ego_df: pd.DataFrame, objects_df: pd.DataFrame, margin_m: float = 12.0
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for v in ego_df["ego_x"].values:
        if not math.isnan(v):
            xs.append(v)
    for v in ego_df["ego_y"].values:
        if not math.isnan(v):
            ys.append(v)
    if len(objects_df):
        xs.extend(objects_df["obj_x"].astype(float).tolist())
        ys.extend(objects_df["obj_y"].astype(float).tolist())
    if not xs:
        return (-50, 50), (-50, 50)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return (xmin - margin_m, xmax + margin_m), (ymin - margin_m, ymax + margin_m)



def compute_validation_and_run_stats(
    samples: List[Dict[str, Any]],
    sample_tl: pd.DataFrame,
    ego_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    planner_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Cross-check extraction consistency for lean CSV-first outputs."""
    n = len(samples)
    ts = sample_tl["timestamp_us"].values
    ts_ok = bool(len(ts) > 1 and np.all(np.diff(ts.astype(np.int64)) > 0))
    ego_ok = len(ego_df) == n
    plan_ok = len(planner_df) == n

    zero_lr = 0
    if len(objects_df):
        for _, r in objects_df.iterrows():
            if lidar_radar_sum(r) == 0:
                zero_lr += 1
    n_fc = int(objects_df["forward_corridor_flag"].sum()) if len(objects_df) and "forward_corridor_flag" in objects_df.columns else 0
    n_fp = int(objects_df["near_future_path_flag"].sum()) if len(objects_df) and "near_future_path_flag" in objects_df.columns else 0
    n_conf = 0
    if len(objects_df) and "ped_conflict_score" in objects_df.columns:
        n_conf = int(
            ((objects_df["coarse_class"] == "pedestrian") & (objects_df["ped_conflict_score"] >= CONFLICT_PED_SCORE_THRESHOLD)).sum()
        )
    fut_fin = 0.0
    if len(objects_df) and "future_path_distance_m" in objects_df.columns:
        s = objects_df["future_path_distance_m"]
        fut_fin = float(s[np.isfinite(s)].shape[0] / max(len(s), 1))

    frames_no_obj = int((planner_df["raw_num_objects_total"] == 0).sum()) if "raw_num_objects_total" in planner_df.columns else 0
    frames_no_ped = int((planner_df["raw_num_pedestrians"] == 0).sum()) if "raw_num_pedestrians" in planner_df.columns else 0
    frames_no_veh = int((planner_df["raw_num_vehicles"] == 0).sum()) if "raw_num_vehicles" in planner_df.columns else 0

    cam_deg_nonzero = bool(objects_df["camera_degradable_flag"].any()) if len(objects_df) and "camera_degradable_flag" in objects_df.columns else False

    validation: Dict[str, Any] = {
        "timestamps_strictly_increasing": ts_ok,
        "ego_timeline_matches_sample_count": ego_ok,
        "planner_observations_matches_sample_count": plan_ok,
        "object_rows_nonzero": len(objects_df) > 0,
        "pedestrian_rows_nonzero": bool((objects_df["coarse_class"] == "pedestrian").any()) if len(objects_df) else False,
        "vehicle_rows_nonzero": bool((objects_df["coarse_class"] == "vehicle").any()) if len(objects_df) else False,
        "camera_degradable_rows_nonzero": cam_deg_nonzero,
        "future_path_finite_fraction": round(fut_fin, 4),
        "relevance_scores_computed_all_rows": bool(
            len(objects_df) == 0 or (objects_df["relevance_score"].notna().all() if "relevance_score" in objects_df.columns else False)
        ),
        "camera_awareness_fields_present": bool(
            len(objects_df) == 0 or {"camera_candidate_flag", "camera_degradable_flag", "camera_dropout_priority"}.issubset(objects_df.columns)
        ),
        "frame_counts_consistent": ego_ok and plan_ok,
    }
    run_stats: Dict[str, Any] = {
        "number_of_frames": n,
        "number_of_frames_with_no_objects": frames_no_obj,
        "number_of_frames_with_no_pedestrians": frames_no_ped,
        "number_of_frames_with_no_vehicles": frames_no_veh,
        "number_of_annotations_with_zero_lidar_and_zero_radar": zero_lr,
        "number_of_objects_classified_in_forward_corridor": n_fc,
        "number_of_objects_classified_near_future_path": n_fp,
        "number_of_pedestrian_annotation_rows_as_conflict_candidates": n_conf,
        "future_path_distance_finite_fraction": fut_fin,
        "relevance_scoring_rows": len(objects_df),
        "total_filtered_planner_true": int(objects_df["filtered_planner_object_flag"].sum()) if len(objects_df) and "filtered_planner_object_flag" in objects_df.columns else 0,
        "total_conflict_pedestrian_observations": int(planner_df["conflict_exists_binary"].sum()) if len(planner_df) and "conflict_exists_binary" in planner_df.columns else 0,
        "camera_candidate_rows": int(objects_df["camera_candidate_flag"].sum()) if len(objects_df) and "camera_candidate_flag" in objects_df.columns else 0,
        "camera_degradable_rows": int(objects_df["camera_degradable_flag"].sum()) if len(objects_df) and "camera_degradable_flag" in objects_df.columns else 0,
    }
    return validation, run_stats


def build_scene_summary(
    scene_name: str,
    ego_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    planner_df: pd.DataFrame,
    log: ExtractionLog,
    scene_description: str,
    validation: Dict[str, Any],
    run_stats: Dict[str, Any],
) -> Dict[str, Any]:
    n_frames = len(ego_df)
    dur = float(ego_df["timestamp_s"].iloc[-1] - ego_df["timestamp_s"].iloc[0]) if n_frames > 1 else 0.0
    dist = float(ego_df["cumulative_distance_m"].iloc[-1]) if n_frames else 0.0
    speeds = ego_df["ego_speed_mps"].replace([np.inf, -np.inf], np.nan).dropna()
    avg_sp = float(speeds.mean()) if len(speeds) else float("nan")
    max_sp = float(speeds.max()) if len(speeds) else float("nan")
    uniq_inst = objects_df["instance_token"].nunique() if len(objects_df) else 0
    ped_inst = (
        objects_df[objects_df["coarse_class"] == "pedestrian"]["instance_token"].nunique() if len(objects_df) else 0
    )
    veh_inst = objects_df[objects_df["coarse_class"] == "vehicle"]["instance_token"].nunique() if len(objects_df) else 0
    min_d = float(objects_df["distance_to_ego_m"].min()) if len(objects_df) else float("nan")
    ped_df = objects_df[objects_df["coarse_class"] == "pedestrian"]
    veh_df = objects_df[objects_df["coarse_class"] == "vehicle"]
    static_df = objects_df[objects_df["coarse_class"] == "static_object"]
    dyn_df = objects_df[objects_df["coarse_class"].isin(["pedestrian", "vehicle", "other"])]
    min_p = float(ped_df["distance_to_ego_m"].min()) if len(ped_df) else float("nan")
    min_v = float(veh_df["distance_to_ego_m"].min()) if len(veh_df) else float("nan")
    min_s = float(static_df["distance_to_ego_m"].min()) if len(static_df) else float("nan")
    min_dyn = float(dyn_df["distance_to_ego_m"].min()) if len(dyn_df) else float("nan")
    most_common = (
        objects_df["category_name"].value_counts().head(8).to_dict() if len(objects_df) else {}
    )
    notes: List[str] = []
    if ego_df["ego_x"].isna().any():
        notes.append("Some frames missing ego pose (check LIDAR_TOP / sample_data).")
    notes.extend(log.warnings[:20])

    objs_per_frame = (
        objects_df.groupby("frame_index").size() if len(objects_df) else pd.Series(dtype=int)
    )
    max_obj_frame = int(objs_per_frame.max()) if len(objs_per_frame) else 0
    avg_obj_frame = float(objs_per_frame.mean()) if len(objs_per_frame) else 0.0
    ped_counts = planner_df["raw_num_pedestrians"] if "raw_num_pedestrians" in planner_df.columns else []
    veh_counts = planner_df["raw_num_vehicles"] if "raw_num_vehicles" in planner_df.columns else []
    peak_ped = int(ped_counts.max()) if len(planner_df) else 0
    peak_veh = int(veh_counts.max()) if len(planner_df) else 0

    ex0, ey0 = float(ego_df["ego_x"].iloc[0]), float(ego_df["ego_y"].iloc[0])
    ex1, ey1 = float(ego_df["ego_x"].iloc[-1]), float(ego_df["ego_y"].iloc[-1])

    scene_desc_lower = (scene_description or "").lower()
    motion_bits: List[str] = []
    if "intersection" in scene_desc_lower or "turn" in scene_desc_lower:
        motion_bits.append("intersection / turn context")
    if peak_ped > 0:
        motion_bits.append("nearby pedestrians")
    if peak_veh > 0:
        motion_bits.append("vehicles")
    if static_df.shape[0] > 100:
        motion_bits.append("construction or roadside objects")
    motion_pattern = "ego moving through an urban scene"
    if motion_bits:
        motion_pattern += " with " + ", ".join(motion_bits)

    eq_notes: List[str] = []
    if validation.get("timestamps_strictly_increasing"):
        eq_notes.append("Sample timestamps are strictly increasing.")
    else:
        eq_notes.append("Timestamp check reported issues; see validation_summary.")
    if validation.get("frame_counts_consistent"):
        eq_notes.append("Timeline row counts match scene sample count.")
    if run_stats.get("relevance_scoring_rows", 0) == len(objects_df):
        eq_notes.append("Relevance scores computed for every annotation row.")
    if run_stats.get("future_path_distance_finite_fraction", 0) >= 0.99:
        eq_notes.append("Future-path distances mostly finite (end-of-scene truncation expected).")

    pr_notes: List[str] = [
        "Forward corridor and future-path metrics are geometric heuristics, not perception.",
        "Filtered subsets favor proximity, sensor support, and visibility rules.",
    ]
    if int(run_stats.get("total_conflict_pedestrian_observations", 0)) > 0:
        pr_notes.append("Conflict-oriented pedestrian scoring flagged at least one high-interest frame.")

    out: Dict[str, Any] = {
        "scene_name": scene_name,
        "total_frames": n_frames,
        "total_duration_seconds": dur,
        "ego_total_distance_m": dist,
        "ego_average_speed_mps": avg_sp,
        "ego_max_speed_mps": max_sp,
        "total_annotation_rows": int(len(objects_df)),
        "total_unique_instances": int(uniq_inst),
        "total_unique_pedestrians": int(ped_inst),
        "total_unique_vehicles": int(veh_inst),
        "minimum_distance_to_any_object_m": min_d,
        "minimum_distance_to_any_pedestrian_m": min_p,
        "minimum_distance_to_any_vehicle_m": min_v,
        "minimum_distance_to_any_static_object_m": min_s,
        "minimum_distance_to_any_dynamic_object_m": min_dyn,
        "maximum_objects_in_frame": max_obj_frame,
        "average_objects_per_frame": avg_obj_frame,
        "peak_pedestrians_in_frame": peak_ped,
        "peak_vehicles_in_frame": peak_veh,
        "ego_start_xy": [ex0, ey0],
        "ego_end_xy": [ex1, ey1],
        "approximate_scene_motion_pattern": motion_pattern,
        "most_common_categories": {k: int(v) for k, v in most_common.items()},
        "notes_and_warnings": notes,
        "extraction_quality_notes": eq_notes,
        "planner_relevance_notes": pr_notes,
        "camera_awareness_summary": {
            "camera_candidate_rows": int(run_stats.get("camera_candidate_rows", 0)),
            "camera_degradable_rows": int(run_stats.get("camera_degradable_rows", 0)),
            "camera_degradation_design_note": "Only tagged dynamic rows are intended for later camera dropout experiments.",
        },
        "validation_summary": validation,
        "run_statistics": run_stats,
    }
    return out


def sensor_metadata_json(
    nusc: Any,
    samples: List[Dict[str, Any]],
    channels: List[str],
    per_ch: Dict[str, Dict[str, int]],
    log: ExtractionLog,
) -> Dict[str, Any]:
    calib: Dict[str, Any] = {}
    try:
        s0 = samples[0]
        sd_tok = pick_lidar_sample_data_token(s0, log)
        if sd_tok:
            sd = nusc.get("sample_data", sd_tok)
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            sens = nusc.get("sensor", cs["sensor_token"])
            calib[LIDAR_CHANNEL] = {
                "sample_data_token": sd_tok,
                "calibrated_sensor_token": sd["calibrated_sensor_token"],
                "sensor_token": cs["sensor_token"],
                "sensor_modality": sens.get("modality"),
                "channel": sens.get("channel"),
            }
    except Exception as exc:
        log.warn(f"sensor calibration summary failed: {exc}")
    return {
        "sensor_channels_present": channels,
        "per_channel_availability": per_ch,
        "calibration_references": calib,
        "primary_channel_ego_pose": LIDAR_CHANNEL,
        "camera_sensor_channels_present": [ch for ch in channels if ch.upper().startswith("CAM")],
        "camera_degradation_protocol": {
            "dynamic_classes_tagged": ["pedestrian", "vehicle"],
            "static_objects_degraded": False,
            "description": "Objects tagged via camera_candidate_flag / camera_degradable_flag are intended for downstream controlled camera-object dropout experiments.",
        },
        "planner_filter_thresholds": {
            "MAX_FILTER_DISTANCE_M": MAX_FILTER_DISTANCE_M,
            "MAX_PEDESTRIAN_DISTANCE_M": MAX_PEDESTRIAN_DISTANCE_M,
            "MAX_VEHICLE_DISTANCE_M": MAX_VEHICLE_DISTANCE_M,
            "MAX_STATIC_DISTANCE_M": MAX_STATIC_DISTANCE_M,
            "FORWARD_CORRIDOR_LATERAL_M": FORWARD_CORRIDOR_LATERAL_M,
            "FUTURE_PATH_DISTANCE_THRESHOLD_M": FUTURE_PATH_DISTANCE_THRESHOLD_M,
            "FUTURE_PATH_LOOKAHEAD_FRAMES": FUTURE_PATH_LOOKAHEAD_FRAMES,
            "CONFLICT_PED_SCORE_THRESHOLD": CONFLICT_PED_SCORE_THRESHOLD,
            "ALLOW_ZERO_POINT_VISIBLE_PEDESTRIANS": ALLOW_ZERO_POINT_VISIBLE_PEDESTRIANS,
            "LOW_SUPPORT_THRESHOLD": LOW_SUPPORT_THRESHOLD,
        },
    }



# ---------------------------------------------------------------------------
# Batch selected-scenes extraction helpers
# ---------------------------------------------------------------------------

DEFAULT_SELECTED_SCENES_JSON: str = "config/three_geometry_candidates/selected_tested_scenes.json"
DEFAULT_HARDCODED_TESTED_JSON: str = "config/three_geometry_candidates/tested_scenes_hardcoded.json"
DEFAULT_BATCH_SUMMARY_REL: str = "data/raw/_batch_extract_summary.json"
DEFAULT_FINAL_SCENES: List[Dict[str, str]] = [
    {"selection_order": "1", "scene_name": "scene-1019", "geometry_case": "straight_approach"},
    {"selection_order": "2", "scene_name": "scene-0061", "geometry_case": "moderate_curvature_approach"},
    {"selection_order": "3", "scene_name": "scene-0102", "geometry_case": "high_curvature_or_circular_approach"},
]


def split_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def normalize_scene_name_cli(text: str) -> str:
    """Accept 0102, 102, scene-0102, or scene-102 and return scene-0102."""
    t = str(text).strip()
    if not t:
        return ""
    if t.startswith("scene-"):
        suffix = t.split("scene-", 1)[1]
        if suffix.isdigit():
            return f"scene-{int(suffix):04d}"
        return t
    digits = "".join(ch for ch in t if ch.isdigit())
    if digits:
        return f"scene-{int(digits):04d}"
    return t


def resolve_relative_path(path_text: Optional[str], default_rel: str) -> Path:
    raw = path_text if path_text else default_rel
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (Path(SCRIPT_DIR) / p).resolve()


def read_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at root: {path}")
    return data


def scene_records_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accept the output formats produced by Script 01."""
    for key in (
        "tested_scene_records",
        "tested_scenes",
        "selected_tested_scenes",
        "selected_new_scenes",
        "candidates",
    ):
        val = data.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict) and str(x.get("scene_name", "")).strip()]
    bc = data.get("best_candidate")
    if isinstance(bc, dict) and str(bc.get("scene_name", "")).strip():
        return [bc]
    return []


def normalize_scene_record(rec: Dict[str, Any], index: int, source: str) -> Dict[str, Any]:
    scene_name = str(rec.get("scene_name") or "").strip()
    if not scene_name:
        raise ValueError(f"Scene record #{index} has no scene_name: {rec}")
    geometry_case = str(
        rec.get("geometry_case")
        or rec.get("geometry_case_candidate")
        or rec.get("geometry_case_hardcoded")
        or rec.get("selection_role")
        or ""
    )
    return {
        "selection_order": int(rec.get("selection_order") or index),
        "scene_name": scene_name,
        "scene_token": str(rec.get("scene_token") or ""),
        "geometry_case": geometry_case,
        "dataset_version": str(rec.get("dataset_version") or rec.get("nuscenes_version") or ""),
        "resolved_dataroot": str(rec.get("resolved_dataroot") or ""),
        "location": str(rec.get("location") or ""),
        "selection_role": str(rec.get("selection_role") or rec.get("role_in_study") or geometry_case),
        "source": source,
    }


def load_selected_scene_records(selected_json: Path, fallback_json: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load the three final scene records.

    Priority:
      1) selected_json if it exists
      2) fallback hard-coded JSON if it exists
      3) built-in hard-coded list: scene-1019, scene-0061, scene-0185
    """
    loaded_from = "built_in_default"
    metadata: Dict[str, Any] = {}
    records_raw: List[Dict[str, Any]] = []

    if selected_json.is_file():
        data = read_json_object(selected_json)
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        records_raw = scene_records_from_json(data)
        loaded_from = str(selected_json)
    elif fallback_json.is_file():
        data = read_json_object(fallback_json)
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        records_raw = scene_records_from_json(data)
        loaded_from = str(fallback_json)
    else:
        records_raw = [dict(x) for x in DEFAULT_FINAL_SCENES]

    if not records_raw:
        raise RuntimeError(
            "No scene records found. Expected selected/tested scenes JSON to contain "
            "tested_scenes, selected_tested_scenes, or selected_new_scenes."
        )

    records = [normalize_scene_record(r, i + 1, loaded_from) for i, r in enumerate(records_raw)]

    # Keep only one record per scene while preserving order.
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for r in sorted(records, key=lambda x: (int(x.get("selection_order", 999)), str(x.get("scene_name", "")))):
        name = str(r["scene_name"])
        if name in seen:
            continue
        seen.add(name)
        unique.append(r)

    return unique, {"loaded_from": loaded_from, "source_metadata": metadata}


def scene_exists_in_version(scene_name: str, version: str, dataroot: str) -> bool:
    scene_json = Path(dataroot) / version / "scene.json"
    if not scene_json.is_file():
        return False
    try:
        rows = json.loads(scene_json.read_text(encoding="utf-8"))
        return any(isinstance(r, dict) and r.get("name") == scene_name for r in rows)
    except Exception:
        return False


def resolve_version_for_scene(scene_rec: Dict[str, Any], versions: Sequence[str], explicit_dataroot: Optional[str]) -> Tuple[str, str]:
    """Return (version, dataroot) for a scene name, searching versions when needed."""
    scene_name = str(scene_rec["scene_name"])
    preferred_version = str(scene_rec.get("dataset_version") or "").strip()
    metadata_dataroot = str(scene_rec.get("resolved_dataroot") or explicit_dataroot or "").strip()

    candidates: List[str] = []
    if preferred_version:
        candidates.append(preferred_version)
    for v in versions:
        if v not in candidates:
            candidates.append(v)
    if not candidates:
        candidates = [DEFAULT_NUSCENES_VERSION]

    last_dataroot: Optional[str] = None
    for version in candidates:
        dataroot = resolve_dataroot(version, {"resolved_dataroot": metadata_dataroot} if metadata_dataroot else None)
        if dataroot is None:
            continue
        last_dataroot = dataroot
        if scene_exists_in_version(scene_name, version, dataroot):
            return version, dataroot

    # Fallback: return the preferred/default version if dataroot exists, so the normal extractor gives a clear error.
    fallback_version = candidates[0]
    fallback_dataroot = last_dataroot or resolve_dataroot(fallback_version, {"resolved_dataroot": metadata_dataroot} if metadata_dataroot else None)
    if fallback_dataroot is None:
        raise RuntimeError(
            f"Could not resolve nuScenes dataroot for {scene_name}. Use --dataroot so that <dataroot>/<version>/scene.json exists."
        )
    raise RuntimeError(
        f"Could not find {scene_name} in versions {list(candidates)} under dataroot {fallback_dataroot}."
    )


def make_single_scene_args(batch_args: argparse.Namespace, scene_rec: Dict[str, Any], version: str, dataroot: str) -> argparse.Namespace:
    return argparse.Namespace(
        config=os.path.join(SCRIPT_DIR, CONFIG_REL_DEFAULT),
        scene_name=str(scene_rec["scene_name"]),
        nuscenes_version=version,
        dataroot=str(dataroot),
        output_root=str(resolve_relative_path(batch_args.output_root, RAW_ROOT_REL)),
        map_dataroot=batch_args.map_dataroot,
    )


def write_batch_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract all final selected/tested nuScenes scenes into data/raw/<scene_name>/ "
            "using the built-in raw extraction logic. No external base extractor is needed."
        )
    )
    p.add_argument(
        "--selected-scenes-json",
        default=DEFAULT_SELECTED_SCENES_JSON,
        help="Selected/tested scenes JSON produced by Script 01. Default: config/three_geometry_candidates/selected_tested_scenes.json",
    )
    p.add_argument(
        "--fallback-tested-json",
        default=DEFAULT_HARDCODED_TESTED_JSON,
        help="Fallback hard-coded tested-scenes JSON. Default: config/three_geometry_candidates/tested_scenes_hardcoded.json",
    )
    p.add_argument(
        "--scene-name",
        "--scene-names",
        dest="direct_scene_names",
        default="",
        help=(
            "Direct scene name(s) to extract, e.g. 0383, scene-0383, or "
            "scene-1019,scene-0061,scene-0102. When provided, the script bypasses "
            "the selected-scenes JSON and extracts these scenes directly."
        ),
    )
    p.add_argument(
        "--versions",
        default=os.environ.get("NUSCENES_VERSIONS", "v1.0-mini,v1.0-trainval"),
        help="Comma-separated versions to search when a scene record has no dataset_version.",
    )
    p.add_argument(
        "--dataroot",
        default=os.environ.get("NUSCENES_DATAROOT", ""),
        help="Optional nuScenes dataroot containing version folders.",
    )
    p.add_argument("--output-root", default=RAW_ROOT_REL, help="Raw output root. Default: data/raw")
    p.add_argument(
        "--map-dataroot",
        default=None,
        help="Optional map dataroot (dataset root, .../maps, or .../maps/expansion).",
    )
    p.add_argument(
        "--only-scenes",
        default="",
        help="Optional comma-separated subset, e.g. scene-1019,scene-0061. Empty = all selected scenes.",
    )
    p.add_argument(
        "--batch-summary",
        default=DEFAULT_BATCH_SUMMARY_REL,
        help="Where to write the batch extraction summary JSON. Default: data/raw/_batch_extract_summary.json",
    )
    return p.parse_args()

def run_single_extraction(args: argparse.Namespace) -> None:
    log = ExtractionLog()
    log.start_wall = datetime.now(timezone.utc).isoformat()

    # Two supported modes:
    #   1) legacy config mode: --config config/best_candidate_scene.json
    #   2) direct scene mode:  --scene-name scene-XXXX --nuscenes-version v1.0-trainval
    if args.scene_name:
        scene_name = str(args.scene_name).strip()
        version = (
            str(args.nuscenes_version).strip()
            if args.nuscenes_version
            else os.environ.get("NUSCENES_VERSION", "").strip() or DEFAULT_NUSCENES_VERSION
        )
        meta: Dict[str, Any] = {
            "nuscenes_version": version,
            "config_source": "direct_scene_name_cli",
        }
        if args.dataroot:
            meta["resolved_dataroot"] = str(args.dataroot).strip()
        log.config_path = "[direct --scene-name mode]"
        log.scene_name = scene_name
        log.version = version

        dataroot = resolve_dataroot(version, meta)
        if dataroot is None:
            print(
                "Error: could not resolve nuScenes dataroot. "
                f"Use --dataroot so that <dataroot>/{version}/scene.json exists, "
                "or set NUSCENES_DATAROOT.",
                file=sys.stderr,
            )
            sys.exit(1)
        log.dataroot = dataroot

        try:
            nusc = init_nuscenes(dataroot, version, log)
        except Exception:
            sys.exit(1)

        try:
            scene = get_scene_record_by_name(nusc, scene_name)
        except Exception as exc:
            print(f"Error: scene not in dataset version {version}: {exc}", file=sys.stderr)
            sys.exit(1)
        scene_token = scene["token"]
        log.scene_token = scene_token
    else:
        config_path = os.path.abspath(args.config)
        log.config_path = config_path

        try:
            cfg = load_best_scene_config(config_path)
        except Exception as exc:
            print(f"Error: cannot load scene config: {exc}", file=sys.stderr)
            sys.exit(1)

        meta = cfg.get("metadata") or {}
        bc = cfg["best_candidate"]
        scene_name = bc["scene_name"]
        scene_token = bc["scene_token"]
        log.scene_name = scene_name
        log.scene_token = scene_token

        version = meta.get("nuscenes_version") if isinstance(meta.get("nuscenes_version"), str) else None
        if not version:
            version = DEFAULT_NUSCENES_VERSION
        log.version = version

        dataroot = resolve_dataroot(version, meta if isinstance(meta, dict) else None)
        if dataroot is None:
            print(
                "Error: could not resolve nuScenes dataroot. "
                f"Set metadata.resolved_dataroot in config or export NUSCENES_DATAROOT "
                f"so that <dataroot>/{version}/scene.json exists.",
                file=sys.stderr,
            )
            sys.exit(1)
        log.dataroot = dataroot

        try:
            nusc = init_nuscenes(dataroot, version, log)
        except Exception:
            sys.exit(1)

        try:
            scene = get_scene_record(nusc, scene_token)
        except Exception as exc:
            print(f"Error: scene not in dataset: {exc}", file=sys.stderr)
            sys.exit(1)

    samples = get_ordered_scene_samples(nusc, scene)
    if not samples:
        print("Error: scene has no samples.", file=sys.stderr)
        sys.exit(1)

    channels = collect_sensor_channels(samples)
    per_ch = per_channel_frame_counts(samples, channels)

    out_root = Path(args.output_root) if args.output_root else Path(SCRIPT_DIR) / RAW_ROOT_REL
    out_dir = out_root / scene_name
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "animations").mkdir(parents=True, exist_ok=True)

    # --- scene_metadata.json
    smd = extract_scene_metadata(nusc, scene, samples, version, dataroot, channels)
    smd_path = out_dir / "scene_metadata.json"
    with open(smd_path, "w", encoding="utf-8") as f:
        json.dump(smd, f, indent=2)
    log.add_file(str(smd_path))

    # --- timelines
    sample_tl = build_sample_timeline(samples)
    sample_tl_path = out_dir / "sample_timeline.csv"
    sample_tl.to_csv(sample_tl_path, index=False)
    log.add_file(str(sample_tl_path))

    ego_df, _poses = extract_ego_timeline(nusc, samples, log)
    # attach delta_t from sample timeline
    ego_df = ego_df.merge(
        sample_tl[["frame_index", "delta_t_s"]], on="frame_index", how="left"
    )
    ego_path = out_dir / "ego_timeline.csv"
    ego_df.to_csv(ego_path, index=False)
    log.add_file(str(ego_path))

    objects_df = extract_objects_timeline(nusc, samples, ego_df, log)
    future_polylines = precompute_future_polylines(ego_df, FUTURE_PATH_LOOKAHEAD_FRAMES)
    objects_df = enrich_objects_timeline_planner(objects_df, ego_df, log, future_polylines)
    agent_df = build_agent_trajectories(objects_df, ego_df)
    objects_df = merge_agent_motion_into_objects(objects_df, agent_df, future_polylines)
    objects_df = add_camera_awareness_layer(objects_df)

    obj_path = out_dir / "objects_timeline.csv"
    objects_df.to_csv(obj_path, index=False)
    log.add_file(str(obj_path))

    agent_path = out_dir / "agent_trajectories.csv"
    agent_df.to_csv(agent_path, index=False)
    log.add_file(str(agent_path))

    planner_df = build_planner_observations(ego_df, objects_df, sample_tl, IN_PATH_LATERAL_THRESHOLD_M)
    plan_path = out_dir / "planner_observations.csv"
    planner_df.to_csv(plan_path, index=False)
    log.add_file(str(plan_path))

    # --- sensor + class counts
    sens_path = out_dir / "sensor_metadata.json"
    with open(sens_path, "w", encoding="utf-8") as f:
        json.dump(sensor_metadata_json(nusc, samples, channels, per_ch, log), f, indent=2)
    log.add_file(str(sens_path))

    cc_data = build_class_counts(objects_df, len(samples))
    if len(objects_df):
        cc_data["planner_annotation_summary"] = {
            "annotations_in_forward_corridor": int(objects_df["forward_corridor_flag"].sum()),
            "annotations_near_future_ego_path": int(objects_df["near_future_path_flag"].sum()),
            "annotations_passing_planner_filter": int(objects_df["filtered_planner_object_flag"].sum()),
            "mean_relevance_score": float(objects_df["relevance_score"].mean()),
            "max_ped_conflict_score": float(objects_df.loc[objects_df["coarse_class"] == "pedestrian", "ped_conflict_score"].max())
            if (objects_df["coarse_class"] == "pedestrian").any()
            else None,
            "camera_candidate_rows": int(objects_df["camera_candidate_flag"].sum()) if "camera_candidate_flag" in objects_df.columns else 0,
            "camera_degradable_rows": int(objects_df["camera_degradable_flag"].sum()) if "camera_degradable_flag" in objects_df.columns else 0,
        }
    cc_path = out_dir / "class_counts.json"
    with open(cc_path, "w", encoding="utf-8") as f:
        json.dump(cc_data, f, indent=2)
    log.add_file(str(cc_path))

    camera_summary = build_camera_detection_summary(objects_df, len(samples))
    cds_path = out_dir / "camera_detection_summary.json"
    with open(cds_path, "w", encoding="utf-8") as f:
        json.dump(camera_summary, f, indent=2)
    log.add_file(str(cds_path))

    validation, run_stats = compute_validation_and_run_stats(
        samples, sample_tl, ego_df, objects_df, planner_df
    )
    run_stats["validation_summary_embedded"] = [f"{k}={v}" for k, v in validation.items()]
    log.run_stats = run_stats

    summary = build_scene_summary(
        scene_name,
        ego_df,
        objects_df,
        planner_df,
        log,
        str(scene.get("description") or ""),
        validation,
        run_stats,
    )
    sum_path = out_dir / "scene_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.add_file(str(sum_path))

    xlim, ylim = compute_xy_bounds(ego_df, objects_df)
    patch_box = patch_box_from_xy_limits(xlim, ylim)
    map_name: Optional[str] = None
    try:
        log_rec = nusc.get("log", scene["log_token"])
        map_name = str(log_rec.get("location") or "")
    except Exception as exc:
        log.warn(f"could not resolve map name from scene log: {exc}")
    map_root = resolve_map_dataroot(args.map_dataroot, dataroot, version)
    nmap = load_nuscenes_map_or_none(map_root, map_name)
    plots_dir = out_dir / "plots"
    make_global_plots(plots_dir, ego_df, objects_df, planner_df, agent_df, xlim, ylim, nmap, patch_box, log)

    gif_path = out_dir / "animations" / "scene_replay.gif"
    make_animation_from_csv(gif_path, ego_df, objects_df, samples, xlim, ylim, nmap, patch_box, log)

    log.end_wall = datetime.now(timezone.utc).isoformat()
    log_path = out_dir / "extraction_log.txt"
    write_log(log, log_path)

    # Console summary
    print("")
    print("nuScenes scene raw extraction complete")
    print(f"  Scene name:     {scene_name}")
    print(f"  Scene token:    {scene_token}")
    print(f"  Frames:         {len(samples)}")
    print(f"  Object rows:    {len(objects_df)}")
    print(f"  Unique instances: {objects_df['instance_token'].nunique() if len(objects_df) else 0}")
    print(
        f"  Pedestrian instances: {objects_df[objects_df['coarse_class']=='pedestrian']['instance_token'].nunique() if len(objects_df) else 0}"
    )
    print(f"  Output directory: {out_dir.resolve()}")
    print(f"  GIF generated:    {log.gif_ok}")
    if not log.gif_ok and log.gif_error:
        print(f"    (GIF error: {log.gif_error[:200]})")
    print("  Major outputs:")
    for rel in (
        "scene_metadata.json",
        "scene_summary.json",
        "sample_timeline.csv",
        "ego_timeline.csv",
        "objects_timeline.csv",
        "agent_trajectories.csv",
        "planner_observations.csv",
        "sensor_metadata.json",
        "class_counts.json",
        "camera_detection_summary.json",
        "extraction_log.txt",
        "plots/",
        "animations/scene_replay.gif",
    ):
        print(f"    - {rel}")
    print("")



def main() -> None:
    args = parse_args()
    selected_json = resolve_relative_path(args.selected_scenes_json, DEFAULT_SELECTED_SCENES_JSON)
    fallback_json = resolve_relative_path(args.fallback_tested_json, DEFAULT_HARDCODED_TESTED_JSON)
    output_root = resolve_relative_path(args.output_root, RAW_ROOT_REL)
    batch_summary_path = resolve_relative_path(args.batch_summary, DEFAULT_BATCH_SUMMARY_REL)
    versions = split_csv(args.versions)
    only = set(split_csv(args.only_scenes))

    direct_names = [normalize_scene_name_cli(x) for x in split_csv(getattr(args, "direct_scene_names", ""))]
    direct_names = [x for x in direct_names if x]

    if direct_names:
        scene_records = [
            {
                "selection_order": i + 1,
                "scene_name": name,
                "scene_token": "",
                "geometry_case": "direct_scene_cli",
                "dataset_version": "",
                "resolved_dataroot": str(args.dataroot or ""),
                "location": "",
                "selection_role": "direct scene extraction requested from CLI",
                "source": "direct --scene-name / --scene-names",
            }
            for i, name in enumerate(direct_names)
        ]
        source_info = {
            "loaded_from": "direct --scene-name / --scene-names",
            "source_metadata": {"note": "Bypassed selected-scenes JSON and extracted the requested scene names directly."},
        }
    else:
        scene_records, source_info = load_selected_scene_records(selected_json, fallback_json)
        if only:
            only_norm = {normalize_scene_name_cli(x) for x in only}
            scene_records = [r for r in scene_records if normalize_scene_name_cli(r["scene_name"]) in only_norm]

    if not scene_records:
        raise RuntimeError(
            "No scenes to extract after applying the filter. "
            "Use --scene-name for arbitrary scenes, or make sure --only-scenes is one of the selected tested scenes."
        )

    print("Batch raw extraction for selected tested scenes")
    print(f"  Selected-scenes source: {source_info.get('loaded_from')}")
    print(f"  Output root: {output_root}")
    print("  Scenes:")
    for r in scene_records:
        print(f"    - {r['scene_name']} | {r.get('geometry_case', '')}")
    print("")

    results: List[Dict[str, Any]] = []
    for i, rec in enumerate(scene_records, start=1):
        scene_name = str(rec["scene_name"])
        print("=" * 78)
        print(f"[{i}/{len(scene_records)}] Extracting {scene_name} ({rec.get('geometry_case', '')})")
        try:
            version, dataroot = resolve_version_for_scene(rec, versions, args.dataroot)
            single_args = make_single_scene_args(args, rec, version, dataroot)
            run_single_extraction(single_args)
            out_dir = output_root / scene_name
            results.append(
                {
                    "scene_name": scene_name,
                    "geometry_case": rec.get("geometry_case", ""),
                    "dataset_version": version,
                    "resolved_dataroot": dataroot,
                    "output_dir": str(out_dir.resolve()),
                    "status": "ok",
                }
            )
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else 1
            results.append(
                {
                    "scene_name": scene_name,
                    "geometry_case": rec.get("geometry_case", ""),
                    "status": "failed",
                    "error": f"single extraction exited with code {code}",
                }
            )
            if code != 0:
                print(f"[failed] {scene_name}: extractor exited with code {code}", file=sys.stderr)
        except Exception as exc:
            results.append(
                {
                    "scene_name": scene_name,
                    "geometry_case": rec.get("geometry_case", ""),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[failed] {scene_name}: {exc}", file=sys.stderr)

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selected_scenes_source": source_info,
        "versions_searched": versions,
        "output_root": str(output_root.resolve()),
        "num_scenes_requested": len(scene_records),
        "num_scenes_extracted_ok": ok_count,
        "results": results,
    }
    write_batch_summary(batch_summary_path, summary)

    print("=" * 78)
    print("Batch raw extraction complete")
    print(f"  OK: {ok_count}/{len(results)}")
    print(f"  Summary: {batch_summary_path}")
    for r in results:
        status = r.get("status")
        scene = r.get("scene_name")
        if status == "ok":
            print(f"    [ok] {scene} -> {r.get('output_dir')}")
        else:
            print(f"    [failed] {scene}: {r.get('error')}")

    if ok_count != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
