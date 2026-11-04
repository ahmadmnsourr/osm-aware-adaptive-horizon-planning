#!/usr/bin/env python3
"""
Unified three-geometry nuScenes scene filtering script for the OSM-error-aware AV crossing project.

This script merges the earlier candidate filtering ideas into ONE discovery script and organizes
results into the three geometry cases used by the final study:

  1) straight_approach
  2) moderate_curvature_approach
  3) high_curvature_or_circular_approach

It also writes a hard-coded tested-scenes JSON for the final scenes that should be used in the
pipeline regardless of ranking:

  - scene-1019 -> straight_approach
  - scene-0061 -> moderate_curvature_approach
  - scene-0383 -> high_curvature_or_circular_approach

The selected_scene_configs outputs keep the same downstream structure used by Script 02:

  {"metadata": {...}, "best_candidate": {...}}

Typical usage:

  python 01_find_candidate_scenes_three_geometries.py

Outputs:

  config/three_geometry_candidates/candidate_scenes_all.json
  config/three_geometry_candidates/candidate_scenes_by_geometry.json
  config/three_geometry_candidates/tested_scenes_hardcoded.json
  config/three_geometry_candidates/selected_tested_scenes.json
  config/three_geometry_candidates/selected_tested_scenes.csv
  config/selected_scene_configs/01_straight_scene-1019.json
  config/selected_scene_configs/02_moderate_curvature_scene-0061.json
  config/selected_scene_configs/03_high_curvature_scene-0383.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LIDAR_CHANNEL = "LIDAR_TOP"

SINGAPORE_LOCATIONS = {
    "singapore-onenorth",
    "singapore-hollandvillage",
    "singapore-queenstown",
}

GEOM_STRAIGHT = "straight_approach"
GEOM_MODERATE = "moderate_curvature_approach"
GEOM_HIGH = "high_curvature_or_circular_approach"

# This is the hard-coded final tested set for the project.
# The script writes this into tested_scenes_hardcoded.json on every run.
TESTED_SCENES: List[Dict[str, Any]] = [
    {
        "selection_order": 1,
        "scene_name": "scene-1019",
        "geometry_case": GEOM_STRAIGHT,
        "role_in_study": "straight ego-approach pedestrian-crossing case",
        "downstream_config_filename": "01_straight_scene-1019.json",
        "refinement_script_hint": "03_refine_scene_1019_straight_same_length_smooth_reaccel.py",
    },
    {
        "selection_order": 2,
        "scene_name": "scene-0061",
        "geometry_case": GEOM_MODERATE,
        "role_in_study": "moderate/slight curvature ego-approach pedestrian-crossing case",
        "downstream_config_filename": "02_moderate_curvature_scene-0061.json",
        "refinement_script_hint": "03_refine_scene_synthetic_0061_only.py",
    },
    {
        "selection_order": 3,
        "scene_name": "scene-0383",
        "geometry_case": GEOM_HIGH,
        "role_in_study": "high-curvature/circular ego-approach pedestrian-crossing case",
        "downstream_config_filename": "03_high_curvature_scene-0383.json",
        "refinement_script_hint": "03_refine_scene_0383.py",
    },
]


@dataclass
class EgoPath:
    sample_tokens: List[str]
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    xy: np.ndarray
    yaw_rad: np.ndarray
    s: np.ndarray

    @property
    def total_distance_m(self) -> float:
        return float(self.s[-1] - self.s[0]) if len(self.s) else 0.0

    @property
    def avg_step_distance_m(self) -> float:
        if len(self.s) < 2:
            return 0.0
        return float(np.mean(np.diff(self.s)))


@dataclass
class ProjectionResult:
    s_m: float
    distance_m: float
    seg_index: int
    qx: float
    qy: float
    signed_lateral_m: float


@dataclass
class PedObservation:
    instance_token: str
    ann_token: str
    sample_token: str
    frame_index: int
    timestamp_s: float
    x: float
    y: float
    category_name: str


@dataclass
class ConflictEvent:
    instance_token: str
    ann_token: str
    sample_token: str
    frame_index: int
    timestamp_s: float
    ped_x: float
    ped_y: float
    crossing_proxy_x: float
    crossing_proxy_y: float
    crossing_s_m: float
    ped_to_future_path_m: float
    ped_to_route_m: float
    ego_arrival_time_s: float
    ego_ped_time_gap_s: float
    signed_lateral_m: float


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find and organize nuScenes crossing scenes into straight, moderate-curvature, and high-curvature cases.")
    p.add_argument("--versions", default=os.environ.get("NUSCENES_VERSIONS", "v1.0-mini,v1.0-trainval"))
    p.add_argument("--dataroot", default=os.environ.get("NUSCENES_DATAROOT", ""), help="Dataset root containing v1.0-mini/ and/or v1.0-trainval/. If omitted, common local folders are searched.")
    p.add_argument("--map-dataroot", default=os.environ.get("NUSCENES_MAP_DATAROOT", ""), help="Optional map dataroot containing maps/expansion. Soft evidence by default.")
    p.add_argument("--output-dir", default=str(SCRIPT_DIR / "config" / "three_geometry_candidates"))
    p.add_argument("--selected-config-dir", default=str(SCRIPT_DIR / "config" / "selected_scene_configs"))
    p.add_argument("--max-scenes-per-version", type=int, default=0, help="Debug limit. 0 = scan all.")
    p.add_argument("--verbose", action="store_true")

    # Interaction filters, intentionally relaxed because Script 03 refines the selected scenes later.
    p.add_argument("--min-ego-total-distance-m", type=float, default=20.0)
    p.add_argument("--min-ego-avg-step-distance-m", type=float, default=0.25)
    p.add_argument("--min-ped-frames", type=int, default=3)
    p.add_argument("--future-lookahead-frames", type=int, default=14)
    p.add_argument("--ped-future-path-distance-m", type=float, default=4.5)
    p.add_argument("--min-intersection-frames", type=int, default=1)
    p.add_argument("--max-conflict-time-gap-s", type=float, default=10.0)
    p.add_argument("--min-ego-before-crossing-m", type=float, default=3.0)
    p.add_argument("--min-ego-after-crossing-m", type=float, default=3.0)
    p.add_argument("--require-lateral-sign-change", action="store_true", help="Optional strict crossing behavior check.")

    # Map evidence. By default, map crossing is recorded/scored but not required.
    p.add_argument("--max-map-crossing-distance-m", type=float, default=20.0)
    p.add_argument("--strict-map-crossing", action="store_true", help="Reject candidates without nearby HD-map ped_crossing evidence.")

    # Geometry thresholds.
    p.add_argument("--geometry-window-before-m", type=float, default=25.0)
    p.add_argument("--geometry-window-after-m", type=float, default=25.0)
    p.add_argument("--straight-heading-change-deg", type=float, default=15.0)
    p.add_argument("--straight-line-rmse-m", type=float, default=1.5)
    p.add_argument("--straight-curvature-score", type=float, default=0.12)
    p.add_argument("--high-heading-change-deg", type=float, default=55.0)
    p.add_argument("--high-rotation-coverage-deg", type=float, default=65.0)
    p.add_argument("--high-tortuosity", type=float, default=1.06)
    p.add_argument("--high-circle-sweep-deg", type=float, default=45.0)
    p.add_argument("--max-circle-fit-rmse-m", type=float, default=8.0)
    p.add_argument("--max-circle-radius-m", type=float, default=90.0)
    return p.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def round_or_none(x: Any, ndigits: int = 3) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return round(v, ndigits)


def wrap_pi(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def quat_wxyz_to_yaw_rad(rot: Sequence[float]) -> float:
    if rot is None or len(rot) < 4:
        return 0.0
    w, x, y, z = [float(v) for v in rot[:4]]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _table_root_ok(dataroot: Path, version: str) -> bool:
    return (dataroot / version / "scene.json").is_file()


def resolve_dataroot_for_version(explicit: str, version: str) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("NUSCENES_DATAROOT", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend([SCRIPT_DIR, Path.cwd(), Path.cwd() / "data", SCRIPT_DIR / "data", SCRIPT_DIR / "nuscenes", Path("/data/sets/nuscenes")])

    seen = set()
    for raw in candidates:
        try:
            p = raw.resolve()
        except Exception:
            continue
        if str(p) in seen:
            continue
        seen.add(str(p))
        if _table_root_ok(p, version):
            return p
        if p.name == version and (p / "scene.json").is_file():
            parent = p.parent
            if _table_root_ok(parent, version):
                return parent
    return None


def install_mapmask_dummy_for_missing_png() -> None:
    """Avoid NuScenes init failure when semantic map PNGs are missing in partial installs."""
    try:
        import nuscenes.utils.map_mask as mm
        import nuscenes.nuscenes as ns_mod
    except Exception:
        return

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
            if getattr(self, "_dummy_map", False):
                return np.zeros((256, 256), dtype=np.uint8)
            return super()._base_mask

    mm.MapMask = MapMaskOrDummy
    ns_mod.MapMask = MapMaskOrDummy


def normalize_nuscenes_map_root(path: Path) -> Path:
    p = path.expanduser().resolve()
    if (p / "maps" / "expansion").is_dir():
        return p
    if p.name == "maps" and (p / "expansion").is_dir():
        return p.parent
    if p.name == "expansion" and p.parent.name == "maps":
        return p.parent.parent
    return p


def resolve_map_dataroot(explicit: str, table_dataroot: Path, version: str) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("NUSCENES_MAP_DATAROOT", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend([table_dataroot, table_dataroot / version, table_dataroot / "v1.0-mini", SCRIPT_DIR, SCRIPT_DIR / version, SCRIPT_DIR / "v1.0-mini", Path.cwd(), Path.cwd() / version, Path.cwd() / "v1.0-mini"])

    seen = set()
    for raw in candidates:
        try:
            p = normalize_nuscenes_map_root(raw)
        except Exception:
            continue
        if str(p) in seen:
            continue
        seen.add(str(p))
        if (p / "maps" / "expansion").is_dir():
            return p
    return None


class MapCrossingCache:
    def __init__(self, dataroot: Path, version: str, explicit_map_dataroot: str = "") -> None:
        self.table_dataroot = dataroot
        self.version = version
        self.dataroot = resolve_map_dataroot(explicit_map_dataroot, dataroot, version)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, location: str) -> Dict[str, Any]:
        if location in self._cache:
            return self._cache[location]
        payload: Dict[str, Any] = {
            "location": location,
            "map_available": False,
            "map_dataroot": str(self.dataroot) if self.dataroot is not None else None,
            "error": "",
            "crossings": [],
        }
        if self.dataroot is None:
            payload["error"] = "nuScenes map JSON files not found. Expected maps/expansion under the dataset root or --map-dataroot."
            self._cache[location] = payload
            return payload
        try:
            # nuScenes may request an old matplotlib style name. Patch it.
            import matplotlib.pyplot as plt
            if not getattr(plt, "_nuscenes_map_style_patched", False):
                old_use = plt.style.use
                def patched(style: Any) -> None:
                    try:
                        old_use(style)
                    except OSError:
                        if style == "seaborn-whitegrid":
                            for alt in ("seaborn-v0_8-whitegrid", "default"):
                                try:
                                    old_use(alt)
                                    return
                                except OSError:
                                    pass
                        raise
                plt.style.use = patched  # type: ignore[assignment]
                plt._nuscenes_map_style_patched = True  # type: ignore[attr-defined]

            from nuscenes.map_expansion.map_api import NuScenesMap
            nmap = NuScenesMap(dataroot=str(self.dataroot), map_name=location)
            crossings: List[Dict[str, Any]] = []
            for rec in getattr(nmap, "ped_crossing", []):
                poly_token = rec.get("polygon_token")
                if not poly_token:
                    continue
                try:
                    poly = nmap.extract_polygon(poly_token)
                    if poly.is_empty:
                        continue
                    crossings.append({"token": str(rec.get("token", poly_token)), "polygon_token": str(poly_token), "polygon": poly})
                except Exception:
                    continue
            payload["map_available"] = True
            payload["crossings"] = crossings
        except Exception as exc:
            payload["error"] = str(exc)
        self._cache[location] = payload
        return payload

    def nearest_crossing(self, location: str, x: float, y: float, near_threshold_m: float) -> Dict[str, Any]:
        info = self.get(location)
        out = {
            "map_check_available": bool(info.get("map_available", False)),
            "resolved_map_dataroot": str(info.get("map_dataroot", "") or ""),
            "map_check_error": str(info.get("error", "") or ""),
            "real_map_crossing_found": False,
            "nearest_map_crossing_distance_m": float("inf"),
            "nearest_map_crossing_token": "",
            "num_map_crossings_near_proxy": 0,
            "num_map_crossings_total_location": len(info.get("crossings", []) or []),
        }
        if not info.get("map_available", False):
            return out
        try:
            from shapely.geometry import Point
            pt = Point(float(x), float(y))
        except Exception as exc:
            out["map_check_available"] = False
            out["map_check_error"] = f"shapely unavailable: {exc}"
            return out
        best_d = float("inf")
        best_token = ""
        near_count = 0
        for c in info.get("crossings", []) or []:
            poly = c.get("polygon")
            if poly is None:
                continue
            try:
                d = float(poly.distance(pt))
            except Exception:
                continue
            if d < best_d:
                best_d = d
                best_token = str(c.get("token", ""))
            if d <= near_threshold_m:
                near_count += 1
        out["nearest_map_crossing_distance_m"] = best_d
        out["nearest_map_crossing_token"] = best_token
        out["num_map_crossings_near_proxy"] = near_count
        out["real_map_crossing_found"] = bool(best_d <= near_threshold_m)
        return out


def point_to_segment_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray, s_a: float, s_b: float) -> Tuple[float, float, np.ndarray]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        t = 0.0
        q = a.copy()
    else:
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        q = a + t * ab
    d = float(np.linalg.norm(p - q))
    s_q = float(s_a + t * (s_b - s_a))
    return s_q, d, q


def project_point_to_path(path: EgoPath, px: float, py: float) -> ProjectionResult:
    p = np.array([float(px), float(py)], dtype=float)
    best_d = float("inf")
    best_s = 0.0
    best_q = path.xy[0].copy()
    best_i = 0
    for i in range(len(path.xy) - 1):
        s_q, d, q = point_to_segment_projection(p, path.xy[i], path.xy[i + 1], float(path.s[i]), float(path.s[i + 1]))
        if d < best_d:
            best_d = d
            best_s = s_q
            best_q = q
            best_i = i
    if best_i < len(path.xy) - 1:
        tangent = path.xy[best_i + 1] - path.xy[best_i]
    else:
        tangent = path.xy[best_i] - path.xy[max(0, best_i - 1)]
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-9:
        signed_lat = 0.0
    else:
        t = tangent / norm
        n = np.array([-t[1], t[0]], dtype=float)
        signed_lat = float(np.dot(p - best_q, n))
    return ProjectionResult(best_s, best_d, best_i, float(best_q[0]), float(best_q[1]), signed_lat)


def min_distance_point_to_polyline(point_xy: np.ndarray, poly_xy: np.ndarray) -> float:
    if len(poly_xy) == 0:
        return float("inf")
    if len(poly_xy) == 1:
        return float(np.linalg.norm(point_xy - poly_xy[0]))
    best = float("inf")
    for i in range(len(poly_xy) - 1):
        _s, d, _q = point_to_segment_projection(point_xy, poly_xy[i], poly_xy[i + 1], 0.0, 1.0)
        best = min(best, d)
    return float(best)


def interp_time_at_s(path: EgoPath, s_query: float) -> float:
    sq = float(np.clip(float(s_query), float(path.s[0]), float(path.s[-1])))
    return float(np.interp(sq, path.s, path.timestamps_s))


def iter_scene_sample_tokens(nusc: Any, scene: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    cur = scene.get("first_sample_token", "")
    while cur:
        tokens.append(cur)
        sample = nusc.get("sample", cur)
        nxt = sample.get("next") or ""
        if not nxt:
            break
        cur = nxt
    return tokens


def get_scene_location(nusc: Any, scene: Dict[str, Any]) -> str:
    try:
        log_token = scene.get("log_token", "")
        if not log_token:
            return ""
        log = nusc.get("log", log_token)
        return str(log.get("location", ""))
    except Exception:
        return ""


def build_ego_path(nusc: Any, sample_tokens: List[str]) -> Optional[EgoPath]:
    xs: List[float] = []
    ys: List[float] = []
    times: List[float] = []
    toks: List[str] = []
    frame_indices: List[int] = []
    t0_us: Optional[float] = None
    for i, st in enumerate(sample_tokens):
        try:
            sample = nusc.get("sample", st)
            sd_token = (sample.get("data") or {}).get(LIDAR_CHANNEL)
            if not sd_token:
                continue
            sd = nusc.get("sample_data", sd_token)
            pose = nusc.get("ego_pose", sd["ego_pose_token"])
            trans = pose.get("translation", [np.nan, np.nan, np.nan])
            timestamp_us = float(sample.get("timestamp", 0.0))
            if t0_us is None:
                t0_us = timestamp_us
            x = float(trans[0])
            y = float(trans[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            xs.append(x)
            ys.append(y)
            times.append((timestamp_us - t0_us) / 1e6)
            toks.append(st)
            frame_indices.append(i)
        except Exception:
            continue
    if len(xs) < 2:
        return None
    xy = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(steps)])

    # Use path tangent headings because we classify route geometry.
    yaw = np.zeros(len(xy), dtype=float)
    if len(xy) >= 2:
        yaw[0] = math.atan2(xy[1, 1] - xy[0, 1], xy[1, 0] - xy[0, 0])
        yaw[-1] = math.atan2(xy[-1, 1] - xy[-2, 1], xy[-1, 0] - xy[-2, 0])
        for i in range(1, len(xy) - 1):
            dx = xy[i + 1, 0] - xy[i - 1, 0]
            dy = xy[i + 1, 1] - xy[i - 1, 1]
            yaw[i] = math.atan2(dy, dx)
    yaw = np.unwrap(yaw)
    return EgoPath(toks, np.asarray(frame_indices, dtype=int), np.asarray(times, dtype=float), xy, yaw, s)


def is_pedestrian_category(category_name: str) -> bool:
    return isinstance(category_name, str) and category_name.startswith("human.pedestrian")


def collect_pedestrian_observations(nusc: Any, path: EgoPath) -> Dict[str, List[PedObservation]]:
    token_to_frame = {tok: int(i) for i, tok in enumerate(path.sample_tokens)}
    token_to_time = {tok: float(path.timestamps_s[i]) for i, tok in enumerate(path.sample_tokens)}
    ped_by_inst: Dict[str, List[PedObservation]] = {}
    for st in path.sample_tokens:
        try:
            sample = nusc.get("sample", st)
        except Exception:
            continue
        for ann_t in sample.get("anns") or []:
            try:
                ann = nusc.get("sample_annotation", ann_t)
            except Exception:
                continue
            cname = ann.get("category_name", "")
            if not is_pedestrian_category(cname):
                continue
            trans = ann.get("translation", [np.nan, np.nan, np.nan])
            x, y = float(trans[0]), float(trans[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            inst = str(ann.get("instance_token") or ann_t)
            ped_by_inst.setdefault(inst, []).append(
                PedObservation(inst, str(ann_t), str(st), token_to_frame[st], token_to_time[st], x, y, str(cname))
            )
    for inst in list(ped_by_inst.keys()):
        ped_by_inst[inst].sort(key=lambda o: (o.frame_index, o.timestamp_s))
    return ped_by_inst


def future_path_polyline(path: EgoPath, frame_index: int, lookahead_frames: int) -> np.ndarray:
    i0 = int(np.clip(frame_index, 0, len(path.xy) - 1))
    i1 = int(np.clip(i0 + max(1, int(lookahead_frames)), i0 + 1, len(path.xy) - 1))
    return path.xy[i0 : i1 + 1]


def has_lateral_sign_change(lateral_values: List[float], eps: float = 0.25) -> bool:
    vals = [float(v) for v in lateral_values if np.isfinite(v) and abs(float(v)) > eps]
    if len(vals) < 2:
        return False
    return (min(vals) < -eps) and (max(vals) > eps)


def find_pedestrian_conflict_events(path: EgoPath, ped_by_inst: Dict[str, List[PedObservation]], args: argparse.Namespace) -> Tuple[List[ConflictEvent], Dict[str, Any]]:
    events: List[ConflictEvent] = []
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for inst, obs_list in ped_by_inst.items():
        if len(obs_list) < int(args.min_ped_frames):
            continue
        lateral_values: List[float] = []
        inst_events: List[ConflictEvent] = []
        min_future = float("inf")
        min_route = float("inf")
        min_gap = float("inf")
        for obs in obs_list:
            proj = project_point_to_path(path, obs.x, obs.y)
            lateral_values.append(proj.signed_lateral_m)
            min_route = min(min_route, proj.distance_m)
            fpoly = future_path_polyline(path, obs.frame_index, int(args.future_lookahead_frames))
            fdist = min_distance_point_to_polyline(np.array([obs.x, obs.y], dtype=float), fpoly)
            min_future = min(min_future, fdist)
            ego_t = interp_time_at_s(path, proj.s_m)
            gap = abs(float(obs.timestamp_s) - float(ego_t))
            min_gap = min(min_gap, gap)
            if fdist <= float(args.ped_future_path_distance_m) and gap <= float(args.max_conflict_time_gap_s):
                inst_events.append(
                    ConflictEvent(
                        instance_token=inst,
                        ann_token=obs.ann_token,
                        sample_token=obs.sample_token,
                        frame_index=int(obs.frame_index),
                        timestamp_s=float(obs.timestamp_s),
                        ped_x=float(obs.x),
                        ped_y=float(obs.y),
                        crossing_proxy_x=float(proj.qx),
                        crossing_proxy_y=float(proj.qy),
                        crossing_s_m=float(proj.s_m),
                        ped_to_future_path_m=float(fdist),
                        ped_to_route_m=float(proj.distance_m),
                        ego_arrival_time_s=float(ego_t),
                        ego_ped_time_gap_s=float(gap),
                        signed_lateral_m=float(proj.signed_lateral_m),
                    )
                )
        sign_change = has_lateral_sign_change(lateral_values)
        if len(inst_events) >= int(args.min_intersection_frames):
            events.extend(inst_events)
        diagnostics[inst] = {
            "num_observations": len(obs_list),
            "num_intersection_events": len(inst_events),
            "min_ped_to_future_path_m": min_future,
            "min_ped_to_route_m": min_route,
            "min_ego_ped_conflict_time_gap_s": min_gap,
            "ped_lateral_sign_change": sign_change,
        }
    events.sort(key=lambda e: (e.ped_to_future_path_m, e.ego_ped_time_gap_s, e.ped_to_route_m))
    return events, {
        "num_pedestrian_instances": len(ped_by_inst),
        "num_pedestrian_instances_with_min_frames": int(sum(1 for v in ped_by_inst.values() if len(v) >= int(args.min_ped_frames))),
        "num_conflict_events_total": len(events),
        "pedestrian_instance_diagnostics": diagnostics,
    }


def line_fit_rmse(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return float("inf")
    pts = points_xy.astype(float)
    center = pts.mean(axis=0)
    z = pts - center
    try:
        _u, _s, vt = np.linalg.svd(z, full_matrices=False)
        direction = vt[0]
        proj = np.outer(z @ direction, direction)
        residual = z - proj
        return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    except Exception:
        return float("inf")


def path_tortuosity(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 1.0
    path_len = float(np.sum(np.linalg.norm(np.diff(points_xy, axis=0), axis=1)))
    chord = float(np.linalg.norm(points_xy[-1] - points_xy[0]))
    if chord < 1e-6:
        return float("inf")
    return float(path_len / chord)


def fit_circle_least_squares(xy: np.ndarray) -> Dict[str, float]:
    if len(xy) < 4:
        return {"circle_radius_m": float("nan"), "circle_fit_rmse_m": float("nan"), "circle_sweep_deg": float("nan"), "circle_center_x": float("nan"), "circle_center_y": float("nan")}
    x = xy[:, 0]
    y = xy[:, 1]
    try:
        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x * x + y * y)
        a, bcoef, c = np.linalg.lstsq(A, b, rcond=None)[0]
        cx = -0.5 * a
        cy = -0.5 * bcoef
        r2 = cx * cx + cy * cy - c
        if r2 <= 1e-9:
            raise ValueError("non-positive radius squared")
        radius = float(math.sqrt(r2))
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        rmse = float(np.sqrt(np.mean((d - radius) ** 2)))
        ang = np.unwrap(np.arctan2(y - cy, x - cx))
        sweep = float(max(abs(ang[-1] - ang[0]), np.sum(np.abs(np.diff(ang)))) * 180.0 / math.pi)
        return {"circle_radius_m": radius, "circle_fit_rmse_m": rmse, "circle_sweep_deg": sweep, "circle_center_x": float(cx), "circle_center_y": float(cy)}
    except Exception:
        return {"circle_radius_m": float("nan"), "circle_fit_rmse_m": float("nan"), "circle_sweep_deg": float("nan"), "circle_center_x": float("nan"), "circle_center_y": float("nan")}


def compute_geometry_metrics(path: EgoPath, crossing_s: float, args: argparse.Namespace) -> Dict[str, Any]:
    s0 = max(float(path.s[0]), float(crossing_s) - float(args.geometry_window_before_m))
    s1 = min(float(path.s[-1]), float(crossing_s) + float(args.geometry_window_after_m))
    idx = np.where((path.s >= s0) & (path.s <= s1))[0]
    if len(idx) < 4:
        ci = int(np.argmin(np.abs(path.s - float(crossing_s))))
        lo = max(0, ci - 6)
        hi = min(len(path.xy), ci + 7)
        idx = np.arange(lo, hi)
    pts = path.xy[idx]
    yaw = np.unwrap(path.yaw_rad[idx]) if len(idx) else np.array([0.0])

    if len(pts) < 2:
        heading_change = 0.0
        rotation_coverage = 0.0
        tort = 1.0
        local_path_len = 0.0
        local_chord = 0.0
    else:
        heading_change = float(abs(yaw[-1] - yaw[0]) * 180.0 / math.pi)
        rotation_coverage = float(np.sum(np.abs(np.diff(yaw))) * 180.0 / math.pi)
        tort = path_tortuosity(pts)
        local_path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        local_chord = float(np.linalg.norm(pts[-1] - pts[0]))

    # Pre-crossing line fit: useful for detecting straight approach.
    pre_idx = np.where((path.s >= max(float(path.s[0]), float(crossing_s) - float(args.geometry_window_before_m))) & (path.s <= float(crossing_s)))[0]
    rmse = line_fit_rmse(path.xy[pre_idx]) if len(pre_idx) >= 2 else float("inf")
    curvature_score = float(max(0.0, tort - 1.0)) if np.isfinite(tort) else float("inf")
    circle = fit_circle_least_squares(pts)

    radius = circle.get("circle_radius_m", float("nan"))
    circle_rmse = circle.get("circle_fit_rmse_m", float("nan"))
    sweep = circle.get("circle_sweep_deg", float("nan"))
    circle_good = False
    try:
        circle_good = bool(np.isfinite(float(radius)) and np.isfinite(float(circle_rmse)) and float(radius) <= float(args.max_circle_radius_m) and float(circle_rmse) <= float(args.max_circle_fit_rmse_m) and float(sweep) >= float(args.high_circle_sweep_deg))
    except Exception:
        circle_good = False

    is_high = bool(
        heading_change >= float(args.high_heading_change_deg)
        or rotation_coverage >= float(args.high_rotation_coverage_deg)
        or (np.isfinite(tort) and tort >= float(args.high_tortuosity))
        or circle_good
    )
    is_straight = bool(
        heading_change <= float(args.straight_heading_change_deg)
        and rmse <= float(args.straight_line_rmse_m)
        and curvature_score <= float(args.straight_curvature_score)
    )

    if is_high:
        geometry_case = GEOM_HIGH
        subtype = "high_curvature_or_circular"
        confidence = np.mean([
            min(1.0, heading_change / max(float(args.high_heading_change_deg), 1e-6)),
            min(1.0, rotation_coverage / max(float(args.high_rotation_coverage_deg), 1e-6)),
            min(1.0, tort / max(float(args.high_tortuosity), 1e-6)) if np.isfinite(tort) else 1.0,
            1.0 if circle_good else 0.0,
        ])
    elif is_straight:
        geometry_case = GEOM_STRAIGHT
        subtype = "clear_straight"
        confidence = np.mean([
            max(0.0, 1.0 - heading_change / max(float(args.straight_heading_change_deg), 1e-6)),
            max(0.0, 1.0 - rmse / max(float(args.straight_line_rmse_m), 1e-6)),
            max(0.0, 1.0 - curvature_score / max(float(args.straight_curvature_score), 1e-6)),
        ])
    else:
        geometry_case = GEOM_MODERATE
        subtype = "moderate_curvature"
        # Confidence is highest for paths that are clearly not straight but not high-curvature.
        low = float(args.straight_heading_change_deg)
        high = float(args.high_heading_change_deg)
        mid_score = 0.0
        if high > low:
            mid_score = min(1.0, max(0.0, (heading_change - low) / (high - low)))
        confidence = 0.45 + 0.45 * mid_score

    return {
        "geometry_case_candidate": geometry_case,
        "geometry_subtype": subtype,
        "geometry_confidence_score": float(np.clip(confidence, 0.0, 1.0)),
        "approach_heading_change_deg": float(heading_change),
        "rotation_coverage_deg": float(rotation_coverage),
        "pre_crossing_line_fit_rmse_m": float(rmse),
        "path_tortuosity": float(tort),
        "path_curvature_score": float(curvature_score),
        "local_path_length_m": float(local_path_len),
        "local_chord_length_m": float(local_chord),
        "num_points_geometry_window": int(len(pts)),
        **circle,
    }


def compute_selection_score(best: ConflictEvent, conflict_events: List[ConflictEvent], map_distance_m: float, ped_sign_change: bool, before_m: float, after_m: float, geom: Dict[str, Any], args: argparse.Namespace) -> float:
    future_term = max(0.0, 1.0 - best.ped_to_future_path_m / max(float(args.ped_future_path_distance_m), 1e-6))
    time_term = max(0.0, 1.0 - best.ego_ped_time_gap_s / max(float(args.max_conflict_time_gap_s), 1e-6))
    route_term = max(0.0, 1.0 - best.ped_to_route_m / max(float(args.ped_future_path_distance_m), 1e-6))
    map_term = max(0.0, 1.0 - map_distance_m / max(float(args.max_map_crossing_distance_m), 1e-6)) if np.isfinite(map_distance_m) else 0.0
    before_term = min(1.0, max(0.0, before_m / max(float(args.min_ego_before_crossing_m), 1e-6)))
    after_term = min(1.0, max(0.0, after_m / max(float(args.min_ego_after_crossing_m), 1e-6)))
    sustained_term = min(1.0, len(conflict_events) / max(float(args.min_intersection_frames) * 3.0, 1.0))
    base = 45.0 * future_term + 25.0 * time_term + 15.0 * route_term + 20.0 * map_term + 12.0 * sustained_term + 8.0 * before_term + 8.0 * after_term + 10.0 * float(bool(ped_sign_change))

    gcase = str(geom.get("geometry_case_candidate", ""))
    gconf = float(geom.get("geometry_confidence_score", 0.0) or 0.0)
    heading = float(geom.get("approach_heading_change_deg", 0.0) or 0.0)
    rot = float(geom.get("rotation_coverage_deg", 0.0) or 0.0)
    tort = float(geom.get("path_tortuosity", 1.0) or 1.0)
    sweep = float(geom.get("circle_sweep_deg", 0.0) or 0.0)
    if gcase == GEOM_STRAIGHT:
        geometry_bonus = 25.0 * gconf
    elif gcase == GEOM_HIGH:
        geometry_bonus = 0.45 * heading + 0.30 * rot + 120.0 * max(0.0, tort - 1.0) + 0.25 * sweep + 18.0 * gconf
    else:
        # Moderate curvature should be meaningfully curved but not circular.
        ideal_mid = 32.0
        mid_bonus = max(0.0, 1.0 - abs(heading - ideal_mid) / max(ideal_mid, 1.0))
        geometry_bonus = 20.0 * mid_bonus + 15.0 * gconf
    return float(base + geometry_bonus)


def analyze_scene(nusc: Any, scene: Dict[str, Any], version: str, dataroot: Path, map_cache: MapCrossingCache, args: argparse.Namespace) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    scene_name = str(scene.get("name", ""))
    location = get_scene_location(nusc, scene)
    if location not in SINGAPORE_LOCATIONS:
        return None, "non_singapore_location"
    sample_tokens = iter_scene_sample_tokens(nusc, scene)
    if len(sample_tokens) < 2:
        return None, "too_few_samples"
    ego_path = build_ego_path(nusc, sample_tokens)
    if ego_path is None or len(ego_path.xy) < 2:
        return None, "unusable_ego_path"
    if ego_path.total_distance_m < float(args.min_ego_total_distance_m):
        return None, "ego_total_distance_too_short"
    if ego_path.avg_step_distance_m < float(args.min_ego_avg_step_distance_m):
        return None, "ego_avg_step_too_small"
    ped_by_inst = collect_pedestrian_observations(nusc, ego_path)
    if len(ped_by_inst) == 0:
        return None, "no_pedestrians"
    conflict_events, ped_info = find_pedestrian_conflict_events(ego_path, ped_by_inst, args)
    if len(conflict_events) < int(args.min_intersection_frames):
        return None, "no_ped_future_path_intersection"
    best = conflict_events[0]
    best_inst_diag = ped_info.get("pedestrian_instance_diagnostics", {}).get(best.instance_token, {})
    ped_sign_change = bool(best_inst_diag.get("ped_lateral_sign_change", False))
    if args.require_lateral_sign_change and not ped_sign_change:
        return None, "no_ped_lateral_sign_change"
    before_m = float(best.crossing_s_m - float(ego_path.s[0]))
    after_m = float(float(ego_path.s[-1]) - best.crossing_s_m)
    if before_m < float(args.min_ego_before_crossing_m) or after_m < float(args.min_ego_after_crossing_m):
        return None, "insufficient_ego_approach_before_or_after_crossing"

    map_info = map_cache.nearest_crossing(location, best.crossing_proxy_x, best.crossing_proxy_y, float(args.max_map_crossing_distance_m))
    map_distance = float(map_info.get("nearest_map_crossing_distance_m", float("inf")))
    real_map_crossing_found = bool(map_info.get("real_map_crossing_found", False))
    if args.strict_map_crossing and not real_map_crossing_found:
        return None, "no_real_map_ped_crossing_near_conflict_point"

    geom = compute_geometry_metrics(ego_path, best.crossing_s_m, args)
    final_score = compute_selection_score(best, conflict_events, map_distance, ped_sign_change, before_m, after_m, geom, args)

    try:
        desc = str(scene.get("description", "") or "")
    except Exception:
        desc = ""

    rec: Dict[str, Any] = {
        "scene_name": scene_name,
        "scene_token": str(scene.get("token", "")),
        "dataset_version": version,
        "resolved_dataroot": str(dataroot),
        "location": location,
        "scene_description": desc,
        "nbr_samples": int(len(sample_tokens)),
        "ego_total_distance_m": round_or_none(ego_path.total_distance_m, 3),
        "ego_avg_step_distance_m": round_or_none(ego_path.avg_step_distance_m, 4),
        "crossing_proxy_x": round_or_none(best.crossing_proxy_x, 3),
        "crossing_proxy_y": round_or_none(best.crossing_proxy_y, 3),
        "crossing_proxy_s_m": round_or_none(best.crossing_s_m, 3),
        "crossing_proxy_frame": int(best.frame_index),
        "ego_distance_before_crossing_m": round_or_none(before_m, 3),
        "ego_distance_after_crossing_m": round_or_none(after_m, 3),
        "ped_future_path_intersection": True,
        "intersection_frames": int(len(conflict_events)),
        "min_ped_to_future_path_m": round_or_none(best.ped_to_future_path_m, 3),
        "min_ped_to_route_m": round_or_none(best.ped_to_route_m, 3),
        "ego_ped_conflict_time_gap_s": round_or_none(best.ego_ped_time_gap_s, 3),
        "crossing_pedestrian_tokens": sorted(list({e.instance_token for e in conflict_events})),
        "best_crossing_pedestrian_token": best.instance_token,
        "best_crossing_ann_token": best.ann_token,
        "ped_lateral_sign_change": bool(ped_sign_change),
        "ped_crossing_behavior_detected": bool(ped_sign_change or len(conflict_events) >= max(int(args.min_intersection_frames), 3)),
        "num_pedestrian_instances": int(ped_info.get("num_pedestrian_instances", 0)),
        "num_pedestrian_instances_with_min_frames": int(ped_info.get("num_pedestrian_instances_with_min_frames", 0)),
        "final_selection_score": round_or_none(final_score, 4),
    }
    rec.update(map_info)
    rec.update({k: round_or_none(v, 4) if isinstance(v, (float, int, np.floating)) else v for k, v in geom.items()})
    return rec, None


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))


def make_scene_index_record(nusc: Any, scene: Dict[str, Any], version: str, dataroot: Path) -> Dict[str, Any]:
    return {
        "scene_name": str(scene.get("name", "")),
        "scene_token": str(scene.get("token", "")),
        "dataset_version": version,
        "resolved_dataroot": str(dataroot),
        "location": get_scene_location(nusc, scene),
        "scene_description": str(scene.get("description", "") or ""),
        "nbr_samples": int(scene.get("nbr_samples", 0) or 0),
        "first_sample_token": str(scene.get("first_sample_token", "") or ""),
        "last_sample_token": str(scene.get("last_sample_token", "") or ""),
    }


def group_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {GEOM_STRAIGHT: [], GEOM_MODERATE: [], GEOM_HIGH: []}
    for rec in candidates:
        g = str(rec.get("geometry_case_candidate", GEOM_MODERATE))
        groups.setdefault(g, []).append(rec)
    for g in groups:
        groups[g].sort(key=lambda r: float(r.get("final_selection_score") or 0.0), reverse=True)
    return groups


def build_metadata(args: argparse.Namespace, versions: List[str], dataroots: Dict[str, Optional[Path]]) -> Dict[str, Any]:
    return {
        "script": Path(__file__).name,
        "generated_utc": utc_now_iso(),
        "purpose": "Unified three-geometry scene discovery and fixed tested-scene configuration for the OSM-error-aware AV crossing project.",
        "versions_requested": versions,
        "resolved_dataroots": {v: str(p) if p is not None else None for v, p in dataroots.items()},
        "allowed_locations": sorted(SINGAPORE_LOCATIONS),
        "geometry_cases": [GEOM_STRAIGHT, GEOM_MODERATE, GEOM_HIGH],
        "tested_scene_names": [x["scene_name"] for x in TESTED_SCENES],
        "hardcoded_tested_scenes_note": "These three scenes are the final scenes used by the project pipeline; candidate ranking is kept for documentation and backup choices.",
        "filters": {
            "singapore_only": True,
            "strict_map_crossing": bool(args.strict_map_crossing),
            "min_ego_total_distance_m": args.min_ego_total_distance_m,
            "min_ego_avg_step_distance_m": args.min_ego_avg_step_distance_m,
            "min_ped_frames": args.min_ped_frames,
            "future_lookahead_frames": args.future_lookahead_frames,
            "ped_future_path_distance_m": args.ped_future_path_distance_m,
            "min_intersection_frames": args.min_intersection_frames,
            "max_conflict_time_gap_s": args.max_conflict_time_gap_s,
            "min_ego_before_crossing_m": args.min_ego_before_crossing_m,
            "min_ego_after_crossing_m": args.min_ego_after_crossing_m,
            "require_lateral_sign_change": bool(args.require_lateral_sign_change),
            "max_map_crossing_distance_m": args.max_map_crossing_distance_m,
        },
        "geometry_thresholds": {
            "straight_heading_change_deg": args.straight_heading_change_deg,
            "straight_line_rmse_m": args.straight_line_rmse_m,
            "straight_curvature_score": args.straight_curvature_score,
            "high_heading_change_deg": args.high_heading_change_deg,
            "high_rotation_coverage_deg": args.high_rotation_coverage_deg,
            "high_tortuosity": args.high_tortuosity,
            "high_circle_sweep_deg": args.high_circle_sweep_deg,
            "max_circle_fit_rmse_m": args.max_circle_fit_rmse_m,
            "max_circle_radius_m": args.max_circle_radius_m,
            "geometry_window_before_m": args.geometry_window_before_m,
            "geometry_window_after_m": args.geometry_window_after_m,
        },
    }


def make_downstream_config(selected: Dict[str, Any], selection_role: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    meta = {
        "generated_utc": utc_now_iso(),
        "selection_role": selection_role,
        "nuscenes_version": selected.get("dataset_version"),
        "resolved_dataroot": selected.get("resolved_dataroot"),
        "geometry_case": selected.get("geometry_case_candidate") or selected.get("geometry_case"),
        "source_selector_script": Path(__file__).name,
        "note": "Compatible with downstream extraction scripts expecting metadata + best_candidate.",
        "hardcoded_tested_scene": bool(selected.get("hardcoded_tested_scene", False)),
        "project_tested_scenes": metadata.get("tested_scene_names", []),
    }
    return {"metadata": meta, "best_candidate": selected}


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys and not isinstance(r.get(k), (list, dict)):
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def build_tested_scene_records(candidates: List[Dict[str, Any]], scene_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_candidate_by_name: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        name = str(c.get("scene_name", ""))
        if name not in best_candidate_by_name or float(c.get("final_selection_score") or 0.0) > float(best_candidate_by_name[name].get("final_selection_score") or 0.0):
            best_candidate_by_name[name] = c

    tested_records: List[Dict[str, Any]] = []
    for spec in TESTED_SCENES:
        name = str(spec["scene_name"])
        gcase = str(spec["geometry_case"])
        if name in best_candidate_by_name:
            rec = dict(best_candidate_by_name[name])
            rec["found_in_dataset"] = True
            rec["passed_current_filters"] = True
        elif name in scene_index:
            rec = dict(scene_index[name])
            rec["found_in_dataset"] = True
            rec["passed_current_filters"] = False
            rec["filter_note"] = "Scene was found in the nuScenes tables but did not pass the current candidate filters. It is still kept because it is hard-coded as a final tested scene."
        else:
            rec = {
                "scene_name": name,
                "scene_token": "",
                "dataset_version": None,
                "resolved_dataroot": None,
                "location": "",
                "scene_description": "",
                "found_in_dataset": False,
                "passed_current_filters": False,
                "filter_note": "Scene was not found in the requested versions/dataroots during this run. Check --versions and --dataroot.",
            }
        rec["geometry_case_candidate"] = gcase
        rec["geometry_case_hardcoded"] = gcase
        rec["hardcoded_tested_scene"] = True
        rec["selection_order"] = int(spec["selection_order"])
        rec["selection_role"] = str(spec["role_in_study"])
        rec["downstream_config_filename"] = str(spec["downstream_config_filename"])
        rec["refinement_script_hint"] = str(spec["refinement_script_hint"])
        tested_records.append(rec)
    tested_records.sort(key=lambda r: int(r.get("selection_order", 999)))
    return tested_records


def write_outputs(output_dir: Path, selected_config_dir: Path, metadata: Dict[str, Any], candidates: List[Dict[str, Any]], rejections: Dict[str, int], tested_records: List[Dict[str, Any]]) -> None:
    ensure_dir(output_dir)
    ensure_dir(selected_config_dir)
    candidates_sorted = sorted(candidates, key=lambda r: float(r.get("final_selection_score") or 0.0), reverse=True)
    grouped = group_candidates(candidates_sorted)

    write_json(output_dir / "candidate_scenes_all.json", {
        "metadata": metadata,
        "rejection_counts": dict(sorted(rejections.items())),
        "num_candidates_passed": len(candidates_sorted),
        "candidates": candidates_sorted,
    })
    write_json(output_dir / "candidate_scenes_by_geometry.json", {
        "metadata": metadata,
        "rejection_counts": dict(sorted(rejections.items())),
        "candidate_counts_by_geometry": {k: len(v) for k, v in grouped.items()},
        "candidates_by_geometry": grouped,
    })
    write_json(output_dir / "tested_scenes_hardcoded.json", {
        "metadata": {
            "generated_utc": utc_now_iso(),
            "note": "Hard-coded final tested scenes used by the project pipeline.",
        },
        "tested_scenes": TESTED_SCENES,
    })
    write_json(output_dir / "selected_tested_scenes.json", {
        "metadata": metadata,
        "tested_scene_records": tested_records,
        "missing_tested_scenes": [r for r in tested_records if not r.get("found_in_dataset")],
    })
    write_csv(output_dir / "selected_tested_scenes.csv", tested_records)

    for rec in tested_records:
        fname = str(rec.get("downstream_config_filename") or f"{safe_name(rec.get('scene_name', 'scene'))}.json")
        payload = make_downstream_config(rec, str(rec.get("selection_role", "tested_scene")), metadata)
        write_json(selected_config_dir / fname, payload)


def main() -> None:
    args = parse_args()
    versions = parse_csv(args.versions)
    output_dir = Path(args.output_dir).expanduser().resolve()
    selected_config_dir = Path(args.selected_config_dir).expanduser().resolve()

    try:
        from nuscenes.nuscenes import NuScenes  # noqa: F401
    except ModuleNotFoundError:
        print("Error: The Python package 'nuscenes' is not installed in this environment.")
        print("Install it with: pip install nuscenes-devkit")
        sys.exit(1)

    install_mapmask_dummy_for_missing_png()
    from nuscenes.nuscenes import NuScenes

    dataroots: Dict[str, Optional[Path]] = {}
    all_candidates: List[Dict[str, Any]] = []
    rejection_counts: Dict[str, int] = {}
    scenes_checked_by_version: Dict[str, int] = {}
    scenes_loaded_by_version: Dict[str, int] = {}
    scene_index: Dict[str, Dict[str, Any]] = {}

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for version in versions:
        dataroot = resolve_dataroot_for_version(args.dataroot, version)
        dataroots[version] = dataroot
        if dataroot is None:
            reject(f"dataroot_not_found_for_{version}")
            print(f"Warning: could not find dataroot for {version}; skipping.")
            continue
        print(f"Loading nuScenes {version} from {dataroot} ...")
        try:
            nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
        except Exception as exc:
            reject(f"load_failed_for_{version}")
            print(f"Warning: failed to load {version}: {exc}")
            continue

        scenes = list(nusc.scene)
        scenes_loaded_by_version[version] = len(scenes)
        if int(args.max_scenes_per_version) > 0:
            scenes = scenes[: int(args.max_scenes_per_version)]
        map_cache = MapCrossingCache(dataroot=dataroot, version=version, explicit_map_dataroot=str(args.map_dataroot or ""))
        checked = 0
        passed = 0

        for scene in scenes:
            checked += 1
            scene_name = str(scene.get("name", ""))
            # Store scene table metadata for hard-coded tested scenes, even if candidate filters reject it.
            if scene_name in {x["scene_name"] for x in TESTED_SCENES}:
                scene_index[scene_name] = make_scene_index_record(nusc, scene, version, dataroot)
            try:
                rec, reason = analyze_scene(nusc, scene, version, dataroot, map_cache, args)
                if rec is None:
                    reject(reason or "unknown_rejection")
                    continue
                all_candidates.append(rec)
                passed += 1
                if args.verbose:
                    print(f"  candidate {rec['scene_name']} | {rec['geometry_case_candidate']} | score={rec['final_selection_score']} | loc={rec['location']}")
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                reject("scene_analysis_exception")
                if args.verbose:
                    print(f"Warning: exception while analyzing {version}/{scene_name}: {exc}")
                    traceback.print_exc()
                continue
        scenes_checked_by_version[version] = checked
        print(f"  checked {checked} scenes, kept {passed} candidates from {version}.")

    metadata = build_metadata(args, versions, dataroots)
    metadata["scenes_loaded_by_version"] = scenes_loaded_by_version
    metadata["scenes_checked_by_version"] = scenes_checked_by_version
    metadata["num_candidates_passed"] = len(all_candidates)
    metadata["scene_index_found_for_tested_scenes"] = {k: bool(k in scene_index) for k in [x["scene_name"] for x in TESTED_SCENES]}

    tested_records = build_tested_scene_records(all_candidates, scene_index)
    write_outputs(output_dir, selected_config_dir, metadata, all_candidates, rejection_counts, tested_records)

    grouped = group_candidates(all_candidates)
    print("\nThree-geometry candidate scan complete")
    print(f"  Output directory: {output_dir}")
    print(f"  Selected config directory: {selected_config_dir}")
    print(f"  Total candidates kept: {len(all_candidates)}")
    print("  Candidate counts:")
    for g in [GEOM_STRAIGHT, GEOM_MODERATE, GEOM_HIGH]:
        print(f"    - {g}: {len(grouped.get(g, []))}")
    print("  Hard-coded tested scenes:")
    for r in tested_records:
        status = "found" if r.get("found_in_dataset") else "NOT FOUND"
        passed = "passed filters" if r.get("passed_current_filters") else "kept by hard-code"
        print(f"    {r.get('selection_order')}. {r.get('scene_name')} -> {r.get('geometry_case_candidate')} ({status}, {passed})")
    print("  Files written:")
    print(f"    - {output_dir / 'candidate_scenes_all.json'}")
    print(f"    - {output_dir / 'candidate_scenes_by_geometry.json'}")
    print(f"    - {output_dir / 'tested_scenes_hardcoded.json'}")
    print(f"    - {output_dir / 'selected_tested_scenes.json'}")
    print(f"    - {selected_config_dir}")


if __name__ == "__main__":
    main()
