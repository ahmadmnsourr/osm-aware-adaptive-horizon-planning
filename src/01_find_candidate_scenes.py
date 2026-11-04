#!/usr/bin/env python3
"""
Scan nuScenes for strict candidate scenes: sufficient ego motion + nearby, path-relevant pedestrians.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration — edit these values for your machine and filtering strictness
# ---------------------------------------------------------------------------

# nuScenes: `dataroot` is the folder that CONTAINS the version subfolder (with JSON
# tables), not the version name itself. Example layout:
#   /data/sets/nuscenes/v1.0-mini/scene.json  -> dataroot=/data/sets/nuscenes
# Override with env: NUSCENES_DATAROOT=/path/to/nuscenes
NUSCENES_DATAROOT = os.environ.get("NUSCENES_DATAROOT", "").strip()
NUSCENES_VERSION = os.environ.get("NUSCENES_VERSION", "v1.0-mini").strip() or "v1.0-mini"

# Output: default relative path under project dir; override with env CANDIDATE_SCENES_JSON
# or optional config/nuscenes_pipeline.json (paths.candidate_scenes_json) if that file exists.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "nuscenes_pipeline.json")
DEFAULT_CANDIDATE_SCENES_JSON_REL = "config/candidate_scenes.json"
# Top-ranked scene only (after sorting by score). Override with env BEST_CANDIDATE_JSON.
DEFAULT_BEST_CANDIDATE_JSON_REL = "config/best_candidate_scene.json"
OUTPUT_JSON = ""

# Ego motion (global frame, meters)
MIN_EGO_TOTAL_DISTANCE_M = 30.0
MIN_EGO_AVG_STEP_DISTANCE_M = 0.8

# Pedestrian relevance
MAX_EGO_PED_DISTANCE_M = 15.0
MIN_PED_FRAMES = 5
MAX_PED_TO_EGO_PATH_M = 5.0

# Scoring weights
SCORE_WEIGHT_EGO_MOVEMENT = 1.5
SCORE_WEIGHT_RELEVANT_PED_COUNT = 20.0
SCORE_WEIGHT_CLOSEST_APPROACH = 12.0
SCORE_WEIGHT_SUSTAINED_FRAMES = 3.0

# Reference scale
SCORE_EGO_DISTANCE_REF_M = 100.0

LIDAR_CHANNEL = "LIDAR_TOP"


def _table_root_ok(dataroot: str, version: str) -> bool:
    """True if nuScenes JSON tables exist at dataroot/version/."""
    root = os.path.join(os.path.abspath(dataroot), version)
    return os.path.isfile(os.path.join(root, "scene.json"))


def resolve_nuscenes_dataroot(version: str) -> Optional[str]:
    """
    Find a valid nuScenes dataroot (folder whose child `version/` contains scene.json).

    Order: NUSCENES_DATAROOT env (already in NUSCENES_DATAROOT), cwd, ./data, /data/sets/nuscenes.
    Also accepts if the user mistakenly set dataroot to .../v1.0-mini (uses parent).
    """
    candidates: List[str] = []
    if NUSCENES_DATAROOT:
        candidates.append(os.path.expanduser(NUSCENES_DATAROOT))
    candidates.extend(
        [
            os.getcwd(),
            os.path.join(os.getcwd(), "data"),
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

        # Standard: dataroot/version/scene.json
        if _table_root_ok(path, version):
            return path

        # User pointed at .../v1.0-mini instead of parent
        if os.path.basename(path.rstrip(os.sep)) == version and os.path.isfile(
            os.path.join(path, "scene.json")
        ):
            parent = os.path.dirname(path)
            if parent and _table_root_ok(parent, version):
                return parent

    return None


def get_candidate_scenes_json_rel() -> str:
    """
    Relative (to project) or absolute path for candidate_scenes output.

    Priority: env CANDIDATE_SCENES_JSON, else optional config/nuscenes_pipeline.json
    (paths.candidate_scenes_json) if present, else DEFAULT_CANDIDATE_SCENES_JSON_REL.
    The pipeline JSON is never required and is never created by this script.
    """
    env = os.environ.get("CANDIDATE_SCENES_JSON", "").strip()
    if env:
        return env
    if os.path.isfile(PIPELINE_CONFIG_PATH):
        try:
            with open(PIPELINE_CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                r = cfg.get("paths", {}).get("candidate_scenes_json")
                if isinstance(r, str) and r.strip():
                    return r.strip()
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return DEFAULT_CANDIDATE_SCENES_JSON_REL


def get_best_candidate_json_rel() -> str:
    """Path for JSON containing only the single best candidate (env BEST_CANDIDATE_JSON overrides)."""
    env = os.environ.get("BEST_CANDIDATE_JSON", "").strip()
    if env:
        return env
    return DEFAULT_BEST_CANDIDATE_JSON_REL


def resolve_output_json_path(rel: str) -> str:
    """Absolute path for candidate JSON."""
    rel = rel.strip()
    if not rel:
        rel = DEFAULT_CANDIDATE_SCENES_JSON_REL
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    return os.path.normpath(os.path.join(SCRIPT_DIR, rel))


def install_mapmask_dummy_for_missing_png() -> None:
    """
    NuScenes loads semantic map PNGs on init; partial installs omit them.
    This script does not use maps. Use in-memory dummy masks instead of writing `maps/` on disk.
    """
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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _xy(p: Sequence[float]) -> np.ndarray:
    """Take global translation, keep XY."""
    t = np.asarray(p, dtype=np.float64)
    return t[:2]


def point_to_segment_distance_sq(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Squared distance from point p to segment ab (2D)."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.sum((p - a) ** 2))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / denom))
    proj = a + t * ab
    return float(np.sum((p - proj) ** 2))


def min_distance_point_to_polyline_xy(point_xy: np.ndarray, polyline_xy: np.ndarray) -> float:
    """
    Minimum Euclidean distance from point to a polyline (piecewise linear) in 2D.
    Degenerate cases: 0 or 1 vertex use point-to-point logic.
    """
    if polyline_xy.shape[0] == 0:
        return float("inf")
    if polyline_xy.shape[0] == 1:
        return float(np.linalg.norm(point_xy - polyline_xy[0]))
    best = float("inf")
    for i in range(polyline_xy.shape[0] - 1):
        d = math.sqrt(point_to_segment_distance_sq(point_xy, polyline_xy[i], polyline_xy[i + 1]))
        if d < best:
            best = d
    return best


def polyline_length_xy(polyline_xy: np.ndarray) -> float:
    """Total arc length of a 2D polyline."""
    if polyline_xy.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(polyline_xy, axis=0), axis=1)))


def avg_step_length_xy(polyline_xy: np.ndarray) -> float:
    """Mean distance between consecutive samples along the polyline."""
    if polyline_xy.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(polyline_xy, axis=0), axis=1)
    return float(np.mean(steps))


# ---------------------------------------------------------------------------
# nuScenes traversal
# ---------------------------------------------------------------------------


def iter_scene_sample_tokens(nusc: Any, scene: Dict[str, Any]) -> List[str]:
    """Ordered list of sample tokens in the scene (key frames)."""
    tokens: List[str] = []
    cur = scene["first_sample_token"]
    while cur:
        tokens.append(cur)
        sample = nusc.get("sample", cur)
        nxt = sample.get("next") or ""
        if not nxt:
            break
        cur = nxt
    return tokens


def ego_xy_from_sample(nusc: Any, sample: Dict[str, Any]) -> Optional[np.ndarray]:
    """Ego vehicle position (XY, global) at this keyframe using LIDAR_TOP pose."""
    try:
        data = sample.get("data") or {}
        sd_token = data.get(LIDAR_CHANNEL)
        if not sd_token:
            return None
        sd = nusc.get("sample_data", sd_token)
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        return _xy(pose["translation"])
    except (KeyError, TypeError, ValueError):
        return None


def is_pedestrian_category(category_name: str) -> bool:
    """nuScenes pedestrian classes share this prefix."""
    return isinstance(category_name, str) and category_name.startswith("human.pedestrian")


# ---------------------------------------------------------------------------
# Per-scene analysis
# ---------------------------------------------------------------------------


def collect_scene_geometry(
    nusc: Any, sample_tokens: List[str]
) -> Tuple[Optional[np.ndarray], Dict[str, List[Tuple[str, np.ndarray, str]]], Dict[str, np.ndarray]]:
    """
    Build ego XY polyline and pedestrian observations per instance.

    Returns:
      ego_polyline_xy: (N, 2) or None if unusable
      ped_by_instance: instance_token -> list of (sample_token, ped_xy, ann_token)
      ego_xy_by_sample: sample_token -> ego XY for that keyframe (alignment for distances)
    """
    ego_points: List[np.ndarray] = []
    ego_xy_by_sample: Dict[str, np.ndarray] = {}
    ped_by_instance: Dict[str, List[Tuple[str, np.ndarray, str]]] = {}

    for st in sample_tokens:
        try:
            sample = nusc.get("sample", st)
        except Exception:
            continue

        e_xy = ego_xy_from_sample(nusc, sample)
        if e_xy is None:
            continue
        ego_points.append(e_xy)
        ego_xy_by_sample[st] = e_xy

        for ann_t in sample.get("anns") or []:
            try:
                ann = nusc.get("sample_annotation", ann_t)
            except Exception:
                continue
            cname = ann.get("category_name", "")
            if not is_pedestrian_category(cname):
                continue
            inst = ann.get("instance_token") or ann_t
            p_xy = _xy(ann["translation"])
            ped_by_instance.setdefault(inst, []).append((st, p_xy, ann_t))

    if len(ego_points) < 2:
        return None, ped_by_instance, ego_xy_by_sample

    ego_poly = np.stack(ego_points, axis=0)
    return ego_poly, ped_by_instance, ego_xy_by_sample


def evaluate_pedestrian_relevance(
    ego_poly_xy: np.ndarray,
    ped_by_instance: Dict[str, List[Tuple[str, np.ndarray, str]]],
    ego_xy_by_sample: Dict[str, np.ndarray],
) -> Tuple[int, int, List[str], int, float]:
    """
    Apply strict proximity + path-distance + minimum frame rules per instance.

    Returns:
      num_ped_instances: pedestrian instances with any annotation in scene
      num_relevant: instances meeting all thresholds
      relevant_instance_tokens: which instances passed
      total_relevant_frames: sum over instances of qualifying frame counts
      min_dist_m: minimum ego–pedestrian XY distance among qualifying observations
    """
    num_ped_instances = len(ped_by_instance)
    relevant_instance_tokens: List[str] = []
    total_relevant_frames = 0
    min_dist_m = float("inf")

    for inst, obs in ped_by_instance.items():
        good_frames = 0
        for sample_tok, p_xy, _ann in obs:
            e_xy = ego_xy_by_sample.get(sample_tok)
            if e_xy is None:
                continue
            d_ego = float(np.linalg.norm(p_xy - e_xy))
            d_path = min_distance_point_to_polyline_xy(p_xy, ego_poly_xy)
            if d_ego <= MAX_EGO_PED_DISTANCE_M and d_path <= MAX_PED_TO_EGO_PATH_M:
                good_frames += 1
                if d_ego < min_dist_m:
                    min_dist_m = d_ego

        if good_frames >= MIN_PED_FRAMES:
            relevant_instance_tokens.append(inst)
            total_relevant_frames += good_frames

    num_relevant = len(relevant_instance_tokens)
    if min_dist_m == float("inf"):
        min_dist_m = float("nan")

    return num_ped_instances, num_relevant, relevant_instance_tokens, total_relevant_frames, min_dist_m


def compute_candidate_score(
    ego_total_m: float,
    ego_avg_step_m: float,
    num_relevant_pedestrians: int,
    min_ego_ped_m: float,
    total_relevant_frames: int,
) -> float:
    """Higher is better; used for ranking candidates."""
    # Movement term: saturate so very long scenes don't dominate
    move = min(ego_total_m / max(SCORE_EGO_DISTANCE_REF_M, 1e-6), 2.0)
    proximity = 0.0
    if not math.isnan(min_ego_ped_m) and min_ego_ped_m < float("inf"):
        proximity = max(0.0, (MAX_EGO_PED_DISTANCE_M - min_ego_ped_m) / max(MAX_EGO_PED_DISTANCE_M, 1e-6))

    score = (
        SCORE_WEIGHT_EGO_MOVEMENT * move
        + SCORE_WEIGHT_RELEVANT_PED_COUNT * float(num_relevant_pedestrians)
        + SCORE_WEIGHT_CLOSEST_APPROACH * proximity
        + SCORE_WEIGHT_SUSTAINED_FRAMES * float(total_relevant_frames)
    )
    # Penalize marginal step motion slightly (noise / crawl)
    step_factor = min(1.0, ego_avg_step_m / max(MIN_EGO_AVG_STEP_DISTANCE_M, 1e-6))
    score *= (0.5 + 0.5 * step_factor)
    return float(score)


def analyze_scene(nusc: Any, scene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Returns a candidate record dict if the scene passes all filters; otherwise None.
    """
    sample_tokens = iter_scene_sample_tokens(nusc, scene)
    if len(sample_tokens) < 2:
        return None

    ego_poly_xy, ped_by_instance, ego_xy_by_sample = collect_scene_geometry(nusc, sample_tokens)
    if ego_poly_xy is None or ego_poly_xy.shape[0] < 2:
        return None

    ego_total_m = polyline_length_xy(ego_poly_xy)
    ego_avg_step_m = avg_step_length_xy(ego_poly_xy)

    if ego_total_m < MIN_EGO_TOTAL_DISTANCE_M or ego_avg_step_m < MIN_EGO_AVG_STEP_DISTANCE_M:
        return None

    num_ped_instances, num_relevant, _rel_tokens, total_rel_frames, min_dist_m = evaluate_pedestrian_relevance(
        ego_poly_xy, ped_by_instance, ego_xy_by_sample
    )

    if num_ped_instances < 1 or num_relevant < 1:
        return None

    score = compute_candidate_score(
        ego_total_m, ego_avg_step_m, num_relevant, min_dist_m, total_rel_frames
    )

    return {
        "scene_name": scene.get("name", ""),
        "scene_token": scene.get("token", ""),
        "nbr_samples": len(sample_tokens),
        "ego_total_distance_m": round(ego_total_m, 3),
        "ego_avg_step_distance_m": round(ego_avg_step_m, 4),
        "num_pedestrian_instances": num_ped_instances,
        "num_relevant_pedestrians": num_relevant,
        "min_ego_ped_distance_m": round(min_dist_m, 3) if not math.isnan(min_dist_m) else None,
        "score": round(score, 4),
    }


def build_metadata(
    candidate_scenes_json_rel: Optional[str] = None,
    best_candidate_json_rel: Optional[str] = None,
) -> Dict[str, Any]:
    """Thresholds and paths written next to candidates for reproducibility."""
    return {
        "nuscenes_dataroot_env": NUSCENES_DATAROOT or None,
        "nuscenes_version": NUSCENES_VERSION,
        "pipeline_config": PIPELINE_CONFIG_PATH
        if os.path.isfile(PIPELINE_CONFIG_PATH)
        else None,
        "candidate_scenes_json": candidate_scenes_json_rel,
        "best_candidate_json": best_candidate_json_rel,
        "output_json": OUTPUT_JSON,
        "lidar_channel": LIDAR_CHANNEL,
        "thresholds": {
            "MIN_EGO_TOTAL_DISTANCE_M": MIN_EGO_TOTAL_DISTANCE_M,
            "MIN_EGO_AVG_STEP_DISTANCE_M": MIN_EGO_AVG_STEP_DISTANCE_M,
            "MAX_EGO_PED_DISTANCE_M": MAX_EGO_PED_DISTANCE_M,
            "MIN_PED_FRAMES": MIN_PED_FRAMES,
            "MAX_PED_TO_EGO_PATH_M": MAX_PED_TO_EGO_PATH_M,
        },
        "score_weights": {
            "SCORE_WEIGHT_EGO_MOVEMENT": SCORE_WEIGHT_EGO_MOVEMENT,
            "SCORE_WEIGHT_RELEVANT_PED_COUNT": SCORE_WEIGHT_RELEVANT_PED_COUNT,
            "SCORE_WEIGHT_CLOSEST_APPROACH": SCORE_WEIGHT_CLOSEST_APPROACH,
            "SCORE_WEIGHT_SUSTAINED_FRAMES": SCORE_WEIGHT_SUSTAINED_FRAMES,
            "SCORE_EGO_DISTANCE_REF_M": SCORE_EGO_DISTANCE_REF_M,
        },
    }


def write_best_candidate_file(
    best_abs: str,
    meta: Dict[str, Any],
    best: Optional[Dict[str, Any]],
) -> None:
    """Write config/best_candidate_scene.json (rank #1 after score sort)."""
    os.makedirs(os.path.dirname(best_abs) or ".", exist_ok=True)
    payload = {"metadata": meta, "best_candidate": best}
    with open(best_abs, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    global OUTPUT_JSON

    rel_out = get_candidate_scenes_json_rel()
    rel_best = get_best_candidate_json_rel()
    OUTPUT_JSON = resolve_output_json_path(rel_out)
    best_path = resolve_output_json_path(rel_best)

    from nuscenes.nuscenes import NuScenes

    out_path = OUTPUT_JSON
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    dataroot = resolve_nuscenes_dataroot(NUSCENES_VERSION)
    if dataroot is None:
        err = (
            f"Could not find nuScenes tables at <dataroot>/{NUSCENES_VERSION}/scene.json. "
            "Set NUSCENES_DATAROOT to your dataset root (the folder that contains the "
            f"'{NUSCENES_VERSION}' subfolder), e.g. export NUSCENES_DATAROOT=/data/sets/nuscenes"
        )
        print(f"Error: {err}")
        meta = build_metadata(rel_out, rel_best)
        meta.update({"error": err, "resolved_dataroot": None, "total_candidates_passed": 0})
        payload = {"metadata": meta, "candidates": []}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        write_best_candidate_file(best_path, meta, None)
        print(f"  Wrote empty candidates to: {os.path.abspath(out_path)}")
        print(f"  Best candidate: {os.path.abspath(best_path)} (none)")
        sys.exit(0)

    try:
        install_mapmask_dummy_for_missing_png()
        nusc = NuScenes(version=NUSCENES_VERSION, dataroot=dataroot, verbose=False)
    except Exception as exc:
        err = f"Failed to load nuScenes at {dataroot} ({NUSCENES_VERSION}): {exc}"
        print(f"Error: {err}")
        meta = build_metadata(rel_out, rel_best)
        meta.update({"error": err, "resolved_dataroot": dataroot, "total_candidates_passed": 0})
        payload = {"metadata": meta, "candidates": []}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        write_best_candidate_file(best_path, meta, None)
        print(f"  Wrote empty candidates to: {os.path.abspath(out_path)}")
        print(f"  Best candidate: {os.path.abspath(best_path)} (none)")
        sys.exit(0)

    scenes = list(nusc.scene)
    candidates: List[Dict[str, Any]] = []
    checked = 0

    for scene in scenes:
        checked += 1
        try:
            rec = analyze_scene(nusc, scene)
            if rec is not None:
                candidates.append(rec)
        except Exception as exc:
            # Skip corrupted / incomplete scenes without stopping the scan
            print(f"Warning: skipped scene {scene.get('name', scene.get('token'))}: {exc}")
            continue

    candidates.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    passed = len(candidates)
    meta = build_metadata(rel_out, rel_best)
    meta["resolved_dataroot"] = dataroot
    meta["total_candidates_passed"] = passed
    payload = {
        "metadata": meta,
        "candidates": candidates,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    best_one = candidates[0] if candidates else None
    write_best_candidate_file(best_path, meta, best_one)

    top_names = [c.get("scene_name", "") for c in candidates[:5]]

    print("nuScenes candidate scan complete")
    print(f"  Dataroot: {dataroot}")
    print(f"  Total scenes checked: {checked}")
    print(f"  Passed filters: {passed}")
    print(f"  Output: {os.path.abspath(out_path)}")
    print(f"  Best candidate: {os.path.abspath(best_path)}")
    if best_one:
        print(f"    → {best_one.get('scene_name', '')} (score={best_one.get('score')})")
    print(f"  Top scenes: {', '.join(top_names) if top_names else '(none)'}")


if __name__ == "__main__":
    main()
