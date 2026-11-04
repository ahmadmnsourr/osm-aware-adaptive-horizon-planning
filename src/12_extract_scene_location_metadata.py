#!/usr/bin/env python3
"""
Extract location-oriented metadata for final paper scenes.

Outputs:
- data/paper_scene_metadata/scene_location_metadata.csv
- data/paper_scene_metadata/scene_location_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_SCENES = ["scene-1019", "scene-0061", "scene-0383"]
GEOMETRY_TYPE_BY_SCENE = {
    "scene-1019": "Straight approach",
    "scene-0061": "Moderate-curvature approach",
    "scene-0383": "High-curvature / circular approach",
}

OUTPUT_COLUMNS = [
    "scene_name",
    "geometry_type",
    "nuscenes_scene_description",
    "nuscenes_log_token",
    "nuscenes_logfile",
    "nuscenes_location_raw",
    "nuscenes_location_readable",
    "nuscenes_date_captured",
    "nuscenes_vehicle",
    "map_name_or_location",
    "first_sample_token",
    "last_sample_token",
    "number_of_samples",
    "refined_crossing_center_x_m",
    "refined_crossing_center_y_m",
    "crossing_zone_arc_length_m",
    "ego_hold_line_arc_length_m",
    "nearest_osm_road_name",
    "nearest_osm_crossing_context",
    "paper_location_phrase",
]

MISSING = "not available"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract paper-ready scene location metadata from nuScenes and local outputs."
    )
    parser.add_argument(
        "--scene-name",
        "--scene-names",
        dest="scene_name",
        default="all",
        help="Use 'all' (default) or comma-separated scene names. Accepts with/without 'scene-' prefix.",
    )
    parser.add_argument("--dataroot", default=".", help="Project root that contains v1.0-* folders.")
    parser.add_argument("--output-dir", default="data/paper_scene_metadata")
    return parser.parse_args()


def normalize_scene_name_token(token: str) -> str:
    t = str(token).strip()
    if not t:
        return ""
    if t.lower() == "all":
        return "all"
    if t.startswith("scene-"):
        return t
    digits = "".join(ch for ch in t if ch.isdigit())
    if digits:
        return f"scene-{digits.zfill(4)}"
    return t


def parse_scene_name_list(text: str) -> List[str]:
    raw = [x.strip() for x in str(text).split(",") if x.strip()]
    if not raw:
        return list(DEFAULT_SCENES)
    out: List[str] = []
    for item in raw:
        norm = normalize_scene_name_token(item)
        if norm == "all":
            out.extend(DEFAULT_SCENES)
        elif norm:
            out.append(norm)
    seen = set()
    final = []
    for scene in out:
        if scene not in seen:
            seen.add(scene)
            final.append(scene)
    return final or list(DEFAULT_SCENES)


def load_json_file(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def merge_dicts(dicts: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for d in dicts:
        if isinstance(d, dict):
            merged.update(d)
    return merged


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def safe_value(value: Any) -> Any:
    return MISSING if is_missing(value) else value


def to_float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def to_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def location_readable(location_raw: str) -> str:
    if not location_raw:
        return MISSING
    mapping = {
        "singapore-onenorth": "Singapore One North",
        "singapore-hollandvillage": "Singapore Holland Village",
        "singapore-queenstown": "Singapore Queenstown",
        "boston-seaport": "Boston Seaport",
    }
    if location_raw in mapping:
        return mapping[location_raw]
    parts = location_raw.replace("_", "-").split("-")
    return " ".join(p.capitalize() for p in parts if p)


def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return float(math.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    qx = ax + t * abx
    qy = ay + t * aby
    return float(math.hypot(px - qx, py - qy))


def distance_point_polyline(px: float, py: float, geom_xy: List[List[float]]) -> float:
    if not geom_xy:
        return float("inf")
    if len(geom_xy) == 1:
        return float(math.hypot(px - float(geom_xy[0][0]), py - float(geom_xy[0][1])))
    best = float("inf")
    for i in range(len(geom_xy) - 1):
        ax, ay = float(geom_xy[i][0]), float(geom_xy[i][1])
        bx, by = float(geom_xy[i + 1][0]), float(geom_xy[i + 1][1])
        best = min(best, point_to_segment_distance(px, py, ax, ay, bx, by))
    return best


class NuScenesIndex:
    def __init__(self, dataroot: Path) -> None:
        self.dataroot = dataroot
        self.versions = ["v1.0-trainval", "v1.0-mini"]
        self.scene_by_version_name: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.log_by_version_token: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.map_by_version_logtoken: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._build()

    def _build(self) -> None:
        for version in self.versions:
            root = self.dataroot / version
            scene_list = load_json_file(root / "scene.json")
            log_list = load_json_file(root / "log.json")
            map_list = load_json_file(root / "map.json")
            if not isinstance(scene_list, list) or not isinstance(log_list, list):
                continue

            scene_by_name: Dict[str, Dict[str, Any]] = {}
            for row in scene_list:
                name = str(row.get("name", ""))
                if name:
                    scene_by_name[name] = row
            self.scene_by_version_name[version] = scene_by_name

            log_by_token: Dict[str, Dict[str, Any]] = {}
            for row in log_list:
                tok = str(row.get("token", ""))
                if tok:
                    log_by_token[tok] = row
            self.log_by_version_token[version] = log_by_token

            map_by_logtoken: Dict[str, Dict[str, Any]] = {}
            if isinstance(map_list, list):
                for m in map_list:
                    for lt in m.get("log_tokens", []) or []:
                        map_by_logtoken[str(lt)] = m
            self.map_by_version_logtoken[version] = map_by_logtoken

    def find_scene(self, scene_name: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        for version in self.versions:
            scene = self.scene_by_version_name.get(version, {}).get(scene_name)
            if scene is not None:
                return version, scene
        return None, None

    def get_log(self, version: str, log_token: str) -> Optional[Dict[str, Any]]:
        return self.log_by_version_token.get(version, {}).get(log_token)

    def get_map_for_log(self, version: str, log_token: str) -> Optional[Dict[str, Any]]:
        return self.map_by_version_logtoken.get(version, {}).get(log_token)


def load_scene_local_metadata(project_root: Path, scene_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_meta = load_json_file(project_root / "data" / "raw" / scene_name / "scene_metadata.json")
    raw_summary = load_json_file(project_root / "data" / "raw" / scene_name / "scene_summary.json")
    refined_meta = load_json_file(project_root / "data" / "refined" / scene_name / "scene_metadata.json")
    refined_summary = load_json_file(project_root / "data" / "refined" / scene_name / "scene_summary.json")

    metadata = merge_dicts([raw_meta, refined_meta])
    summary = merge_dicts([raw_summary, refined_summary])
    return metadata, summary


def compact_tags(tags: Dict[str, Any]) -> str:
    preferred = [
        "name",
        "highway",
        "crossing",
        "crossing_ref",
        "crossing:markings",
        "traffic_signals",
        "junction",
    ]
    pairs = []
    for k in preferred:
        if k in tags and not is_missing(tags[k]):
            pairs.append(f"{k}={tags[k]}")
    if not pairs:
        return MISSING
    return "; ".join(pairs)


def load_osm_context(project_root: Path, scene_name: str, ref_x: Optional[float], ref_y: Optional[float]) -> Tuple[str, str]:
    features_csv = project_root / "data" / "osm" / scene_name / "converted" / "osm_features_local.csv"
    alignment_json = project_root / "data" / "osm" / scene_name / "alignment" / "osm_alignment_summary.json"

    if not features_csv.exists():
        return MISSING, MISSING

    rows: List[Dict[str, Any]] = []
    try:
        with features_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception:
        return MISSING, MISSING

    by_feature_id = {str(r.get("feature_id", "")): r for r in rows}

    alignment = load_json_file(alignment_json) if alignment_json.exists() else None
    nearest_crossing_context = MISSING
    nearest_crossing_id = None

    if isinstance(alignment, dict):
        nearest_crossing_id = alignment.get("nearest_crossing_feature_id")
        if isinstance(nearest_crossing_id, str) and nearest_crossing_id in by_feature_id:
            r = by_feature_id[nearest_crossing_id]
            tags = {}
            try:
                tags = json.loads(r.get("source_tags_json") or "{}")
            except Exception:
                tags = {}
            nearest_crossing_context = (
                f"{r.get('feature_type', MISSING)} | osm_id={r.get('osm_id', MISSING)} | "
                f"tags: {compact_tags(tags)}"
            )
        else:
            near_type = alignment.get("nearest_crossing_feature_type")
            if not is_missing(near_type):
                nearest_crossing_context = f"{near_type}"

    road_name = MISSING
    road_rows = [r for r in rows if str(r.get("feature_type", "")) == "road_centerlike"]
    if road_rows:
        if ref_x is not None and ref_y is not None:
            best_name = None
            best_dist = float("inf")
            for r in road_rows:
                try:
                    geom = json.loads(r.get("geometry_local_json") or "[]")
                    if not isinstance(geom, list):
                        geom = []
                except Exception:
                    geom = []
                dist = distance_point_polyline(ref_x, ref_y, geom)
                try:
                    tags = json.loads(r.get("source_tags_json") or "{}")
                except Exception:
                    tags = {}
                candidate_name = (
                    tags.get("name")
                    or tags.get("addr:street")
                    or tags.get("official_name")
                    or tags.get("ref")
                )
                if candidate_name and dist < best_dist:
                    best_dist = dist
                    best_name = str(candidate_name)
            if best_name:
                road_name = best_name

        if road_name == MISSING:
            for r in road_rows:
                try:
                    tags = json.loads(r.get("source_tags_json") or "{}")
                except Exception:
                    tags = {}
                candidate_name = (
                    tags.get("name")
                    or tags.get("addr:street")
                    or tags.get("official_name")
                    or tags.get("ref")
                )
                if candidate_name:
                    road_name = str(candidate_name)
                    break

    return road_name if road_name else MISSING, nearest_crossing_context


def build_paper_phrase(scene_name: str, readable_loc: str) -> str:
    if readable_loc == MISSING:
        return f"nuScenes {scene_name}, location not available"
    return f"nuScenes {scene_name}, {readable_loc} map area"


def build_scene_row(project_root: Path, nusc_index: NuScenesIndex, scene_name: str) -> Dict[str, Any]:
    metadata_local, summary_local = load_scene_local_metadata(project_root, scene_name)
    version, scene_row = nusc_index.find_scene(scene_name)

    log_row = None
    map_row = None
    if scene_row is not None and version is not None:
        log_token = scene_row.get("log_token")
        log_row = nusc_index.get_log(version, str(log_token))
        map_row = nusc_index.get_map_for_log(version, str(log_token))

    scene_description = (
        (scene_row or {}).get("description")
        or metadata_local.get("scene_description")
        or metadata_local.get("description")
    )
    log_token = (scene_row or {}).get("log_token")
    logfile = (log_row or {}).get("logfile")
    location_raw = (log_row or {}).get("location") or metadata_local.get("location") or metadata_local.get("map_name")
    location_human = location_readable(str(location_raw)) if not is_missing(location_raw) else MISSING
    date_captured = (log_row or {}).get("date_captured")
    vehicle = (log_row or {}).get("vehicle")
    map_name_or_location = (
        metadata_local.get("map_name")
        or (map_row or {}).get("filename")
        or location_raw
    )
    first_sample = (scene_row or {}).get("first_sample_token") or metadata_local.get("first_sample_token")
    last_sample = (scene_row or {}).get("last_sample_token") or metadata_local.get("last_sample_token")
    num_samples = (scene_row or {}).get("nbr_samples") or metadata_local.get("nbr_samples")

    crossing_x = to_float_or_none(summary_local.get("crossing_center_x"))
    crossing_y = to_float_or_none(summary_local.get("crossing_center_y"))
    crossing_arc = to_float_or_none(summary_local.get("crossing_zone_arc_length_m"))
    hold_arc = to_float_or_none(summary_local.get("ego_hold_line_arc_length_m"))

    nearest_road_name, nearest_crossing_context = load_osm_context(
        project_root=project_root,
        scene_name=scene_name,
        ref_x=crossing_x,
        ref_y=crossing_y,
    )

    row = {
        "scene_name": scene_name,
        "geometry_type": GEOMETRY_TYPE_BY_SCENE.get(scene_name, MISSING),
        "nuscenes_scene_description": safe_value(scene_description),
        "nuscenes_log_token": safe_value(log_token),
        "nuscenes_logfile": safe_value(logfile),
        "nuscenes_location_raw": safe_value(location_raw),
        "nuscenes_location_readable": safe_value(location_human),
        "nuscenes_date_captured": safe_value(date_captured),
        "nuscenes_vehicle": safe_value(vehicle),
        "map_name_or_location": safe_value(map_name_or_location),
        "first_sample_token": safe_value(first_sample),
        "last_sample_token": safe_value(last_sample),
        "number_of_samples": safe_value(to_int_or_none(num_samples)),
        "refined_crossing_center_x_m": safe_value(crossing_x),
        "refined_crossing_center_y_m": safe_value(crossing_y),
        "crossing_zone_arc_length_m": safe_value(crossing_arc),
        "ego_hold_line_arc_length_m": safe_value(hold_arc),
        "nearest_osm_road_name": safe_value(nearest_road_name),
        "nearest_osm_crossing_context": safe_value(nearest_crossing_context),
        "paper_location_phrase": build_paper_phrase(scene_name, safe_value(location_human)),
    }

    if scene_row is None:
        row["nuscenes_scene_description"] = safe_value(metadata_local.get("scene_description"))
        row["paper_location_phrase"] = f"nuScenes {scene_name}, location not available"

    return row


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scene_location_metadata.csv"
    json_path = out_dir / "scene_location_metadata.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, MISSING) for k in OUTPUT_COLUMNS})

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def print_terminal_summary(rows: List[Dict[str, Any]], nusc_index: NuScenesIndex) -> None:
    print("paper_scene_location_summary:")
    for r in rows:
        scene_name = str(r["scene_name"])
        location = r.get("nuscenes_location_raw", MISSING)
        desc = r.get("nuscenes_scene_description", MISSING)
        geometry = r.get("geometry_type", MISSING)
        found_version, found_scene = nusc_index.find_scene(scene_name)
        if found_scene is None:
            extra = " [scene not found in v1.0-trainval or v1.0-mini]"
        else:
            extra = f" [source={found_version}]"
        print(f"{scene_name} | {geometry} | {location} | description: {desc}{extra}")


def main() -> None:
    args = parse_args()
    project_root = Path(args.dataroot).resolve()
    scene_names = parse_scene_name_list(args.scene_name)

    nusc_index = NuScenesIndex(project_root)

    rows = [build_scene_row(project_root, nusc_index, scene) for scene in scene_names]
    csv_path, json_path = write_outputs(rows, project_root / args.output_dir)
    print_terminal_summary(rows, nusc_index)
    print(f"output_csv: {csv_path}")
    print(f"output_json: {json_path}")


if __name__ == "__main__":
    main()
