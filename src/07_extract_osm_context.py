#!/usr/bin/env python3
"""
Step 1 OSM extraction + alignment stage.

This script:
1) Loads refined-scene ego trajectory in nuScenes local map coordinates.
2) Converts local (x, y) <-> lat/lon using nuScenes map origin assumptions.
3) Queries OSM around the ego route.
4) Classifies and converts relevant OSM features back to local frame.
5) Compares OSM crossing / traffic-control context against the refined scenario.
6) Saves raw OSM, converted features, alignment summaries, and validation plots.

IMPORTANT:
- No planner is invoked here.
- No map priors are injected into planner in this step.
- No OSM reliability score is computed here.
- This script only extracts OSM context and reports geometric alignment fields needed by later OSM error-case generation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

try:
    from nuscenes.scripts.export_poses import REFERENCE_COORDINATES
except Exception:
    REFERENCE_COORDINATES = {
        "boston-seaport": [42.336849169438615, -71.05785369873047],
        "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
        "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
        "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
    }


@dataclass
class OSMConfig:
    scene_name: str
    refined_root: Path
    output_root: Path
    route_buffer_m: float
    zoom_buffer_m: float
    overwrite: bool

    # Alignment / proximity thresholds used only for geometric diagnostics.
    traffic_control_near_crossing_radius_m: float
    hold_line_support_radius_m: float
    route_crossing_support_radius_m: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract OSM context and assess local-frame geometric alignment.")
    p.add_argument("--scene-name", default="scene-0061")
    p.add_argument("--refined-root", default="data/refined")
    p.add_argument("--output-root", default="data/osm")
    p.add_argument("--route-buffer-m", type=float, default=40.0)
    p.add_argument("--zoom-buffer-m", type=float, default=20.0)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--traffic-control-near-crossing-radius-m", type=float, default=18.0)
    p.add_argument("--hold-line-support-radius-m", type=float, default=12.0)
    p.add_argument("--route-crossing-support-radius-m", type=float, default=12.0)
    return p.parse_args()


def ensure_dirs(scene_out: Path) -> Dict[str, Path]:
    paths = {
        "root": scene_out,
        "query": scene_out / "osm_query",
        "converted": scene_out / "converted",
        "alignment": scene_out / "alignment",
        "validation": scene_out / "validation",
        "logs": scene_out / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def get_scene_inputs(refined_scene_dir: Path) -> Dict[str, Any]:
    ego_df = pd.read_csv(refined_scene_dir / "ego_timeline.csv").sort_values("frame_index").reset_index(drop=True)
    sample_df = pd.read_csv(refined_scene_dir / "sample_timeline.csv").sort_values("frame_index").reset_index(drop=True)
    scene_metadata = json.loads((refined_scene_dir / "scene_metadata.json").read_text(encoding="utf-8"))
    scene_summary = json.loads((refined_scene_dir / "scene_summary.json").read_text(encoding="utf-8"))
    return {
        "ego_df": ego_df,
        "sample_df": sample_df,
        "scene_metadata": scene_metadata,
        "scene_summary": scene_summary,
    }


def meters_to_latlon_deltas(buffer_m: float, center_lat: float) -> Tuple[float, float]:
    dlat = buffer_m / 111_320.0
    dlon = buffer_m / max(1e-6, 111_320.0 * math.cos(math.radians(center_lat)))
    return dlat, dlon


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso_utc(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    t = str(text).strip()
    try:
        if t.endswith("Z"):
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


class LocalLatLonTransform:
    """
    Convert between nuScenes local map XY and WGS84 lat/lon.

    Assumption:
    - local map origin is at map south-west corner
    - map coordinates are meters in Web Mercator space
    """

    def __init__(self, location: str) -> None:
        if location not in REFERENCE_COORDINATES:
            raise KeyError(f"No reference coordinates for location: {location}")
        ref_lat, ref_lon = REFERENCE_COORDINATES[location]
        self.location = location
        self.ref_lat = float(ref_lat)
        self.ref_lon = float(ref_lon)
        self.to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.to_wgs = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        self.ref_merc_x, self.ref_merc_y = self.to_merc.transform(self.ref_lon, self.ref_lat)

    def local_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        mx = self.ref_merc_x + float(x)
        my = self.ref_merc_y + float(y)
        lon, lat = self.to_wgs.transform(mx, my)
        return float(lat), float(lon)

    def latlon_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        mx, my = self.to_merc.transform(float(lon), float(lat))
        x = mx - self.ref_merc_x
        y = my - self.ref_merc_y
        return float(x), float(y)


def build_route_latlon(ego_df: pd.DataFrame, tfm: LocalLatLonTransform) -> pd.DataFrame:
    rows = []
    for _, r in ego_df.iterrows():
        x = float(r["ego_x"])
        y = float(r["ego_y"])
        lat, lon = tfm.local_to_latlon(x, y)
        rows.append(
            {
                "frame_index": int(r["frame_index"]),
                "timestamp_s": float(r["timestamp_s"]),
                "local_x": x,
                "local_y": y,
                "lat": lat,
                "lon": lon,
            }
        )
    return pd.DataFrame(rows)


def build_osm_bbox(route_ll: pd.DataFrame, route_buffer_m: float) -> Dict[str, float]:
    lat_min = float(route_ll["lat"].min())
    lat_max = float(route_ll["lat"].max())
    lon_min = float(route_ll["lon"].min())
    lon_max = float(route_ll["lon"].max())
    center_lat = float(0.5 * (lat_min + lat_max))
    center_lon = float(0.5 * (lon_min + lon_max))
    dlat, dlon = meters_to_latlon_deltas(route_buffer_m, center_lat)
    return {
        "south": lat_min - dlat,
        "west": lon_min - dlon,
        "north": lat_max + dlat,
        "east": lon_max + dlon,
        "center_lat": center_lat,
        "center_lon": center_lon,
    }


def overpass_query(bbox: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    south, west, north, east = bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    q = f"""
[out:json][timeout:60];
(
  node["highway"="crossing"]({south},{west},{north},{east});
  way["highway"="crossing"]({south},{west},{north},{east});
  node["crossing"]({south},{west},{north},{east});
  way["crossing"]({south},{west},{north},{east});
  node["highway"="traffic_signals"]({south},{west},{north},{east});
  node["traffic_signals"]({south},{west},{north},{east});
  node["highway"="stop"]({south},{west},{north},{east});
  node["highway"="give_way"]({south},{west},{north},{east});
  node["traffic_sign"~"stop|give_way|yield",i]({south},{west},{north},{east});
  node["traffic_signals:direction"]({south},{west},{north},{east});
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|service|unclassified|living_street"]({south},{west},{north},{east});
  way["junction"]({south},{west},{north},{east});
);
out body geom;
>;
out skel qt;
""".strip()
    encoded = urllib.parse.urlencode({"data": q}).encode("utf-8")
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter",
    ]
    last_exc: Optional[Exception] = None
    query_started_utc = utc_now_iso()
    query_started_dt = parse_iso_utc(query_started_utc)

    for endpoint in endpoints:
        req = urllib.request.Request(endpoint, data=encoded, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = resp.read().decode("utf-8")
            raw = json.loads(payload)
            query_finished_utc = utc_now_iso()
            query_finished_dt = parse_iso_utc(query_finished_utc)
            duration_s = (
                round((query_finished_dt - query_started_dt).total_seconds(), 3)
                if query_started_dt is not None and query_finished_dt is not None
                else float("nan")
            )
            osm3s = raw.get("osm3s", {}) if isinstance(raw, dict) else {}
            overpass_osm_base_utc = osm3s.get("timestamp_osm_base")
            overpass_areas_base_utc = osm3s.get("timestamp_areas_base")
            overpass_generator = osm3s.get("copyright") or (raw.get("generator") if isinstance(raw, dict) else None)
            response_remark = raw.get("remark") if isinstance(raw, dict) else None
            osm_base_dt = parse_iso_utc(overpass_osm_base_utc)
            freshness_s = (
                round((query_finished_dt - osm_base_dt).total_seconds(), 3)
                if query_finished_dt is not None and osm_base_dt is not None
                else None
            )
            query_meta = {
                "query_started_utc": query_started_utc,
                "query_finished_utc": query_finished_utc,
                "query_duration_s": duration_s,
                "endpoint_used": endpoint,
                "overpass_osm_base_utc": overpass_osm_base_utc,
                "overpass_areas_base_utc": overpass_areas_base_utc,
                "overpass_generator": overpass_generator,
                "response_remark": response_remark,
                "osm_snapshot_age_s_at_query_finish": freshness_s,
                "osm_snapshot_age_h_at_query_finish": round(float(freshness_s) / 3600.0, 6) if freshness_s is not None else None,
            }
            return raw, query_meta
        except Exception as exc:
            last_exc = exc
            try:
                insecure_ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, timeout=120, context=insecure_ctx) as resp:
                    payload = resp.read().decode("utf-8")
                raw = json.loads(payload)
                query_finished_utc = utc_now_iso()
                query_finished_dt = parse_iso_utc(query_finished_utc)
                duration_s = (
                    round((query_finished_dt - query_started_dt).total_seconds(), 3)
                    if query_started_dt is not None and query_finished_dt is not None
                    else float("nan")
                )
                osm3s = raw.get("osm3s", {}) if isinstance(raw, dict) else {}
                overpass_osm_base_utc = osm3s.get("timestamp_osm_base")
                overpass_areas_base_utc = osm3s.get("timestamp_areas_base")
                overpass_generator = osm3s.get("copyright") or (raw.get("generator") if isinstance(raw, dict) else None)
                response_remark = raw.get("remark") if isinstance(raw, dict) else None
                osm_base_dt = parse_iso_utc(overpass_osm_base_utc)
                freshness_s = (
                    round((query_finished_dt - osm_base_dt).total_seconds(), 3)
                    if query_finished_dt is not None and osm_base_dt is not None
                    else None
                )
                query_meta = {
                    "query_started_utc": query_started_utc,
                    "query_finished_utc": query_finished_utc,
                    "query_duration_s": duration_s,
                    "endpoint_used": endpoint,
                    "overpass_ssl_verify_relaxed": True,
                    "overpass_osm_base_utc": overpass_osm_base_utc,
                    "overpass_areas_base_utc": overpass_areas_base_utc,
                    "overpass_generator": overpass_generator,
                    "response_remark": response_remark,
                    "osm_snapshot_age_s_at_query_finish": freshness_s,
                    "osm_snapshot_age_h_at_query_finish": round(float(freshness_s) / 3600.0, 6) if freshness_s is not None else None,
                }
                return raw, query_meta
            except Exception as exc2:
                last_exc = exc2
                continue
    raise RuntimeError(f"All Overpass queries failed: {last_exc}")


def tags_to_json(tags: Optional[Dict[str, Any]]) -> str:
    return json.dumps(tags or {}, ensure_ascii=True, sort_keys=True)


def classify_feature(element: Dict[str, Any]) -> str:
    tags = element.get("tags") or {}
    etype = element.get("type", "")
    highway = str(tags.get("highway", ""))
    crossing = str(tags.get("crossing", ""))
    traffic_sign = str(tags.get("traffic_sign", "")).lower()
    junction = str(tags.get("junction", ""))

    if highway == "crossing" or crossing:
        return "crossing_node" if etype == "node" else "crossing_way"
    if highway == "traffic_signals" or "traffic_signal" in "".join([str(tags.get(k, "")) for k in tags]).lower():
        return "traffic_signal"
    if highway == "stop" or "stop" in traffic_sign:
        return "stop_sign"
    if highway == "give_way" or "yield" in traffic_sign or "give_way" in traffic_sign:
        return "give_way"
    if junction:
        return "junction_hint"
    if etype == "way" and highway:
        return "road_centerlike"
    return "other"


def element_geometry_latlon(element: Dict[str, Any]) -> Tuple[str, List[List[float]], float, float]:
    etype = element.get("type")
    if etype == "node":
        lat = float(element["lat"])
        lon = float(element["lon"])
        return "point", [[lat, lon]], lat, lon
    geom = element.get("geometry") or []
    if not geom:
        center = element.get("center")
        if center:
            lat = float(center["lat"])
            lon = float(center["lon"])
            return "center_point", [[lat, lon]], lat, lon
        return "unknown", [], float("nan"), float("nan")
    coords = [[float(p["lat"]), float(p["lon"])] for p in geom]
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    lat_c = float(np.mean(lats))
    lon_c = float(np.mean(lons))
    if len(coords) >= 4 and np.allclose(coords[0], coords[-1]):
        gtype = "polygon"
    else:
        gtype = "polyline"
    return gtype, coords, lat_c, lon_c


def convert_elements_to_features(
    elements: List[Dict[str, Any]],
    tfm: LocalLatLonTransform,
    scene_name: str,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    rich: List[Dict[str, Any]] = []
    for idx, el in enumerate(elements):
        if el.get("type") not in ("node", "way", "relation"):
            continue
        tags = el.get("tags") or {}
        ftype = classify_feature(el)
        if ftype == "other":
            continue

        gtype, geom_ll, lat_c, lon_c = element_geometry_latlon(el)
        geom_local: List[List[float]] = []
        for lat, lon in geom_ll:
            x, y = tfm.latlon_to_local(lat, lon)
            geom_local.append([x, y])

        if np.isfinite(lat_c) and np.isfinite(lon_c):
            lx, ly = tfm.latlon_to_local(lat_c, lon_c)
        else:
            lx, ly = float("nan"), float("nan")

        fid = f"{scene_name}_{ftype}_{el.get('type','u')}_{el.get('id','x')}_{idx}"
        row = {
            "feature_id": fid,
            "feature_type": ftype,
            "osm_element_type": str(el.get("type", "")),
            "osm_id": str(el.get("id", "")),
            "source_tags_json": tags_to_json(tags),
            "lat": lat_c,
            "lon": lon_c,
            "local_x": lx,
            "local_y": ly,
            "geometry_type": gtype,
            "geometry_latlon_json": json.dumps(geom_ll, ensure_ascii=True),
            "geometry_local_json": json.dumps(geom_local, ensure_ascii=True),
            "scene_name": scene_name,
        }
        rows.append(row)
        rich.append(row | {"source_tags": tags, "geometry_latlon": geom_ll, "geometry_local": geom_local})

    df = pd.DataFrame(rows)
    return df, rich


OSM_LAYER_STYLE: Dict[str, Tuple[str, Optional[str]]] = {
    "crossing_node": ("#e11d48", "o"),
    "crossing_way": ("#f43f5e", None),
    "traffic_signal": ("#f59e0b", "^"),
    "stop_sign": ("#dc2626", "P"),
    "give_way": ("#7c3aed", "D"),
    "road_centerlike": ("#6b7280", None),
    "junction_hint": ("#0ea5e9", "s"),
}
OSM_LAYER_LEGEND: Dict[str, str] = {
    "crossing_node": "Crossing (node)",
    "crossing_way": "Crossing (way)",
    "traffic_signal": "Traffic signals",
    "stop_sign": "Stop",
    "give_way": "Give way / yield",
    "road_centerlike": "Road (centerline)",
    "junction_hint": "Junction hint",
}
OSM_MAP_MAX_ZOOM = 18
LOCAL_VIEW_MARGIN_M = 22.0
MAP_VIEW_MARGIN_M = 30.0


def save_raw_osm(raw: Dict[str, Any], paths: Dict[str, Path]) -> None:
    (paths["query"] / "osm_raw_features.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    elements = raw.get("elements", [])
    nodes = []
    ways = []
    for el in elements:
        if el.get("type") == "node":
            nodes.append(
                {
                    "id": el.get("id"),
                    "lat": el.get("lat"),
                    "lon": el.get("lon"),
                    "tags_json": tags_to_json(el.get("tags")),
                }
            )
        elif el.get("type") == "way":
            ways.append(
                {
                    "id": el.get("id"),
                    "node_count": len(el.get("nodes", []) or []),
                    "geom_count": len(el.get("geometry", []) or []),
                    "tags_json": tags_to_json(el.get("tags")),
                }
            )
    pd.DataFrame(nodes).to_csv(paths["query"] / "osm_raw_nodes.csv", index=False)
    pd.DataFrame(ways).to_csv(paths["query"] / "osm_raw_ways.csv", index=False)


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


def point_to_polyline_distance(px: float, py: float, poly_xy: np.ndarray) -> float:
    if len(poly_xy) == 0:
        return float("inf")
    if len(poly_xy) == 1:
        return float(math.hypot(px - poly_xy[0, 0], py - poly_xy[0, 1]))
    best = float("inf")
    for i in range(len(poly_xy) - 1):
        d = point_to_segment_distance(px, py, poly_xy[i, 0], poly_xy[i, 1], poly_xy[i + 1, 0], poly_xy[i + 1, 1])
        if d < best:
            best = d
    return best


def load_geometry_local(row: pd.Series) -> np.ndarray:
    geom = json.loads(row["geometry_local_json"])
    if not geom:
        x = float(row["local_x"])
        y = float(row["local_y"])
        return np.array([[x, y]], dtype=float)
    return np.array(geom, dtype=float)


def feature_distance_to_point(row: pd.Series, px: float, py: float) -> float:
    geom = load_geometry_local(row)
    gtype = str(row["geometry_type"])
    if gtype in ("polyline", "polygon"):
        return point_to_polyline_distance(px, py, geom)
    return float(np.min(np.sqrt((geom[:, 0] - px) ** 2 + (geom[:, 1] - py) ** 2)))


def build_route_polyline(ego_df: pd.DataFrame) -> np.ndarray:
    return ego_df[["ego_x", "ego_y"]].to_numpy(dtype=float)


def scenario_reference_points(scene_summary: Dict[str, Any], ego_df: pd.DataFrame) -> Dict[str, float]:
    cx = float(scene_summary["crossing_center_x"])
    cy = float(scene_summary["crossing_center_y"])
    hold_arc = float(scene_summary.get("ego_hold_line_arc_length_m", float("nan")))
    hold_x = float("nan")
    hold_y = float("nan")
    if np.isfinite(hold_arc) and "cumulative_distance_m" in ego_df.columns:
        s = ego_df["cumulative_distance_m"].to_numpy(dtype=float)
        ex = ego_df["ego_x"].to_numpy(dtype=float)
        ey = ego_df["ego_y"].to_numpy(dtype=float)
        hold_x = float(np.interp(hold_arc, s, ex))
        hold_y = float(np.interp(hold_arc, s, ey))
    return {
        "crossing_center_x": cx,
        "crossing_center_y": cy,
        "hold_x": hold_x,
        "hold_y": hold_y,
        "hold_arc_m": hold_arc,
    }


def assess_osm_alignment(
    features_df: pd.DataFrame,
    ego_df: pd.DataFrame,
    scene_summary: Dict[str, Any],
    cfg: OSMConfig,
) -> Dict[str, Any]:
    """
    Compute geometric OSM-to-scenario alignment diagnostics only.

    This function intentionally does not compute a reliability score. It only
    reports where the nearest OSM crossing and traffic-control features are
    relative to the refined crossing reference and ego route.
    """
    route_xy = build_route_polyline(ego_df)
    ref = scenario_reference_points(scene_summary, ego_df)
    cx = ref["crossing_center_x"]
    cy = ref["crossing_center_y"]
    hold_x = ref["hold_x"]
    hold_y = ref["hold_y"]

    base_empty = {
        "crossing_candidates_found": 0,
        "nearest_crossing_feature_id": "",
        "nearest_crossing_feature_type": "",
        "nearest_crossing_local_x": None,
        "nearest_crossing_local_y": None,
        "nearest_crossing_distance_to_scenario_center_m": None,
        "nearest_crossing_distance_to_route_m": None,
        "traffic_signal_near_crossing_count": 0,
        "stop_sign_near_crossing_count": 0,
        "give_way_near_crossing_count": 0,
        "traffic_control_near_hold_line_count": 0,
        "missing_crossing_flag": True,
        "missing_traffic_control_flag": True,
        "route_supported_crossing_flag": False,
        "scenario_reference": ref,
        "notes": ["No relevant OSM features returned."],
        "crossing_candidates_ranked": [],
    }

    if len(features_df) == 0:
        return base_empty

    crossings = features_df[features_df["feature_type"].isin(["crossing_node", "crossing_way"])].copy()
    traffic = features_df[features_df["feature_type"].isin(["traffic_signal", "stop_sign", "give_way"])].copy()

    notes: List[str] = []

    nearest_crossing_feature_id = ""
    nearest_crossing_feature_type = ""
    nearest_crossing_local_x: Optional[float] = None
    nearest_crossing_local_y: Optional[float] = None
    nearest_crossing_dist_center = float("inf")
    nearest_crossing_dist_route = float("inf")
    route_supported_crossing_flag = False

    if len(crossings):
        d_center = []
        d_route = []
        representative_x = []
        representative_y = []
        for _, row in crossings.iterrows():
            dc = feature_distance_to_point(row, cx, cy)
            geom = load_geometry_local(row)

            if len(geom) == 1:
                dr = point_to_polyline_distance(geom[0, 0], geom[0, 1], route_xy)
                rx, ry = float(geom[0, 0]), float(geom[0, 1])
            else:
                dr = min(point_to_polyline_distance(p[0], p[1], route_xy) for p in geom)
                # Use the geometry point closest to the refined crossing center as the representative local point.
                dist_to_ref = np.sqrt((geom[:, 0] - cx) ** 2 + (geom[:, 1] - cy) ** 2)
                rep_idx = int(np.argmin(dist_to_ref))
                rx, ry = float(geom[rep_idx, 0]), float(geom[rep_idx, 1])

            d_center.append(dc)
            d_route.append(dr)
            representative_x.append(rx)
            representative_y.append(ry)

        crossings = crossings.copy()
        crossings["distance_to_scenario_crossing_center_m"] = d_center
        crossings["distance_to_ego_route_m"] = d_route
        crossings["representative_local_x"] = representative_x
        crossings["representative_local_y"] = representative_y
        crossings = crossings.sort_values(
            ["distance_to_scenario_crossing_center_m", "distance_to_ego_route_m"]
        ).reset_index(drop=True)

        best = crossings.iloc[0]
        nearest_crossing_feature_id = str(best["feature_id"])
        nearest_crossing_feature_type = str(best["feature_type"])
        nearest_crossing_local_x = float(best["representative_local_x"])
        nearest_crossing_local_y = float(best["representative_local_y"])
        nearest_crossing_dist_center = float(best["distance_to_scenario_crossing_center_m"])
        nearest_crossing_dist_route = float(best["distance_to_ego_route_m"])
        route_supported_crossing_flag = nearest_crossing_dist_route <= cfg.route_crossing_support_radius_m

        if not route_supported_crossing_flag:
            notes.append("Nearest OSM crossing is not tightly supported by the ego route corridor.")
    else:
        notes.append("No crossing features found in queried OSM context.")

    traffic_signal_near_crossing_count = 0
    stop_sign_near_crossing_count = 0
    give_way_near_crossing_count = 0
    traffic_control_near_hold_line_count = 0

    if len(traffic):
        d_cross = []
        d_hold = []
        for _, row in traffic.iterrows():
            d_cross.append(feature_distance_to_point(row, cx, cy))
            if np.isfinite(hold_x) and np.isfinite(hold_y):
                d_hold.append(feature_distance_to_point(row, hold_x, hold_y))
            else:
                d_hold.append(float("inf"))

        traffic = traffic.copy()
        traffic["distance_to_scenario_crossing_center_m"] = d_cross
        traffic["distance_to_hold_line_m"] = d_hold

        near_cross = traffic[traffic["distance_to_scenario_crossing_center_m"] <= cfg.traffic_control_near_crossing_radius_m]
        near_hold = traffic[traffic["distance_to_hold_line_m"] <= cfg.hold_line_support_radius_m]

        traffic_signal_near_crossing_count = int((near_cross["feature_type"] == "traffic_signal").sum())
        stop_sign_near_crossing_count = int((near_cross["feature_type"] == "stop_sign").sum())
        give_way_near_crossing_count = int((near_cross["feature_type"] == "give_way").sum())
        traffic_control_near_hold_line_count = int(len(near_hold))

    missing_crossing_flag = len(crossings) == 0
    missing_traffic_control_flag = (
        traffic_signal_near_crossing_count + stop_sign_near_crossing_count + give_way_near_crossing_count
    ) == 0

    if missing_traffic_control_flag:
        notes.append("No OSM traffic-control feature found near the scenario crossing.")

    crossings_with_alignment = crossings.to_dict(orient="records") if len(crossings) else []

    return {
        "crossing_candidates_found": int(len(crossings)),
        "nearest_crossing_feature_id": nearest_crossing_feature_id,
        "nearest_crossing_feature_type": nearest_crossing_feature_type,
        "nearest_crossing_local_x": nearest_crossing_local_x,
        "nearest_crossing_local_y": nearest_crossing_local_y,
        "nearest_crossing_distance_to_scenario_center_m": None
        if not np.isfinite(nearest_crossing_dist_center)
        else nearest_crossing_dist_center,
        "nearest_crossing_distance_to_route_m": None
        if not np.isfinite(nearest_crossing_dist_route)
        else nearest_crossing_dist_route,
        "traffic_signal_near_crossing_count": int(traffic_signal_near_crossing_count),
        "stop_sign_near_crossing_count": int(stop_sign_near_crossing_count),
        "give_way_near_crossing_count": int(give_way_near_crossing_count),
        "traffic_control_near_hold_line_count": int(traffic_control_near_hold_line_count),
        "missing_crossing_flag": bool(missing_crossing_flag),
        "missing_traffic_control_flag": bool(missing_traffic_control_flag),
        "route_supported_crossing_flag": bool(route_supported_crossing_flag),
        "scenario_reference": ref,
        "notes": notes,
        "crossing_candidates_ranked": crossings_with_alignment,
    }

def enrich_features_with_alignment(features_df: pd.DataFrame, scene_summary: Dict[str, Any], ego_df: pd.DataFrame) -> pd.DataFrame:
    if len(features_df) == 0:
        return features_df.copy()

    route_xy = build_route_polyline(ego_df)
    ref = scenario_reference_points(scene_summary, ego_df)
    cx = ref["crossing_center_x"]
    cy = ref["crossing_center_y"]

    out = features_df.copy()
    d_center = []
    d_route = []
    for _, row in out.iterrows():
        dc = feature_distance_to_point(row, cx, cy)
        geom = load_geometry_local(row)
        if len(geom) == 1:
            dr = point_to_polyline_distance(geom[0, 0], geom[0, 1], route_xy)
        else:
            dr = min(point_to_polyline_distance(p[0], p[1], route_xy) for p in geom)
        d_center.append(dc)
        d_route.append(dr)
    out["distance_to_scenario_crossing_center_m"] = d_center
    out["distance_to_ego_route_m"] = d_route
    return out


def local_xy_to_webmerc(tfm: LocalLatLonTransform, x: float, y: float) -> Tuple[float, float]:
    return tfm.ref_merc_x + float(x), tfm.ref_merc_y + float(y)


def _infer_tile_zoom_mercator_bbox(xmin: float, xmax: float, ymin: float, ymax: float, max_zoom: int) -> int:
    merc_to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lons: List[float] = []
    lats: List[float] = []
    for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)):
        lon, lat = merc_to_ll.transform(float(x), float(y))
        lons.append(lon)
        lats.append(lat)
    w, e = min(lons), max(lons)
    s, n = min(lats), max(lats)
    lon_length = max(abs(e - w), 1e-12)
    lat_length = max(abs(n - s), 1e-12)
    zoom_lon = int(np.ceil(np.log2(360 * 2.0 / lon_length)))
    zoom_lat = int(np.ceil(np.log2(360 * 2.0 / lat_length)))
    z = int(min(zoom_lon, zoom_lat))
    return max(0, min(z, max_zoom))


def plot_validation_local(
    ego_df: pd.DataFrame,
    features_df: pd.DataFrame,
    scene_summary: Dict[str, Any],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ex = ego_df["ego_x"].to_numpy(dtype=float)
    ey = ego_df["ego_y"].to_numpy(dtype=float)
    ax.plot(ex, ey, "-", color="navy", lw=2.2, label="Ego path", zorder=4)
    ax.scatter([ex[0]], [ey[0]], c="green", s=70, marker="s", zorder=5, label="Start")
    ax.scatter([ex[-1]], [ey[-1]], c="red", s=70, marker="X", zorder=5, label="End")

    # scenario crossing center
    cx = float(scene_summary["crossing_center_x"])
    cy = float(scene_summary["crossing_center_y"])
    ax.scatter([cx], [cy], c="yellow", s=110, marker="*", edgecolors="black", linewidths=0.6, zorder=6, label="Scenario crossing center")

    for ftype, g in features_df.groupby("feature_type"):
        color, marker = OSM_LAYER_STYLE.get(ftype, ("black", None))
        labeled = False
        for _, r in g.iterrows():
            geom = json.loads(r["geometry_local_json"])
            if not geom:
                continue
            xs = [p[0] for p in geom]
            ys = [p[1] for p in geom]
            gtype = r["geometry_type"]
            lbl = f"OSM {ftype}" if not labeled else None
            if gtype in ("polyline", "polygon"):
                ax.plot(xs, ys, "-", color=color, lw=1.3, alpha=0.85, label=lbl, zorder=2)
            else:
                ax.scatter([r["local_x"]], [r["local_y"]], c=color, s=45, marker=marker or "o", label=lbl, zorder=3)
            labeled = True

    margin = LOCAL_VIEW_MARGIN_M
    xmin = min(np.min(ex), cx) - margin
    xmax = max(np.max(ex), cx) + margin
    ymin = min(np.min(ey), cy) - margin
    ymax = max(np.max(ey), cy) + margin

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.92)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_validation_map(
    ego_df: pd.DataFrame,
    features_df: pd.DataFrame,
    scene_summary: Dict[str, Any],
    tfm: LocalLatLonTransform,
    out_path: Path,
    title: str,
) -> Optional[str]:
    try:
        ctx = importlib.import_module("contextily")
    except ImportError:
        return "contextily not installed (pip install contextily)"

    fig, ax = plt.subplots(figsize=(11, 11))
    ex = ego_df["ego_x"].to_numpy(dtype=float)
    ey = ego_df["ego_y"].to_numpy(dtype=float)
    ex_m = np.array([local_xy_to_webmerc(tfm, x, y)[0] for x, y in zip(ex, ey)])
    ey_m = np.array([local_xy_to_webmerc(tfm, x, y)[1] for x, y in zip(ex, ey)])

    cx = float(scene_summary["crossing_center_x"])
    cy = float(scene_summary["crossing_center_y"])
    cx_m, cy_m = local_xy_to_webmerc(tfm, cx, cy)

    margin = MAP_VIEW_MARGIN_M
    xmin = min(np.min(ex), cx) - margin
    xmax = max(np.max(ex), cx) + margin
    ymin = min(np.min(ey), cy) - margin
    ymax = max(np.max(ey), cy) + margin

    x0_m, y0_m = local_xy_to_webmerc(tfm, xmin, ymin)
    x1_m, y1_m = local_xy_to_webmerc(tfm, xmax, ymax)
    left_m, right_m = min(x0_m, x1_m), max(x0_m, x1_m)
    bottom_m, top_m = min(y0_m, y1_m), max(y0_m, y1_m)

    ax.set_xlim(left_m, right_m)
    ax.set_ylim(bottom_m, top_m)
    ax.set_aspect("equal", adjustable="box")

    map_zoom = _infer_tile_zoom_mercator_bbox(left_m, right_m, bottom_m, top_m, OSM_MAP_MAX_ZOOM)

    basemap_note: Optional[str] = None
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom=map_zoom,
            attribution_size=6,
            attribution="© OpenStreetMap contributors",
            zorder=1,
        )
    except Exception as exc:
        try:
            ctx.add_basemap(
                ax,
                crs="EPSG:3857",
                source=ctx.providers.CartoDB.Positron,
                zoom=map_zoom,
                attribution_size=6,
                zorder=1,
            )
            basemap_note = f"OSM Mapnik tiles unavailable; used CartoDB light basemap ({exc})"
        except Exception as exc2:
            return f"basemap failed: {exc}; fallback failed: {exc2}"

    ax.plot(ex_m, ey_m, "-", color="cyan", lw=2.8, solid_capstyle="round", zorder=6, label="Ego path")
    ax.scatter([ex_m[0]], [ey_m[0]], c="lime", s=85, marker="s", edgecolors="black", linewidths=0.6, zorder=7, label="Start")
    ax.scatter([ex_m[-1]], [ey_m[-1]], c="red", s=95, marker="X", edgecolors="black", linewidths=0.5, zorder=7, label="End")
    ax.scatter([cx_m], [cy_m], c="yellow", s=120, marker="*", edgecolors="black", linewidths=0.7, zorder=8, label="Scenario crossing center")

    present_types = set(features_df["feature_type"].astype(str).unique()) if len(features_df) else set()
    ftype_order = [k for k in OSM_LAYER_LEGEND if k in present_types]
    ftype_order += sorted(present_types - set(ftype_order))

    for ftype in ftype_order:
        g = features_df[features_df["feature_type"].eq(ftype)]
        color, marker = OSM_LAYER_STYLE.get(ftype, ("#ffffff", "o"))
        leg = OSM_LAYER_LEGEND.get(ftype, str(ftype))
        labeled = False
        for _, r in g.iterrows():
            geom = json.loads(r["geometry_local_json"])
            lbl = leg if not labeled else None
            if not geom:
                continue
            xs_m = []
            ys_m = []
            for lx, ly in geom:
                mx, my = local_xy_to_webmerc(tfm, lx, ly)
                xs_m.append(mx)
                ys_m.append(my)
            gtype = str(r["geometry_type"])
            if gtype in ("polyline", "polygon"):
                ax.plot(xs_m, ys_m, "-", color=color, lw=2.0, alpha=0.95, zorder=5, label=lbl)
            else:
                mx, my = local_xy_to_webmerc(tfm, float(r["local_x"]), float(r["local_y"]))
                ax.scatter([mx], [my], c=color, s=55, marker=marker or "o", edgecolors="black", linewidths=0.35, zorder=6, label=lbl)
            if lbl is not None:
                labeled = True

    ax.set_axis_off()
    ax.set_title(
        title,
        fontsize=12,
        pad=12,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 4},
    )
    fig.patch.set_facecolor("#1a1a1a")
    ax.legend(
        loc="upper left",
        fontsize=7,
        ncol=2,
        framealpha=0.92,
        facecolor="white",
        edgecolor="gray",
        title="Layers (OSM + ego + scenario)",
        title_fontsize=8,
    )
    fig.text(
        0.5,
        0.02,
        f"Basemap: OpenStreetMap Mapnik · z≤{OSM_MAP_MAX_ZOOM} · CRS EPSG:3857",
        ha="center",
        fontsize=6,
        color="lightgray",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.patch.get_facecolor())
    plt.close(fig)
    return basemap_note


def main() -> None:
    args = parse_args()
    cfg = OSMConfig(
        scene_name=args.scene_name,
        refined_root=Path(args.refined_root),
        output_root=Path(args.output_root),
        route_buffer_m=float(args.route_buffer_m),
        zoom_buffer_m=float(args.zoom_buffer_m),
        overwrite=bool(args.overwrite),
        traffic_control_near_crossing_radius_m=float(args.traffic_control_near_crossing_radius_m),
        hold_line_support_radius_m=float(args.hold_line_support_radius_m),
        route_crossing_support_radius_m=float(args.route_crossing_support_radius_m),
    )

    refined_scene_dir = cfg.refined_root / cfg.scene_name
    scene_out = cfg.output_root / cfg.scene_name
    paths = ensure_dirs(scene_out)

    data = get_scene_inputs(refined_scene_dir)
    ego_df = data["ego_df"]
    scene_metadata = data["scene_metadata"]
    scene_summary = data["scene_summary"]

    location = str(scene_metadata.get("location") or scene_metadata.get("map_name") or "")
    if location not in REFERENCE_COORDINATES:
        raise KeyError(f"Scene location '{location}' not in known nuScenes map reference coordinates.")

    tfm = LocalLatLonTransform(location)
    route_ll = build_route_latlon(ego_df, tfm)
    bbox = build_osm_bbox(route_ll, cfg.route_buffer_m)

    (paths["query"] / "osm_query_bbox.json").write_text(json.dumps(bbox, indent=2), encoding="utf-8")
    (paths["query"] / "osm_query_center.json").write_text(
        json.dumps({"lat": bbox["center_lat"], "lon": bbox["center_lon"], "scene_name": cfg.scene_name}, indent=2),
        encoding="utf-8",
    )

    raw: Dict[str, Any]
    query_meta: Dict[str, Any]
    query_error: Optional[str] = None
    try:
        raw, query_meta = overpass_query(bbox)
    except Exception as exc:
        raw = {"elements": [], "error": str(exc)}
        now_utc = utc_now_iso()
        query_meta = {
            "query_started_utc": now_utc,
            "query_finished_utc": now_utc,
            "query_duration_s": 0.0,
            "endpoint_used": None,
            "overpass_osm_base_utc": None,
            "overpass_areas_base_utc": None,
            "overpass_generator": None,
            "response_remark": None,
            "osm_snapshot_age_s_at_query_finish": None,
            "osm_snapshot_age_h_at_query_finish": None,
        }
        query_error = str(exc)

    save_raw_osm(raw, paths)

    elements = raw.get("elements", []) or []
    features_df, features_rich = convert_elements_to_features(elements, tfm, cfg.scene_name)

    if len(features_df) == 0:
        features_df = pd.DataFrame(
            columns=[
                "feature_id", "feature_type", "osm_element_type", "osm_id", "source_tags_json",
                "lat", "lon", "local_x", "local_y", "geometry_type",
                "geometry_latlon_json", "geometry_local_json", "scene_name",
            ]
        )
        features_rich = []

    features_df = enrich_features_with_alignment(features_df, scene_summary, ego_df)

    # Save converted outputs
    features_df.to_csv(paths["converted"] / "osm_features_latlon.csv", index=False)
    features_df.to_csv(paths["converted"] / "osm_features_local.csv", index=False)
    (paths["converted"] / "osm_features_latlon.json").write_text(
        json.dumps(features_rich, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (paths["converted"] / "osm_features_local.json").write_text(
        json.dumps(features_rich, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    alignment = assess_osm_alignment(features_df, ego_df, scene_summary, cfg)
    (paths["alignment"] / "osm_alignment_summary.json").write_text(
        json.dumps(alignment, indent=2, allow_nan=True), encoding="utf-8"
    )


    plot_validation_local(
        ego_df=ego_df,
        features_df=features_df,
        scene_summary=scene_summary,
        out_path=paths["validation"] / "osm_validation_local.png",
        title=f"OSM validation local frame ({cfg.scene_name})",
    )

    map_note = plot_validation_map(
        ego_df=ego_df,
        features_df=features_df,
        scene_summary=scene_summary,
        tfm=tfm,
        out_path=paths["validation"] / "osm_validation_map.png",
        title=f"OSM map view ({cfg.scene_name})",
    )

    route_len = float(ego_df["cumulative_distance_m"].iloc[-1] - ego_df["cumulative_distance_m"].iloc[0]) if len(ego_df) else 0.0
    feature_counts = features_df["feature_type"].value_counts().to_dict()
    for key in ["crossing_node", "crossing_way", "traffic_signal", "stop_sign", "give_way", "road_centerlike", "junction_hint"]:
        feature_counts.setdefault(key, 0)
    (paths["logs"] / "osm_feature_counts.json").write_text(json.dumps(feature_counts, indent=2), encoding="utf-8")

    conversion_ok = bool(
        len(features_df) == 0
        or (np.isfinite(features_df["local_x"]).all() and np.isfinite(features_df["local_y"]).all())
    )

    summary = {
        "scene_name": cfg.scene_name,
        "location": location,
        "reference_coordinate_latlon": [tfm.ref_lat, tfm.ref_lon],
        "route_length_m": route_len,
        "route_buffer_m": cfg.route_buffer_m,
        "zoom_buffer_m": cfg.zoom_buffer_m,
        "query_bbox": bbox,
        "query_center": {"lat": bbox["center_lat"], "lon": bbox["center_lon"]},
        "query_metadata": query_meta,
        "raw_osm_element_count": int(len(elements)),
        "relevant_feature_count": int(len(features_df)),
        "feature_types_found": sorted([k for k, v in feature_counts.items() if int(v) > 0]),
        "local_frame_conversion_succeeded": conversion_ok,
        "query_error": query_error,
        "alignment_summary_file": str(paths["alignment"] / "osm_alignment_summary.json"),
        "map_validation": {
            "local_png": str(paths["validation"] / "osm_validation_local.png"),
            "osm_map_png": str(paths["validation"] / "osm_validation_map.png"),
            "note": map_note,
        },
        "output_dir": str(scene_out),
    }
    (paths["logs"] / "osm_extraction_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )

    print("osm_extraction_summary:")
    print(f"  scene: {cfg.scene_name}")
    print(f"  location: {location}")
    if query_error:
        print(f"  query_error: {query_error}")
        print("  Note: No OSM data was returned. Public Overpass servers often time out; retry later if needed.")
    print(f"  raw_osm_elements: {len(elements)}")
    print(f"  relevant_features: {len(features_df)}")
    print(f"  conversion_ok: {conversion_ok}")
    print("  feature_counts:")
    for k in sorted(feature_counts.keys()):
        print(f"    - {k}: {feature_counts[k]}")
    print("  alignment:")
    print(f"    - nearest_crossing_distance_to_scenario_center_m: {alignment['nearest_crossing_distance_to_scenario_center_m']}")
    print(f"    - nearest_crossing_distance_to_route_m: {alignment['nearest_crossing_distance_to_route_m']}")
    print(f"    - missing_crossing_flag: {alignment['missing_crossing_flag']}")
    print(f"    - missing_traffic_control_flag: {alignment['missing_traffic_control_flag']}")
    print(f"    - route_supported_crossing_flag: {alignment['route_supported_crossing_flag']}")
    print(f"  output_dir: {scene_out}")


if __name__ == "__main__":
    main()