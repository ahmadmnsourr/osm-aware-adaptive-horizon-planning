#!/usr/bin/env python3
"""
Generate controlled OSM crosswalk positional-error cases from extracted OSM priors for one or more scenes.

Purpose
-------
This script reads the OSM crossing extracted by 07_extract_osm_context.py and creates
controlled forward/backward longitudinal offsets anchored at the extracted OSM crossing and applied along the refined ego route. These
cases are used later by the OSM-aware planner to test how map-prior positional error
interacts with degraded camera perception.

Important design choice
-----------------------
The simulated crossings are shifted along the ego route/path, not by simply adding
x/y offsets. This keeps the perturbed OSM crossing attached to the road/path shape.

Reads
-----
- data/refined/<scene_name>/ego_timeline.csv
- data/refined/<scene_name>/scene_summary.json
- data/osm/<scene_name>/converted/osm_features_local.csv
- data/osm/<scene_name>/alignment/osm_alignment_summary.json

Writes
------
- data/osm/<scene_name>/error_cases/osm_error_cases.csv
- data/osm/<scene_name>/error_cases/osm_error_cases.json
- data/osm/<scene_name>/error_cases/osm_error_cases_summary.json
- data/osm/<scene_name>/error_cases/osm_error_cases_visualization.png

Default methodology
-------------------
- clean_osm: extracted OSM crossing projected to the ego route.
- shifted cases: offsets are scaled by each scene's crossing length, not fixed arbitrary meters.
  Example: lambda = 0.25 means one quarter of the scenario crossing-zone length.
- Backward and forward cases are generated for each multiplier and shifted along the ego route.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PathReference:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    s_min: float
    s_max: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_float_list_csv(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate OSM crosswalk positional-error cases shifted along the ego route."
    )
    p.add_argument(
        "--scene-name",
        default="all",
        help="Scene to process, short scene number, comma-separated scenes, or 'all'. Default: all three selected scenes.",
    )
    p.add_argument(
        "--scene-names",
        default="scene-1019,scene-0061,scene-0383",
        help="Comma-separated scene list used when --scene-name all."
    )
    p.add_argument("--refined-root", default="data/refined")
    p.add_argument("--osm-root", default="data/osm")
    p.add_argument(
        "--offset-method",
        choices=["crossing_length_multipliers", "fixed_meters"],
        default="crossing_length_multipliers",
        help="Default uses scenario-scaled offsets: offset = lambda * crossing_length_m. Use fixed_meters to reproduce the old 1-5 m sweep.",
    )
    p.add_argument(
        "--error-multipliers",
        default="0.25,0.50,0.75,1.00,1.25,1.50",
        help="Comma-separated lambda values used when --offset-method crossing_length_multipliers.",
    )
    p.add_argument(
        "--offsets-m",
        default="1,2,3,4,5",
        help="Comma-separated absolute shift magnitudes in meters used only when --offset-method fixed_meters.",
    )
    p.add_argument(
        "--use-reference-anchor",
        action="store_true",
        help="Optional fallback/debug mode: anchor offsets at the refined reference crossing instead of the extracted OSM crossing.",
    )
    p.add_argument(
        "--plot-window-m",
        type=float,
        default=45.0,
        help="Half-window around crossing for visualization bounds.",
    )
    p.add_argument(
        "--zoom-margin-m",
        type=float,
        default=10.0,
        help="Margin around reference/OSM/error points for the zoomed visualization.",
    )
    p.add_argument(
        "--plot-dpi",
        type=int,
        default=320,
        help="DPI for saved PNG figures.",
    )
    p.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also save PDF versions of the visualizations.",
    )
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_path_reference(ego_df: pd.DataFrame) -> PathReference:
    df = ego_df.sort_values("frame_index").reset_index(drop=True).copy()
    if "cumulative_distance_m" in df.columns:
        s = pd.to_numeric(df["cumulative_distance_m"], errors="coerce").to_numpy(dtype=float)
    else:
        xy = df[["ego_x", "ego_y"]].to_numpy(dtype=float)
        step = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
        s = np.concatenate([[0.0], np.cumsum(step)])
    x = pd.to_numeric(df["ego_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["ego_y"], errors="coerce").to_numpy(dtype=float)
    if "ego_yaw_rad" in df.columns:
        yaw = np.unwrap(pd.to_numeric(df["ego_yaw_rad"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    else:
        dx = np.gradient(x)
        dy = np.gradient(y)
        yaw = np.unwrap(np.arctan2(dy, dx))

    valid = np.isfinite(s) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        raise ValueError("ego_timeline.csv must contain at least two valid ego path points.")
    s = s[valid]
    x = x[valid]
    y = y[valid]
    yaw = yaw[valid]

    # Ensure monotonically nondecreasing path progress for interpolation.
    order = np.argsort(s)
    s = s[order]
    x = x[order]
    y = y[order]
    yaw = yaw[order]
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    s = s[keep]
    x = x[keep]
    y = y[keep]
    yaw = yaw[keep]
    if len(s) < 2:
        raise ValueError("Ego path progress has fewer than two unique points.")

    return PathReference(s=s, x=x, y=y, yaw=yaw, s_min=float(s[0]), s_max=float(s[-1]))


def interp_path(pref: PathReference, s_query: float) -> Tuple[float, float, float]:
    sq = float(np.clip(float(s_query), pref.s_min, pref.s_max))
    x = float(np.interp(sq, pref.s, pref.x))
    y = float(np.interp(sq, pref.s, pref.y))
    yaw = float(np.interp(sq, pref.s, pref.yaw))
    return x, y, yaw


def nearest_path_s_for_point(pref: PathReference, px: float, py: float) -> Tuple[float, float, int]:
    """
    Project a point to the nearest segment of the ego route and return s, distance, and segment index.
    """
    p = np.array([float(px), float(py)], dtype=float)
    best_dist = float("inf")
    best_s = pref.s_min
    best_seg = 0

    for i in range(len(pref.s) - 1):
        a = np.array([pref.x[i], pref.y[i]], dtype=float)
        b = np.array([pref.x[i + 1], pref.y[i + 1]], dtype=float)
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            t = 0.0
            q = a
        else:
            t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
            q = a + t * ab
        dist = float(np.linalg.norm(p - q))
        if dist < best_dist:
            best_dist = dist
            best_s = float(pref.s[i] + t * (pref.s[i + 1] - pref.s[i]))
            best_seg = i

    return best_s, best_dist, best_seg


def scenario_reference(
    scene_summary: Dict[str, Any],
    pref: PathReference,
    features_df: Optional[pd.DataFrame] = None,
    alignment: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Build the reference crossing used for measuring OSM error.

    Preferred source is the refined crossing center from scene_summary.json.
    For pass-through/no-refinement scenes, this can be missing; in that case,
    we use the route-supported OSM crossing as a safe geometric fallback so
    Script 08 can still generate OSM prior perturbations without modifying the
    refined scene.
    """
    cx_raw = scene_summary.get("crossing_center_x", np.nan)
    cy_raw = scene_summary.get("crossing_center_y", np.nan)
    try:
        cx = float(cx_raw)
        cy = float(cy_raw)
    except Exception:
        cx = float("nan")
        cy = float("nan")

    reference_source = "scene_summary_refined_crossing"

    if not (np.isfinite(cx) and np.isfinite(cy)):
        # First fallback: use the nearest crossing selected by Script 07 alignment if it exists.
        alignment = alignment or {}
        ax = alignment.get("nearest_crossing_local_x", None)
        ay = alignment.get("nearest_crossing_local_y", None)
        try:
            axf = float(ax)
            ayf = float(ay)
        except Exception:
            axf = float("nan")
            ayf = float("nan")
        if np.isfinite(axf) and np.isfinite(ayf):
            cx, cy = axf, ayf
            reference_source = "alignment_nearest_osm_crossing_fallback_no_refined_crossing"

    if not (np.isfinite(cx) and np.isfinite(cy)) and features_df is not None and len(features_df):
        # Second fallback: choose the OSM crossing closest to the ego route.
        crossings = features_df[features_df.get("feature_type", pd.Series(dtype=str)).isin(["crossing_node", "crossing_way"])].copy()
        if len(crossings):
            candidate_rows: List[Tuple[float, float, float]] = []
            for _, row in crossings.iterrows():
                x = float(row.get("local_x", np.nan))
                y = float(row.get("local_y", np.nan))
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                _, d_route, _ = nearest_path_s_for_point(pref, x, y)
                candidate_rows.append((float(d_route), x, y))
            if candidate_rows:
                candidate_rows.sort(key=lambda r: r[0])
                _, cx, cy = candidate_rows[0]
                reference_source = "nearest_osm_crossing_to_route_fallback_no_refined_crossing"

    if not (np.isfinite(cx) and np.isfinite(cy)):
        # Final fallback: route midpoint. This is only to avoid a hard crash when
        # both refined crossing and OSM crossing are unavailable.
        s_mid = float(0.5 * (pref.s_min + pref.s_max))
        cx, cy, _ = interp_path(pref, s_mid)
        reference_source = "route_midpoint_fallback_no_crossing_available"

    s_ref = scene_summary.get("crossing_zone_arc_length_m", None)
    if s_ref is None or not np.isfinite(float(s_ref)):
        s_ref, _, _ = nearest_path_s_for_point(pref, cx, cy)
    else:
        s_ref = float(s_ref)

    crossing_length_raw = scene_summary.get("crossing_length_m", 10.0)
    crossing_width_raw = scene_summary.get("crossing_width_m", 6.5)
    try:
        crossing_length_m = float(crossing_length_raw)
    except Exception:
        crossing_length_m = 10.0
    try:
        crossing_width_m = float(crossing_width_raw)
    except Exception:
        crossing_width_m = 6.5
    if not np.isfinite(crossing_length_m) or crossing_length_m <= 0.0:
        crossing_length_m = 10.0
    if not np.isfinite(crossing_width_m) or crossing_width_m <= 0.0:
        crossing_width_m = 6.5

    return {
        "reference_crossing_x": float(cx),
        "reference_crossing_y": float(cy),
        "reference_crossing_s_m": float(np.clip(float(s_ref), pref.s_min, pref.s_max)),
        "crossing_length_m": crossing_length_m,
        "crossing_width_m": crossing_width_m,
        "reference_source": reference_source,
    }


def get_nearest_osm_crossing(
    features_df: pd.DataFrame,
    alignment: Dict[str, Any],
    ref: Dict[str, float],
    pref: PathReference,
) -> Dict[str, Any]:
    """
    Select the clean OSM crossing anchor.

    Priority:
    1) nearest_crossing_feature_id from alignment summary if present.
    2) nearest crossing feature by distance_to_scenario_crossing_center_m.
    3) reference crossing center if no valid OSM crossing exists.
    """
    default = {
        "clean_source": "reference_fallback_no_osm_crossing",
        "clean_osm_feature_id": "",
        "clean_osm_feature_type": "reference_fallback",
        "clean_osm_crossing_x": float(ref["reference_crossing_x"]),
        "clean_osm_crossing_y": float(ref["reference_crossing_y"]),
        "clean_osm_crossing_s_m": float(ref["reference_crossing_s_m"]),
        "clean_osm_distance_to_reference_xy_m": 0.0,
        "clean_osm_distance_to_route_m": 0.0,
    }

    if len(features_df) == 0 or "feature_type" not in features_df.columns:
        return default
    crossings = features_df[features_df["feature_type"].isin(["crossing_node", "crossing_way"])].copy()
    if len(crossings) == 0:
        return default

    fid = str(alignment.get("nearest_crossing_feature_id", "") or "")
    chosen: Optional[pd.Series] = None
    if fid:
        match = crossings[crossings["feature_id"].astype(str).eq(fid)]
        if len(match):
            chosen = match.iloc[0]

    if chosen is None:
        if "distance_to_scenario_crossing_center_m" in crossings.columns:
            crossings["_d"] = pd.to_numeric(crossings["distance_to_scenario_crossing_center_m"], errors="coerce")
            crossings = crossings.sort_values("_d", na_position="last")
        chosen = crossings.iloc[0]

    x = float(chosen.get("local_x", np.nan))
    y = float(chosen.get("local_y", np.nan))
    if not (np.isfinite(x) and np.isfinite(y)):
        return default

    clean_s, dist_route, _ = nearest_path_s_for_point(pref, x, y)
    dx = x - float(ref["reference_crossing_x"])
    dy = y - float(ref["reference_crossing_y"])
    dist_ref = float(math.hypot(dx, dy))

    return {
        "clean_source": "nearest_osm_crossing",
        "clean_osm_feature_id": str(chosen.get("feature_id", "")),
        "clean_osm_feature_type": str(chosen.get("feature_type", "")),
        "clean_osm_crossing_x": x,
        "clean_osm_crossing_y": y,
        "clean_osm_crossing_s_m": float(clean_s),
        "clean_osm_distance_to_reference_xy_m": dist_ref,
        "clean_osm_distance_to_route_m": float(dist_route),
    }


def _multiplier_label(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p").replace("-", "m")


def build_error_cases(
    pref: PathReference,
    ref: Dict[str, float],
    clean: Dict[str, Any],
    offsets_m: List[float],
    use_reference_anchor: bool,
    offset_method: str = "fixed_meters",
    error_multipliers: Optional[List[float]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    crossing_length = max(float(ref.get("crossing_length_m", 10.0)), 1e-6)

    # Main design: offsets are applied to the extracted OSM crossing anchor, not the reference crossing.
    # The reference crossing is only used afterward to measure the resulting error.
    # If no valid OSM crossing exists, the script falls back to the reference crossing so it can still run.
    if (not use_reference_anchor) and clean["clean_source"] == "nearest_osm_crossing":
        anchor_s = float(clean["clean_osm_crossing_s_m"])
        anchor_x = float(clean["clean_osm_crossing_x"])
        anchor_y = float(clean["clean_osm_crossing_y"])
        anchor_source = "nearest_osm_crossing"
    else:
        anchor_s = float(ref["reference_crossing_s_m"])
        anchor_x = float(ref["reference_crossing_x"])
        anchor_y = float(ref["reference_crossing_y"])
        anchor_source = "reference_crossing_center_fallback_or_debug"

    def add_case(name: str, direction: str, signed_offset: float, multiplier: Optional[float]) -> None:
        s_case = float(np.clip(anchor_s + signed_offset, pref.s_min, pref.s_max))
        x_case, y_case, yaw_case = interp_path(pref, s_case)
        injected_offset_abs = abs(float(signed_offset))
        ref_xy_error = float(math.hypot(x_case - float(ref["reference_crossing_x"]), y_case - float(ref["reference_crossing_y"])))
        ref_s_error = float(s_case - float(ref["reference_crossing_s_m"]))
        total_abs_error_to_reference = abs(ref_s_error)
        normalized = float(min(1.0, total_abs_error_to_reference / crossing_length))
        rows.append(
            {
                "osm_error_case": name,
                "offset_direction": direction,
                "offset_m": float(injected_offset_abs),
                "injected_offset_signed_m": float(signed_offset),
                "injected_offset_abs_m": injected_offset_abs,
                "error_multiplier_lambda": None if multiplier is None else float(multiplier),
                "offset_method": str(offset_method),
                "reference_scale_m": crossing_length,
                "osm_error_signed_m": ref_s_error,
                "osm_error_abs_m": total_abs_error_to_reference,
                "normalized_osm_error": normalized,
                "anchor_source": anchor_source,
                "reference_source": str(ref.get("reference_source", "unknown")),
                "reference_crossing_s_m": float(ref["reference_crossing_s_m"]),
                "reference_crossing_x": float(ref["reference_crossing_x"]),
                "reference_crossing_y": float(ref["reference_crossing_y"]),
                "clean_osm_source": clean["clean_source"],
                "clean_osm_feature_id": clean["clean_osm_feature_id"],
                "clean_osm_feature_type": clean["clean_osm_feature_type"],
                "clean_osm_crossing_s_m": float(clean["clean_osm_crossing_s_m"]),
                "clean_osm_crossing_x": float(clean["clean_osm_crossing_x"]),
                "clean_osm_crossing_y": float(clean["clean_osm_crossing_y"]),
                "clean_osm_distance_to_reference_xy_m": float(clean["clean_osm_distance_to_reference_xy_m"]),
                "clean_osm_distance_to_route_m": float(clean["clean_osm_distance_to_route_m"]),
                "perturbed_crossing_s_m": s_case,
                "perturbed_crossing_x": x_case,
                "perturbed_crossing_y": y_case,
                "perturbed_crossing_yaw_rad": yaw_case,
                "perturbed_crossing_xy_error_to_reference_m": ref_xy_error,
                "perturbed_crossing_s_error_to_reference_m": ref_s_error,
                "crossing_length_m": float(ref["crossing_length_m"]),
                "crossing_width_m": float(ref["crossing_width_m"]),
            }
        )

    add_case("clean_osm", "none", 0.0, 0.0)

    if offset_method == "crossing_length_multipliers":
        lambdas = sorted(set(float(v) for v in (error_multipliers or []) if float(v) > 0.0))
        for lam in lambdas:
            off = float(lam * crossing_length)
            label = _multiplier_label(lam)
            add_case(f"shift_backward_{label}L", "backward", -off, lam)
            add_case(f"shift_forward_{label}L", "forward", off, lam)
    else:
        for off in sorted(set(float(o) for o in offsets_m if float(o) > 0.0)):
            add_case(f"shift_backward_{int(round(off)):02d}m", "backward", -off, None)
            add_case(f"shift_forward_{int(round(off)):02d}m", "forward", off, None)

    return pd.DataFrame(rows)


OSM_LAYER_STYLE: Dict[str, Tuple[str, Optional[str], float, float]] = {
    "crossing_node": ("#e11d48", "o", 62.0, 0.95),
    "crossing_way": ("#f43f5e", None, 1.9, 0.95),
    "traffic_signal": ("#f59e0b", "^", 58.0, 0.95),
    "stop_sign": ("#dc2626", "P", 58.0, 0.95),
    "give_way": ("#7c3aed", "D", 54.0, 0.95),
    "road_centerlike": ("#6b7280", None, 1.1, 0.45),
    "junction_hint": ("#0ea5e9", "s", 42.0, 0.80),
}

OSM_LAYER_LABELS: Dict[str, str] = {
    "crossing_node": "OSM crossing node",
    "crossing_way": "OSM crossing way",
    "traffic_signal": "OSM traffic signal",
    "stop_sign": "OSM stop sign",
    "give_way": "OSM give-way/yield",
    "road_centerlike": "OSM road geometry",
    "junction_hint": "OSM junction hint",
}


def parse_geometry_local(value: Any, fallback_x: float, fallback_y: float) -> List[List[float]]:
    try:
        geom = json.loads(value) if isinstance(value, str) else value
        if isinstance(geom, list) and len(geom):
            return [[float(p[0]), float(p[1])] for p in geom if len(p) >= 2]
    except Exception:
        pass
    if np.isfinite(fallback_x) and np.isfinite(fallback_y):
        return [[float(fallback_x), float(fallback_y)]]
    return []


def plot_osm_layers(ax: plt.Axes, features_df: pd.DataFrame) -> None:
    """Draw all extracted OSM layers in the same spirit as the Script 07 validation plot."""
    if features_df is None or len(features_df) == 0 or "feature_type" not in features_df.columns:
        return

    plotted_labels = set()
    type_order = [
        "road_centerlike",
        "junction_hint",
        "crossing_way",
        "crossing_node",
        "traffic_signal",
        "stop_sign",
        "give_way",
    ]
    present = features_df["feature_type"].astype(str).unique().tolist()
    ordered_types = [t for t in type_order if t in present] + sorted([t for t in present if t not in type_order])

    for ftype in ordered_types:
        layer = features_df[features_df["feature_type"].astype(str).eq(ftype)]
        color, marker, size_or_lw, alpha = OSM_LAYER_STYLE.get(ftype, ("#111827", "o", 42.0, 0.75))
        label_base = OSM_LAYER_LABELS.get(ftype, f"OSM {ftype}")
        for _, r in layer.iterrows():
            geom = parse_geometry_local(
                r.get("geometry_local_json", "[]"),
                float(r.get("local_x", np.nan)),
                float(r.get("local_y", np.nan)),
            )
            if not geom:
                continue
            xs = [p[0] for p in geom]
            ys = [p[1] for p in geom]
            gtype = str(r.get("geometry_type", "point"))
            label = label_base if label_base not in plotted_labels else None
            if label is not None:
                plotted_labels.add(label_base)
            if gtype in ("polyline", "polygon") and len(xs) >= 2:
                ax.plot(xs, ys, "-", color=color, lw=float(size_or_lw), alpha=float(alpha), label=label, zorder=1)
            else:
                ax.scatter(
                    [xs[0]],
                    [ys[0]],
                    c=color,
                    s=float(size_or_lw),
                    marker=marker or "o",
                    edgecolors="black",
                    linewidths=0.35,
                    alpha=float(alpha),
                    label=label,
                    zorder=3,
                )


def add_error_case_markers(ax: plt.Axes, cases_df: pd.DataFrame, ref: Dict[str, float], clean: Dict[str, Any]) -> None:
    cx = float(ref["reference_crossing_x"])
    cy = float(ref["reference_crossing_y"])

    ax.scatter(
        [cx], [cy],
        c="gold", s=210, marker="*", edgecolors="black", linewidths=0.9,
        label="Reference crossing", zorder=8,
    )

    if clean["clean_source"] == "nearest_osm_crossing":
        ax.scatter(
            [float(clean["clean_osm_crossing_x"])],
            [float(clean["clean_osm_crossing_y"])],
            c="black", s=105, marker="o", edgecolors="white", linewidths=0.45,
            label="Extracted OSM crossing", zorder=7,
        )

    clean_case = cases_df[cases_df["osm_error_case"].eq("clean_osm")]
    if len(clean_case):
        r = clean_case.iloc[0]
        ax.scatter(
            [r["perturbed_crossing_x"]], [r["perturbed_crossing_y"]],
            c="lime", s=125, marker="P", edgecolors="black", linewidths=0.65,
            label="Clean OSM prior projected to route", zorder=9,
        )

    forward = cases_df[cases_df["offset_direction"].eq("forward")].sort_values("offset_m")
    backward = cases_df[cases_df["offset_direction"].eq("backward")].sort_values("offset_m")

    if len(forward):
        ax.scatter(
            forward["perturbed_crossing_x"], forward["perturbed_crossing_y"],
            c="#ef4444", s=82, marker=">", edgecolors="black", linewidths=0.25,
            label="Forward-shifted OSM prior", zorder=9,
        )
        for i, (_, r) in enumerate(forward.iterrows()):
            ax.annotate(
                f"+{r['offset_m']:.0f}m",
                (r["perturbed_crossing_x"], r["perturbed_crossing_y"]),
                xytext=(7, -4 - 2 * i), textcoords="offset points",
                fontsize=8.6, color="#7f1d1d", weight="semibold",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.0},
                zorder=10,
            )

    if len(backward):
        ax.scatter(
            backward["perturbed_crossing_x"], backward["perturbed_crossing_y"],
            c="#2563eb", s=82, marker="<", edgecolors="black", linewidths=0.25,
            label="Backward-shifted OSM prior", zorder=9,
        )
        for i, (_, r) in enumerate(backward.iterrows()):
            ax.annotate(
                f"-{r['offset_m']:.0f}m",
                (r["perturbed_crossing_x"], r["perturbed_crossing_y"]),
                xytext=(-31, 3 + 2 * i), textcoords="offset points",
                fontsize=8.6, color="#1e3a8a", weight="semibold",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.0},
                zorder=10,
            )

    ordered = cases_df.sort_values("perturbed_crossing_s_m")
    ax.plot(
        ordered["perturbed_crossing_x"], ordered["perturbed_crossing_y"],
        "--", color="#374151", lw=1.35, alpha=0.85,
        label="Offset sequence along route", zorder=5,
    )


def compute_plot_bounds(
    ego_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    ref: Dict[str, float],
    clean: Dict[str, Any],
    margin_m: float,
    zoom: bool,
) -> Tuple[float, float, float, float]:
    ex = ego_df["ego_x"].to_numpy(dtype=float)
    ey = ego_df["ego_y"].to_numpy(dtype=float)
    xs: List[float] = []
    ys: List[float] = []

    if zoom:
        xs.extend(cases_df["perturbed_crossing_x"].to_numpy(dtype=float).tolist())
        ys.extend(cases_df["perturbed_crossing_y"].to_numpy(dtype=float).tolist())
        xs.append(float(ref["reference_crossing_x"])); ys.append(float(ref["reference_crossing_y"]))
        if clean["clean_source"] == "nearest_osm_crossing":
            xs.append(float(clean["clean_osm_crossing_x"])); ys.append(float(clean["clean_osm_crossing_y"]))
    else:
        smin = float(cases_df["perturbed_crossing_s_m"].min() - margin_m)
        smax = float(cases_df["perturbed_crossing_s_m"].max() + margin_m)
        mask = np.ones(len(ego_df), dtype=bool)
        if "cumulative_distance_m" in ego_df.columns:
            s = pd.to_numeric(ego_df["cumulative_distance_m"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(s) & (s >= smin) & (s <= smax)
            if not mask.any():
                mask = np.ones(len(ego_df), dtype=bool)
        xs.extend(ex[mask].tolist()); ys.extend(ey[mask].tolist())
        xs.extend(cases_df["perturbed_crossing_x"].to_numpy(dtype=float).tolist())
        ys.extend(cases_df["perturbed_crossing_y"].to_numpy(dtype=float).tolist())
        xs.append(float(ref["reference_crossing_x"])); ys.append(float(ref["reference_crossing_y"]))

    xmin, xmax = min(xs) - margin_m, max(xs) + margin_m
    ymin, ymax = min(ys) - margin_m, max(ys) + margin_m
    w = xmax - xmin
    h = ymax - ymin
    if w > h:
        extra = 0.5 * (w - h)
        ymin -= extra; ymax += extra
    elif h > w:
        extra = 0.5 * (h - w)
        xmin -= extra; xmax += extra
    return xmin, xmax, ymin, ymax


def plot_error_cases(
    ego_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    ref: Dict[str, float],
    clean: Dict[str, Any],
    out_path: Path,
    scene_name: str,
    margin_m: float,
    dpi: int,
    zoom: bool,
    save_pdf: bool,
) -> None:
    ex = ego_df["ego_x"].to_numpy(dtype=float)
    ey = ego_df["ego_y"].to_numpy(dtype=float)

    fig_size = (11.0, 9.2) if not zoom else (9.6, 8.8)
    fig, ax = plt.subplots(figsize=fig_size)

    plot_osm_layers(ax, features_df)
    ax.plot(ex, ey, "-", color="navy", lw=2.8, label="Ego route / road shape", zorder=4)
    ax.scatter([ex[0]], [ey[0]], c="green", s=82, marker="s", edgecolors="black", linewidths=0.4, label="Start", zorder=6)
    ax.scatter([ex[-1]], [ey[-1]], c="red", s=92, marker="X", edgecolors="black", linewidths=0.4, label="End", zorder=6)

    add_error_case_markers(ax, cases_df, ref, clean)

    xmin, xmax, ymin, ymax = compute_plot_bounds(ego_df, cases_df, ref, clean, margin_m=margin_m, zoom=zoom)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    title_suffix = "zoomed view" if zoom else "context view"
    ax.set_title(f"OSM crossing positional-error cases with extracted OSM priors ({scene_name}) — {title_suffix}")
    ax.grid(True, alpha=0.25, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    by_label: Dict[str, Any] = {}
    for h, l in zip(handles, labels):
        if l and l not in by_label:
            by_label[l] = h
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="best", fontsize=7.7, ncol=2, framealpha=0.94,
        title="Layers and simulated OSM-error cases", title_fontsize=8.4,
    )

    note = (
        "Offsets are applied in path-progress coordinates from the extracted OSM crossing; "
        "error is measured relative to the refined reference crossing."
    )
    fig.text(0.5, 0.015, note, ha="center", fontsize=8.3, color="#374151")
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
def normalize_scene_name(scene: str) -> str:
    s = str(scene).strip()
    if not s:
        return s
    if s.lower().startswith("scene-"):
        return s
    if s.isdigit():
        return f"scene-{int(s):04d}"
    return s


def resolve_scene_names(args: argparse.Namespace) -> List[str]:
    raw = str(args.scene_name).strip()
    if raw.lower() in {"all", "three", "selected", ""}:
        scenes = [normalize_scene_name(x) for x in str(args.scene_names).split(",") if x.strip()]
    else:
        scenes = [normalize_scene_name(x) for x in raw.split(",") if x.strip()]
    seen = set()
    out: List[str] = []
    for s in scenes:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    if not out:
        raise ValueError("No scenes selected.")
    return out


def process_scene(scene_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    refined_scene_dir = Path(args.refined_root) / scene_name
    osm_scene_dir = Path(args.osm_root) / scene_name
    out_dir = osm_scene_dir / "error_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    ego_path = refined_scene_dir / "ego_timeline.csv"
    summary_path = refined_scene_dir / "scene_summary.json"
    features_path = osm_scene_dir / "converted" / "osm_features_local.csv"
    alignment_path = osm_scene_dir / "alignment" / "osm_alignment_summary.json"

    if not ego_path.is_file():
        raise FileNotFoundError(f"Missing ego timeline: {ego_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing scene summary: {summary_path}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Missing OSM local features: {features_path}. Run 07_extract_osm_context_all_scenes.py first.")

    ego_df = pd.read_csv(ego_path).sort_values("frame_index").reset_index(drop=True)
    scene_summary = load_json(summary_path)
    features_df = pd.read_csv(features_path)
    alignment = load_json(alignment_path)

    pref = build_path_reference(ego_df)
    ref = scenario_reference(scene_summary, pref, features_df=features_df, alignment=alignment)
    clean = get_nearest_osm_crossing(features_df, alignment, ref, pref)

    offsets = parse_float_list_csv(args.offsets_m)
    multipliers = parse_float_list_csv(args.error_multipliers)

    cases_df = build_error_cases(
        pref=pref,
        ref=ref,
        clean=clean,
        offsets_m=offsets,
        use_reference_anchor=bool(args.use_reference_anchor),
        offset_method=str(args.offset_method),
        error_multipliers=multipliers,
    )
    cases_df = cases_df.sort_values(["offset_direction", "offset_m", "osm_error_case"]).reset_index(drop=True)

    csv_path = out_dir / "osm_error_cases.csv"
    json_path = out_dir / "osm_error_cases.json"
    summary_out_path = out_dir / "osm_error_cases_summary.json"
    plot_path = out_dir / "osm_error_cases_visualization.png"
    zoom_plot_path = out_dir / "osm_error_cases_visualization_zoomed.png"

    cases_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(cases_df.to_dict(orient="records"), indent=2, allow_nan=True), encoding="utf-8")

    plot_error_cases(
        ego_df=ego_df,
        features_df=features_df,
        cases_df=cases_df,
        ref=ref,
        clean=clean,
        out_path=plot_path,
        scene_name=scene_name,
        margin_m=float(args.plot_window_m),
        dpi=int(args.plot_dpi),
        zoom=False,
        save_pdf=bool(args.save_pdf),
    )
    plot_error_cases(
        ego_df=ego_df,
        features_df=features_df,
        cases_df=cases_df,
        ref=ref,
        clean=clean,
        out_path=zoom_plot_path,
        scene_name=scene_name,
        margin_m=float(args.zoom_margin_m),
        dpi=int(args.plot_dpi),
        zoom=True,
        save_pdf=bool(args.save_pdf),
    )

    if str(args.offset_method) == "crossing_length_multipliers":
        methodology_note = (
            "Scene-scaled OSM error sweep: offset_m = lambda * crossing_length_m, shifted along the ego route. "
            "This keeps errors comparable across straight, moderate-curvature, and circular/high-curvature scenes."
        )
    else:
        methodology_note = "Fixed-meter OSM error sweep, shifted along the ego route."

    payload = {
        "scene_name": scene_name,
        "generated_at_utc": utc_now_iso(),
        "script": "08_generate_osm_error_cases_all_scenes.py",
        "purpose": "Generate controlled OSM crossing positional offsets along the refined ego route.",
        "offset_method": str(args.offset_method),
        "offsets_m": offsets,
        "error_multipliers_lambda": multipliers if str(args.offset_method) == "crossing_length_multipliers" else [],
        "num_cases": int(len(cases_df)),
        "anchor_policy": "extracted_osm_crossing" if not args.use_reference_anchor else "reference_crossing_debug",
        "reference_crossing": ref,
        "clean_osm_crossing": clean,
        "output_files": {
            "csv": str(csv_path),
            "json": str(json_path),
            "summary": str(summary_out_path),
            "visualization_png": str(plot_path),
            "visualization_zoomed_png": str(zoom_plot_path),
        },
        "methodology": methodology_note,
        "notes": [
            "Offsets are applied in path-progress coordinates, so perturbed crossings follow the ego route shape.",
            "Positive signed error means the OSM prior is shifted forward along the ego route.",
            "Negative signed error means the OSM prior is shifted backward along the ego route.",
            "Offsets are anchored at the extracted OSM crossing when available, not the reference crossing.",
            "osm_error_signed_m is measured relative to the refined reference crossing after applying the offset.",
            "normalized_osm_error = min(1, abs(osm_error_signed_m) / crossing_length_m).",
            "For pass-through/no-refinement scenes with no crossing center, the script uses a route-supported OSM crossing fallback.",
        ],
    }
    summary_out_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    print("osm_error_case_generation_summary:")
    print(f"  scene: {scene_name}")
    print(f"  reference_source: {ref.get('reference_source', 'unknown')}")
    print(f"  clean_source: {clean['clean_source']}")
    if (not args.use_reference_anchor) and clean['clean_source'] == 'nearest_osm_crossing':
        anchor_used = 'nearest_osm_crossing'
    else:
        anchor_used = 'reference_crossing_center_fallback_or_debug'
    print(f"  anchor_used: {anchor_used}")
    print(f"  offset_method: {args.offset_method}")
    if str(args.offset_method) == "crossing_length_multipliers":
        print(f"  crossing_length_m: {float(ref['crossing_length_m']):.3f}")
        print(f"  error_multipliers_lambda: {multipliers}")
    else:
        print(f"  offsets_m: {offsets}")
    print(f"  num_cases: {len(cases_df)}")
    print(f"  output_csv: {csv_path}")
    print(f"  visualization: {plot_path}")
    print(f"  zoomed_visualization: {zoom_plot_path}")
    return {
        "scene_name": scene_name,
        "status": "ok",
        "num_cases": int(len(cases_df)),
        "reference_source": ref.get("reference_source", "unknown"),
        "clean_source": clean["clean_source"],
        "offset_method": str(args.offset_method),
        "output_csv": str(csv_path),
        "visualization_png": str(plot_path),
        "zoomed_visualization_png": str(zoom_plot_path),
    }


def main() -> None:
    args = parse_args()
    scenes = resolve_scene_names(args)
    batch_rows: List[Dict[str, Any]] = []

    print("batch_osm_error_case_generation:")
    print(f"  scenes: {scenes}")
    print(f"  offset_method: {args.offset_method}")

    for scene_name in scenes:
        try:
            row = process_scene(scene_name, args)
        except Exception as exc:
            row = {
                "scene_name": scene_name,
                "status": "failed",
                "error": str(exc),
            }
            print(f"[failed] {scene_name}: {exc}")
        batch_rows.append(row)

    osm_root = Path(args.osm_root)
    osm_root.mkdir(parents=True, exist_ok=True)
    batch_json = osm_root / "all_scenes_osm_error_cases_summary.json"
    batch_csv = osm_root / "all_scenes_osm_error_cases_summary.csv"
    batch_payload = {
        "generated_at_utc": utc_now_iso(),
        "script": "08_generate_osm_error_cases_all_scenes.py",
        "scenes": scenes,
        "offset_method": str(args.offset_method),
        "error_multipliers_lambda": parse_float_list_csv(args.error_multipliers),
        "offsets_m": parse_float_list_csv(args.offsets_m),
        "scene_results": batch_rows,
    }
    batch_json.write_text(json.dumps(batch_payload, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(batch_rows).to_csv(batch_csv, index=False)

    failures = [r for r in batch_rows if r.get("status") != "ok"]
    print("batch_summary:")
    print(f"  ok: {len(batch_rows) - len(failures)}")
    print(f"  failed: {len(failures)}")
    print(f"  batch_json: {batch_json}")
    print(f"  batch_csv: {batch_csv}")
    if failures:
        raise RuntimeError(f"OSM error-case generation failed for: {[r['scene_name'] for r in failures]}")


if __name__ == "__main__":
    main()
