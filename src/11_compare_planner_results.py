#!/usr/bin/env python3
"""
Final paper-level comparison script for the 3-geometry OSM-aware planner study.

Goal
----
Generate a compact set of contribution-focused figures, not a large plot archive.
The main story is:

    OSM error + camera degradation  ->  learned/adaptive prediction horizon
                                    ->  safety change
                                    ->  performance/efficiency change

Final tested scenes
-------------------
- scene-1019 : Straight approach
- scene-0061 : Moderate-curvature approach
- scene-0383 : High-curvature / circular approach

Required inputs
---------------
For each scene:
    data/planner_perception_only/<scene>/per_run_metrics.csv
    data/planner_osm_aware/<scene>/per_run_metrics.csv

Optional inputs
---------------
    data/osm/<scene>/error_cases/*.png
    data/learned_horizon_policy/<scene>/adaptive_horizon_policy.json

Main outputs
------------
    data/planner_comparison_paper/plots/main/
        01_scene-1019_straight_approach_horizon_safety_performance_heatmap.png
        02_scene-0061_moderate_curvature_horizon_safety_performance_heatmap.png
        03_scene-0383_high_curvature_circular_horizon_safety_performance_heatmap.png
        05_horizon_bar_points_across_geometries.png
        06_planner_robustness_by_degradation.png

    data/planner_comparison_paper/plots/osm_error_cases/
        copied OSM error case plots from Script 08, if available

    data/planner_comparison_paper/tables/
        all_paired_key_metrics.csv
        geometry_summary_key_metrics.csv
        geometry_uncertainty_horizon_summary.csv
        figure_index.csv

    data/planner_comparison_paper/reports/
        recommended_figures_and_captions.md
        comparison_summary.json

Typical usage
-------------
    python 11_compare_planner_results_geometry_heatmap_story.py

If your roots are not under data/:
    python 11_compare_planner_results_geometry_heatmap_story.py \
        --baseline-root planner_perception_only \
        --osm-root planner_osm_aware \
        --learned-policy-root learned_horizon_policy
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-9


@dataclass(frozen=True)
class SceneSpec:
    scene: str
    geometry: str
    short: str
    order: int


DEFAULT_SCENES: List[SceneSpec] = [
    SceneSpec("scene-1019", "Straight approach", "straight_approach", 1),
    SceneSpec("scene-0061", "Moderate-curvature approach", "moderate_curvature", 2),
    SceneSpec("scene-0383", "High-curvature / circular approach", "high_curvature_circular", 3),
]

# Metrics used in the compact final story.
SAFETY_PREFERRED = [
    # Primary paper safety signal: TTC is continuous and avoids the misleading
    # case where stopping near a pedestrian increases the number of unsafe-distance
    # frames even though the temporal collision margin improves.
    "min_ped_ttc_s",
    "critical_ped_ttc_frames",
    "ped_conflict_zone_violation_frames",
    "unsafe_frames_total",
]
PERFORMANCE_PREFERRED = [
    "effective_travel_time_s",
    "travel_time_s",
    "stop_time_s",
    "average_speed_mps",
    "final_progress_m",
]
COMFORT_PREFERRED = [
    "comfort_jerk_rms_mps3",
    "max_braking_mps2",
]
HORIZON_METRIC = "mean_adaptive_ped_horizon_s"

HIGHER_IS_BETTER = {
    "min_ped_ttc_s",
    "post_encroachment_time_s",
    "min_ped_distance_m",
    "min_relevant_ped_distance_m",
    "average_speed_mps",
    "final_progress_m",
}
LOWER_IS_BETTER = {
    "effective_travel_time_s",
    "unsafe_frames_total",
    "unsafe_ped_frames",
    "unsafe_vehicle_frames",
    "ped_conflict_zone_violation_frames",
    "critical_ped_ttc_frames",
    "critical_ped_ttc_episodes",
    "travel_time_s",
    "stop_time_s",
    "comfort_jerk_rms_mps3",
    "max_braking_mps2",
    "hard_brake_frames",
    "mean_braking_mps2_during_brake",
}
METRIC_LABELS = {
    "effective_travel_time_s": "Effective travel-time reduction (%)",
    "mean_adaptive_ped_horizon_s": "Adaptive pedestrian horizon (s)",
    "min_adaptive_ped_horizon_s": "Min adaptive pedestrian horizon (s)",
    "max_adaptive_ped_horizon_s": "Max adaptive pedestrian horizon (s)",
    "mean_adaptive_vehicle_horizon_s": "Adaptive vehicle horizon (s)",
    "unsafe_frames_total": "Unsafe-frame reduction (frames)",
    "ped_conflict_zone_violation_frames": "Conflict-zone violation reduction (frames)",
    "critical_ped_ttc_frames": "Critical TTC-frame reduction (frames)",
    "min_ped_ttc_s": "Min pedestrian TTC improvement (%)",
    "travel_time_s": "Travel-time reduction (%)",
    "stop_time_s": "Stop-time reduction (%)",
    "average_speed_mps": "Average-speed improvement (%)",
    "final_progress_m": "Final-progress improvement (m)",
    "comfort_jerk_rms_mps3": "Jerk RMS reduction (%)",
    "max_braking_mps2": "Max-braking reduction (%)",
}

OSM_ERROR_PLOT_CANDIDATES = [
    "osm_error_cases_visualization_zoomed.png",
    "osm_error_cases_visualization.png",
    "osm_error_cases_map.png",
    "osm_error_cases.png",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create final compact geometry-based comparison plots for the paper.")
    p.add_argument("--baseline-root", default="data/planner_perception_only")
    p.add_argument("--osm-root", default="data/planner_osm_aware")
    p.add_argument("--learned-policy-root", default="data/learned_horizon_policy")
    p.add_argument("--osm-context-root", default="data/osm")
    p.add_argument("--output-root", default="data/planner_comparison_paper")
    p.add_argument("--scene-name", default="all", help="all, or comma-separated scene names such as scene-1019,scene-0061")
    p.add_argument("--focus-mode", default="burst_dropout", help="Mode used for the per-geometry 3-panel heatmaps. Falls back automatically if missing.")
    p.add_argument("--osm-error-axis", default="signed", choices=["signed", "absolute"])
    p.add_argument("--degradation-levels", default="10,30,50,70,90")
    p.add_argument("--dpi", type=int, default=240)
    p.add_argument("--no-copy-osm-plots", action="store_true", help="Do not copy Script 08 OSM error case plots.")
    return p.parse_args()


def normalize_scene_name(s: str) -> str:
    s = str(s).strip()
    if not s:
        return s
    return s if s.startswith("scene-") else f"scene-{s}"


def parse_scene_specs(text: str) -> List[SceneSpec]:
    if str(text).strip().lower() == "all":
        return list(DEFAULT_SCENES)
    wanted = [normalize_scene_name(x) for x in str(text).split(",") if x.strip()]
    lookup = {s.scene: s for s in DEFAULT_SCENES}
    out: List[SceneSpec] = []
    for i, scene in enumerate(wanted, start=1):
        if scene in lookup:
            out.append(lookup[scene])
        else:
            out.append(SceneSpec(scene, scene, scene.replace("scene-", "scene_"), i))
    return out


def parse_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (10.5, 6.2),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
    })


def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "root": root,
        "plots": root / "plots",
        "main": root / "plots" / "main",
        "osm_error_cases": root / "plots" / "osm_error_cases",
        "tables": root / "tables",
        "reports": root / "reports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def finite_or_nan(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def safe_mean(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def safe_std(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    return float(np.nanstd(arr, ddof=1)) if np.isfinite(arr).sum() > 1 else float("nan")


def metric_direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    return "neutral"


def first_available(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def x_axis_col(axis_mode: str) -> str:
    return "osm_error_abs_m" if axis_mode == "absolute" else "osm_error_signed_m"


def pretty_float_label(v: Any) -> str:
    f = finite_or_nan(v)
    if not np.isfinite(f):
        return "n/a"
    if abs(f) < 0.005:
        f = 0.0
    return f"{f:.2f}".rstrip("0").rstrip(".")


def pretty_signed_meter_label(v: Any) -> str:
    f = finite_or_nan(v)
    if not np.isfinite(f):
        return "n/a"
    if abs(f) < 0.005:
        return "0"
    return f"{f:+.0f}"


def build_osm_error_axis_label_map(df: pd.DataFrame, axis_mode: str) -> Dict[float, str]:
    xcol = x_axis_col(axis_mode)
    if df is None or len(df) == 0 or xcol not in df.columns:
        return {}
    work = df.copy()
    work[xcol] = pd.to_numeric(work[xcol], errors="coerce")
    if "injected_offset_signed_m" not in work.columns:
        work["injected_offset_signed_m"] = np.nan
    work["injected_offset_signed_m"] = pd.to_numeric(work["injected_offset_signed_m"], errors="coerce")
    labels: Dict[float, str] = {}
    for xval, g in work[np.isfinite(work[xcol])].groupby(xcol, dropna=False):
        x = float(xval)
        inj_vals = sorted({float(v) for v in g["injected_offset_signed_m"].dropna().to_numpy(dtype=float) if np.isfinite(v)})
        if len(inj_vals) == 1:
            first = pretty_signed_meter_label(inj_vals[0])
        elif len(inj_vals) > 1:
            first = "/".join(pretty_signed_meter_label(v) for v in inj_vals[:3])
            if len(inj_vals) > 3:
                first += "..."
        else:
            first = "OSM"
        labels[x] = f"{first}\n{pretty_float_label(x)}"
    return labels


def lookup_axis_label(label_map: Dict[float, str], x: Any) -> str:
    xv = finite_or_nan(x)
    for k, label in label_map.items():
        if np.isfinite(k) and abs(float(k) - xv) <= 1e-6:
            return label
    return pretty_float_label(xv)


def remove_parenthesized_text(text: str) -> str:
    s = str(text)
    out_chars: List[str] = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            continue
        if ch == ")" and depth > 0:
            depth -= 1
            continue
        if depth == 0:
            out_chars.append(ch)
    cleaned = "".join(out_chars)
    # Collapse repeated spaces created by removals while preserving new lines.
    return "\n".join(" ".join(line.split()) for line in cleaned.split("\n")).strip()


def load_scene_metrics(scene: SceneSpec, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = Path(args.baseline_root) / scene.scene / "per_run_metrics.csv"
    osm_path = Path(args.osm_root) / scene.scene / "per_run_metrics.csv"
    missing = [str(p) for p in [baseline_path, osm_path] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n  " + "\n  ".join(missing))
    baseline = pd.read_csv(baseline_path)
    osm = pd.read_csv(osm_path)
    for df_name, df, req in [
        ("baseline", baseline, ["degradation_mode", "degradation_level_pct", "seed"]),
        ("osm", osm, ["osm_error_case", "degradation_mode", "degradation_level_pct", "seed", "normalized_osm_error"]),
    ]:
        missing_cols = [c for c in req if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{df_name} {scene.scene} is missing columns: {missing_cols}")
    if "osm_error_signed_m" not in osm.columns:
        osm["osm_error_signed_m"] = osm.get("osm_error_abs_m", np.nan)
    if "osm_error_abs_m" not in osm.columns:
        osm["osm_error_abs_m"] = pd.to_numeric(osm["osm_error_signed_m"], errors="coerce").abs()
    if "injected_offset_signed_m" not in osm.columns:
        osm["injected_offset_signed_m"] = np.nan
    if "offset_direction" not in osm.columns:
        osm["offset_direction"] = "unknown"

    expected = parse_int_csv(args.degradation_levels)
    if expected:
        baseline = baseline[baseline["degradation_level_pct"].isin(expected)].copy()
        osm = osm[osm["degradation_level_pct"].isin(expected)].copy()
    baseline["scene_name"] = scene.scene
    baseline["geometry_type"] = scene.geometry
    osm["scene_name"] = scene.scene
    osm["geometry_type"] = scene.geometry
    baseline = enrich_with_final_progress(scene=scene, df=baseline, root=Path(args.baseline_root), is_osm=False)
    osm = enrich_with_final_progress(scene=scene, df=osm, root=Path(args.osm_root), is_osm=True)
    baseline, osm = enrich_with_effective_travel_time(
        scene=scene,
        baseline_df=baseline,
        osm_df=osm,
        baseline_root=Path(args.baseline_root),
        osm_root=Path(args.osm_root),
    )
    return baseline, osm


def read_final_progress_from_rollout(path: Path) -> float:
    if not path.is_file():
        return float("nan")
    try:
        df = pd.read_csv(path)
    except Exception:
        return float("nan")
    for col in ["s_progress_m", "cumulative_distance_m"]:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(arr):
                return float(arr.iloc[-1])
    return float("nan")


def read_rollout_df(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def detect_progress_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["s_progress_m", "cumulative_distance_m"]:
        if c in df.columns:
            return c
    return None


def rollout_mean_speed(df: pd.DataFrame) -> float:
    if "speed_mps" in df.columns:
        s = pd.to_numeric(df["speed_mps"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().any():
            v = float(np.nanmean(s.to_numpy(dtype=float)))
            if np.isfinite(v) and v > 0.1:
                return v
    col = detect_progress_column(df)
    if col is None or "timestamp_s" not in df.columns:
        return float("nan")
    p = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(df["timestamp_s"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(p) & np.isfinite(t)
    if valid.sum() < 2:
        return float("nan")
    p = p[valid]
    t = t[valid]
    dt = np.diff(t)
    dp = np.diff(p)
    keep = dt > 1e-6
    if not keep.any():
        return float("nan")
    v = np.nanmean(dp[keep] / dt[keep])
    return float(v) if np.isfinite(v) and v > 0.1 else float("nan")


def estimate_effective_travel_time_s(df: pd.DataFrame, target_s: float, nominal_speed_mps: float) -> float:
    if not np.isfinite(target_s):
        return float("nan")
    pcol = detect_progress_column(df)
    if pcol is None or "timestamp_s" not in df.columns:
        return float("nan")
    p = pd.to_numeric(df[pcol], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(df["timestamp_s"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(p) & np.isfinite(t)
    if valid.sum() < 2:
        return float("nan")
    p = p[valid]
    t = t[valid]
    t0 = float(t[0])
    tr = t - t0
    hit = np.where(p >= float(target_s))[0]
    if len(hit):
        i = int(hit[0])
        if i == 0:
            return float(max(0.0, tr[0]))
        p0, p1 = float(p[i - 1]), float(p[i])
        t0r, t1r = float(tr[i - 1]), float(tr[i])
        if p1 <= p0 + EPS:
            return float(max(0.0, t1r))
        alpha = float(np.clip((float(target_s) - p0) / (p1 - p0), 0.0, 1.0))
        return float(max(0.0, t0r + alpha * (t1r - t0r)))
    last_t = float(tr[-1])
    last_p = float(p[-1])
    remaining = max(0.0, float(target_s) - last_p)
    nom = max(0.1, float(nominal_speed_mps))
    return float(max(0.0, last_t + remaining / nom))


def resolve_scene_target_progress(scene: SceneSpec, baseline_root: Path, osm_root: Path) -> Tuple[float, float]:
    refined_scene = Path("data/refined") / scene.scene
    crossing_arc = float("nan")
    route_len = float("nan")
    summary_path = refined_scene / "scene_summary.json"
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            crossing_arc = finite_or_nan(payload.get("crossing_zone_arc_length_m", np.nan))
            route_len = finite_or_nan(payload.get("ego_total_distance_m", np.nan))
        except Exception:
            pass
    ego_path = refined_scene / "ego_timeline.csv"
    if (not np.isfinite(route_len)) and ego_path.is_file():
        try:
            ego = pd.read_csv(ego_path)
            if "cumulative_distance_m" in ego.columns:
                route_len = finite_or_nan(pd.to_numeric(ego["cumulative_distance_m"], errors="coerce").max())
        except Exception:
            pass
    if not np.isfinite(route_len):
        candidates = [
            baseline_root / scene.scene / "runs",
            osm_root / scene.scene / "runs",
        ]
        for base in candidates:
            if not base.exists():
                continue
            try:
                for p in base.rglob("ego_rollout.csv"):
                    v = read_final_progress_from_rollout(p)
                    if np.isfinite(v):
                        route_len = max(route_len, v) if np.isfinite(route_len) else v
                        break
            except Exception:
                pass
            if np.isfinite(route_len):
                break

    if np.isfinite(crossing_arc):
        target_s = float(crossing_arc + 15.0)
    elif np.isfinite(route_len):
        target_s = float(0.95 * route_len)
    else:
        target_s = float("nan")

    if np.isfinite(route_len) and np.isfinite(target_s):
        target_s = float(np.clip(target_s, 0.0, max(0.0, route_len - 1e-3)))
    return target_s, route_len


def enrich_with_effective_travel_time(
    scene: SceneSpec,
    baseline_df: pd.DataFrame,
    osm_df: pd.DataFrame,
    baseline_root: Path,
    osm_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    b = baseline_df.copy()
    o = osm_df.copy()
    target_s, _route_len = resolve_scene_target_progress(scene, baseline_root, osm_root)

    # Nominal speed from baseline reference rollout when available.
    nominal_speed = float("nan")
    if len(b):
        r0 = b.iloc[0]
        path0 = (
            baseline_root
            / scene.scene
            / "runs"
            / str(r0.get("degradation_mode", ""))
            / f"degradation_{int(r0.get('degradation_level_pct', 0))}"
            / f"seed_{int(r0.get('seed', 0)):03d}"
            / "ego_rollout.csv"
        )
        d0 = read_rollout_df(path0)
        if d0 is not None:
            nominal_speed = rollout_mean_speed(d0)
    if not np.isfinite(nominal_speed):
        nominal_speed = 5.0

    b_times: List[float] = []
    for _, row in b.iterrows():
        run_path = (
            baseline_root
            / scene.scene
            / "runs"
            / str(row.get("degradation_mode", ""))
            / f"degradation_{int(row.get('degradation_level_pct', 0))}"
            / f"seed_{int(row.get('seed', 0)):03d}"
            / "ego_rollout.csv"
        )
        d = read_rollout_df(run_path)
        b_times.append(estimate_effective_travel_time_s(d, target_s, nominal_speed) if d is not None else float("nan"))
    b["effective_travel_time_s"] = b_times

    o_times: List[float] = []
    for _, row in o.iterrows():
        run_path = (
            osm_root
            / scene.scene
            / "runs"
            / str(row.get("osm_error_case", ""))
            / str(row.get("degradation_mode", ""))
            / f"degradation_{int(row.get('degradation_level_pct', 0))}"
            / f"seed_{int(row.get('seed', 0)):03d}"
            / "ego_rollout.csv"
        )
        d = read_rollout_df(run_path)
        o_times.append(estimate_effective_travel_time_s(d, target_s, nominal_speed) if d is not None else float("nan"))
    o["effective_travel_time_s"] = o_times
    return b, o


def enrich_with_final_progress(scene: SceneSpec, df: pd.DataFrame, root: Path, is_osm: bool) -> pd.DataFrame:
    out = df.copy()
    if "final_progress_m" in out.columns:
        vals = pd.to_numeric(out["final_progress_m"], errors="coerce")
        if vals.notna().any():
            return out
    keys = ["degradation_mode", "degradation_level_pct", "seed"]
    if is_osm:
        keys = ["osm_error_case", "degradation_mode", "degradation_level_pct", "seed"]
    missing = [k for k in keys if k not in out.columns]
    if missing:
        out["final_progress_m"] = np.nan
        return out
    cache: Dict[Tuple[Any, ...], float] = {}
    values: List[float] = []
    for _, row in out.iterrows():
        if is_osm:
            key = (
                str(row.get("osm_error_case", "")),
                str(row.get("degradation_mode", "")),
                int(row.get("degradation_level_pct", 0)),
                int(row.get("seed", 0)),
            )
            run_dir = (
                root
                / scene.scene
                / "runs"
                / key[0]
                / key[1]
                / f"degradation_{key[2]}"
                / f"seed_{key[3]:03d}"
            )
        else:
            key = (
                str(row.get("degradation_mode", "")),
                int(row.get("degradation_level_pct", 0)),
                int(row.get("seed", 0)),
            )
            run_dir = (
                root
                / scene.scene
                / "runs"
                / key[0]
                / f"degradation_{key[1]}"
                / f"seed_{key[2]:03d}"
            )
        if key not in cache:
            cache[key] = read_final_progress_from_rollout(run_dir / "ego_rollout.csv")
        values.append(cache[key])
    out["final_progress_m"] = values
    return out


def choose_focus_mode(osm: pd.DataFrame, preferred: str) -> str:
    modes = sorted(osm["degradation_mode"].astype(str).unique().tolist())
    if preferred in modes:
        return preferred
    if modes:
        return modes[0]
    return preferred


def numeric_metrics_for_pairing(baseline: pd.DataFrame, osm: pd.DataFrame) -> List[str]:
    candidates = list(dict.fromkeys(SAFETY_PREFERRED + PERFORMANCE_PREFERRED + COMFORT_PREFERRED + [
        "min_ped_distance_m", "min_relevant_ped_distance_m", "post_encroachment_time_s", "final_progress_m",
    ]))
    out = []
    for m in candidates:
        if m in baseline.columns and m in osm.columns:
            b = pd.to_numeric(baseline[m], errors="coerce")
            o = pd.to_numeric(osm[m], errors="coerce")
            if b.notna().any() and o.notna().any():
                out.append(m)
    return out


def build_paired_rows(scene: SceneSpec, baseline: pd.DataFrame, osm: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    b_id = ["degradation_mode", "degradation_level_pct", "seed"]
    o_id = [
        "osm_error_case", "offset_direction", "injected_offset_signed_m", "osm_error_signed_m", "osm_error_abs_m",
        "normalized_osm_error", "degradation_mode", "degradation_level_pct", "seed",
    ]
    b = baseline[b_id + metrics].copy()
    o = osm[[c for c in o_id if c in osm.columns] + metrics].copy()
    for m in metrics:
        b[m] = pd.to_numeric(b[m], errors="coerce")
        o[m] = pd.to_numeric(o[m], errors="coerce")
    b = b.groupby(b_id, dropna=False).mean(numeric_only=True).reset_index()
    o = o.groupby([c for c in o_id if c in o.columns], dropna=False).mean(numeric_only=True).reset_index()
    merged = o.merge(b, on=b_id, how="inner", suffixes=("_osm", "_baseline"))
    rows: List[Dict[str, Any]] = []
    for _, r in merged.iterrows():
        base_row: Dict[str, Any] = {
            "scene_name": scene.scene,
            "geometry_type": scene.geometry,
            "geometry_short": scene.short,
            "geometry_order": scene.order,
            "osm_error_case": str(r.get("osm_error_case", "")),
            "offset_direction": str(r.get("offset_direction", "")),
            "injected_offset_signed_m": finite_or_nan(r.get("injected_offset_signed_m", np.nan)),
            "osm_error_signed_m": finite_or_nan(r.get("osm_error_signed_m", np.nan)),
            "osm_error_abs_m": finite_or_nan(r.get("osm_error_abs_m", np.nan)),
            "normalized_osm_error": finite_or_nan(r.get("normalized_osm_error", np.nan)),
            "degradation_mode": str(r.get("degradation_mode", "")),
            "degradation_level_pct": int(r.get("degradation_level_pct", 0)),
            "seed": int(r.get("seed", 0)),
        }
        for m in metrics:
            bval = finite_or_nan(r.get(f"{m}_baseline", np.nan))
            oval = finite_or_nan(r.get(f"{m}_osm", np.nan))
            raw = oval - bval if np.isfinite(bval) and np.isfinite(oval) else float("nan")
            direction = metric_direction(m)
            if direction == "higher_is_better":
                signed_abs = raw
                signed_pct = 100.0 * raw / abs(bval) if np.isfinite(bval) and abs(bval) > EPS else float("nan")
            elif direction == "lower_is_better":
                signed_abs = -raw
                signed_pct = 100.0 * (-raw) / abs(bval) if np.isfinite(bval) and abs(bval) > EPS else float("nan")
            else:
                signed_abs = float("nan")
                signed_pct = float("nan")
            row = dict(base_row)
            row.update({
                "metric_name": m,
                "baseline_value": bval,
                "osm_value": oval,
                "raw_difference_osm_minus_baseline": raw,
                "signed_improvement_absolute": signed_abs,
                "signed_improvement_percent": signed_pct,
                "direction_of_improvement": direction,
                "osm_helped_flag": bool(np.isfinite(signed_abs) and signed_abs > 0.0),
            })
            row["baseline_effective_travel_time_s"] = bval if m == "effective_travel_time_s" else float("nan")
            row["osm_effective_travel_time_s"] = oval if m == "effective_travel_time_s" else float("nan")
            row["effective_travel_time_improvement_s"] = signed_abs if m == "effective_travel_time_s" else float("nan")
            row["effective_travel_time_reduction_pct"] = signed_pct if m == "effective_travel_time_s" else float("nan")
            row["travel_time_metric_source"] = "effective_route_completion_time" if m == "effective_travel_time_s" else ""
            rows.append(row)
    return pd.DataFrame(rows)


def compute_osm_condition_summary(scene: SceneSpec, osm: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "scene_name", "geometry_type", "osm_error_case", "offset_direction", "injected_offset_signed_m",
        "osm_error_signed_m", "osm_error_abs_m", "normalized_osm_error", "degradation_mode", "degradation_level_pct",
    ]
    group_cols = [c for c in group_cols if c in osm.columns]
    metrics = [m for m in [
        HORIZON_METRIC, "min_adaptive_ped_horizon_s", "max_adaptive_ped_horizon_s",
        "mean_adaptive_vehicle_horizon_s", "mean_crossing_proximity_factor", "mean_osm_context_weight",
        "min_ped_ttc_s", "unsafe_frames_total", "comfort_jerk_rms_mps3", "travel_time_s", "stop_time_s",
    ] if m in osm.columns]
    if not metrics:
        return pd.DataFrame()
    work = osm.copy()
    for m in metrics:
        work[m] = pd.to_numeric(work[m], errors="coerce")
    agg = {f"{m}_mean": (m, safe_mean) for m in metrics}
    agg.update({f"{m}_std": (m, safe_std) for m in metrics})
    out = work.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    out["geometry_short"] = scene.short
    out["geometry_order"] = scene.order
    return out


def pivot_for_heatmap(df: pd.DataFrame, value_col: str, x_col: str, y_col: str = "degradation_level_pct") -> pd.DataFrame:
    if df is None or len(df) == 0 or value_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d[np.isfinite(d[value_col]) & np.isfinite(d[x_col]) & np.isfinite(d[y_col])].copy()
    if len(d) == 0:
        return pd.DataFrame()
    return d.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc=safe_mean).sort_index(axis=0).sort_index(axis=1)


def draw_heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    title: str,
    cmap: str,
    center_zero: bool,
    x_label_map: Dict[float, str],
    cbar_label: str,
    fig: plt.Figure,
) -> None:
    if pivot.empty:
        ax.axis("off")
        ax.set_title(title + "\n(no data)", fontsize=11)
        return
    arr = pivot.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        ax.axis("off")
        ax.set_title(title + "\n(no finite data)", fontsize=11)
        return
    if center_zero:
        vmax = max(1.0, float(np.nanpercentile(np.abs(finite), 95)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if abs(vmax - vmin) < EPS:
            pad = max(1.0, abs(vmax) * 0.05)
            vmin -= pad
            vmax += pad
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    xlabels = [lookup_axis_label(x_label_map, x) for x in pivot.columns]
    # Keep the lower row of x-axis labels readable by removing parenthesized text.
    xlabels = [remove_parenthesized_text(lbl) for lbl in xlabels]
    ax.set_xticklabels(xlabels, rotation=0, fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([pretty_float_label(y) for y in pivot.index])
    ax.set_xlabel("Injected OSM shift\nactual error (m)", fontsize=9)
    ax.set_ylabel("Camera degradation (%)", fontsize=9)
    ax.set_title(title, fontsize=12)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=8)


def choose_safety_metric(paired: pd.DataFrame) -> str:
    available = sorted(set(paired["metric_name"].astype(str))) if len(paired) else []
    # Required order:
    # 1) min_ped_ttc_s if finite for enough rows
    # 2) critical_ped_ttc_frames
    # 3) ped_conflict_zone_violation_frames
    # 4) unsafe_frames_total
    if "min_ped_ttc_s" in available:
        d = paired[paired["metric_name"].eq("min_ped_ttc_s")].copy()
        ttc_base = pd.to_numeric(d["baseline_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ttc_osm = pd.to_numeric(d["osm_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = ttc_base.notna() & ttc_osm.notna()
        finite_count = int(finite.sum())
        finite_ratio = float(finite.mean()) if len(finite) else 0.0
        if finite_count >= 3 and finite_ratio >= 0.35:
            return "min_ped_ttc_s"
    for m in ["critical_ped_ttc_frames", "ped_conflict_zone_violation_frames", "unsafe_frames_total"]:
        if m in available:
            return m
    for m in SAFETY_PREFERRED:
        if m in available:
            return m
    return available[0] if available else ""


def scene_has_weak_safety_stress_test(paired_scene: pd.DataFrame) -> bool:
    if len(paired_scene) == 0:
        return True
    unsafe = paired_scene[paired_scene["metric_name"].eq("unsafe_frames_total")].copy()
    ttc = paired_scene[paired_scene["metric_name"].eq("min_ped_ttc_s")].copy()
    if len(unsafe) == 0 or len(ttc) == 0:
        return True
    baseline_unsafe = pd.to_numeric(unsafe["baseline_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    ttc_base = pd.to_numeric(ttc["baseline_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    ttc_osm = pd.to_numeric(ttc["osm_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean_unsafe = float(np.nanmean(baseline_unsafe.to_numpy(dtype=float))) if baseline_unsafe.notna().any() else 0.0
    ttc_finite_ratio = float(((ttc_base.notna()) & (ttc_osm.notna())).mean()) if len(ttc) else 0.0
    return bool(mean_unsafe <= 1.0 and ttc_finite_ratio < 0.35)


def choose_performance_metric(paired: pd.DataFrame) -> str:
    available = sorted(set(paired["metric_name"].astype(str))) if len(paired) else []
    if not available:
        return ""
    if "effective_travel_time_s" in available:
        return "effective_travel_time_s"
    if "travel_time_s" in available:
        return "travel_time_s"
    return available[0]


def metric_varies(paired: pd.DataFrame, metric: str, eps: float = 1e-6, fixed_threshold: float = 0.95) -> bool:
    d = paired[paired["metric_name"].eq(metric)].copy()
    if len(d) == 0:
        return False
    diff = (
        pd.to_numeric(d["osm_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        - pd.to_numeric(d["baseline_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    )
    diff = diff[np.isfinite(diff.to_numpy(dtype=float))]
    if len(diff) == 0:
        return False
    fixed_ratio = float((np.abs(diff.to_numpy(dtype=float)) < eps).mean())
    return bool(fixed_ratio <= fixed_threshold)


def is_travel_time_fixed_horizon(paired: pd.DataFrame, eps: float = 1e-6, fixed_threshold: float = 0.95) -> bool:
    d = paired[paired["metric_name"].eq("travel_time_s")].copy()
    if len(d) == 0:
        return False
    diff = (
        pd.to_numeric(d["osm_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        - pd.to_numeric(d["baseline_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    )
    diff = diff[np.isfinite(diff.to_numpy(dtype=float))]
    if len(diff) == 0:
        return False
    fixed_ratio = float((np.abs(diff.to_numpy(dtype=float)) < eps).mean())
    return bool(fixed_ratio > fixed_threshold)


def performance_panel_title(metric: str) -> str:
    if metric == "effective_travel_time_s":
        return "Effective travel-time reduction (%)"
    return "Effective travel-time reduction (%)" if metric == "travel_time_s" else METRIC_LABELS.get(metric, metric)


def paired_metric_value_for_heatmap(paired: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, str, str, bool]:
    """Return dataframe/value column/label/center_zero for one paired metric."""
    d = paired[paired["metric_name"].eq(metric)].copy()
    if len(d) == 0:
        return d, "", metric, True

    if metric == "effective_travel_time_s":
        d["heatmap_value"] = pd.to_numeric(d["signed_improvement_percent"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return d, "heatmap_value", METRIC_LABELS.get(metric, "Effective travel-time reduction (%)"), True

    if metric == "final_progress_m":
        d["heatmap_value"] = pd.to_numeric(d["signed_improvement_absolute"], errors="coerce")
        return d, "heatmap_value", METRIC_LABELS.get(metric, "Final-progress improvement (m)"), True

    # For count-based safety metrics, absolute frame reduction is clearer than a percent
    # when the baseline value is near zero.
    if metric in {"unsafe_frames_total", "ped_conflict_zone_violation_frames", "critical_ped_ttc_frames"}:
        d["heatmap_value"] = pd.to_numeric(d["signed_improvement_absolute"], errors="coerce")
        label = METRIC_LABELS.get(metric, f"{metric} improvement")
        return d, "heatmap_value", label, True

    # For time/performance metrics, percent improvement is usually more comparable.
    pct = pd.to_numeric(d["signed_improvement_percent"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if pct.notna().any():
        d["heatmap_value"] = pct
        label = METRIC_LABELS.get(metric, f"{metric} improvement (%)")
        return d, "heatmap_value", label, True

    d["heatmap_value"] = pd.to_numeric(d["signed_improvement_absolute"], errors="coerce")
    if metric_direction(metric) == "higher_is_better":
        unit = "absolute increase"
    else:
        unit = "absolute reduction"
    label = f"{metric} {unit}"
    return d, "heatmap_value", label, True


def save_scene_three_panel_heatmap(
    scene: SceneSpec,
    osm_summary: pd.DataFrame,
    paired: pd.DataFrame,
    focus_mode: str,
    axis_mode: str,
    out_path: Path,
    dpi: int,
) -> Dict[str, Any]:
    mode = choose_focus_mode(osm_summary, focus_mode) if len(osm_summary) else focus_mode
    xcol = x_axis_col(axis_mode)
    osm_m = osm_summary[osm_summary["degradation_mode"].astype(str).eq(mode)].copy()
    paired_m = paired[paired["degradation_mode"].astype(str).eq(mode)].copy()
    x_label_map = build_osm_error_axis_label_map(osm_m if len(osm_m) else paired_m, axis_mode)

    h_col = f"{HORIZON_METRIC}_mean" if f"{HORIZON_METRIC}_mean" in osm_m.columns else HORIZON_METRIC
    h_pivot = pivot_for_heatmap(osm_m, h_col, xcol)

    safety_metric = choose_safety_metric(paired_m)
    perf_metric = choose_performance_metric(paired_m)
    travel_fixed = is_travel_time_fixed_horizon(paired_m)
    safety_df, safety_col, safety_label, safety_center = paired_metric_value_for_heatmap(paired_m, safety_metric)
    perf_df, perf_col, perf_label, perf_center = paired_metric_value_for_heatmap(paired_m, perf_metric)
    safety_pivot = pivot_for_heatmap(safety_df, safety_col, xcol) if safety_col else pd.DataFrame()
    perf_pivot = pivot_for_heatmap(perf_df, perf_col, xcol) if perf_col else pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(20.5, 5.9), constrained_layout=True)
    draw_heatmap(
        axes[0], h_pivot,
        "Learned/adaptive\npedestrian horizon",
        cmap="viridis", center_zero=False, x_label_map=x_label_map,
        cbar_label="seconds", fig=fig,
    )
    draw_heatmap(
        axes[1], safety_pivot,
        "Safety outcome\n" + safety_label,
        cmap="RdYlGn", center_zero=safety_center, x_label_map=x_label_map,
        cbar_label="positive = OSM-aware better", fig=fig,
    )
    draw_heatmap(
        axes[2], perf_pivot,
        performance_panel_title(perf_metric),
        cmap="RdYlGn", center_zero=perf_center, x_label_map=x_label_map,
        cbar_label="positive = OSM-aware better", fig=fig,
    )
    fig.suptitle(f"{scene.geometry}: horizon adaptation, safety, and performance ({mode})", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "scene_name": scene.scene,
        "geometry_type": scene.geometry,
        "plot_file": str(out_path),
        "focus_mode_used": mode,
        "safety_metric_used": safety_metric,
        "performance_metric_used": perf_metric,
        "travel_time_metric_is_fixed_horizon": bool(travel_fixed),
    }


def build_comfort_tradeoff_summary(scene_specs: List[SceneSpec], paired_all: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scene in scene_specs:
        d = paired_all[paired_all["scene_name"].eq(scene.scene)].copy()
        if len(d) == 0:
            continue
        selected = choose_performance_metric(d)
        travel_fixed = is_travel_time_fixed_horizon(d)
        perf_d = d[d["metric_name"].eq(selected)].copy()
        perf_mean = safe_mean(perf_d["signed_improvement_percent"]) if selected not in {"final_progress_m"} else safe_mean(perf_d["signed_improvement_absolute"])
        eff_d = d[d["metric_name"].eq("effective_travel_time_s")].copy()
        b_eff = safe_mean(eff_d["baseline_value"]) if len(eff_d) else float("nan")
        o_eff = safe_mean(eff_d["osm_value"]) if len(eff_d) else float("nan")
        i_eff = safe_mean(eff_d["signed_improvement_absolute"]) if len(eff_d) else float("nan")
        r_eff = safe_mean(eff_d["signed_improvement_percent"]) if len(eff_d) else float("nan")
        stop_d = d[d["metric_name"].eq("stop_time_s")].copy()
        spd_d = d[d["metric_name"].eq("average_speed_mps")].copy()
        prog_d = d[d["metric_name"].eq("final_progress_m")].copy()
        jerk_d = d[d["metric_name"].eq("comfort_jerk_rms_mps3")].copy()
        brake_d = d[d["metric_name"].eq("max_braking_mps2")].copy()
        rows.append(
            {
                "scene_name": scene.scene,
                "geometry_type": scene.geometry,
                "travel_time_metric_is_fixed_horizon": bool(travel_fixed),
                "selected_performance_metric": selected,
                "performance_improvement_mean": perf_mean,
                "baseline_effective_travel_time_s": b_eff,
                "osm_effective_travel_time_s": o_eff,
                "effective_travel_time_improvement_s": i_eff,
                "effective_travel_time_reduction_pct": r_eff,
                "travel_time_metric_source": "effective_route_completion_time",
                "stop_time_change_mean": safe_mean(stop_d["raw_difference_osm_minus_baseline"]) if len(stop_d) else float("nan"),
                "average_speed_change_mean": safe_mean(spd_d["raw_difference_osm_minus_baseline"]) if len(spd_d) else float("nan"),
                "final_progress_change_mean": safe_mean(prog_d["raw_difference_osm_minus_baseline"]) if len(prog_d) else float("nan"),
                "comfort_jerk_improvement_mean": safe_mean(jerk_d["signed_improvement_percent"]) if len(jerk_d) else float("nan"),
                "max_braking_improvement_mean": safe_mean(brake_d["signed_improvement_percent"]) if len(brake_d) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def save_safety_behavior_tradeoff_plot(summary_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    if len(summary_df) == 0:
        return
    d = summary_df.copy()
    x = np.arange(len(d))
    safety = pd.to_numeric(d["performance_improvement_mean"], errors="coerce").to_numpy(dtype=float)
    jerk = pd.to_numeric(d["comfort_jerk_improvement_mean"], errors="coerce").to_numpy(dtype=float)
    brake = pd.to_numeric(d["max_braking_improvement_mean"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(x, safety, marker="o", label="Selected behavior/perf improvement")
    ax.plot(x, jerk, marker="s", label="Jerk improvement")
    ax.plot(x, brake, marker="^", label="Max-braking improvement")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in d["geometry_type"].tolist()], rotation=0)
    ax.set_ylabel("Improvement (signed, % unless noted)")
    ax.set_title("Safety-behavior tradeoff by geometry")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def uncertainty_bin(v: float) -> str:
    if not np.isfinite(v):
        return "Unknown"
    if v < 0.34:
        return "Low"
    if v < 0.67:
        return "Medium"
    return "High"


def build_geometry_uncertainty_horizon_summary(all_osm: pd.DataFrame) -> pd.DataFrame:
    if len(all_osm) == 0 or HORIZON_METRIC not in all_osm.columns:
        return pd.DataFrame()
    d = all_osm.copy()
    d["degradation_fraction"] = pd.to_numeric(d["degradation_level_pct"], errors="coerce") / 100.0
    d["normalized_osm_error"] = pd.to_numeric(d["normalized_osm_error"], errors="coerce").clip(0.0, 1.0)
    d["combined_uncertainty_index"] = 0.5 * d["degradation_fraction"] + 0.5 * d["normalized_osm_error"]
    d["uncertainty_bin"] = d["combined_uncertainty_index"].map(uncertainty_bin)
    d[HORIZON_METRIC] = pd.to_numeric(d[HORIZON_METRIC], errors="coerce")
    out = (
        d.groupby(["geometry_order", "geometry_type", "uncertainty_bin"], dropna=False)
        .agg(
            mean_adaptive_ped_horizon_s=(HORIZON_METRIC, safe_mean),
            std_adaptive_ped_horizon_s=(HORIZON_METRIC, safe_std),
            mean_combined_uncertainty=("combined_uncertainty_index", safe_mean),
            num_runs=(HORIZON_METRIC, "count"),
        )
        .reset_index()
    )
    bin_order = {"Low": 1, "Medium": 2, "High": 3, "Unknown": 4}
    out["uncertainty_bin_order"] = out["uncertainty_bin"].map(bin_order).fillna(99).astype(int)
    return out.sort_values(["geometry_order", "uncertainty_bin_order"]).reset_index(drop=True)


def save_geometry_vs_uncertainty_heatmap(summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    if len(summary) == 0:
        return
    work = summary.copy()
    pivot = work.pivot_table(
        index="geometry_type",
        columns="uncertainty_bin",
        values="mean_adaptive_ped_horizon_s",
        aggfunc=safe_mean,
    )
    wanted_cols = [c for c in ["Low", "Medium", "High"] if c in pivot.columns]
    pivot = pivot.reindex(columns=wanted_cols)
    # preserve scene order
    geom_order = work[["geometry_type", "geometry_order"]].drop_duplicates().sort_values("geometry_order")["geometry_type"].tolist()
    pivot = pivot.reindex(index=[g for g in geom_order if g in pivot.index])
    if pivot.empty:
        return
    arr = pivot.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if abs(vmax - vmin) < EPS:
        vmin -= 1.0
        vmax += 1.0
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Combined uncertainty bin\n(camera degradation + OSM positional error)")
    ax.set_ylabel("Ego approach geometry")
    ax.set_title("Geometry-dependent learned/adaptive horizon behavior")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}s", ha="center", va="center", fontsize=10, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Mean pedestrian prediction horizon (s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def save_horizon_bar_points_across_geometries(summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    """
    Paper-level compact horizon summary.

    Bars show the overall mean pedestrian prediction horizon per geometry.
    Points show the uncertainty-bin means (Low / Medium / High) for the same geometry.
    This keeps the plot focused on the contribution mechanism: geometry-dependent
    horizon adaptation.
    """
    if summary is None or len(summary) == 0:
        return
    req = {"geometry_order", "geometry_type", "uncertainty_bin", "mean_adaptive_ped_horizon_s", "num_runs"}
    if not req.issubset(set(summary.columns)):
        return
    d = summary.copy()
    d["mean_adaptive_ped_horizon_s"] = pd.to_numeric(d["mean_adaptive_ped_horizon_s"], errors="coerce")
    d["num_runs"] = pd.to_numeric(d["num_runs"], errors="coerce").fillna(1.0).clip(lower=1.0)
    d = d[np.isfinite(d["mean_adaptive_ped_horizon_s"].to_numpy(dtype=float))].copy()
    if len(d) == 0:
        return

    geom_order = (
        d[["geometry_order", "geometry_type"]]
        .drop_duplicates()
        .sort_values("geometry_order")
        .reset_index(drop=True)
    )
    if len(geom_order) == 0:
        return

    labels = geom_order["geometry_type"].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)

    overall_means: List[float] = []
    for geom in labels:
        g = d[d["geometry_type"].astype(str).eq(geom)].copy()
        weights = g["num_runs"].to_numpy(dtype=float)
        vals = g["mean_adaptive_ped_horizon_s"].to_numpy(dtype=float)
        if np.isfinite(vals).any():
            overall_means.append(float(np.average(vals, weights=weights)))
        else:
            overall_means.append(float("nan"))

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(x, overall_means, width=0.56, alpha=0.35, label="Overall mean horizon")

    bin_offsets = {"Low": -0.18, "Medium": 0.0, "High": 0.18}
    bin_markers = {"Low": "o", "Medium": "s", "High": "^"}
    for b, off in bin_offsets.items():
        ys: List[float] = []
        xs: List[float] = []
        for i, geom in enumerate(labels):
            g = d[(d["geometry_type"].astype(str).eq(geom)) & (d["uncertainty_bin"].astype(str).eq(b))]
            if len(g) == 0:
                continue
            weights = g["num_runs"].to_numpy(dtype=float)
            vals = g["mean_adaptive_ped_horizon_s"].to_numpy(dtype=float)
            if np.isfinite(vals).any():
                xs.append(float(i) + off)
                ys.append(float(np.average(vals, weights=weights)))
        if xs:
            ax.scatter(xs, ys, s=85, marker=bin_markers.get(b, "o"), label=f"{b} uncertainty", zorder=3)
            for xi, yi in zip(xs, ys):
                ax.text(xi, yi + 0.035, f"{yi:.2f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Mean pedestrian prediction horizon (s)")
    ax.set_xlabel("Ego approach geometry")
    ax.set_title("Geometry-dependent horizon adaptation across tested scenes")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="best", frameon=True)

    finite_means = [v for v in overall_means if np.isfinite(v)]
    if finite_means:
        ymin = max(0.0, min(finite_means) - 0.5)
        ymax = max(finite_means) + 0.8
        for b in bin_offsets:
            vals = d[d["uncertainty_bin"].astype(str).eq(b)]["mean_adaptive_ped_horizon_s"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                ymin = min(ymin, float(np.nanmin(vals)) - 0.5)
                ymax = max(ymax, float(np.nanmax(vals)) + 0.8)
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_uncertainty_response_curves_summary(paired_all: pd.DataFrame) -> pd.DataFrame:
    if len(paired_all) == 0:
        return pd.DataFrame()
    d = paired_all.copy()
    d["degradation_fraction"] = pd.to_numeric(d["degradation_level_pct"], errors="coerce") / 100.0
    d["normalized_osm_error"] = pd.to_numeric(d["normalized_osm_error"], errors="coerce").clip(0.0, 1.0)
    d["combined_uncertainty_index"] = 0.5 * d["degradation_fraction"] + 0.5 * d["normalized_osm_error"]
    d["uncertainty_bin"] = d["combined_uncertainty_index"].map(uncertainty_bin)
    rows: List[Dict[str, Any]] = []
    for (scene_name, geometry_type, geometry_order), scene_df in d.groupby(
        ["scene_name", "geometry_type", "geometry_order"], dropna=False
    ):
        safety_metric = choose_safety_metric(scene_df)
        s = scene_df[scene_df["metric_name"].eq(safety_metric)].copy()
        p = scene_df[scene_df["metric_name"].eq("effective_travel_time_s")].copy()
        if len(s) == 0 and len(p) == 0:
            continue
        for ub in ["Low", "Medium", "High"]:
            sb = s[s["uncertainty_bin"].astype(str).eq(ub)]
            pb = p[p["uncertainty_bin"].astype(str).eq(ub)]
            rows.append(
                {
                    "scene_name": str(scene_name),
                    "geometry_type": str(geometry_type),
                    "geometry_order": int(geometry_order),
                    "uncertainty_bin": ub,
                    "safety_metric_used": str(safety_metric),
                    "mean_safety_improvement_pct": safe_mean(sb["signed_improvement_percent"]) if len(sb) else float("nan"),
                    "mean_effective_travel_time_reduction_pct": safe_mean(pb["signed_improvement_percent"]) if len(pb) else float("nan"),
                    "num_safety_rows": int(len(sb)),
                    "num_performance_rows": int(len(pb)),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        bin_order = {"Low": 1, "Medium": 2, "High": 3}
        out["uncertainty_bin_order"] = out["uncertainty_bin"].map(bin_order).fillna(99).astype(int)
        out = out.sort_values(["geometry_order", "uncertainty_bin_order"]).reset_index(drop=True)
    return out


def save_uncertainty_response_curves_by_geometry(
    uncertainty_horizon_summary: pd.DataFrame,
    response_summary: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    if len(uncertainty_horizon_summary) == 0 and len(response_summary) == 0:
        return
    geom_order = (
        pd.concat(
            [
                uncertainty_horizon_summary[["geometry_type", "geometry_order"]] if len(uncertainty_horizon_summary) else pd.DataFrame(columns=["geometry_type", "geometry_order"]),
                response_summary[["geometry_type", "geometry_order"]] if len(response_summary) else pd.DataFrame(columns=["geometry_type", "geometry_order"]),
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values("geometry_order")
    )
    if len(geom_order) == 0:
        return
    x_bins = ["Low", "Medium", "High"]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6), constrained_layout=True, gridspec_kw={"wspace": 0.20})
    panels = [
        ("mean_adaptive_ped_horizon_s", "Mean adaptive pedestrian horizon (s)", "Mean adaptive pedestrian horizon vs uncertainty"),
        ("mean_safety_improvement_pct", "Mean safety improvement (%)", "Mean safety improvement vs uncertainty"),
        ("mean_effective_travel_time_reduction_pct", "Mean effective travel-time reduction (%)", "Mean effective travel-time reduction vs uncertainty"),
    ]
    for ax, (value_col, ylabel, title) in zip(axes, panels):
        for _, g in geom_order.iterrows():
            geom = str(g["geometry_type"])
            if value_col == "mean_adaptive_ped_horizon_s":
                src = uncertainty_horizon_summary[uncertainty_horizon_summary["geometry_type"].eq(geom)].copy()
            else:
                src = response_summary[response_summary["geometry_type"].eq(geom)].copy()
            y_vals: List[float] = []
            for ub in x_bins:
                pick = src[src["uncertainty_bin"].astype(str).eq(ub)]
                y_vals.append(safe_mean(pick[value_col]) if len(pick) else float("nan"))
            ax.plot(x_bins, y_vals, marker="o", linewidth=2.0, label=geom)
        if value_col != "mean_adaptive_ped_horizon_s":
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Uncertainty bin")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.tick_params(labelsize=9)
    axes[0].legend(loc="best", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def build_horizon_safety_coupling_summary(paired_all: pd.DataFrame, osm_raw_all: pd.DataFrame) -> pd.DataFrame:
    """
    Build the paper-level coupling dataset:
        adaptive horizon (mechanism)  vs  paired safety improvement (outcome).

    Each point is aggregated over seeds for one geometry / degradation / OSM-error case.
    This is intentionally different from the horizon bar chart: it links the learned
    horizon to the safety benefit rather than repeating horizon levels only.
    """
    if paired_all is None or len(paired_all) == 0 or osm_raw_all is None or len(osm_raw_all) == 0:
        return pd.DataFrame()
    if HORIZON_METRIC not in osm_raw_all.columns:
        return pd.DataFrame()

    rows: List[pd.DataFrame] = []
    merge_keys = ["scene_name", "osm_error_case", "degradation_mode", "degradation_level_pct", "seed"]

    osm = osm_raw_all.copy()
    for c in merge_keys:
        if c in osm.columns:
            if c in {"scene_name", "osm_error_case", "degradation_mode"}:
                osm[c] = osm[c].astype(str)
            else:
                osm[c] = pd.to_numeric(osm[c], errors="coerce")
    osm[HORIZON_METRIC] = pd.to_numeric(osm[HORIZON_METRIC], errors="coerce")

    for scene_name, scene_df in paired_all.groupby("scene_name", dropna=False):
        safety_metric = choose_safety_metric(scene_df)
        if not safety_metric:
            continue
        safety = scene_df[scene_df["metric_name"].eq(safety_metric)].copy()
        if len(safety) == 0:
            continue

        for c in merge_keys:
            if c in safety.columns:
                if c in {"scene_name", "osm_error_case", "degradation_mode"}:
                    safety[c] = safety[c].astype(str)
                else:
                    safety[c] = pd.to_numeric(safety[c], errors="coerce")

        osm_cols = [
            c for c in merge_keys + [
                HORIZON_METRIC, "normalized_osm_error", "osm_error_signed_m",
                "osm_error_abs_m", "injected_offset_signed_m", "geometry_type",
                "geometry_order",
            ]
            if c in osm.columns
        ]
        merged = safety.merge(osm[osm_cols], on=merge_keys, how="left", suffixes=("", "_osmrow"))
        if len(merged) == 0:
            continue

        merged["safety_metric_used"] = safety_metric
        merged["safety_improvement_pct"] = (
            pd.to_numeric(merged["signed_improvement_percent"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        merged["safety_improvement_abs"] = (
            pd.to_numeric(merged["signed_improvement_absolute"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        merged[HORIZON_METRIC] = pd.to_numeric(merged[HORIZON_METRIC], errors="coerce")

        group_cols = [
            "scene_name", "geometry_type", "geometry_order", "safety_metric_used",
            "osm_error_case", "degradation_mode", "degradation_level_pct",
        ]
        for optional in ["normalized_osm_error", "osm_error_signed_m", "osm_error_abs_m", "injected_offset_signed_m"]:
            if optional in merged.columns and optional not in group_cols:
                group_cols.append(optional)

        keep = merged[np.isfinite(merged[HORIZON_METRIC].to_numpy(dtype=float))].copy()
        if len(keep) == 0:
            continue

        agg = (
            keep.groupby(group_cols, dropna=False)
            .agg(
                mean_adaptive_ped_horizon_s=(HORIZON_METRIC, safe_mean),
                mean_safety_improvement_pct=("safety_improvement_pct", safe_mean),
                mean_safety_improvement_abs=("safety_improvement_abs", safe_mean),
                num_points=("safety_improvement_abs", "count"),
            )
            .reset_index()
        )
        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["degradation_level_pct"] = pd.to_numeric(out["degradation_level_pct"], errors="coerce")
    return out.sort_values(["geometry_order", "degradation_level_pct", "osm_error_case"]).reset_index(drop=True)


def save_horizon_safety_coupling_by_geometry(summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    """
    Replacement for the previous uncertainty-response curve.

    This figure directly tests the contribution mechanism:
        Does the adaptive horizon selected by the OSM-aware planner correspond to
        improved pedestrian safety?

    It is not just a repetition of the horizon values. It couples the planner's
    learned horizon with the paired baseline-vs-OSM safety improvement.
    """
    if summary is None or len(summary) == 0:
        return
    req = {"geometry_order", "geometry_type", "mean_adaptive_ped_horizon_s", "mean_safety_improvement_pct"}
    if not req.issubset(set(summary.columns)):
        return

    d = summary.copy()
    d["mean_adaptive_ped_horizon_s"] = pd.to_numeric(d["mean_adaptive_ped_horizon_s"], errors="coerce")
    d["mean_safety_improvement_pct"] = pd.to_numeric(d["mean_safety_improvement_pct"], errors="coerce")
    d["mean_safety_improvement_abs"] = pd.to_numeric(d.get("mean_safety_improvement_abs", np.nan), errors="coerce")
    d["degradation_level_pct"] = pd.to_numeric(d.get("degradation_level_pct", np.nan), errors="coerce")
    d = d[np.isfinite(d["mean_adaptive_ped_horizon_s"].to_numpy(dtype=float))].copy()
    if len(d) == 0:
        return

    geom_order = (
        d[["geometry_order", "geometry_type"]]
        .drop_duplicates()
        .sort_values("geometry_order")
        .reset_index(drop=True)
    )
    n = len(geom_order)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.4), sharey=False, constrained_layout=True)
    if n == 1:
        axes = [axes]

    deg_values = pd.to_numeric(d["degradation_level_pct"], errors="coerce")
    finite_deg = deg_values[np.isfinite(deg_values)]
    vmin = float(finite_deg.min()) if len(finite_deg) else 0.0
    vmax = float(finite_deg.max()) if len(finite_deg) else 100.0

    last_scatter = None
    for ax, (_, grow) in zip(axes, geom_order.iterrows()):
        geom = str(grow["geometry_type"])
        g = d[d["geometry_type"].astype(str).eq(geom)].copy()

        y = pd.to_numeric(g["mean_safety_improvement_pct"], errors="coerce")
        ylabel = "Safety improvement (%)"
        if not y.notna().any():
            y = pd.to_numeric(g["mean_safety_improvement_abs"], errors="coerce")
            ylabel = "Safety improvement (absolute)"

        x = pd.to_numeric(g["mean_adaptive_ped_horizon_s"], errors="coerce")
        c = pd.to_numeric(g["degradation_level_pct"], errors="coerce")
        valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
        if valid.sum() == 0:
            ax.axis("off")
            ax.set_title(f"{geom}\n(no coupling data)")
            continue

        xv = x.to_numpy(dtype=float)[valid]
        yv = y.to_numpy(dtype=float)[valid]
        cv = c.to_numpy(dtype=float)[valid]
        if not np.isfinite(cv).any():
            cv = np.zeros_like(xv)

        last_scatter = ax.scatter(
            xv, yv, c=cv, cmap="viridis", vmin=vmin, vmax=vmax,
            s=42, alpha=0.82, edgecolor="black", linewidth=0.25,
        )

        if len(xv) >= 3 and np.nanstd(xv) > EPS:
            try:
                coef = np.polyfit(xv, yv, deg=1)
                xs = np.linspace(float(np.nanmin(xv)), float(np.nanmax(xv)), 80)
                ax.plot(xs, coef[0] * xs + coef[1], linestyle="--", linewidth=1.6, alpha=0.85)
                corr = float(np.corrcoef(xv, yv)[0, 1]) if len(xv) > 1 else float("nan")
                if np.isfinite(corr):
                    ax.text(
                        0.03, 0.95, f"r = {corr:.2f}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
                    )
            except Exception:
                pass

        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.55)
        ax.set_title(geom, fontsize=12)
        ax.set_xlabel("Mean adaptive pedestrian horizon (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22, linestyle="--")

    if last_scatter is not None:
        cb = fig.colorbar(last_scatter, ax=axes, fraction=0.025, pad=0.015)
        cb.set_label("Camera degradation (%)")

    fig.suptitle("Adaptive-horizon safety coupling across geometries", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def choose_robustness_safety_metric(baseline_all: pd.DataFrame, osm_all: pd.DataFrame) -> str:
    """Choose an actual safety metric that can be compared directly by degradation.

    Preference is given to metrics where higher is safer, especially min pedestrian TTC.
    If only count-style risk metrics are available, the plotting function converts them
    to a safety score by negating the value so higher still means safer.
    """
    candidates = [
        "min_ped_ttc_s",
        "post_encroachment_time_s",
        "min_relevant_ped_distance_m",
        "min_ped_distance_m",
        "critical_ped_ttc_frames",
        "ped_conflict_zone_violation_frames",
        "unsafe_frames_total",
    ]
    for metric in candidates:
        if metric not in baseline_all.columns or metric not in osm_all.columns:
            continue
        b = pd.to_numeric(baseline_all[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
        o = pd.to_numeric(osm_all[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if len(b) == 0 or len(o) == 0:
            continue
        if min(float(b.notna().mean()), float(o.notna().mean())) >= 0.35:
            return metric
    return "min_ped_ttc_s"


def robustness_panel_axis_info(metric: str) -> Tuple[str, bool]:
    """Return y-label and whether to negate a risk/count metric.

    The plotted axis always uses the convention: higher = safer.
    """
    if metric in {"unsafe_frames_total", "critical_ped_ttc_frames", "ped_conflict_zone_violation_frames"}:
        return f"Safety score (-{metric})", True
    if metric == "min_ped_ttc_s":
        return "Min pedestrian TTC (s)", False
    if metric == "post_encroachment_time_s":
        return "Post-encroachment time (s)", False
    if metric in {"min_relevant_ped_distance_m", "min_ped_distance_m"}:
        return "Minimum pedestrian distance (m)", False
    return str(metric), False


def build_planner_robustness_by_degradation_summary(
    baseline_all: pd.DataFrame,
    osm_all: pd.DataFrame,
    safety_metric: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build mean and seed-level robustness data for Figure 06.

    The seed-level table is intentionally aggregated by scene, geometry,
    degradation, and seed so the plot shows seed-to-seed variability without
    overplotting every OSM-error case. Baseline points are averaged across
    degradation modes for the seed; OSM-aware points are averaged across OSM
    error cases, degradation modes, and the seed.
    """
    if len(baseline_all) == 0 or len(osm_all) == 0:
        return pd.DataFrame(), pd.DataFrame()
    if safety_metric not in baseline_all.columns or safety_metric not in osm_all.columns:
        return pd.DataFrame(), pd.DataFrame()

    common_cols = ["scene_name", "geometry_type", "geometry_order", "degradation_level_pct", "seed"]
    b_cols = [c for c in common_cols if c in baseline_all.columns] + [safety_metric]
    o_cols = [c for c in common_cols if c in osm_all.columns] + [safety_metric]
    b = baseline_all[b_cols].copy()
    o = osm_all[o_cols].copy()
    b[safety_metric] = pd.to_numeric(b[safety_metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    o[safety_metric] = pd.to_numeric(o[safety_metric], errors="coerce").replace([np.inf, -np.inf], np.nan)

    seed_keys = ["scene_name", "geometry_type", "geometry_order", "degradation_level_pct", "seed"]
    b_seed = (
        b.groupby(seed_keys, dropna=False)
        .agg(seed_safety_value=(safety_metric, safe_mean), num_raw_runs=(safety_metric, "count"))
        .reset_index()
    )
    b_seed["planner"] = "Perception-only baseline"
    o_seed = (
        o.groupby(seed_keys, dropna=False)
        .agg(seed_safety_value=(safety_metric, safe_mean), num_raw_runs=(safety_metric, "count"))
        .reset_index()
    )
    o_seed["planner"] = "OSM-aware learned-horizon"
    seed_df = pd.concat([b_seed, o_seed], ignore_index=True)
    seed_df["safety_metric_used"] = safety_metric

    mean_keys = ["scene_name", "geometry_type", "geometry_order", "degradation_level_pct", "planner"]
    mean_df = (
        seed_df.groupby(mean_keys, dropna=False)
        .agg(
            mean_safety=("seed_safety_value", safe_mean),
            std_safety=("seed_safety_value", safe_std),
            num_seeds=("seed_safety_value", "count"),
        )
        .reset_index()
    )
    mean_df["safety_metric_used"] = safety_metric
    return (
        mean_df.sort_values(["geometry_order", "degradation_level_pct", "planner"]).reset_index(drop=True),
        seed_df.sort_values(["geometry_order", "degradation_level_pct", "planner", "seed"]).reset_index(drop=True),
    )


def save_planner_robustness_by_degradation_seed_points(
    mean_summary: pd.DataFrame,
    seed_points: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    """Save Figure 06 in a mean ± std shaded-band style.

    Each geometry panel shows the robustness safety metric versus camera
    degradation. The mean trend is shown as a solid line, and the shaded band
    shows mean ± standard deviation across seeds. Baseline is blue and
    OSM-aware learned-horizon is orange.
    """
    if mean_summary is None or len(mean_summary) == 0:
        return

    d_mean = mean_summary.copy()
    d_mean["degradation_level_pct"] = pd.to_numeric(d_mean["degradation_level_pct"], errors="coerce")
    metric = (
        str(d_mean["safety_metric_used"].dropna().iloc[0])
        if "safety_metric_used" in d_mean.columns and d_mean["safety_metric_used"].notna().any()
        else "min_ped_ttc_s"
    )
    ylabel, negate = robustness_panel_axis_info(metric)

    if negate:
        d_mean["plot_mean"] = -pd.to_numeric(d_mean["mean_safety"], errors="coerce")
    else:
        d_mean["plot_mean"] = pd.to_numeric(d_mean["mean_safety"], errors="coerce")
    d_mean["plot_std"] = pd.to_numeric(d_mean["std_safety"], errors="coerce").abs()

    geom_order = (
        d_mean[["geometry_type", "geometry_order"]]
        .drop_duplicates()
        .sort_values("geometry_order")
        .reset_index(drop=True)
    )
    if len(geom_order) == 0:
        return

    colors = {
        "Perception-only baseline": "#2F6BDE",
        "OSM-aware learned-horizon": "#E67E22",
    }
    markers = {
        "Perception-only baseline": "o",
        "OSM-aware learned-horizon": "s",
    }
    planner_order = ["Perception-only baseline", "OSM-aware learned-horizon"]
    x_vals = [10, 30, 50, 70, 90]

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), constrained_layout=True, gridspec_kw={"wspace": 0.17})
    if len(geom_order) == 1:
        axes = [axes]

    for ax, (_, ginfo) in zip(axes, geom_order.iterrows()):
        geom = str(ginfo["geometry_type"])
        geom_df = d_mean[d_mean["geometry_type"].astype(str).eq(geom)].copy()
        if len(geom_df) == 0:
            ax.axis("off")
            ax.set_title(f"{geom}\n(no data)")
            continue

        for planner in planner_order:
            p = geom_df[geom_df["planner"].astype(str).eq(planner)].copy().sort_values("degradation_level_pct")
            if len(p) == 0:
                continue
            x = pd.to_numeric(p["degradation_level_pct"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(p["plot_mean"], errors="coerce").to_numpy(dtype=float)
            ystd = pd.to_numeric(p["plot_std"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() == 0:
                continue
            x = x[valid]
            y = y[valid]
            ystd = ystd[valid]
            ystd = np.nan_to_num(ystd, nan=0.0, posinf=0.0, neginf=0.0)
            color = colors.get(planner, None)
            marker = markers.get(planner, "o")

            ax.fill_between(
                x,
                y - ystd,
                y + ystd,
                color=color,
                alpha=0.18,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(
                x,
                y,
                marker=marker,
                linewidth=2.5,
                markersize=6.5,
                color=color,
                label=planner,
                zorder=3,
            )

        ax.set_xticks(x_vals)
        ax.set_xlabel("Camera degradation (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(geom)
        ax.grid(axis="both", alpha=0.24, linewidth=0.8, linestyle="--")
        ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best", fontsize=9, frameon=True)

    fig.suptitle("Planner robustness under camera degradation across tested geometries", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def copy_osm_error_case_plots(scene_specs: List[SceneSpec], args: argparse.Namespace, paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    if args.no_copy_osm_plots:
        return copied
    for scene in scene_specs:
        base = Path(args.osm_context_root) / scene.scene / "error_cases"
        found: Optional[Path] = None
        for name in OSM_ERROR_PLOT_CANDIDATES:
            p = base / name
            if p.is_file():
                found = p
                break
        if found is None and base.is_dir():
            pngs = sorted(base.glob("*.png"))
            if pngs:
                found = pngs[0]
        if found is None:
            copied.append({
                "scene_name": scene.scene,
                "geometry_type": scene.geometry,
                "source_file": "",
                "copied_file": "",
                "status": "not_found",
            })
            continue
        dst = paths["osm_error_cases"] / f"{scene.order:02d}_{scene.short}_osm_error_cases.png"
        shutil.copy2(found, dst)
        copied.append({
            "scene_name": scene.scene,
            "geometry_type": scene.geometry,
            "source_file": str(found),
            "copied_file": str(dst),
            "status": "copied",
        })
    return copied


def build_geometry_summary(paired_all: pd.DataFrame, osm_all: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if len(osm_all) == 0:
        return pd.DataFrame()
    for (order, geom, scene), g_osm in osm_all.groupby(["geometry_order", "geometry_type", "scene_name"], dropna=False):
        row: Dict[str, Any] = {
            "geometry_order": int(order),
            "geometry_type": geom,
            "scene_name": scene,
            "num_osm_runs": int(len(g_osm)),
            "mean_adaptive_ped_horizon_s": safe_mean(g_osm[HORIZON_METRIC]) if HORIZON_METRIC in g_osm.columns else float("nan"),
        }
        g_pair = paired_all[(paired_all["scene_name"].eq(scene))].copy()
        row["scene_has_weak_safety_stress_test"] = bool(scene_has_weak_safety_stress_test(g_pair))
        for metric in ["unsafe_frames_total", "min_ped_ttc_s", "travel_time_s", "stop_time_s", "comfort_jerk_rms_mps3"]:
            d = g_pair[g_pair["metric_name"].eq(metric)].copy()
            if len(d) == 0:
                continue
            row[f"{metric}_mean_baseline"] = safe_mean(d["baseline_value"])
            row[f"{metric}_mean_osm"] = safe_mean(d["osm_value"])
            row[f"{metric}_mean_improvement_abs"] = safe_mean(d["signed_improvement_absolute"])
            row[f"{metric}_mean_improvement_pct"] = safe_mean(d["signed_improvement_percent"])
            row[f"{metric}_help_rate_pct"] = 100.0 * pd.Series(d["osm_helped_flag"]).fillna(False).astype(bool).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("geometry_order").reset_index(drop=True)


def read_policy_summary(scene: SceneSpec, root: Path) -> Dict[str, Any]:
    p = root / scene.scene / "adaptive_horizon_policy.json"
    out: Dict[str, Any] = {
        "scene_name": scene.scene,
        "geometry_type": scene.geometry,
        "policy_file": str(p),
        "policy_found": p.is_file(),
    }
    if not p.is_file():
        return out
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        out["policy_error"] = str(exc)
        return out
    out["policy_type"] = payload.get("policy_type", "")
    out["policy_name"] = payload.get("policy_name", "")
    out["pedestrian_mae_s"] = payload.get("pedestrian", {}).get("fit_metrics", {}).get("mae_s", np.nan)
    out["vehicle_mae_s"] = payload.get("vehicle", {}).get("fit_metrics", {}).get("mae_s", np.nan)
    return out


def write_reports(
    paths: Dict[str, Path],
    figure_rows: List[Dict[str, Any]],
    osm_copy_rows: List[Dict[str, Any]],
    geom_summary: pd.DataFrame,
    uncertainty_summary: pd.DataFrame,
    policy_rows: List[Dict[str, Any]],
    safety_stress_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    figure_index = pd.DataFrame(figure_rows + osm_copy_rows)
    figure_index.to_csv(paths["tables"] / "figure_index.csv", index=False)

    captions: List[str] = []
    captions.append("# Recommended paper figures and captions")
    captions.append("")
    captions.append("## Main result figures")
    captions.append("")
    captions.append("**Figure A–C: Geometry-specific mechanism heatmaps.** Each row shows camera degradation and each column shows the injected OSM crossing offset. The three panels connect the mechanism to outcomes: the selected pedestrian prediction horizon, a TTC-based safety metric, and an efficiency/performance metric. Positive safety/performance values mean the OSM-aware learned-horizon planner improved over the perception-only IDM baseline. TTC is used as the main safety panel because it measures temporal collision margin and is less misleading than raw unsafe-frame counts when a vehicle stops near a crossing.")
    captions.append("")
    captions.append("**Figure D: Horizon bar/point summary across scenes.** Bars show overall mean adaptive pedestrian horizon by geometry, with low/medium/high uncertainty point overlays.")
    captions.append("")
    captions.append("**Figure E: Planner robustness by degradation.** Each panel compares the perception-only baseline and the OSM-aware learned-horizon planner as camera degradation increases. Solid blue/orange lines show the mean trend, and the shaded bands show mean ± standard deviation across seeds.")
    captions.append("")
    captions.append("## Suggested final paper figure set")
    captions.append("")
    captions.append("1. Three copied OSM error-case plots, one per geometry.")
    captions.append("2. Three core geometry-specific heatmaps.")
    captions.append("3. One horizon bar/point summary across geometries.")
    captions.append("4. One planner-robustness-by-degradation figure with mean ± standard-deviation shaded bands.")
    captions.append("")
    captions.append("This keeps the paper focused on OSM error definition, horizon adaptation behavior, and robustness under sensor degradation.")
    captions.append("")
    (paths["reports"] / "recommended_figures_and_captions.md").write_text("\n".join(captions) + "\n", encoding="utf-8")

    payload = {
        "output_root": str(paths["root"]),
        "focus_mode_requested": args.focus_mode,
        "osm_error_axis": args.osm_error_axis,
        "figure_index": figure_rows,
        "osm_error_case_plot_copy_status": osm_copy_rows,
        "geometry_summary": geom_summary.to_dict(orient="records") if len(geom_summary) else [],
        "uncertainty_horizon_summary": uncertainty_summary.to_dict(orient="records") if len(uncertainty_summary) else [],
        "learned_policy_summary": policy_rows,
        "safety_stress_test_warnings": safety_stress_rows,
    }
    (paths["reports"] / "comparison_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def main() -> None:
    apply_plot_style()
    args = parse_args()
    scene_specs = parse_scene_specs(args.scene_name)
    paths = ensure_dirs(Path(args.output_root))

    all_paired: List[pd.DataFrame] = []
    all_osm_summary: List[pd.DataFrame] = []
    all_baseline_raw: List[pd.DataFrame] = []
    all_osm_raw: List[pd.DataFrame] = []
    figure_rows: List[Dict[str, Any]] = []
    policy_rows: List[Dict[str, Any]] = []

    for scene in scene_specs:
        baseline, osm = load_scene_metrics(scene, args)
        baseline_raw = baseline.copy()
        baseline_raw["geometry_short"] = scene.short
        baseline_raw["geometry_order"] = scene.order
        all_baseline_raw.append(baseline_raw)
        # Store raw OSM rows for cross-geometry horizon/uncertainty analysis.
        osm_raw = osm.copy()
        osm_raw["geometry_short"] = scene.short
        osm_raw["geometry_order"] = scene.order
        all_osm_raw.append(osm_raw)

        metrics = numeric_metrics_for_pairing(baseline, osm)
        paired = build_paired_rows(scene, baseline, osm, metrics)
        osm_summary = compute_osm_condition_summary(scene, osm)
        all_paired.append(paired)
        all_osm_summary.append(osm_summary)
        policy_rows.append(read_policy_summary(scene, Path(args.learned_policy_root)))

        out_name = f"{scene.order:02d}_{scene.scene}_{scene.short}_horizon_safety_performance_heatmap.png"
        info = save_scene_three_panel_heatmap(
            scene=scene,
            osm_summary=osm_summary,
            paired=paired,
            focus_mode=args.focus_mode,
            axis_mode=args.osm_error_axis,
            out_path=paths["main"] / out_name,
            dpi=args.dpi,
        )
        figure_rows.append({
            "figure_type": "main_result_three_panel_heatmap",
            "scene_name": scene.scene,
            "geometry_type": scene.geometry,
            "file": str(paths["main"] / out_name),
            "status": "created",
            **{k: v for k, v in info.items() if k not in {"scene_name", "geometry_type", "plot_file"}},
        })

    paired_all = pd.concat(all_paired, ignore_index=True) if all_paired else pd.DataFrame()
    osm_summary_all = pd.concat(all_osm_summary, ignore_index=True) if all_osm_summary else pd.DataFrame()
    baseline_raw_all = pd.concat(all_baseline_raw, ignore_index=True) if all_baseline_raw else pd.DataFrame()
    osm_raw_all = pd.concat(all_osm_raw, ignore_index=True) if all_osm_raw else pd.DataFrame()

    paired_all.to_csv(paths["tables"] / "all_paired_key_metrics.csv", index=False)
    osm_summary_all.to_csv(paths["tables"] / "all_osm_condition_summary.csv", index=False)
    tradeoff_summary = build_comfort_tradeoff_summary(scene_specs, paired_all)
    tradeoff_summary.to_csv(paths["tables"] / "comfort_tradeoff_summary.csv", index=False)

    geom_summary = build_geometry_summary(paired_all, osm_raw_all)
    geom_summary.to_csv(paths["tables"] / "geometry_summary_key_metrics.csv", index=False)
    safety_stress_rows: List[Dict[str, Any]] = []
    for scene in scene_specs:
        weak = bool(scene_has_weak_safety_stress_test(paired_all[paired_all["scene_name"].eq(scene.scene)].copy()))
        safety_stress_rows.append(
            {
                "scene_name": scene.scene,
                "geometry_type": scene.geometry,
                "scene_has_weak_safety_stress_test": weak,
            }
        )

    uncertainty_summary = build_geometry_uncertainty_horizon_summary(osm_raw_all)
    uncertainty_summary.to_csv(paths["tables"] / "geometry_uncertainty_horizon_summary.csv", index=False)

    horizon_bar_points_plot = paths["main"] / "05_horizon_bar_points_across_geometries.png"
    save_horizon_bar_points_across_geometries(uncertainty_summary, horizon_bar_points_plot, args.dpi)
    figure_rows.append({
        "figure_type": "main_result_horizon_bar_points_across_geometries",
        "scene_name": "all",
        "geometry_type": "all geometries",
        "file": str(horizon_bar_points_plot),
        "status": "created" if horizon_bar_points_plot.is_file() else "not_created_no_data",
        "focus_mode_used": "all modes",
        "safety_metric_used": "n/a",
        "performance_metric_used": "n/a",
    })
    robustness_metric = choose_robustness_safety_metric(baseline_raw_all, osm_raw_all)
    robustness_mean_summary, robustness_seed_points = build_planner_robustness_by_degradation_summary(
        baseline_all=baseline_raw_all,
        osm_all=osm_raw_all,
        safety_metric=robustness_metric,
    )
    robustness_mean_summary.to_csv(paths["tables"] / "planner_robustness_by_degradation_summary.csv", index=False)
    robustness_seed_points.to_csv(paths["tables"] / "planner_robustness_seed_values.csv", index=False)
    robustness_plot = paths["main"] / "06_planner_robustness_by_degradation.png"
    save_planner_robustness_by_degradation_seed_points(
        mean_summary=robustness_mean_summary,
        seed_points=robustness_seed_points,
        out_path=robustness_plot,
        dpi=args.dpi,
    )
    figure_rows.append({
        "figure_type": "main_result_planner_robustness_by_degradation_seed_points",
        "scene_name": "all",
        "geometry_type": "all geometries",
        "file": str(robustness_plot),
        "status": "created" if robustness_plot.is_file() else "not_created_no_data",
        "focus_mode_used": "all modes",
        "safety_metric_used": robustness_metric,
        "performance_metric_used": "n/a",
    })

    pd.DataFrame(policy_rows).to_csv(paths["tables"] / "learned_policy_summary.csv", index=False)
    osm_copy_rows = copy_osm_error_case_plots(scene_specs, args, paths)
    write_reports(paths, figure_rows, osm_copy_rows, geom_summary, uncertainty_summary, policy_rows, safety_stress_rows, args)

    print("final_geometry_comparison_done:")
    print(f"  scenes: {', '.join(s.scene for s in scene_specs)}")
    print(f"  output_root: {paths['root']}")
    print(f"  main_plots: {paths['main']}")
    print(f"  osm_error_case_plots: {paths['osm_error_cases']}")
    print(f"  tables: {paths['tables']}")
    print(f"  report: {paths['reports'] / 'recommended_figures_and_captions.md'}")
    print("  suggested paper plots:")
    print("    - 3 copied OSM error-case plots, one per geometry")
    print("    - 3 three-panel heatmaps: horizon + safety + performance")
    print("    - 1 horizon bar/point summary across the three scenes")
    print("    - 1 planner robustness-by-degradation plot with mean ± std shaded bands")


if __name__ == "__main__":
    main()
