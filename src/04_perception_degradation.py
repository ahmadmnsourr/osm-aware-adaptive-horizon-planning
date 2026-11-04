#!/usr/bin/env python3
"""
Generate camera-based degraded perception inputs from a refined scene package.

Reads:
- data/refined/<scene>/sample_timeline.csv
- data/refined/<scene>/ego_timeline.csv
- data/refined/<scene>/objects_timeline.csv
- data/refined/<scene>/scene_summary.json

Writes (default root: data/planner_perception_only):
- data/planner_perception_only/<scene>/degraded_inputs/<mode>/degradation_XX/seed_YYY/objects_timeline_degraded.csv
- data/planner_perception_only/<scene>/degraded_inputs/<mode>/degradation_XX/seed_YYY/degradation_audit.json
- data/planner_perception_only/<scene>/degraded_inputs/<mode>/degradation_XX/seed_YYY/degradation_audit_per_frame.csv
- data/planner_perception_only/<scene>/degraded_inputs/degradation_audit_per_run.csv
- data/planner_perception_only/<scene>/degraded_inputs/degradation_formula_summary.json

Design intent:
- Degrade only objects explicitly marked as camera-degradable by upstream extraction/refinement.
- Keep the three modes:
    1) random_dropout
    2) burst_dropout
    3) distance_based_dropout
- Use logical, report-ready formulas tied to camera visibility, off-axis position, and distance.
- Center burst dropout around the refined crossing interaction window instead of random scene-wide placement.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DYNAMIC_CLASSES = {"pedestrian", "vehicle"}
SUPPORTED_MODES = {"random_dropout", "burst_dropout", "distance_based_dropout"}


@dataclass
class SceneCrossingWindow:
    crossing_start_frame: int
    crossing_end_frame: int
    candidate_start_frame: int
    candidate_end_frame: int


def parse_int_list_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate logical camera-based degraded perception inputs."
    )
    p.add_argument("--scene-name", default="scene-0061")
    p.add_argument("--refined-root", default="data/refined")
    p.add_argument(
        "--results-root",
        default="data/planner_perception_only",
        help="Output root under data/ (default: data/planner_perception_only).",
    )
    p.add_argument(
        "--degradation-modes",
        default="random_dropout,burst_dropout,distance_based_dropout",
        help="Comma-separated modes.",
    )
    p.add_argument("--degradation-levels", default="10,30,50,70,90")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument(
        "--distance-ref-m",
        type=float,
        default=25.0,
        help="Reference distance scale for distance-based degradation.",
    )
    p.add_argument(
        "--burst-base-extra-buffer",
        type=int,
        default=2,
        help="Base number of frames added before/after crossing for burst candidate zone.",
    )
    p.add_argument(
        "--keep-at-least-one-close-object-under-30",
        action="store_true",
        help="Optional realism guard for low degradation levels (<30%%): preserve one close central eligible object when possible.",
    )
    return p.parse_args()


def load_refined_scene(scene_dir: Path) -> Dict[str, object]:
    sample_df = pd.read_csv(scene_dir / "sample_timeline.csv")
    ego_df = pd.read_csv(scene_dir / "ego_timeline.csv")
    objects_df = pd.read_csv(scene_dir / "objects_timeline.csv")
    scene_summary = json.loads((scene_dir / "scene_summary.json").read_text(encoding="utf-8"))
    return {
        "sample_df": sample_df,
        "ego_df": ego_df,
        "objects_df": objects_df,
        "scene_summary": scene_summary,
    }


def safe_bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=bool)
    s = df[col]
    if s.dtype == bool:
        return s
    return s.fillna(default).astype(bool)


def camera_eligible_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Prefer upstream camera-aware flags. Fall back conservatively only if needed.
    """
    if "camera_degradable_flag" in df.columns:
        degr = safe_bool_series(df, "camera_degradable_flag", default=False)
        include = safe_bool_series(df, "degradation_include_flag", default=True)
        return (degr & include).to_numpy(dtype=bool)

    # Conservative fallback if older upstream files are used.
    dyn = df["coarse_class"].isin(DYNAMIC_CLASSES)
    return dyn.to_numpy(dtype=bool)


def visibility_multiplier_one(row: pd.Series) -> float:
    q = str(row.get("camera_visibility_quality", "") or "").strip().lower()
    if q == "good":
        return 0.80
    if q == "fair":
        return 1.00
    if q == "poor":
        return 1.25
    return 1.10


def lateral_multiplier_one(row: pd.Series) -> float:
    lat = abs(float(row.get("relative_lateral_m", 0.0)))
    if lat <= 1.5:
        return 1.00
    if lat <= 3.0:
        return 1.15
    return 1.30


def class_multiplier_one(row: pd.Series) -> float:
    cc = str(row.get("coarse_class", ""))
    if cc == "pedestrian":
        return 1.05
    if cc == "vehicle":
        return 1.00
    return 1.00


def distance_factor_one(row: pd.Series, distance_ref_m: float) -> float:
    d = float(row.get("distance_to_ego_m", np.nan))
    if not np.isfinite(d):
        return 1.00
    ref = max(float(distance_ref_m), 1e-6)
    return float(np.clip(d / ref, 0.10, 1.00))


def random_dropout_probability(row: pd.Series, severity: float) -> float:
    p = severity * visibility_multiplier_one(row) * class_multiplier_one(row)
    return float(np.clip(p, 0.0, 0.98))


def distance_based_dropout_probability(row: pd.Series, severity: float, distance_ref_m: float) -> float:
    p = (
        severity
        * distance_factor_one(row, distance_ref_m)
        * visibility_multiplier_one(row)
        * lateral_multiplier_one(row)
        * class_multiplier_one(row)
    )
    return float(np.clip(p, 0.0, 0.98))


def burst_inside_probability(row: pd.Series, severity: float, distance_ref_m: float) -> float:
    """
    Stronger dropout inside crossing-centered burst interval.
    Uses a severity-anchored burst base plus camera plausibility multipliers.
    """
    burst_base = 0.45 + 0.55 * severity
    distance_term = 0.60 + 0.40 * distance_factor_one(row, distance_ref_m)
    p = (
        burst_base
        * distance_term
        * visibility_multiplier_one(row)
        * lateral_multiplier_one(row)
        * class_multiplier_one(row)
    )
    return float(np.clip(p, 0.0, 0.98))


def burst_outside_probability(row: pd.Series, severity: float) -> float:
    """
    Keep a small background camera miss rate outside burst interval so the mode is not perfectly clean elsewhere.
    """
    p = 0.20 * random_dropout_probability(row, severity)
    return float(np.clip(p, 0.0, 0.40))


def choose_crossing_centered_burst_window(
    scene_summary: Dict[str, object],
    frame_indices: List[int],
    level_pct: int,
    seed: int,
    base_extra_buffer: int,
) -> SceneCrossingWindow:
    if not frame_indices:
        return SceneCrossingWindow(0, 0, 0, 0)

    n_frames = len(frame_indices)
    frame_min = int(min(frame_indices))
    frame_max = int(max(frame_indices))

    crossing_start = int(scene_summary.get("crossing_start_frame", frame_min))
    crossing_end = int(scene_summary.get("crossing_end_frame", frame_max))
    crossing_start = int(np.clip(crossing_start, frame_min, frame_max))
    crossing_end = int(np.clip(crossing_end, crossing_start, frame_max))

    crossing_len = crossing_end - crossing_start + 1
    severity = float(np.clip(level_pct / 100.0, 0.0, 1.0))

    extra_buffer = int(base_extra_buffer + round(3.0 * severity))
    candidate_start = max(frame_min, crossing_start - extra_buffer)
    candidate_end = min(frame_max, crossing_end + extra_buffer)

    candidate_len = candidate_end - candidate_start + 1
    if candidate_len <= 0:
        candidate_start = crossing_start
        candidate_end = crossing_end
        candidate_len = candidate_end - candidate_start + 1

    # Shorter at low severity, longer at high severity, always tied to crossing window.
    nominal_len = int(round(crossing_len * (0.55 + 0.65 * severity)))
    nominal_len = max(2, min(candidate_len, nominal_len))

    rng = np.random.default_rng(seed)
    start_low = candidate_start
    start_high = candidate_end - nominal_len + 1
    if start_high < start_low:
        burst_start = candidate_start
        burst_end = candidate_end
    else:
        burst_start = int(rng.integers(start_low, start_high + 1))
        burst_end = burst_start + nominal_len - 1

    return SceneCrossingWindow(
        crossing_start_frame=crossing_start,
        crossing_end_frame=crossing_end,
        candidate_start_frame=burst_start,
        candidate_end_frame=burst_end,
    )


def enforce_low_severity_realism_guard(
    frame_df: pd.DataFrame,
    keep_mask: np.ndarray,
    eligible_mask: np.ndarray,
    level_pct: int,
) -> np.ndarray:
    """
    For mild degradation, avoid collapsing a frame that still has a very close, central eligible object.
    Optional, off by default unless user enables it.
    """
    if level_pct >= 30:
        return keep_mask
    if not eligible_mask.any():
        return keep_mask
    if keep_mask[eligible_mask].any():
        return keep_mask

    elig = frame_df.loc[eligible_mask].copy()
    if len(elig) == 0:
        return keep_mask

    # Keep the closest, most central candidate if such an object exists.
    elig["centrality_score"] = elig["distance_to_ego_m"].astype(float) + 0.8 * elig["relative_lateral_m"].abs().astype(float)
    best_idx = elig["centrality_score"].idxmin()
    pos = frame_df.index.get_loc(best_idx)
    keep_mask[pos] = True
    return keep_mask


def per_frame_summary_row(
    frame_idx: int,
    frame_df_before: pd.DataFrame,
    frame_df_after: pd.DataFrame,
    level_pct: int,
    seed: int,
    mode: str,
    eligible_before: int,
    eligible_after: int,
    mean_drop_probability: float,
    burst_active: bool,
    burst_window: Optional[SceneCrossingWindow],
) -> Dict[str, float | int | bool | str]:
    b_ped = int((frame_df_before["coarse_class"] == "pedestrian").sum())
    b_veh = int((frame_df_before["coarse_class"] == "vehicle").sum())
    a_ped = int((frame_df_after["coarse_class"] == "pedestrian").sum())
    a_veh = int((frame_df_after["coarse_class"] == "vehicle").sum())

    row: Dict[str, float | int | bool | str] = {
        "frame_index": int(frame_idx),
        "degradation_mode": mode,
        "degradation_level_pct": int(level_pct),
        "seed": int(seed),
        "dynamic_rows_before": b_ped + b_veh,
        "dynamic_rows_after": a_ped + a_veh,
        "ped_rows_before": b_ped,
        "ped_rows_after": a_ped,
        "veh_rows_before": b_veh,
        "veh_rows_after": a_veh,
        "camera_eligible_rows_before": int(eligible_before),
        "camera_eligible_rows_after": int(eligible_after),
        "camera_drop_fraction": float((eligible_before - eligible_after) / eligible_before) if eligible_before > 0 else 0.0,
        "mean_drop_probability": float(mean_drop_probability),
        "burst_active": bool(burst_active),
    }
    if burst_window is not None:
        row["crossing_start_frame"] = int(burst_window.crossing_start_frame)
        row["crossing_end_frame"] = int(burst_window.crossing_end_frame)
        row["burst_start_frame"] = int(burst_window.candidate_start_frame)
        row["burst_end_frame"] = int(burst_window.candidate_end_frame)
    return row


def degrade_random_dropout(
    objects_df: pd.DataFrame,
    level_pct: int,
    seed: int,
    distance_ref_m: float,
    realism_guard: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float | int | str]]:
    severity = float(np.clip(level_pct / 100.0, 0.0, 1.0))
    rng = np.random.default_rng(seed)

    kept_chunks: List[pd.DataFrame] = []
    frame_rows: List[Dict[str, float | int | bool | str]] = []

    for frame_idx, frame_df in objects_df.groupby("frame_index", sort=True):
        f = frame_df.copy()
        eligible = camera_eligible_mask(f)
        keep_mask = np.ones(len(f), dtype=bool)

        probs = np.zeros(len(f), dtype=float)
        for pos, (_, row) in enumerate(f.iterrows()):
            if eligible[pos]:
                probs[pos] = random_dropout_probability(row, severity)

        draws = rng.random(len(f))
        keep_mask[eligible] = draws[eligible] >= probs[eligible]

        if realism_guard:
            keep_mask = enforce_low_severity_realism_guard(f, keep_mask, eligible, level_pct)

        kept = f.loc[keep_mask].copy()
        kept_chunks.append(kept)

        frame_rows.append(
            per_frame_summary_row(
                frame_idx=int(frame_idx),
                frame_df_before=f,
                frame_df_after=kept,
                level_pct=level_pct,
                seed=seed,
                mode="random_dropout",
                eligible_before=int(eligible.sum()),
                eligible_after=int((eligible & keep_mask).sum()),
                mean_drop_probability=float(probs[eligible].mean()) if eligible.any() else 0.0,
                burst_active=False,
                burst_window=None,
            )
        )

    degraded_df = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else objects_df.iloc[0:0].copy()
    frame_audit_df = pd.DataFrame(frame_rows).sort_values("frame_index").reset_index(drop=True)
    meta: Dict[str, float | int | str] = {
        "degradation_mode": "random_dropout",
        "severity_fraction": float(severity),
        "formula": "p_drop = severity * visibility_multiplier * class_multiplier",
        "camera_only": "true",
    }
    return degraded_df, frame_audit_df, meta


def degrade_distance_based_dropout(
    objects_df: pd.DataFrame,
    level_pct: int,
    seed: int,
    distance_ref_m: float,
    realism_guard: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float | int | str]]:
    severity = float(np.clip(level_pct / 100.0, 0.0, 1.0))
    rng = np.random.default_rng(seed)

    kept_chunks: List[pd.DataFrame] = []
    frame_rows: List[Dict[str, float | int | bool | str]] = []

    for frame_idx, frame_df in objects_df.groupby("frame_index", sort=True):
        f = frame_df.copy()
        eligible = camera_eligible_mask(f)
        keep_mask = np.ones(len(f), dtype=bool)

        probs = np.zeros(len(f), dtype=float)
        for pos, (_, row) in enumerate(f.iterrows()):
            if eligible[pos]:
                probs[pos] = distance_based_dropout_probability(row, severity, distance_ref_m)

        draws = rng.random(len(f))
        keep_mask[eligible] = draws[eligible] >= probs[eligible]

        if realism_guard:
            keep_mask = enforce_low_severity_realism_guard(f, keep_mask, eligible, level_pct)

        kept = f.loc[keep_mask].copy()
        kept_chunks.append(kept)

        frame_rows.append(
            per_frame_summary_row(
                frame_idx=int(frame_idx),
                frame_df_before=f,
                frame_df_after=kept,
                level_pct=level_pct,
                seed=seed,
                mode="distance_based_dropout",
                eligible_before=int(eligible.sum()),
                eligible_after=int((eligible & keep_mask).sum()),
                mean_drop_probability=float(probs[eligible].mean()) if eligible.any() else 0.0,
                burst_active=False,
                burst_window=None,
            )
        )

    degraded_df = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else objects_df.iloc[0:0].copy()
    frame_audit_df = pd.DataFrame(frame_rows).sort_values("frame_index").reset_index(drop=True)
    meta: Dict[str, float | int | str] = {
        "degradation_mode": "distance_based_dropout",
        "severity_fraction": float(severity),
        "distance_ref_m": float(distance_ref_m),
        "formula": "p_drop = severity * distance_factor * visibility_multiplier * lateral_multiplier * class_multiplier",
        "camera_only": "true",
    }
    return degraded_df, frame_audit_df, meta


def degrade_burst_dropout(
    objects_df: pd.DataFrame,
    scene_summary: Dict[str, object],
    level_pct: int,
    seed: int,
    distance_ref_m: float,
    base_extra_buffer: int,
    realism_guard: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float | int | str]]:
    severity = float(np.clip(level_pct / 100.0, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    grouped = list(objects_df.groupby("frame_index", sort=True))
    frame_indices = [int(k) for k, _ in grouped]
    burst_window = choose_crossing_centered_burst_window(
        scene_summary=scene_summary,
        frame_indices=frame_indices,
        level_pct=level_pct,
        seed=seed,
        base_extra_buffer=base_extra_buffer,
    )

    kept_chunks: List[pd.DataFrame] = []
    frame_rows: List[Dict[str, float | int | bool | str]] = []

    for frame_idx, frame_df in grouped:
        f = frame_df.copy()
        eligible = camera_eligible_mask(f)
        keep_mask = np.ones(len(f), dtype=bool)

        inside_burst = burst_window.candidate_start_frame <= int(frame_idx) <= burst_window.candidate_end_frame

        probs = np.zeros(len(f), dtype=float)
        for pos, (_, row) in enumerate(f.iterrows()):
            if not eligible[pos]:
                continue
            if inside_burst:
                probs[pos] = burst_inside_probability(row, severity, distance_ref_m)
            else:
                probs[pos] = burst_outside_probability(row, severity)

        draws = rng.random(len(f))
        keep_mask[eligible] = draws[eligible] >= probs[eligible]

        if realism_guard:
            keep_mask = enforce_low_severity_realism_guard(f, keep_mask, eligible, level_pct)

        kept = f.loc[keep_mask].copy()
        kept_chunks.append(kept)

        frame_rows.append(
            per_frame_summary_row(
                frame_idx=int(frame_idx),
                frame_df_before=f,
                frame_df_after=kept,
                level_pct=level_pct,
                seed=seed,
                mode="burst_dropout",
                eligible_before=int(eligible.sum()),
                eligible_after=int((eligible & keep_mask).sum()),
                mean_drop_probability=float(probs[eligible].mean()) if eligible.any() else 0.0,
                burst_active=inside_burst,
                burst_window=burst_window,
            )
        )

    degraded_df = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else objects_df.iloc[0:0].copy()
    frame_audit_df = pd.DataFrame(frame_rows).sort_values("frame_index").reset_index(drop=True)

    meta: Dict[str, float | int | str] = {
        "degradation_mode": "burst_dropout",
        "severity_fraction": float(severity),
        "crossing_start_frame": int(burst_window.crossing_start_frame),
        "crossing_end_frame": int(burst_window.crossing_end_frame),
        "burst_start_frame": int(burst_window.candidate_start_frame),
        "burst_end_frame": int(burst_window.candidate_end_frame),
        "formula_inside_burst": "p_drop = (0.45 + 0.55*severity) * (0.60 + 0.40*distance_factor) * visibility_multiplier * lateral_multiplier * class_multiplier",
        "formula_outside_burst": "p_drop = 0.20 * [severity * visibility_multiplier * class_multiplier]",
        "camera_only": "true",
    }
    return degraded_df, frame_audit_df, meta


def degrade_objects(
    objects_df: pd.DataFrame,
    scene_summary: Dict[str, object],
    mode: str,
    level_pct: int,
    seed: int,
    distance_ref_m: float,
    base_extra_buffer: int,
    realism_guard: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float | int | str]]:
    if mode == "random_dropout":
        return degrade_random_dropout(objects_df, level_pct, seed, distance_ref_m, realism_guard)
    if mode == "distance_based_dropout":
        return degrade_distance_based_dropout(objects_df, level_pct, seed, distance_ref_m, realism_guard)
    if mode == "burst_dropout":
        return degrade_burst_dropout(
            objects_df=objects_df,
            scene_summary=scene_summary,
            level_pct=level_pct,
            seed=seed,
            distance_ref_m=distance_ref_m,
            base_extra_buffer=base_extra_buffer,
            realism_guard=realism_guard,
        )
    raise ValueError(f"Unsupported degradation mode: {mode}")


def make_run_audit(
    base_df: pd.DataFrame,
    degraded_df: pd.DataFrame,
    frame_audit_df: pd.DataFrame,
    mode: str,
    level_pct: int,
    seed: int,
    extra_meta: Dict[str, float | int | str],
) -> Dict[str, float | int | str]:
    base_eligible = camera_eligible_mask(base_df)
    deg_eligible = camera_eligible_mask(degraded_df)

    b_dyn = int(base_df["coarse_class"].isin(DYNAMIC_CLASSES).sum())
    d_dyn = int(degraded_df["coarse_class"].isin(DYNAMIC_CLASSES).sum())
    b_ped = int((base_df["coarse_class"] == "pedestrian").sum())
    d_ped = int((degraded_df["coarse_class"] == "pedestrian").sum())
    b_veh = int((base_df["coarse_class"] == "vehicle").sum())
    d_veh = int((degraded_df["coarse_class"] == "vehicle").sum())

    audit: Dict[str, float | int | str] = {
        "degradation_mode": mode,
        "degradation_level_pct": int(level_pct),
        "seed": int(seed),
        "rows_total_before": int(len(base_df)),
        "rows_total_after": int(len(degraded_df)),
        "dynamic_rows_before": b_dyn,
        "dynamic_rows_after": d_dyn,
        "ped_rows_before": b_ped,
        "ped_rows_after": d_ped,
        "veh_rows_before": b_veh,
        "veh_rows_after": d_veh,
        "camera_eligible_rows_before": int(base_eligible.sum()),
        "camera_eligible_rows_after": int(deg_eligible.sum()),
        "actual_camera_drop_fraction": float((base_eligible.sum() - deg_eligible.sum()) / base_eligible.sum()) if base_eligible.sum() > 0 else 0.0,
        "frames_with_any_camera_eligible_rows_before": int((frame_audit_df["camera_eligible_rows_before"] > 0).sum()),
        "frames_with_all_camera_eligible_rows_lost": int(
            ((frame_audit_df["camera_eligible_rows_before"] > 0) & (frame_audit_df["camera_eligible_rows_after"] == 0)).sum()
        ),
        "mean_frame_camera_drop_fraction": float(frame_audit_df["camera_drop_fraction"].mean()) if len(frame_audit_df) else 0.0,
        "mean_drop_probability_across_frames": float(frame_audit_df["mean_drop_probability"].mean()) if len(frame_audit_df) else 0.0,
    }
    audit.update(extra_meta)
    return audit


def write_formula_summary(path: Path) -> None:
    summary = {
        "camera_only_degradation": True,
        "eligibility_rule": "Only rows with camera_degradable_flag=True and degradation_include_flag=True are degraded; if these columns are absent, fallback is dynamic classes only.",
        "random_dropout": {
            "formula": "p_drop = severity * visibility_multiplier * class_multiplier",
            "notes": "Independent per eligible camera object per frame. Visibility adjusts fragility; no scene-wide spatial structure."
        },
        "distance_based_dropout": {
            "formula": "p_drop = severity * distance_factor * visibility_multiplier * lateral_multiplier * class_multiplier",
            "distance_factor": "clip(distance_to_ego_m / distance_ref_m, 0.10, 1.00)",
            "notes": "Farther, off-axis, and poorer-visibility objects are more likely to disappear."
        },
        "burst_dropout": {
            "location_rule": "Burst interval is centered around refined crossing_start_frame to crossing_end_frame with severity-scaled temporal buffer; seed controls exact start within that crossing-focused zone.",
            "inside_formula": "p_drop = (0.45 + 0.55*severity) * (0.60 + 0.40*distance_factor) * visibility_multiplier * lateral_multiplier * class_multiplier",
            "outside_formula": "p_drop = 0.20 * [severity * visibility_multiplier * class_multiplier]",
            "notes": "Represents temporary camera/detector instability near the safety-critical crossing interaction."
        },
        "visibility_multipliers": {
            "good": 0.80,
            "fair": 1.00,
            "poor": 1.25,
            "unknown": 1.10
        },
        "lateral_multipliers": {
            "|lat|<=1.5m": 1.00,
            "1.5m<|lat|<=3.0m": 1.15,
            "|lat|>3.0m": 1.30
        },
        "class_multipliers": {
            "pedestrian": 1.05,
            "vehicle": 1.00
        }
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    scene_name = args.scene_name
    levels = parse_int_list_csv(args.degradation_levels)
    seeds = parse_int_list_csv(args.seeds)
    modes = parse_str_list_csv(args.degradation_modes)

    bad_modes = [m for m in modes if m not in SUPPORTED_MODES]
    if bad_modes:
        raise ValueError(
            f"Unsupported degradation modes: {bad_modes}. Supported: {sorted(SUPPORTED_MODES)}"
        )

    refined_scene_dir = Path(args.refined_root) / scene_name
    result_scene_dir = Path(args.results_root) / scene_name
    degraded_root = result_scene_dir / "degraded_inputs"
    degraded_root.mkdir(parents=True, exist_ok=True)

    data = load_refined_scene(refined_scene_dir)
    objects_df = data["objects_df"]  # type: ignore[index]
    scene_summary = data["scene_summary"]  # type: ignore[index]

    all_run_audits: List[Dict[str, float | int | str]] = []

    for mode in modes:
        for level in levels:
            for seed in seeds:
                run_dir = degraded_root / mode / f"degradation_{level:02d}" / f"seed_{seed:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                degraded_df, frame_audit_df, extra_meta = degrade_objects(
                    objects_df=objects_df,  # type: ignore[arg-type]
                    scene_summary=scene_summary,  # type: ignore[arg-type]
                    mode=mode,
                    level_pct=level,
                    seed=seed,
                    distance_ref_m=args.distance_ref_m,
                    base_extra_buffer=args.burst_base_extra_buffer,
                    realism_guard=bool(args.keep_at_least_one_close_object_under_30),
                )

                degraded_path = run_dir / "objects_timeline_degraded.csv"
                degraded_df.to_csv(degraded_path, index=False)

                frame_audit_path = run_dir / "degradation_audit_per_frame.csv"
                frame_audit_df.to_csv(frame_audit_path, index=False)

                run_audit = make_run_audit(
                    base_df=objects_df,  # type: ignore[arg-type]
                    degraded_df=degraded_df,
                    frame_audit_df=frame_audit_df,
                    mode=mode,
                    level_pct=level,
                    seed=seed,
                    extra_meta=extra_meta,
                )

                audit_path = run_dir / "degradation_audit.json"
                audit_path.write_text(json.dumps(run_audit, indent=2), encoding="utf-8")
                all_run_audits.append(run_audit)

                print(
                    f"[ok] mode={mode} level={level:02d}% seed={seed:03d} rows={len(degraded_df)}"
                )

    summary_df = pd.DataFrame(all_run_audits)
    if len(summary_df):
        summary_df = summary_df.sort_values(
            ["degradation_mode", "degradation_level_pct", "seed"]
        ).reset_index(drop=True)
    summary_df.to_csv(degraded_root / "degradation_audit_per_run.csv", index=False)

    write_formula_summary(degraded_root / "degradation_formula_summary.json")

    print(f"[done] degraded inputs written to {degraded_root}")


if __name__ == "__main__":
    main()
