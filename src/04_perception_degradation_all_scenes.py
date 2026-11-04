#!/usr/bin/env python3
"""
Batch wrapper for 04_perception_degradation.py.

It keeps the original Script 04 logic unchanged, but adds safe batch handling:
  --scene-name all
  --scene-name scene-1019,scene-0061,scene-0383
  --scene-names scene-1019,scene-0061,scene-0383

Place this file beside 04_perception_degradation.py and run it from the src folder.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

DEFAULT_BATCH_SCENES = ["scene-1019", "scene-0061", "scene-0383"]


def normalize_scene_name_token(token: str) -> str:
    t = str(token).strip()
    if not t:
        return ""
    if t.lower() in {"all", "three", "selected"}:
        return "all"
    if t.startswith("scene-"):
        return t
    digits = "".join(ch for ch in t if ch.isdigit())
    if digits:
        return f"scene-{digits.zfill(4)}"
    return t


def parse_scene_list(scene_name_text: str, scene_names_text: str) -> List[str]:
    primary = str(scene_name_text or "").strip()
    if primary.lower() in {"", "all", "three", "selected"}:
        raw_items = [x.strip() for x in str(scene_names_text or "").split(",") if x.strip()]
    else:
        raw_items = [x.strip() for x in primary.split(",") if x.strip()]

    out: List[str] = []
    for item in raw_items:
        norm = normalize_scene_name_token(item)
        if norm == "all":
            out.extend(DEFAULT_BATCH_SCENES)
        elif norm:
            out.append(norm)

    if not out:
        out = list(DEFAULT_BATCH_SCENES)

    final: List[str] = []
    seen = set()
    for scene in out:
        if scene not in seen:
            final.append(scene)
            seen.add(scene)
    return final


def parse_wrapper_args(argv: List[str]) -> tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(
        description="Batch-safe launcher for 04_perception_degradation.py",
        add_help=True,
    )
    p.add_argument("--scene-name", default="all", help="Scene, comma-separated scenes, or all.")
    p.add_argument("--scene-names", default=",".join(DEFAULT_BATCH_SCENES), help="Scene list used when --scene-name all.")
    p.add_argument("--base-script", default="04_perception_degradation.py", help="Original Script 04 filename/path.")
    p.add_argument("--skip-missing", action="store_true", help="Continue to the next scene if one scene fails.")
    args, remaining = p.parse_known_args(argv)
    return args, remaining


def main() -> None:
    args, remaining = parse_wrapper_args(sys.argv[1:])
    scenes = parse_scene_list(args.scene_name, args.scene_names)
    base_script = Path(args.base_script)
    if not base_script.is_absolute():
        base_script = Path(__file__).resolve().parent / base_script
    if not base_script.is_file():
        raise FileNotFoundError(f"Cannot find original Script 04: {base_script}")

    print("batch_perception_degradation_start")
    print(f"  scenes: {scenes}")
    print(f"  base_script: {base_script}")
    failures = []
    for scene in scenes:
        cmd = [sys.executable, str(base_script), "--scene-name", scene] + remaining
        print("\n" + "=" * 88)
        print(f"[script04] running {scene}")
        print(" ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((scene, result.returncode))
            if not args.skip_missing:
                raise SystemExit(result.returncode)
    if failures:
        print(f"batch_perception_degradation_done_with_failures: {failures}")
        raise SystemExit(1)
    print("batch_perception_degradation_complete")


if __name__ == "__main__":
    main()
