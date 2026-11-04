# Experiment artifacts and provenance

This repository includes a **curated, read-only extract** of the original experiment outputs supplied by the developer. These records support quantitative inspection without committing all ~46,000 archived files or redistributing original nuScenes datasets.

## Scenes and configurations

The three selected-scene configuration files are in [`../src/config/selected_scene_configs/`](../src/config/selected_scene_configs/):

| Scene | Geometry | nuScenes split |
|---|---|---|
| `scene-1019` | Straight approach | `v1.0-trainval` |
| `scene-0061` | Moderate curvature | `v1.0-mini` |
| `scene-0383` | High-curvature / circular | `v1.0-trainval` |

The rest of `src/config/` retains the scene-selection artifacts, including candidate lists and tested-scene records. Any former machine-specific `C:\Users\...` locations in exported JSON/CSV were replaced by empty strings. Set `NUSCENES_DATAROOT` to point to the appropriate local dataset root; the scrubbed settings are not a substitute for installing nuScenes.

## Data layout

Selected outputs from the source archive's `data/` directory are stored under `results/experiment_data/`:

| Directory | Contents |
|---|---|
| [`idm_calibration/`](../results/experiment_data/idm_calibration/) | Scene-specific IDM parameters and calibration summaries |
| [`learned_horizon_policy/`](../results/experiment_data/learned_horizon_policy/) | Bounded policy JSONs, training records, predictions, plots and fit diagnostics |
| [`planner_perception_only/`](../results/experiment_data/planner_perception_only/) | Per-run baseline measurements and batch summaries |
| [`planner_osm_aware_rulebased/`](../results/experiment_data/planner_osm_aware_rulebased/) | OSM-aware bootstrap runs using the rule-based prediction horizon |
| [`planner_osm_aware/`](../results/experiment_data/planner_osm_aware/) | OSM-aware runs using the learned horizon policy |
| [`planner_comparison/`](../results/experiment_data/planner_comparison/) | Paired experimental metrics, geometry-level summaries, trade-offs and results metadata |
| [`osm/`](../results/experiment_data/osm/) | Selected batch-level crossing-offset and map-prior summaries |

In the source experiment, intermediate data lived under `src/data/` and the comparison script used a `data/planner_comparison_paper/` output directory. Here those records have been grouped in `results/experiment_data/` for public inspection and to avoid accidentally committing large generated files; the original contents and keys of the selected numeric tables are retained. **Scripts still write to their original `src/data/` defaults** if rerun.

## Summary metric traceability

The README's mean minimum pedestrian TTC values are rounded from [`geometry_summary_key_metrics.csv`](../results/experiment_data/planner_comparison/tables/geometry_summary_key_metrics.csv). The detailed paired rows are in [`all_paired_key_metrics.csv`](../results/experiment_data/planner_comparison/tables/all_paired_key_metrics.csv). These are historical output artifacts, not a fresh independent reproduction.

The paired comparisons are specific to the described experiment conditions and depend on the chosen definitions of TTC, OSM perturbations and controlled scene refinements. The study is not a general-purpose autonomous-driving safety certification or on-road validation.

## Reproducing the experiment

To rerun the full pipeline, download the required nuScenes `v1.0-mini` and `v1.0-trainval` data under their applicable licenses, point `NUSCENES_DATAROOT` to the local dataset, and run the documented pipeline from `src/`. OpenStreetMap extraction may require network access and may return changed map data over time. The bundled selected summaries and policies allow the previously generated outputs to be inspected but do **not** guarantee that the full experiment will rerun identically without the original datasets, map snapshots and software environment.

The source package's complete trajectory-level output archive was deliberately not copied into this public-style repository draft (the supplied file included approximately 46,000 ZIP entries and ~1.0 GB of uncompressed content). Retain the original archive privately for full experiment provenance.
