# Data and reproducibility

## Required upstream inputs

The pipeline reads nuScenes metadata and trajectory/sensor records, plus OpenStreetMap context. The original nuScenes data files are **not bundled** with this source-code package. Obtain the appropriate dataset releases under their own terms. Source files expect the dataset root through `NUSCENES_DATAROOT`; the multi-geometry scripts may use both `v1.0-mini` and `v1.0-trainval`, depending on the scene.

The scene-selector creates configuration JSON under `src/config/three_geometry_candidates/`; the multi-scene extractor reads `selected_tested_scenes.json` (or a fallback file). The selected configurations are included under `src/config/`; machine-specific dataset paths have been cleared, so set the `NUSCENES_DATAROOT` environment variable. Three scenes documented for the experiments are `scene-1019`, `scene-0061`, and `scene-0383`.

## Output contracts

With the shell's current working directory set to `src/`, script defaults generally write beneath `src/data/`:

| Directory | Purpose |
|---|---|
| `data/raw/<scene>/` | NuScenes extraction and timeline metadata |
| `data/refined/<scene>/` | Controlled pedestrian-crossing scene tables and metadata |
| `data/idm_calibration/<scene>/` | Shared baseline/OSM-aware IDM parameter calibration |
| `data/planner_perception_only/<scene>/` | Camera degradation inputs, baseline rollouts and metrics |
| `data/osm/<scene>/` | OpenStreetMap context and shifted crossing-prior cases |
| `data/planner_osm_aware/<scene>/` | OSM-aware runs and metrics |
| `data/learned_horizon_policy/<scene>/` | Policy JSON, training samples, diagnostics |
| `data/planner_comparison_paper/` | Paired comparison CSVs, figure exports, summaries |

The `src/data/` tree is ignored by Git because it can contain large extracted datasets and trajectory-level products. Curated, source-generated JSON/CSV metrics and policy files are included under `results/experiment_data/` for inspection. These do **not** replace the raw nuScenes scenes or every intermediate file; see [Experiment artifacts](EXPERIMENT_ARTIFACTS.md). Avoid exposing raw nuScenes content or private data without checking distribution conditions.

## Two-pass learned-policy process

The OSM-aware script can load a trained JSON policy when it exists; otherwise it falls back to a bounded rule-based horizon. This makes the following sequence necessary for a fresh training run:

1. Prepare scene refinements, perception-degraded observations, IDM calibration and OSM error cases.
2. Run the perception-only baseline.
3. Run the OSM-aware planner **without** the learned policy to create training rollouts.
4. Train the adaptive-horizon policy using the resulting baseline and OSM-aware outcomes.
5. Rerun the OSM-aware planner using the policy JSON, then compare paired results.

The provided scripts were syntax-checked, but the complete workflow was **not executed** in this review environment. The package contains scene configurations and selected experiment outputs, but lacks the full nuScenes dataset and complete raw/intermediate outputs.

## Dataset and method constraints

- The crossing interactions are controlled/refined variants of nuScenes scenes; they must not be described as unmodified real-world ground truth.
- Perception degradation and route-aligned OSM offsets are experimental perturbations, not independently measured sensor failures or certified map accuracy values.
- The planner controls longitudinal motion via IDM; it does not constitute a full production-grade autonomous-driving stack or validate lateral planning in traffic.
- Quantitative tables in the README are taken from the supplied research materials. They have not been independently reproduced from this archive.
- Use appropriate attribution for nuScenes and OpenStreetMap and follow their source-data terms.
