# Terrain reconstruction and planning

## The five things that are never mixed

1. **Land cover**: visual classes on the imagery. Never an input to elevation values.
2. **Observed surface (DSM)**: the highest observed point per cell (canopy tops, roofs, ground where visible). Cells with no observation stay empty; gaps are **not** filled.
3. **Classified ground points**: points a ground filter judged to be bare earth. (No filter can see through dense canopy; it only labels which *observed* points look like ground.)
4. **Estimated bare earth (DTM)**: measured where ground points exist, interpolated elsewhere within a search radius. Cells with no evidence within the radius stay empty (no global fill, no flattening).
5. **Insufficient-evidence areas**: the confidence and evidence layers.

These are separate files, separate API layers, and are labelled **OBSERVED** / **ESTIMATED** wherever they appear (map legends, inspector, 3D labels, exports' file names `dsm_OBSERVED_surface.tif` / `dtm_ESTIMATED_bare_earth.tif`).

## Pipeline (`services/terrain_job.py`, `terrain_pipeline.py`, `groundfilter.py`, `core/dtm.py`)

1. **Point cloud** from photogrammetry (ODM). Statistical outlier removal.
2. **Ground filter**: `pdal_smrf` (default), `pdal_pmf`, or `builtin_pmf` (Zhang et al. progressive morphological filter, numpy/scipy). PDAL engines fail with install instructions if PDAL is missing; there is **never** a silent substitution. `builtin_pmf` must be chosen explicitly and shaves terrain features wider than its window; prefer SMRF for production.
3. **DSM**: max z per cell of non-noise points. **DTM**: measured cells keep their measured value; others are interpolated (`local_plane` default, `idw`, `tin`, `tps`, `nearest`) within `max_radius_m`.
4. **Supplemental sources** (LiDAR, external DEM, survey points) are merged by **priority** (higher wins), with a per-source `z_offset` (datum shift), `trust` and enable switch. The evidence raster records which cell came from where.
5. **Safety check with land cover**: photogrammetric "ground" cells under mapped canopy whose surface lies within `canopy_clearance_m` of the DSM are probably canopy mislabelled as ground; they are demoted from *measured* to *estimated* (lowering their confidence). This is the only use of the class mask, and it can only make results *less* confident.
6. **Cross-validation**: block hold-out of measured ground gives RMSE and bias by distance-to-nearest-ground. It measures error where ground was observed; it says nothing about bias where ground was never observed (the report says so).

### Evidence raster

0 none · 1 measured ground · 2 interpolated · 3 external model.

### Confidence raster

Heuristic, evidence-based (not a statistical interval): by distance to measured ground (inflated on steep slopes) and ground-point density. Defaults: **high** ≤ 2 m and dense, **moderate** ≤ 6 m, **low** ≤ 15 m, **unknown** beyond / no data. Thresholds are job parameters. When a large share of the site is low or unknown, the summary
states that **a supplemental survey (total station / GNSS rover) or LiDAR is advised** and the UI repeats it.

## Planning tools (`core/planning.py`, `core/terrain_analysis.py`)

* **Inspect**: DSM, DTM, object height, evidence, confidence, source, visual class at a point; OBSERVED / ESTIMATED labelled.
* **Profile**: estimated ground and observed surface along a line with per-sample confidence; CSV export carries the disclaimer; vertical exaggeration is shown.
* **Footprint report**: ground min/mean/max, relief, percentiles, best-fit plane and downslope direction, mean/max slope, steep fraction, confidence mix, coverage, corner elevations, warnings.
* **Cut / fill** against a flat pad, a sloped plane, or a plane through three points. It **refuses** (and says why) when under 95 % of the polygon has an estimate or over 25 % rests on low/unknown ground. Otherwise it returns cut, fill, net, by-confidence split and a worst-case volume uncertainty from the cross-validated RMSE.
* **Steep slopes** above a threshold; **drainage** (D8 flow paths and filled depressions) from the *estimated* surface, labelled indicative under canopy.
* **CAD export** (DXF R12): boundary, footprints, profiles, measurements, control points, and contours of the *estimated* DTM (lightly smoothed). Contours are split onto `TS_CONTOUR_EST_*` and `TS_CONTOUR_LOWCONF_*` layers by terrain confidence; a visible "PRELIMINARY PLANNING ESTIMATE" note is drawn in the file.

Every planning output is labelled **preliminary planning estimate**: not for final foundation design, permitting or construction staking without validation by a licensed surveyor / engineer.

## Editing the 3D point cloud

In **Terrain 3D → Edit points** you can lasso or box-select points (selection runs through the whole cloud along the view direction; un-ticked classes are protected) and set their class (ground, vegetation classes, building, noise, unclassified), with persistent undo/redo.
Edits are stored on the server (`terrain/cloud_edits/`, an operation log plus a per-point override) and survive reloads.

Important limits, stated in the UI too: the edits apply to the **display sample** (about 1.5 M points; the full photogrammetry cloud is never sent to the browser), not to the full cloud, and they do **not** change the DSM, DTM, confidence or evidence rasters — re-running terrain reconstruction starts again from the full cloud.
If the display sample is rebuilt, old edits are not applied to it (the point order would not match). *Export → Edited point cloud (LAS, display SAMPLE)* writes the sample with the edited classes. Feeding manual ground marks back into the terrain run is not implemented.

## Known limits

* The whole pipeline has been exercised only on synthetic data and with `builtin_pmf`; PDAL and ODM have not been run.
* A camera cannot see soil under dense canopy. In such areas the DTM is an interpolation from the nearest visible ground and the confidence layer says so.
* Slope-adaptive behaviour of `builtin_pmf` is limited; steep, wide terrain can be shaved.
* No vertical datum transformations (geoid models); `z_offset` on a source is a manual shift.
* Drainage and cut/fill assume the DTM is right; their uncertainty is only as good as the cross-validation.
