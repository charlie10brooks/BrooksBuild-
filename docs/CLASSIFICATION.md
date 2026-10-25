# Land-cover classification

**Classification is a visual interpretation of the orthomosaic. It is not elevation evidence.** Nothing in this subsystem writes to, or feeds values into, any elevation raster.

## Classes

Ten defaults, editable per project (name, colour, and an "obscures ground" flag): 0 Unknown / unclassified, 1 Trees and canopy, 2 Brush and shrubs, 3 Grass and low vegetation,
4 Bare soil, 5 Gravel and rock, 6 Buildings and structures, 7 Roads and pavement, 8 Water, 9 Other objects. Up to 64 classes (ids 0-63). The "obscures ground" flag is used *only* by the terrain
step as a safety check (see `TERRAIN.md`).

## Storage: the mask is separate from the imagery

The orthomosaic is read-only. The mask is a tiled sparse raster (`core/rasterstore.py`): one value per pixel, **low 6 bits = class id, bit 6 (0x40) = set manually, bit 7 (0x80) = set automatically**.
Missing tiles read as unknown, so an untouched mask costs almost nothing. Each edit records before/after patches for undo/redo (the imagery is never rewritten), class histograms update incrementally,
writes are serialised by a lock, and an interrupted commit is detected and repaired on open. Autosave is immediate: every edit is persisted before the API responds, and the status bar shows saved / saving / error.

Mask tiles are served as PNG with **R = class, G = provenance (0 none, 1 manual, 2 automatic), B = confidence 0-100**, at levels 0-2 plus an overview, and coloured in the browser — so class colours, hiding classes, opacity,
provenance view and confidence view are all instant and need no re-render.

## Tools

Sample (pixel colour, class, provenance, confidence, similar-region size) · brush · polygon · rectangle · lasso · flood fill with colour tolerance (4/8-connected) · select-similar (every pixel within tolerance in a search window) ·
reclassify (class A → B in view or whole image) · erase to unknown. Fill and select-similar **preview** the selection and apply on confirmation. Every selection tool shares *tidy* options (drop small regions, fill holes, smooth edges) and
*restrict* / *protect* class lists ("only change pixels that are currently…", "never change pixels that are…"). Zoom, pan, pixel inspect, class visibility, overlay opacity, side-by-side view, live legend statistics.

## Automatic classification

Models implement a small interface and plug into one tiled, overlapped inference loop (tile size and context overlap come from the model; results are blended so tile seams do not show). Built in:

| Model | What it is | Limits |
| --- | --- | --- |
| `heuristic` | Colour/texture rules. **Not a trained model.** | Only vegetation (trees/brush/grass), bare soil, rock, road, water. Leaves anything uncertain *unclassified*. Cannot recognise buildings, species, or what is under canopy. |
| `learned` | Multinomial logistic regression trained from **this project's manual corrections** ("Learn from my corrections"). Accuracy is measured on held-out tiles and reported. | Only as good and as general as your labels; one project. |
| `onnx` | Any ONNX semantic-segmentation network you supply (`TERRASCOPE_SEGMENTATION_MODEL` + a JSON class map). | **No pretrained network ships with this repo.** Never tested with a real network. |

Rules: automatic labels get `TAG_AUTO` and a per-pixel confidence; manual edits are never overwritten unless you pick *everything* as the overwrite scope; a snapshot of the mask is taken before every automatic run and can be restored;
results report counts and the share left unclassified. Automatic output is a starting point for review, never a finished classification, and the UI says so.

## Classifying the uploaded drone photos

After drone images are uploaded and ingested, the **Upload** view offers *Classify drone photos* (all photos, or one batch; heuristic / learned / ONNX, same models as above).
A job (`classify_images`) reads each photo at a reduced working size (default 1600 px long side), runs the tiled model, and stores a per-photo mask (`imgclass/<image_id>.png`, R = class, B = confidence) plus class fractions in the `image_classification` table.
The gallery shows thumbnails with the class overlay; opening a photo gives opacity, per-class show/hide and percentages. `photo_classification_stats.csv` can be exported.

What it is not: the labels are in each photo's own pixels. Photos are not orthorectified, so they are **not georeferenced** and they do **not** feed the orthomosaic mask or any elevation product.
The heuristic assumes roughly nadir photos (sky, haze or oblique views are misread: blue sky looks like water to colour rules). There is no per-photo manual editor yet; to correct labels, use the orthomosaic classification editor.
