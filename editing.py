"""Applies classification edit operations to a TiledRaster mask.

Edits only ever touch the categorical mask. The orthomosaic is read-only, and nothing here
touches elevation data: reclassifying pixels is *visual interpretation*, never terrain evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from . import regions
from .rasterstore import TiledRaster
from .tiling import Window


class OrthoReader(Protocol):
    width: int
    height: int

    def read_rgb(self, window: Window) -> np.ndarray:
        """uint8 (H, W, 3) for ``window`` (must lie inside the raster)."""


class ArrayOrthoReader:
    """In-memory reader; used by tests and the synthetic development sample."""

    def __init__(self, rgb: np.ndarray):
        self.rgb = rgb
        self.height, self.width = rgb.shape[:2]

    def read_rgb(self, window: Window) -> np.ndarray:
        return self.rgb[window.slices()][..., :3]


OP_TYPES = {"brush", "polygon", "lasso", "rect", "fill", "similar"}
DEFAULT_ANALYSIS_SIDE = 2048      # fill/similar search window around the seed, pixels
MAX_ANALYSIS_PIXELS = 16_000_000  # hard server-side guard


class EditError(ValueError):
    pass


@dataclass
class EditOp:
    type: str
    class_id: int
    points: List[Tuple[float, float]] = field(default_factory=list)
    radius: float = 8.0
    tolerance: float = 12.0
    connectivity: int = 4
    rect: Optional[Tuple[float, float, float, float]] = None
    analysis_window: Optional[Window] = None
    only_classes: Optional[Sequence[int]] = None      # modify only pixels currently in these classes
    protect_classes: Optional[Sequence[int]] = None   # never modify pixels currently in these classes
    # tidy-up applied to polygon/fill/similar selections
    min_area: int = 0
    fill_holes: bool = False
    smooth_radius: int = 0


@dataclass
class EditResult:
    window: Optional[Window]
    changed_pixels: int
    truncated: bool = False
    message: str = ""


def _validate(op: EditOp, mask: TiledRaster) -> None:
    if op.type not in OP_TYPES:
        raise EditError(f"unknown operation {op.type!r}")
    if not 0 <= int(op.class_id) <= 255:
        raise EditError("class_id must be 0..255")
    if op.type in ("polygon", "lasso") and len(op.points) < 3:
        raise EditError("polygon/lasso needs at least 3 points")
    if op.type in ("brush", "fill", "similar") and not op.points:
        raise EditError("operation needs at least one point")
    if op.type == "rect" and op.rect is None:
        raise EditError("rect operation needs rect=(x0,y0,x1,y1)")
    if op.type == "brush" and not (0 < op.radius <= 2048):
        raise EditError("brush radius must be in (0, 2048]")
    if not (0 <= op.tolerance <= 441):
        raise EditError("tolerance must be in [0, 441]")
    if len(op.points) > 200_000:
        raise EditError("too many points")


def _bounds(points: Sequence[Tuple[float, float]], pad: float, mask: TiledRaster) -> Optional[Window]:
    p = np.asarray(points, dtype=float)
    w = Window(int(np.floor(p[:, 0].min() - pad)), int(np.floor(p[:, 1].min() - pad)),
               int(np.ceil(p[:, 0].max() + pad)) + 1, int(np.ceil(p[:, 1].max() + pad)) + 1)
    return w.clip(mask.width, mask.height)


def compute_selection(op: EditOp, mask: TiledRaster, ortho: Optional[OrthoReader],
                      class_mask: int = 0xFF) -> Tuple[Optional[Window], Optional[np.ndarray], bool]:
    """Return (window, selection mask within window, truncated).

    ``class_mask`` selects the class bits of stored values (the rest may carry provenance tags)."""
    _validate(op, mask)
    truncated = False
    if op.type == "brush":
        win = _bounds(op.points, op.radius + 1, mask)
        if win is None:
            return None, None, False
        sel = regions.brush_mask(op.points, op.radius, win)
    elif op.type in ("polygon", "lasso"):
        win = _bounds(op.points, 1, mask)
        if win is None:
            return None, None, False
        if win.area > MAX_ANALYSIS_PIXELS * 4:
            raise EditError("selection too large; reduce the area")
        sel = regions.polygon_mask(op.points, win)
    elif op.type == "rect":
        x0, y0, x1, y1 = op.rect  # type: ignore[misc]
        win = Window(int(np.floor(min(x0, x1))), int(np.floor(min(y0, y1))),
                     int(np.ceil(max(x0, x1))), int(np.ceil(max(y0, y1)))).clip(mask.width, mask.height)
        if win is None:
            return None, None, False
        if win.area > MAX_ANALYSIS_PIXELS * 4:
            raise EditError("selection too large; reduce the area")
        sel = regions.rect_mask(x0, y0, x1, y1, win)
    else:  # fill / similar need real pixels
        if ortho is None:
            raise EditError("flood fill and select-by-color need the orthomosaic, which is not available yet")
        sx, sy = op.points[0]
        ix, iy = int(np.floor(sx)), int(np.floor(sy))
        if not (0 <= ix < mask.width and 0 <= iy < mask.height):
            raise EditError("seed outside the raster")
        if op.analysis_window is not None:
            win = op.analysis_window.clip(mask.width, mask.height)
            if win is None or not (win.x0 <= ix < win.x1 and win.y0 <= iy < win.y1):
                raise EditError("seed outside analysis window")
        else:
            half = DEFAULT_ANALYSIS_SIDE // 2
            win = Window(ix - half, iy - half, ix + half, iy + half).clip(mask.width, mask.height)
        assert win is not None
        if win.area > MAX_ANALYSIS_PIXELS:
            raise EditError(f"analysis window exceeds {MAX_ANALYSIS_PIXELS:,} pixels; zoom in or shrink it")
        rgb = ortho.read_rgb(win)
        current = mask.read(win) & class_mask
        restrict = None
        if op.only_classes is not None:
            restrict = np.isin(current, list(op.only_classes))
        seed_rc = (iy - win.y0, ix - win.x0)
        if op.type == "fill":
            sel, truncated = regions.flood_select(rgb, seed_rc, op.tolerance, op.connectivity, restrict)
        else:
            sel = regions.similar_mask(rgb, seed_rc, op.tolerance, restrict)
            truncated = False
    if op.min_area or op.fill_holes or op.smooth_radius:
        sel = regions.clean_mask(sel, op.min_area, op.fill_holes, op.smooth_radius)
    return win, sel, truncated


def apply_edit(op: EditOp, mask: TiledRaster, ortho: Optional[OrthoReader] = None,
               label: Optional[str] = None, class_mask: int = 0xFF, tag: int = 0) -> EditResult:
    """Compute the selection and write the class into the mask as one undoable edit.

    Stored value = ``class_id | tag``. ``class_mask`` isolates the class bits for comparisons (``only_classes``,
    ``protect_classes``); ``tag`` marks provenance in the remaining bits (e.g. 'manual edit'). A pixel is rewritten if
    its stored value differs, so re-painting an auto-classified pixel with the same class still marks it manual.
    """
    win, sel, truncated = compute_selection(op, mask, ortho, class_mask)
    if win is None or sel is None:
        return EditResult(None, 0, False, "selection is outside the raster")
    if int(op.class_id) & ~class_mask:
        raise EditError(f"class_id {op.class_id} does not fit in class bits {class_mask:#x}")
    current = mask.read(win)
    cur_cls = current & class_mask
    target = sel
    if op.only_classes is not None and op.type not in ("fill", "similar"):
        target = target & np.isin(cur_cls, list(op.only_classes))
    if op.protect_classes:
        target = target & ~np.isin(cur_cls, list(op.protect_classes))
    value = int(op.class_id) | int(tag)
    target = target & (current != value)
    n = int(target.sum())
    if n:
        with mask.edit(label or f"{op.type} -> class {op.class_id}"):
            mask.paint(win, target, value)
    msg = "fill reached the analysis-window edge and may continue; zoom out or enlarge the window" if truncated else ""
    return EditResult(win if n else None, n, truncated, msg)
