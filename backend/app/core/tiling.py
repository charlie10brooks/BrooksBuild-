"""Pixel-window and tile-grid arithmetic. Pure Python/NumPy, no framework deps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Window:
    """Half-open pixel window [x0, x1) x [y0, y1) in raster (column, row) space."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"invalid window {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    def intersect(self, other: "Window") -> Optional["Window"]:
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return Window(x0, y0, x1, y1)

    def clip(self, width: int, height: int) -> Optional["Window"]:
        return self.intersect(Window(0, 0, width, height))

    def translate(self, dx: int, dy: int) -> "Window":
        return Window(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def pad(self, n: int) -> "Window":
        return Window(self.x0 - n, self.y0 - n, self.x1 + n, self.y1 + n)

    def slices(self) -> Tuple[slice, slice]:
        return (slice(self.y0, self.y1), slice(self.x0, self.x1))

    def union(self, other: "Window") -> "Window":
        return Window(min(self.x0, other.x0), min(self.y0, other.y0),
                      max(self.x1, other.x1), max(self.y1, other.y1))

    def as_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


def tile_origins(length: int, tile: int, stride: int) -> List[int]:
    """Start offsets along one axis. The last tile is shifted back so it ends on the edge."""
    if tile <= 0 or stride <= 0:
        raise ValueError("tile and stride must be positive")
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, stride))
    if origins[-1] + tile < length:
        origins.append(length - tile)
    return origins


def axis_segments(length: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Per-axis (tile_start, tile_end, keep_start, keep_end).

    ``keep`` ranges partition [0, length) exactly: each seam sits at the midpoint of the
    overlap between neighbouring tiles, which is where predictions are least affected
    by missing context. Every keep range lies inside its tile.
    """
    if not 0 <= overlap < tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")
    origins = tile_origins(length, tile, tile - overlap)
    bounds = []
    for a, b in zip(origins[:-1], origins[1:]):
        bounds.append((b + (a + tile)) // 2)
    out = []
    for i, o in enumerate(origins):
        start = 0 if i == 0 else bounds[i - 1]
        end = length if i == len(origins) - 1 else bounds[i]
        out.append((o, min(o + tile, length), start, end))
    return out


@dataclass(frozen=True)
class TileSpec:
    window: Window  # pixels the model sees (includes overlap context)
    keep: Window    # pixels this tile is responsible for in the merged output


def iter_tiles(width: int, height: int, tile: int, overlap: int = 0) -> Iterator[TileSpec]:
    """Row-major overlapping tiles with non-overlapping ``keep`` regions."""
    ys = axis_segments(height, tile, overlap)
    xs = axis_segments(width, tile, overlap)
    for (ty0, ty1, ky0, ky1) in ys:
        for (tx0, tx1, kx0, kx1) in xs:
            yield TileSpec(Window(tx0, ty0, tx1, ty1), Window(kx0, ky0, kx1, ky1))


def blend_weights(h: int, w: int, overlap: int) -> np.ndarray:
    """2-D feathering weights: linear ramp across ``overlap`` pixels at every edge.

    Used with weight-normalised accumulation, so image borders (covered by one tile) are
    unaffected by the ramp.
    """
    ramp = max(int(overlap), 1)

    def axis(n: int) -> np.ndarray:
        d = np.minimum(np.arange(n), np.arange(n)[::-1]).astype(np.float32) + 1.0
        return np.minimum(1.0, d / ramp)

    return np.outer(axis(h), axis(w)).astype(np.float32)
