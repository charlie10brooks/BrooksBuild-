"""Default land-cover classes. Ids are stable and stored in the mask raster (uint8)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np

UNKNOWN = 0
TREES = 1
BRUSH = 2
GRASS = 3
SOIL = 4
ROCK = 5
BUILDING = 6
ROAD = 7
WATER = 8
OTHER = 9


@dataclass
class LandCoverClass:
    id: int
    key: str
    name: str
    color: str          # #RRGGBB, user-configurable per project
    vegetation: bool    # true for classes that obscure ground in the DSM
    visible: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CLASSES: List[LandCoverClass] = [
    LandCoverClass(UNKNOWN, "unknown", "Unknown / unclassified", "#9aa0a6", False),
    LandCoverClass(TREES, "trees", "Trees and canopy", "#1b5e20", True),
    LandCoverClass(BRUSH, "brush", "Brush and shrubs", "#7cb342", True),
    LandCoverClass(GRASS, "grass", "Grass and low vegetation", "#d4e157", True),
    LandCoverClass(SOIL, "soil", "Bare soil", "#8d6e63", False),
    LandCoverClass(ROCK, "rock", "Gravel and rock", "#b0a99f", False),
    LandCoverClass(BUILDING, "building", "Buildings and structures", "#d32f2f", False),
    LandCoverClass(ROAD, "road", "Roads and pavement", "#455a64", False),
    LandCoverClass(WATER, "water", "Water", "#1e88e5", False),
    LandCoverClass(OTHER, "other", "Other objects", "#ab47bc", False),
]


def hex_to_rgb(color: str) -> tuple:
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"bad color {color!r}")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def build_lut(classes: List[dict], opacity: float = 0.6, hidden: set = frozenset()) -> np.ndarray:
    """256x4 RGBA lookup table. Unknown (0) and hidden classes are fully transparent."""
    lut = np.zeros((256, 4), dtype=np.uint8)
    a = int(round(255 * min(max(opacity, 0.0), 1.0)))
    for c in classes:
        cid = int(c["id"])
        if cid == UNKNOWN or cid in hidden or not (0 <= cid < 256):
            continue
        r, g, b = hex_to_rgb(c["color"])
        lut[cid] = (r, g, b, a)
    return lut


def colorize(mask: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """uint8 class mask -> RGBA image using a LUT."""
    return lut[mask]
