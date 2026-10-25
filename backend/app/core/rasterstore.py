"""Tiled, sparse, persistent raster store with undo/redo.

Used for the categorical classification mask, the per-pixel confidence layer, and any
other raster that must be editable without loading the whole dataset.

Design
------
* The raster is split into ``tile`` x ``tile`` blocks; each existing block is one
  compressed ``.npz`` file written atomically. Missing blocks read as ``fill`` so an
  untouched raster costs almost nothing on disk.
* Every mutation is an *edit*. An edit records before/after pixels of the changed
  region of each touched tile (a patch), so undo/redo never rewrites the original
  imagery and never needs the full raster in memory.
* Aggregate class histograms are maintained incrementally (deltas) so statistics for a
  multi-gigapixel mask are O(1) to read.
* A single-writer file lock serialises edits across processes. A pending-journal file
  lets ``open`` detect an interrupted commit and rebuild the histogram.
* Linux/macOS only (uses ``fcntl``).
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .tiling import Window

META = "meta.json"
PENDING = "pending.json"
Key = Tuple[int, int]


class RasterStoreError(Exception):
    pass


class EditTooLarge(RasterStoreError):
    pass


def _hist(arr: np.ndarray) -> Dict[int, int]:
    vals, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


class _Edit:
    def __init__(self, label: str, record: bool):
        self.label = label
        self.record = record
        self.before: Dict[Key, np.ndarray] = {}
        self.before_hist: Dict[Key, Dict[int, int]] = {}
        self.delta: Dict[int, int] = {}
        self.touched: set = set()


class TiledRaster:
    def __init__(self, root: Union[str, Path], cache_tiles: int = 256,
                 max_edit_tiles: int = 2048, history_limit: int = 200):
        self.root = Path(root)
        if not (self.root / META).exists():
            raise RasterStoreError(f"no raster store at {self.root}")
        self.cache_tiles = cache_tiles
        self.max_edit_tiles = max_edit_tiles
        self.history_limit = history_limit
        self._cache: "OrderedDict[Key, np.ndarray]" = OrderedDict()
        self._dirty: Dict[Key, np.ndarray] = {}
        self._rlock = threading.RLock()
        self._active: Optional[_Edit] = None
        self._lock_fd = os.open(str(self.root / ".lock"), os.O_CREAT | os.O_RDWR)
        self._load_meta()
        if (self.root / PENDING).exists():
            self.rebuild_histogram()

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, root: Union[str, Path], width: int, height: int, tile: int = 256,
               dtype: str = "uint8", fill: int = 0, exist_ok: bool = False, **kw) -> "TiledRaster":
        root = Path(root)
        if (root / META).exists():
            if not exist_ok:
                raise RasterStoreError(f"raster store already exists at {root}")
            return cls(root, **kw)
        if width <= 0 or height <= 0 or tile <= 0:
            raise ValueError("width, height and tile must be positive")
        (root / "tiles").mkdir(parents=True, exist_ok=True)
        (root / "history").mkdir(exist_ok=True)
        dt = np.dtype(dtype)
        track = bool(np.issubdtype(dt, np.integer))
        meta = {
            "width": int(width), "height": int(height), "tile": int(tile),
            "dtype": dt.name, "fill": int(fill) if track else float(fill),
            "version": 0,
            "hist": {str(int(fill)): int(width) * int(height)} if track else None,
            "history": {"seqs": [], "pos": 0, "next": 1},
        }
        _atomic_json(root / META, meta)
        return cls(root, **kw)

    def close(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None  # type: ignore[assignment]

    def _load_meta(self) -> None:
        with open(self.root / META) as f:
            m = json.load(f)
        self.width, self.height, self.tile = m["width"], m["height"], m["tile"]
        self.dtype = np.dtype(m["dtype"])
        self.fill = m["fill"]
        self._version = m["version"]
        self._hist = None if m["hist"] is None else {int(k): v for k, v in m["hist"].items()}
        self._hmeta = m["history"]

    def _save_meta(self) -> None:
        self._version += 1
        _atomic_json(self.root / META, {
            "width": self.width, "height": self.height, "tile": self.tile,
            "dtype": self.dtype.name, "fill": self.fill, "version": self._version,
            "hist": None if self._hist is None else {str(k): v for k, v in self._hist.items() if v},
            "history": self._hmeta,
        })

    @property
    def version(self) -> int:
        """Monotonic edit counter; clients use it to invalidate cached tiles."""
        with open(self.root / META) as f:
            return json.load(f)["version"]

    def _sync(self) -> None:
        """Reload metadata written by another process; drop stale tile cache."""
        before = self._version
        self._load_meta()
        if self._version != before:
            self._cache.clear()

    # ------------------------------------------------------------------ geometry
    def _tile_shape(self, ty: int, tx: int) -> Tuple[int, int]:
        return (min(self.tile, self.height - ty * self.tile),
                min(self.tile, self.width - tx * self.tile))

    def _path(self, ty: int, tx: int) -> Path:
        return self.root / "tiles" / str(ty) / f"{tx}.npz"

    def tile_exists(self, ty: int, tx: int) -> bool:
        return (ty, tx) in self._dirty or self._path(ty, tx).exists()

    def _load_disk(self, ty: int, tx: int) -> Optional[np.ndarray]:
        p = self._path(ty, tx)
        if not p.exists():
            return None
        with np.load(p) as z:
            return z["a"]

    def _get_ro(self, ty: int, tx: int) -> Optional[np.ndarray]:
        key = (ty, tx)
        if key in self._dirty:
            return self._dirty[key]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        a = self._load_disk(ty, tx)
        if a is not None:
            self._cache[key] = a
            while len(self._cache) > self.cache_tiles:
                self._cache.popitem(last=False)
        return a

    # ------------------------------------------------------------------ reading
    def read(self, window: Window) -> np.ndarray:
        """Return ``window`` pixels; anything outside the raster reads as ``fill``."""
        with self._rlock:
            if self._active is None:
                self._sync()
            out = np.full(window.shape, self.fill, dtype=self.dtype)
            w = window.clip(self.width, self.height)
            if w is None:
                return out
            T = self.tile
            for ty in range(w.y0 // T, (w.y1 - 1) // T + 1):
                for tx in range(w.x0 // T, (w.x1 - 1) // T + 1):
                    a = self._get_ro(ty, tx)
                    if a is None:
                        continue
                    oy, ox = ty * T, tx * T
                    gy0, gy1 = max(w.y0, oy), min(w.y1, oy + a.shape[0])
                    gx0, gx1 = max(w.x0, ox), min(w.x1, ox + a.shape[1])
                    out[gy0 - window.y0:gy1 - window.y0, gx0 - window.x0:gx1 - window.x0] = \
                        a[gy0 - oy:gy1 - oy, gx0 - ox:gx1 - ox]
            return out

    def read_tile(self, ty: int, tx: int) -> np.ndarray:
        h, w = self._tile_shape(ty, tx)
        return self.read(Window(tx * self.tile, ty * self.tile, tx * self.tile + w, ty * self.tile + h))

    def iter_existing_tiles(self) -> Iterator[Key]:
        base = self.root / "tiles"
        for d in sorted(base.iterdir()) if base.exists() else []:
            if d.is_dir() and d.name.isdigit():
                for f in sorted(d.glob("*.npz")):
                    yield (int(d.name), int(f.stem))

    # ------------------------------------------------------------------ editing
    @contextmanager
    def edit(self, label: str = "edit", record_history: bool = True) -> Iterator["TiledRaster"]:
        """Group writes into one undoable step. All-or-nothing: an exception rolls back."""
        with self._rlock:
            if self._active is not None:
                raise RasterStoreError("nested edit")
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                self._sync()
                ed = _Edit(label, record_history)
                self._active = ed
                _atomic_json(self.root / PENDING, {"label": label})
                try:
                    yield self
                except BaseException:
                    self._dirty.clear()
                    self._cache.clear()
                    self._active = None
                    self._load_meta()
                    # tiles already flushed in bulk mode cannot be rolled back
                    if not record_history and ed.touched:
                        self.rebuild_histogram()
                    else:
                        (self.root / PENDING).unlink(missing_ok=True)
                    raise
                else:
                    self._commit(ed)
            finally:
                self._active = None
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def write(self, window: Window, data, mask: Optional[np.ndarray] = None) -> None:
        """Write a scalar or window-shaped array; ``mask`` (window-shaped bool) limits it."""
        if self._active is None:
            with self.edit("write"):
                self._write(window, data, mask)
        else:
            with self._rlock:
                self._write(window, data, mask)

    def paint(self, window: Window, mask: np.ndarray, value: int) -> None:
        self.write(window, value, mask)

    def _touch(self, ty: int, tx: int) -> np.ndarray:
        ed = self._active
        assert ed is not None
        key = (ty, tx)
        if key in self._dirty:
            return self._dirty[key]
        if ed.record and len(ed.touched | {key}) > self.max_edit_tiles:
            raise EditTooLarge(
                f"edit touches more than {self.max_edit_tiles} tiles; "
                "use a smaller area or a bulk (non-undoable) operation")
        a = self._get_ro(ty, tx)
        arr = a.copy() if a is not None else np.full(self._tile_shape(ty, tx), self.fill, dtype=self.dtype)
        if self._hist is not None:
            ed.before_hist[key] = _hist(arr)
        if ed.record:
            ed.before[key] = arr.copy()
        ed.touched.add(key)
        self._dirty[key] = arr
        return arr

    def _write(self, window: Window, data, mask: Optional[np.ndarray]) -> None:
        w = window.clip(self.width, self.height)
        if w is None:
            return
        scalar = np.isscalar(data) or np.ndim(data) == 0
        if not scalar and tuple(np.shape(data)) != window.shape:
            raise ValueError("data shape must equal window shape")
        if mask is not None and mask.shape != window.shape:
            raise ValueError("mask shape must equal window shape")
        T = self.tile
        for ty in range(w.y0 // T, (w.y1 - 1) // T + 1):
            for tx in range(w.x0 // T, (w.x1 - 1) // T + 1):
                th, tw = self._tile_shape(ty, tx)
                oy, ox = ty * T, tx * T
                gy0, gy1 = max(w.y0, oy), min(w.y1, oy + th)
                gx0, gx1 = max(w.x0, ox), min(w.x1, ox + tw)
                sy, sx = slice(gy0 - window.y0, gy1 - window.y0), slice(gx0 - window.x0, gx1 - window.x0)
                m = None if mask is None else mask[sy, sx]
                if m is not None and not m.any():
                    continue
                if scalar and m is None and data == self.fill and not self.tile_exists(ty, tx):
                    continue
                arr = self._touch(ty, tx)
                sub = arr[gy0 - oy:gy1 - oy, gx0 - ox:gx1 - ox]
                src = data if scalar else np.asarray(data[sy, sx], dtype=self.dtype)
                if m is None:
                    sub[...] = src
                elif scalar:
                    sub[m] = data
                else:
                    sub[m] = src[m]
        if not self._active.record and len(self._dirty) >= 64:  # type: ignore[union-attr]
            self._flush_dirty(self._active)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ persistence
    def _flush_tile(self, key: Key, arr: np.ndarray, ed: _Edit) -> None:
        if self._hist is not None:
            ah, bh = _hist(arr), ed.before_hist.pop(key, {})
            for v in set(ah) | set(bh):
                ed.delta[v] = ed.delta.get(v, 0) + ah.get(v, 0) - bh.get(v, 0)
        p = self._path(*key)
        if bool((arr == self.fill).all()):
            p.unlink(missing_ok=True)
            self._cache.pop(key, None)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            _atomic_npz(p, a=arr)
            self._cache[key] = arr
            self._cache.move_to_end(key)

    def _flush_dirty(self, ed: _Edit) -> None:
        for key, arr in list(self._dirty.items()):
            self._flush_tile(key, arr, ed)
        self._dirty.clear()
        while len(self._cache) > self.cache_tiles:
            self._cache.popitem(last=False)

    def _apply_delta(self, delta: Dict[int, int]) -> None:
        if self._hist is None:
            return
        for v, d in delta.items():
            self._hist[v] = self._hist.get(v, 0) + d
        self._hist = {k: v for k, v in self._hist.items() if v}

    def _commit(self, ed: _Edit) -> None:
        patch_arrays: Dict[str, np.ndarray] = {}
        keys, boxes = [], []
        if ed.record:
            for key, before in ed.before.items():
                after = self._dirty.get(key)
                if after is None:
                    continue
                diff = after != before
                if not diff.any():
                    self._dirty.pop(key)
                    ed.before_hist.pop(key, None)
                    continue
                rows, cols = np.where(diff.any(axis=1))[0], np.where(diff.any(axis=0))[0]
                y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
                i = len(keys)
                keys.append(key)
                boxes.append((y0, y1, x0, x1))
                patch_arrays[f"b{i}"] = before[y0:y1, x0:x1]
                patch_arrays[f"a{i}"] = after[y0:y1, x0:x1]
        self._flush_dirty(ed)
        self._apply_delta(ed.delta)
        h = self._hmeta
        if ed.record:
            if keys:
                seq = h["next"]
                _atomic_npz(self.root / "history" / f"{seq:06d}.npz",
                            keys=np.array(keys, dtype=np.int32),
                            boxes=np.array(boxes, dtype=np.int32), **patch_arrays)
                for dropped in h["seqs"][h["pos"]:]:          # redo branch is discarded
                    (self.root / "history" / f"{dropped:06d}.npz").unlink(missing_ok=True)
                h["seqs"] = h["seqs"][:h["pos"]] + [seq]
                h["next"] = seq + 1
                while len(h["seqs"]) > self.history_limit:
                    (self.root / "history" / f"{h['seqs'].pop(0):06d}.npz").unlink(missing_ok=True)
                h["pos"] = len(h["seqs"])
        else:
            self._reset_history_files()
        changed = bool(keys) if ed.record else bool(ed.touched)
        if changed:
            self._save_meta()
        (self.root / PENDING).unlink(missing_ok=True)

    # ------------------------------------------------------------------ undo / redo
    def can_undo(self) -> bool:
        self._sync()
        return self._hmeta["pos"] > 0

    def can_redo(self) -> bool:
        self._sync()
        return self._hmeta["pos"] < len(self._hmeta["seqs"])

    def undo(self) -> Optional[Window]:
        return self._step(-1)

    def redo(self) -> Optional[Window]:
        return self._step(+1)

    def _step(self, direction: int) -> Optional[Window]:
        with self._rlock:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                self._sync()
                h = self._hmeta
                if direction < 0 and h["pos"] == 0:
                    return None
                if direction > 0 and h["pos"] >= len(h["seqs"]):
                    return None
                seq = h["seqs"][h["pos"] - 1] if direction < 0 else h["seqs"][h["pos"]]
                prefix = "b" if direction < 0 else "a"
                _atomic_json(self.root / PENDING, {"label": "undo" if direction < 0 else "redo"})
                ed = _Edit("step", False)
                self._active = ed
                region: Optional[Window] = None
                with np.load(self.root / "history" / f"{seq:06d}.npz") as z:
                    keys, boxes = z["keys"], z["boxes"]
                    for i, ((ty, tx), (y0, y1, x0, x1)) in enumerate(zip(keys.tolist(), boxes.tolist())):
                        arr = self._touch(ty, tx)
                        arr[y0:y1, x0:x1] = z[f"{prefix}{i}"]
                        ow = Window(tx * self.tile + x0, ty * self.tile + y0, tx * self.tile + x1, ty * self.tile + y1)
                        region = ow if region is None else region.union(ow)
                self._flush_dirty(ed)
                self._apply_delta(ed.delta)
                h["pos"] += direction
                self._save_meta()
                (self.root / PENDING).unlink(missing_ok=True)
                return region
            finally:
                self._active = None
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _reset_history_files(self) -> None:
        for s in self._hmeta["seqs"]:
            (self.root / "history" / f"{s:06d}.npz").unlink(missing_ok=True)
        self._hmeta["seqs"], self._hmeta["pos"] = [], 0

    # ------------------------------------------------------------------ statistics
    def histogram(self) -> Dict[int, int]:
        """Pixel count per value, maintained incrementally (integer rasters only)."""
        with self._rlock:
            self._sync()
            if self._hist is None:
                raise RasterStoreError("histogram not tracked for float rasters")
            return dict(self._hist)

    def rebuild_histogram(self) -> Dict[int, int]:
        """Rescan every tile. Run after a crash or if counts are suspected to drift."""
        with self._rlock:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                self._load_meta()
                if self._hist is None:
                    (self.root / PENDING).unlink(missing_ok=True)
                    return {}
                counts: Dict[int, int] = {}
                covered = 0
                for (ty, tx) in self.iter_existing_tiles():
                    a = self._load_disk(ty, tx)
                    if a is None:
                        continue
                    covered += a.size
                    for v, c in _hist(a).items():
                        counts[v] = counts.get(v, 0) + c
                fill = int(self.fill)
                counts[fill] = counts.get(fill, 0) + self.width * self.height - covered
                self._hist = {k: v for k, v in counts.items() if v}
                self._cache.clear()
                self._save_meta()
                (self.root / PENDING).unlink(missing_ok=True)
                return dict(self._hist)
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def disk_usage(self) -> int:
        return sum(f.stat().st_size for f in (self.root).rglob("*") if f.is_file())


# ---------------------------------------------------------------------- helpers
def _atomic_json(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, **arrays) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
