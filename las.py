"""Chunked LAS reader/writer (uncompressed LAS 1.2-1.4) using only numpy/struct.

* Reading streams in chunks, so multi-hundred-million-point clouds never need to fit in memory.
* LAZ is *not* decoded here (needs laszip). ``ensure_las`` converts via PDAL when available and raises a
  ``MissingDependency`` with install instructions otherwise.
* The writer emits LAS 1.2 point format 3 (XYZ, intensity, return numbers, classification, GPS time, RGB) with a
  GeoTIFF-key VLR holding the EPSG code so the CRS travels with the file.
"""
from __future__ import annotations

import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

SIG = b"LASF"
REC_LEN = {0: 20, 1: 28, 2: 26, 3: 34, 4: 57, 5: 63, 6: 30, 7: 36, 8: 38, 9: 59, 10: 67}
RGB_OFFSET = {2: 20, 3: 28, 5: 28, 7: 30, 8: 30, 10: 30}
ASPRS_GROUND, ASPRS_LOW_VEG, ASPRS_MED_VEG, ASPRS_HIGH_VEG, ASPRS_BUILDING, ASPRS_NOISE = 2, 3, 4, 5, 6, 7
ASPRS_UNCLASSIFIED, ASPRS_WATER = 1, 9


class LasError(ValueError):
    pass


class MissingDependency(RuntimeError):
    """A required external tool is absent. The message always says how to install it."""


@dataclass
class LasHeader:
    version: Tuple[int, int]
    point_format: int
    record_length: int
    count: int
    scale: Tuple[float, float, float]
    offset: Tuple[float, float, float]
    mins: Tuple[float, float, float]
    maxs: Tuple[float, float, float]
    offset_to_points: int
    epsg: Optional[int]
    wkt: Optional[str]
    compressed: bool = False

    @property
    def bounds_xy(self):
        return (self.mins[0], self.mins[1], self.maxs[0], self.maxs[1])


def _read_vlrs(buf: bytes, header_size: int, n: int):
    pos, out = header_size, []
    for _ in range(n):
        _, uid, rid, length = struct.unpack_from("<H16sHH", buf, pos)[0:4]
        user = uid.split(b"\x00")[0].decode("ascii", "replace")
        data = buf[pos + 54: pos + 54 + length]
        out.append((user, rid, data))
        pos += 54 + length
    return out


def read_header(path: Path) -> LasHeader:
    with open(path, "rb") as f:
        head = f.read(375)
        if head[:4] != SIG or len(head) < 227:
            raise LasError(f"{Path(path).name} is not a LAS file")
        vmaj, vmin = head[24], head[25]
        header_size, off_pts, n_vlr = struct.unpack_from("<HII", head, 94)
        pf_raw, rec_len, n_legacy = struct.unpack_from("<BHI", head, 104)
        sx, sy, sz, ox, oy, oz = struct.unpack_from("<6d", head, 131)
        maxx, minx, maxy, miny, maxz, minz = struct.unpack_from("<6d", head, 179)
        count = n_legacy
        if (vmaj, vmin) >= (1, 4) and len(head) >= 255:
            count = struct.unpack_from("<Q", head, 247)[0] or n_legacy
        compressed = bool(pf_raw & 0x80)
        pf = pf_raw & 0x3F
        f.seek(header_size)
        vlr_blob = f.read(max(off_pts - header_size, 0))
    vlrs = _read_vlrs(b"\x00" * header_size + vlr_blob, header_size, n_vlr) if n_vlr else []
    epsg, wkt = None, None
    for user, rid, data in vlrs:
        if user == "LASF_Projection" and rid == 34735 and len(data) >= 8:
            nkeys = struct.unpack_from("<H", data, 6)[0]
            for i in range(nkeys):
                key, loc, cnt, val = struct.unpack_from("<4H", data, 8 + 8 * i)
                if key in (3072, 2048) and loc == 0:
                    epsg = val
        elif user == "LASF_Projection" and rid in (2112, 2111):
            wkt = data.split(b"\x00")[0].decode("utf-8", "replace")
    if pf not in REC_LEN:
        raise LasError(f"unsupported LAS point format {pf}")
    return LasHeader((vmaj, vmin), pf, rec_len, count, (sx, sy, sz), (ox, oy, oz), (minx, miny, minz),
                     (maxx, maxy, maxz), off_pts, epsg, wkt, compressed)


def _dtype(h: LasHeader) -> np.dtype:
    pf = h.point_format
    names, fmts, offs = ["X", "Y", "Z", "intensity"], ["<i4", "<i4", "<i4", "<u2"], [0, 4, 8, 12]
    if pf <= 5:
        names += ["flags", "cls_byte"]; fmts += ["u1", "u1"]; offs += [14, 15]
    else:
        names += ["flags", "flags2", "cls_byte"]; fmts += ["u1", "u1", "u1"]; offs += [14, 15, 16]
    if pf in RGB_OFFSET:
        o = RGB_OFFSET[pf]
        names += ["red", "green", "blue"]; fmts += ["<u2"] * 3; offs += [o, o + 2, o + 4]
    return np.dtype({"names": names, "formats": fmts, "offsets": offs, "itemsize": h.record_length})


def iter_points(path: Path, chunk: int = 2_000_000) -> Iterator[Dict[str, np.ndarray]]:
    """Yield dicts with x,y,z (float64), classification (uint8), intensity, and rgb (uint16 Nx3) if present."""
    h = read_header(path)
    if h.compressed:
        raise LasError("LAZ-compressed file: convert with ensure_las() (needs PDAL) before reading")
    dt = _dtype(h)
    remaining = h.count
    with open(path, "rb") as f:
        f.seek(h.offset_to_points)
        while remaining > 0:
            n = min(chunk, remaining)
            buf = f.read(n * h.record_length)
            got = len(buf) // h.record_length
            if got == 0:
                raise LasError(f"file truncated: expected {h.count} points, header says more than present")
            a = np.frombuffer(buf[:got * h.record_length], dtype=dt)
            out = {"x": a["X"] * h.scale[0] + h.offset[0], "y": a["Y"] * h.scale[1] + h.offset[1],
                   "z": a["Z"] * h.scale[2] + h.offset[2],
                   "classification": (a["cls_byte"] & 0x1F) if h.point_format <= 5 else a["cls_byte"].copy(),
                   "intensity": a["intensity"].copy()}
            if "red" in a.dtype.names:
                out["rgb"] = np.stack([a["red"], a["green"], a["blue"]], axis=1)
            yield out
            remaining -= got
            if got < n:
                raise LasError(f"file truncated: {h.count - remaining} of {h.count} points present")


def read_all(path: Path, max_points: int = 200_000_000) -> Dict[str, np.ndarray]:
    h = read_header(path)
    if h.count > max_points:
        raise LasError(f"{h.count:,} points exceeds the in-memory limit of {max_points:,}; process in chunks")
    parts = list(iter_points(path))
    if not parts:
        return {"x": np.zeros(0), "y": np.zeros(0), "z": np.zeros(0), "classification": np.zeros(0, np.uint8),
                "intensity": np.zeros(0, np.uint16)}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def ensure_las(path: Path, workdir: Path, pdal_bin: str = "pdal") -> Path:
    """Return a readable uncompressed LAS path, converting LAZ with PDAL if needed."""
    path = Path(path)
    if path.suffix.lower() == ".las":
        return path
    if path.suffix.lower() != ".laz":
        raise LasError(f"unsupported point-cloud extension {path.suffix!r}; use .las or .laz")
    if shutil.which(pdal_bin) is None:
        raise MissingDependency(
            "PDAL is required to read .laz files but was not found on PATH. Install it with "
            "`conda install -c conda-forge pdal` or `apt install pdal` (or run inside the provided Docker image).")
    out = Path(workdir) / (path.stem + ".las")
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([pdal_bin, "translate", str(path), str(out), "--writers.las.compression=false"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise LasError(f"pdal translate failed: {r.stderr.strip()[:500]}")
    return out


class LasWriter:
    """Append chunks, finalize on close. LAS 1.2, point format 3, GeoTIFF-key VLR for the EPSG code."""

    def __init__(self, path: Path, epsg: Optional[int] = None, scale: float = 0.001,
                 offset: Tuple[float, float, float] = (0.0, 0.0, 0.0), system_id: str = "terrascope"):
        self.path, self.epsg, self.scale, self.offset = Path(path), epsg, scale, offset
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "wb")
        self._vlr = self._build_vlr()
        self._off_pts = 227 + len(self._vlr)
        self._f.write(b"\x00" * self._off_pts)
        self.count = 0
        self._mins = np.array([np.inf] * 3)
        self._maxs = np.array([-np.inf] * 3)
        self._by_return = np.zeros(5, dtype=np.uint32)
        self._sysid = system_id.encode("ascii", "replace")[:32].ljust(32, b"\x00")

    def _build_vlr(self) -> bytes:
        if not self.epsg:
            return b""
        geographic = 4000 <= self.epsg < 4900
        keys = [(1024, 0, 1, 2 if geographic else 1), (1025, 0, 1, 1), ((2048 if geographic else 3072), 0, 1, self.epsg)]
        data = struct.pack("<4H", 1, 1, 0, len(keys)) + b"".join(struct.pack("<4H", *k) for k in keys)
        return struct.pack("<H16sHH32s", 0, b"LASF_Projection".ljust(16, b"\x00"), 34735, len(data),
                           b"GeoTiff projection keys".ljust(32, b"\x00")) + data

    def write(self, x, y, z, classification=None, rgb=None, intensity=None) -> None:
        x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
        n = len(x)
        if n == 0:
            return
        if not (len(y) == n == len(z)):
            raise LasError("x, y, z must have equal length")
        if not np.all(np.isfinite(x) & np.isfinite(y) & np.isfinite(z)):
            raise LasError("non-finite coordinates cannot be written to LAS")
        rec = np.zeros(n, dtype=np.dtype({"names": ["X", "Y", "Z", "i", "flags", "cls", "scan", "ud", "ps", "gps", "r", "g", "b"],
                                          "formats": ["<i4", "<i4", "<i4", "<u2", "u1", "u1", "i1", "u1", "<u2", "<f8", "<u2", "<u2", "<u2"],
                                          "offsets": [0, 4, 8, 12, 14, 15, 16, 17, 18, 20, 28, 30, 32], "itemsize": 34}))
        lim = 2 ** 31 - 1
        for arr, off, name in ((x, self.offset[0], "X"), (y, self.offset[1], "Y"), (z, self.offset[2], "Z")):
            q = np.round((arr - off) / self.scale)
            if np.abs(q).max() > lim:
                raise LasError(f"coordinate range too large for scale {self.scale}; subtract an offset or raise the scale")
            rec[name] = q.astype(np.int32)
        rec["flags"] = 0b00001001                      # return 1 of 1
        if intensity is not None:
            rec["i"] = np.asarray(intensity, dtype=np.uint16)
        if classification is not None:
            c = np.asarray(classification)
            if c.min() < 0 or c.max() > 31:
                raise LasError("classification must be 0..31 for point format 3")
            rec["cls"] = c.astype(np.uint8)
        if rgb is not None:
            rgb = np.asarray(rgb, dtype=np.uint16)
            rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        self._f.write(rec.tobytes())
        self.count += n
        self._by_return[0] += n
        self._mins = np.minimum(self._mins, [x.min(), y.min(), z.min()])
        self._maxs = np.maximum(self._maxs, [x.max(), y.max(), z.max()])

    def close(self) -> None:
        if self._f.closed:
            return
        mins = self._mins if self.count else np.zeros(3)
        maxs = self._maxs if self.count else np.zeros(3)
        import datetime as _dt
        today = _dt.date.today()
        h = bytearray(227)
        h[0:4] = SIG
        h[24], h[25] = 1, 2
        h[26:58] = b"terrascope".ljust(32, b"\x00")
        h[58:90] = self._sysid
        struct.pack_into("<HH", h, 90, today.timetuple().tm_yday, today.year)
        struct.pack_into("<HII", h, 94, 227, self._off_pts, 1 if self._vlr else 0)
        struct.pack_into("<BHI", h, 104, 3, 34, self.count)
        struct.pack_into("<5I", h, 111, *self._by_return)
        struct.pack_into("<6d", h, 131, self.scale, self.scale, self.scale, *self.offset)
        struct.pack_into("<6d", h, 179, maxs[0], mins[0], maxs[1], mins[1], maxs[2], mins[2])
        self._f.seek(0)
        self._f.write(bytes(h))
        self._f.seek(227)
        self._f.write(self._vlr)
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_las(path: Path, x, y, z, classification=None, rgb=None, epsg: Optional[int] = None,
              scale: float = 0.001, offset: Optional[Tuple[float, float, float]] = None) -> Path:
    x = np.asarray(x, float)
    if offset is None:
        offset = (float(np.floor(x.min())) if len(x) else 0.0, float(np.floor(np.min(y))) if len(x) else 0.0, 0.0)
    with LasWriter(path, epsg, scale, offset) as w:
        w.write(x, y, z, classification, rgb)
    return Path(path)
