"""Image header/EXIF/XMP metadata extraction (Pillow only; never decodes pixels)."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

_IFD_EXIF, _IFD_GPS = 0x8769, 0x8825
_TAG_MAKE, _TAG_MODEL, _TAG_DATETIME, _TAG_ORIENT = 271, 272, 306, 274
_TAG_DTO, _TAG_FOCAL, _TAG_FOCAL35, _TAG_EXPO, _TAG_FNUM, _TAG_ISO = 36867, 37386, 41989, 33434, 33437, 34855
_TAG_SUBSEC_DTO = 37521

_XMP_KEYS = {
    "rel_alt": r"(?:drone-dji:)?RelativeAltitude",
    "abs_alt": r"(?:drone-dji:)?AbsoluteAltitude",
    "gimbal_yaw": r"(?:drone-dji:)?GimbalYawDegree",
    "gimbal_pitch": r"(?:drone-dji:)?GimbalPitchDegree",
    "gimbal_roll": r"(?:drone-dji:)?GimbalRollDegree",
    "flight_yaw": r"(?:drone-dji:)?FlightYawDegree",
}

JPEG_MAGIC = b"\xff\xd8\xff"
TIFF_MAGICS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def sniff_format(head: bytes) -> Optional[str]:
    """Return 'jpeg' / 'tiff' from the first bytes, or None. File extensions are never trusted."""
    if head.startswith(JPEG_MAGIC):
        return "jpeg"
    if any(head.startswith(m) for m in TIFF_MAGICS):
        return "tiff"
    return None


def _rat(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms(v, ref) -> Optional[float]:
    try:
        d, m, s = (float(x) for x in v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    out = d + m / 60.0 + s / 3600.0
    return -out if str(ref).upper() in ("S", "W") else out


def _parse_dt(s: Any, subsec: Any = None) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip().strip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if subsec:
                try:
                    dt = dt.replace(microsecond=int(str(subsec).strip()[:6].ljust(6, "0")))
                except ValueError:
                    pass
            return dt.isoformat()
        except ValueError:
            continue
    return None


def _xmp(path: Path, limit: int = 262_144) -> Dict[str, float]:
    try:
        with open(path, "rb") as f:
            blob = f.read(limit)
    except OSError:
        return {}
    out: Dict[str, float] = {}
    for key, name in _XMP_KEYS.items():
        m = re.search(name.encode() + rb'(?:="|>)\s*([+-]?\d+(?:\.\d+)?)', blob)
        if m:
            out[key] = float(m.group(1))
    return out


def read_image_metadata(path: Path, max_pixels: int = 400_000_000) -> Dict[str, Any]:
    """Header-only read. Raises ValueError with a user-facing message for unreadable/oversized images."""
    path = Path(path)
    prev = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None          # we apply our own, explicit limit below
    try:
        with Image.open(path) as im:
            width, height = im.size
            fmt = (im.format or "").lower()
            if width * height > max_pixels:
                raise ValueError(f"image is {width}x{height} ({width * height / 1e6:.0f} MP), above the "
                                 f"configured limit of {max_pixels / 1e6:.0f} MP")
            exif = im.getexif()
            base = {k: exif.get(k) for k in (_TAG_MAKE, _TAG_MODEL, _TAG_DATETIME, _TAG_ORIENT)}
            ex = exif.get_ifd(_IFD_EXIF) if exif else {}
            gps = exif.get_ifd(_IFD_GPS) if exif else {}
            bands = len(im.getbands())
            mode = im.mode
    except (OSError, SyntaxError, Image.DecompressionBombError) as e:
        raise ValueError(f"unreadable image: {e}") from e
    finally:
        Image.MAX_IMAGE_PIXELS = prev

    meta: Dict[str, Any] = {
        "format": fmt, "width": width, "height": height, "mode": mode, "bands": bands,
        "make": (str(base[_TAG_MAKE]).strip("\x00 ") if base[_TAG_MAKE] else None),
        "model": (str(base[_TAG_MODEL]).strip("\x00 ") if base[_TAG_MODEL] else None),
        "orientation": base[_TAG_ORIENT],
        "captured_at": _parse_dt(ex.get(_TAG_DTO) or base[_TAG_DATETIME], ex.get(_TAG_SUBSEC_DTO)),
        "focal_mm": _rat(ex.get(_TAG_FOCAL)), "focal35_mm": _rat(ex.get(_TAG_FOCAL35)),
        "exposure_s": _rat(ex.get(_TAG_EXPO)), "f_number": _rat(ex.get(_TAG_FNUM)),
        "iso": ex.get(_TAG_ISO) if isinstance(ex.get(_TAG_ISO), int) else None,
        "lat": None, "lon": None, "alt_msl": None,
    }
    if gps:
        lat, lon = _dms(gps.get(2), gps.get(1)), _dms(gps.get(4), gps.get(3))
        meta["lat"], meta["lon"] = lat, lon
        alt = _rat(gps.get(6))
        if alt is not None:
            below = gps.get(5) in (1, b"\x01")
            meta["alt_msl"] = -alt if below else alt
    meta.update({"rel_alt": None, "gimbal_yaw": None, "gimbal_pitch": None, "gimbal_roll": None, "flight_yaw": None})
    if fmt == "jpeg":
        x = _xmp(path)
        meta.update({k: x.get(k) for k in ("rel_alt", "gimbal_yaw", "gimbal_pitch", "gimbal_roll", "flight_yaw")})
        if meta["alt_msl"] is None and "abs_alt" in x:
            meta["alt_msl"] = x["abs_alt"]
    return meta


def georef_assessment(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Judge whether an image carries usable georeferencing. Never guesses missing values."""
    lat, lon = meta.get("lat"), meta.get("lon")
    issues = []
    if lat is None or lon is None:
        return {"status": "missing", "has_gps": False,
                "issues": ["No GPS position in EXIF. Provide ground control points, or ODM can only produce "
                           "a local, non-georeferenced model."]}
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {"status": "invalid", "has_gps": False, "issues": [f"GPS position out of range ({lat}, {lon})."]}
    if abs(lat) < 1e-6 and abs(lon) < 1e-6:
        return {"status": "invalid", "has_gps": False,
                "issues": ["GPS position is exactly 0,0 (no satellite fix when the photo was taken)."]}
    status = "ok"
    if meta.get("alt_msl") is None:
        status = "partial"
        issues.append("No GPS altitude: vertical georeferencing will be weak; use ground control points.")
    if meta.get("rel_alt") is None:
        issues.append("No relative flight altitude: footprint and overlap cannot be estimated for this image.")
    return {"status": status, "has_gps": True, "issues": issues}
