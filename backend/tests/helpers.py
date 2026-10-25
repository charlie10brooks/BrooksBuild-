import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings
from app.context import AppContext
from app.workers.runner import Worker


def make_app(tmp: Path = None, **over) -> AppContext:
    tmp = tmp or Path(tempfile.mkdtemp(prefix="ts_test_"))
    s = Settings(data_dir=tmp, min_free_bytes=0, **over)
    s.stale_job_seconds = over.get("stale_job_seconds", 2)
    return AppContext(s)


def jpeg_bytes(seed=0, size=(160, 120), gps=None, rel_alt=None, when="2026:05:01 10:00:00"):
    """Small real JPEG with optional EXIF GPS (lat, lon, alt) and DJI-style XMP relative altitude."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    ex = Image.Exif()
    ex[0x010F], ex[0x0110] = "TestCam", "T1"
    ex_ifd = ex.get_ifd(0x8769)
    ex_ifd[0x9003] = when
    ex_ifd[0xA405] = 24
    ex_ifd[0x920A] = 8.8
    if gps:
        lat, lon, alt = gps
        g = ex.get_ifd(0x8825)

        def dms(v):
            v = abs(v)
            d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
            return (float(d), float(m), float(s))
        g[1], g[2] = ("N" if lat >= 0 else "S"), dms(lat)
        g[3], g[4] = ("E" if lon >= 0 else "W"), dms(lon)
        g[5], g[6] = 0, float(alt)
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=ex, quality=85)
    data = buf.getvalue()
    if rel_alt is not None:
        xmp = (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
               f'<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" drone-dji:RelativeAltitude="+{rel_alt}" '
               f'drone-dji:GimbalPitchDegree="-90.0" drone-dji:FlightYawDegree="0.0"/></rdf:RDF></x:xmpmeta>').encode()
        seg = b"\xff\xe1" + (len(xmp) + 31).to_bytes(2, "big") + b"http://ns.adobe.com/xap/1.0/\x00" + xmp
        data = data[:2] + seg + data[2:]
    return data


def drain(app: AppContext, worker: Worker = None, limit=500):
    """Run queued jobs inline until the queue is empty (ignores run_after delays)."""
    app.db.execute("UPDATE jobs SET run_after=0 WHERE status='queued'")
    w = worker or Worker(app)
    n = 0
    while w.run_one() and n < limit:
        app.db.execute("UPDATE jobs SET run_after=0 WHERE status='queued'")
        n += 1
    return n
