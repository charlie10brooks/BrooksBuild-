"""PDF site report (reportlab). Pure presentation of what is already in the project: nothing here computes new terrain.

Every page carries the "preliminary planning estimate" footer. Observed and estimated surfaces are labelled wherever they appear.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..core import classes as C
from ..core.dtm import GridSpec
from ..core.tiling import Window
from . import classify as K
from . import ortho as O
from . import render as R

FIG_PX = 900


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _png(img_bgr_or_rgba: np.ndarray) -> io.BytesIO:
    ok, buf = cv2.imencode(".png", img_bgr_or_rgba)
    return io.BytesIO(buf.tobytes())


def _extent(ex) -> Optional[tuple]:
    """(grid, bbox) of the first available georeferenced raster, preferring the orthomosaic."""
    for name in ("orthophoto", "dtm", "dsm"):
        p = ex.product(name)
        g = (p or {}).get("meta", {}).get("grid")
        if g:
            gs = GridSpec(**{k: g[k] for k in ("x0", "y0", "res", "width", "height")})
            return gs, (gs.x0, gs.y0 - gs.height * gs.res, gs.x0 + gs.width * gs.res, gs.y0)
    return None


def _canvas(bbox, maxpx=FIG_PX):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    s = maxpx / max(w, h)
    return max(2, int(round(w * s))), max(2, int(round(h * s)))


def _ortho_rgba(ex, bbox, w, h) -> Optional[np.ndarray]:
    prod = ex.product("orthophoto")
    if not prod or not prod["meta"].get("grid"):
        return None
    g = prod["meta"]["grid"]
    pyr = O.OrthoPyramid(ex.app.storage.path(ex.pid, "ortho/pyramid"))
    T = pyr.info["tile"]
    z = int(max(0, min(pyr.info["levels"] - 1, np.floor(np.log2(max(g["width"] / w, g["height"] / h, 1.0))))))
    s = 2 ** z
    nx, ny = int(np.ceil(g["width"] / s / T)), int(np.ceil(g["height"] / s / T))
    if nx * ny > 400:
        return None
    canvas = np.zeros((ny * T, nx * T, 4), np.uint8)
    for ty in range(ny):
        for tx in range(nx):
            pth = pyr.tile_path(z, tx, ty)
            if pth:
                t = O._tile_decode(pth)
                if t.shape[2] == 3:
                    t = np.dstack([t, np.full(t.shape[:2], 255, np.uint8)])
                canvas[ty * T:ty * T + t.shape[0], tx * T:tx * T + t.shape[1]] = t
    xs = bbox[0] + (np.arange(w) + .5) * (bbox[2] - bbox[0]) / w
    ys = bbox[3] - (np.arange(h) + .5) * (bbox[3] - bbox[1]) / h
    cc = np.floor((xs - g["x0"]) / g["res"] / s).astype(int)
    rr = np.floor((g["y0"] - ys) / g["res"] / s).astype(int)
    inside = (rr[:, None] >= 0) & (rr[:, None] < canvas.shape[0]) & (cc[None, :] >= 0) & (cc[None, :] < canvas.shape[1])
    out = canvas[np.clip(rr, 0, canvas.shape[0] - 1)[:, None], np.clip(cc, 0, canvas.shape[1] - 1)[None, :]]
    out[~inside] = 0
    return out


def _classes_rgba(ex, bbox, w, h) -> Optional[np.ndarray]:
    prod = ex.product("orthophoto")
    if not prod or not prod["meta"].get("grid"):
        return None
    g = prod["meta"]["grid"]
    mask, _ = O.class_rasters(ex.app, ex.pid)
    xs = bbox[0] + (np.arange(w) + .5) * (bbox[2] - bbox[0]) / w
    ys = bbox[3] - (np.arange(h) + .5) * (bbox[3] - bbox[1]) / h
    cc = np.floor((xs - g["x0"]) / g["res"]).astype(int)
    rr = np.floor((g["y0"] - ys) / g["res"]).astype(int)
    out = np.zeros((h, w, 4), np.uint8)
    lut = C.build_lut(ex.classes, 0.75)
    T = mask.tile
    vr = (rr >= 0) & (rr < mask.height)
    vc = (cc >= 0) & (cc < mask.width)
    key = np.where(vr[:, None] & vc[None, :], (rr // T)[:, None] * 1_000_000 + (cc // T)[None, :], -1)
    for k in np.unique(key):
        if k < 0:
            continue
        tile = mask.read_tile(int(k // 1_000_000), int(k % 1_000_000))
        ii, jj = np.nonzero(key == k)
        out[ii, jj] = lut[tile[rr[ii] % T, cc[jj] % T] & K.CLASS_MASK]
    return out


def _over(base: Optional[np.ndarray], top: np.ndarray) -> np.ndarray:
    h, w = top.shape[:2]
    bg = np.full((h, w, 3), 245, np.float32) if base is None else base[..., :3].astype(np.float32)
    a = top[..., 3:4].astype(np.float32) / 255.0
    return np.clip(bg * (1 - a) + top[..., :3].astype(np.float32) * a, 0, 255).astype(np.uint8)


def _figures(ex) -> List[tuple]:
    ext = _extent(ex)
    if not ext:
        return []
    grid, bbox = ext
    w, h = _canvas(bbox)
    figs = []
    ortho = _ortho_rgba(ex, bbox, w, h)
    if ortho is not None:
        figs.append(("Orthomosaic (imagery)", cv2.cvtColor(ortho[..., :3], cv2.COLOR_RGB2BGR), "Visual imagery. Not elevation."))
        hist = O.class_rasters(ex.app, ex.pid)[0].histogram()
        if any((v & K.CLASS_MASK) for v in hist):
            cl = _classes_rgba(ex, bbox, w, h)
            figs.append(("Land-cover classification over the orthomosaic",
                         cv2.cvtColor(_over(ortho, cl), cv2.COLOR_RGB2BGR), "VISUAL interpretation. Not elevation evidence."))
    for layer, title, note, extra in (
            ("dsm", "Observed surface (DSM), hillshade", "OBSERVED: canopy and structure tops. This is not the ground.", {"shade": True}),
            ("dtm", "Estimated bare earth (DTM), hillshade", "ESTIMATED: partly interpolated under canopy. See confidence below.", {"shade": True}),
            ("confidence", "Terrain confidence", "Green = high, orange = moderate, red = low, grey = unknown (insufficient evidence). Estimated terrain in grey or red areas should not be relied on.", {})):
        p = ex.product(layer)
        if not p:
            continue
        try:
            img = R.render_layer(p["path"], layer, bbox, w, h, "terrain", None, None, extra.get("shade", False), 1.0, 1.0)
        except Exception as e:                                                       # noqa: BLE001
            ex.ctx.log(f"figure {layer} skipped: {e}", "warning")
            continue
        bg = np.full((h, w, 3), 245, np.uint8)
        figs.append((title, cv2.cvtColor(_over(bg, img), cv2.COLOR_RGB2BGR), note))
    return figs


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(0.75 * inch, 0.45 * inch, "PRELIMINARY PLANNING ESTIMATE - not for final foundation design, permitting or staking.")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_report(ex, path: Path) -> None:
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], alignment=TA_LEFT, fontSize=20, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], spaceBefore=12, spaceAfter=4, textColor=colors.HexColor("#1f2d3d"))
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=12)
    S = ParagraphStyle("S", parent=B, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"))
    cell = ParagraphStyle("cell", parent=B, fontSize=8, leading=10)

    def tbl(rows, widths=None, head=True):
        t = Table([[Paragraph(_esc(c), cell) if not isinstance(c, Paragraph) else c for c in r] for r in rows], colWidths=widths, repeatRows=1 if head else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8bec6")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        if head:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e9ee")))
        t.setStyle(TableStyle(st))
        return t

    p = ex.proj
    el: list = []
    el.append(Paragraph(_esc(p["name"]), H1))
    el.append(Paragraph(f"Terrain reconstruction and vegetation-removal report &nbsp;|&nbsp; generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", S))
    el.append(Spacer(1, 8))
    box = Table([[Paragraph("<b>PRELIMINARY PLANNING ESTIMATE.</b> " + _esc(
        "Not suitable for final foundation design, permitting, staking or any use that requires a licensed survey. "
        "Where this report shows an estimated bare-earth surface under vegetation, those heights are inferred, not measured."), B)]],
        colWidths=[7.0 * inch])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#b3261e")), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf2f1")),
                             ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    el.append(box)
    if p["is_sample"]:
        el += [Spacer(1, 6), Paragraph("<b>SYNTHETIC SAMPLE DATA</b> - generated for development and testing; this is not a real site.", B)]

    el.append(Paragraph("Project and coordinate system", H2))
    el.append(tbl([["Item", "Value"],
                   ["Address", p["address"] or "-"],
                   ["Horizontal CRS", f"EPSG:{p['epsg']}" if p["epsg"] else "NOT SET"],
                   ["Vertical datum", f"EPSG:{p['vertical_epsg']}" if p["vertical_epsg"] else "not specified"],
                   ["Horizontal / vertical units", f"{p['horizontal_units']} / {p['vertical_units']}"],
                   ["Images", str(ex.app.db.one("SELECT COUNT(*) n FROM images WHERE project_id=?", (ex.pid,))["n"])]],
                  [2.0 * inch, 5.0 * inch]))

    el.append(Paragraph("What each layer means", H2))
    el.append(tbl([["Layer", "Meaning", "Elevation evidence?"],
                   ["Land cover", "Visual classification of the orthomosaic (trees, brush, grass, soil, ...)", "No"],
                   ["Observed surface (DSM)", "Tops of canopy and structures as seen by photogrammetry", "Observed, but not ground under vegetation"],
                   ["Classified ground points", "Point-cloud returns the ground filter accepted as ground", "Yes: measured ground"],
                   ["Estimated bare earth (DTM)", "Measured where ground points exist, interpolated elsewhere", "Partly (see evidence and confidence)"],
                   ["Insufficient evidence", "Cells where no defensible elevation estimate exists", "None: shown as unknown"]],
                  [1.7 * inch, 3.6 * inch, 1.7 * inch]))

    # ---- terrain
    summ = (ex.product("terrain_summary") or {}).get("meta")
    el.append(Paragraph("Terrain evidence summary", H2))
    if not summ:
        el.append(Paragraph("The terrain pipeline has not been run, so there is no bare-earth estimate in this project.", B))
    else:
        cov = summ.get("coverage", {})
        conf = summ.get("confidence", {})
        rows = [["Measure", "Value"],
                ["Ground engine", str(summ.get("ground_engine"))],
                ["Resolution", f"{summ.get('resolution_m')} m"],
                ["Ground share of points", f"{100 * summ.get('ground_fraction_of_points', 0):.1f}%"],
                ["Cells with MEASURED ground", f"{100 * cov.get('measured', 0):.1f}%"],
                ["Cells INTERPOLATED", f"{100 * cov.get('interpolated', 0):.1f}%"],
                ["Cells from external data", f"{100 * cov.get('external', 0):.1f}%"],
                ["Cells with NO estimate", f"{100 * cov.get('none', 0):.1f}%"]]
        for k in ("high", "moderate", "low", "unknown"):
            if k in conf:
                rows.append([f"Confidence: {k}", f"{100 * conf[k]['fraction']:.1f}% ({conf[k]['area_m2']:.0f} m2)"])
        cvv = summ.get("cross_validation")
        if isinstance(cvv, dict) and cvv.get("rmse") is not None:
            rows.append(["Leave-out cross-validation RMSE", f"{cvv['rmse']:.3f} m (interpolation skill where ground exists; not accuracy under canopy)"])
        cp = summ.get("check_points") or {}
        rows.append(["Independent check points", (f"n={cp['n']}, RMSE {cp['rmse']:.3f}, mean dz {cp['mean_dz']:.3f}" if cp.get("n") else cp.get("note", "none"))])
        el.append(tbl(rows, [2.6 * inch, 4.4 * inch]))
        if conf.get("advice"):
            el += [Spacer(1, 6), Paragraph("<b>Recommendation:</b> " + _esc(conf["advice"]), B)]
        for w in summ.get("warnings", []):
            el.append(Paragraph("Warning: " + _esc(w), S))
        srcs = summ.get("sources") or []
        if srcs:
            el.append(Paragraph("Elevation sources used", H2))
            el.append(tbl([["Source", "Kind", "Priority", "Cells used"]] + [[s.get("name"), s.get("kind"), s.get("priority"), s.get("cells_used", s.get("cells", ""))] for s in srcs],
                          [3.0 * inch, 1.5 * inch, 1.0 * inch, 1.5 * inch]))

    # ---- figures
    figs = []
    try:
        figs = _figures(ex)
    except Exception as e:                                                          # noqa: BLE001
        ex.ctx.log(f"figures skipped: {type(e).__name__}: {e}", "warning")
    if figs:
        el.append(Paragraph("Maps", H2))
        for title, img, note in figs:
            h, w = img.shape[:2]
            width = 4.3 * inch
            fig = [Paragraph("<b>" + _esc(title) + "</b>", B), Image(_png(img), width=width, height=width * h / w),
                   Paragraph(_esc(note), S), Spacer(1, 8)]
            el.append(KeepTogether(fig))

    # ---- land cover
    mask = None
    if ex.product("orthophoto"):
        mask, _ = O.class_rasters(ex.app, ex.pid)
    lc_head = Paragraph("Land-cover statistics (visual classification)", H2)
    if mask is None:
        el += [lc_head, Paragraph("No orthomosaic, so no classification.", B)]
    else:
        g = ex.product("orthophoto")["meta"].get("grid")
        unit = None
        if g and p["horizontal_units"] != "degree":
            f = {"metre": 1.0, "foot": 0.3048, "us_survey_foot": 1200 / 3937}.get(p["horizontal_units"], 1.0)
            unit = (g["res"] * f) ** 2
        h = mask.histogram()
        total = mask.width * mask.height
        rows = [["Class", "Share", "Area (m2)", "Manual px", "Auto px"]]
        for c in ex.classes:
            n = {"none": 0, "manual": 0, "automatic": 0}
            for v, cnt in h.items():
                if v & K.CLASS_MASK == c["id"]:
                    n["manual" if v & K.TAG_MANUAL else ("automatic" if v & K.TAG_AUTO else "none")] += cnt
            px = sum(n.values())
            rows.append([c["name"], f"{100 * px / total:.1f}%" if total else "-", f"{px * unit:,.0f}" if unit else "n/a", f"{n['manual']:,}", f"{n['automatic']:,}"])
        t = tbl(rows, [2.6 * inch, 0.9 * inch, 1.3 * inch, 1.1 * inch, 1.1 * inch])
        sty = []
        for i, c in enumerate(ex.classes, 1):
            sty.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(c["color"])))
            sty.append(("TEXTCOLOR", (0, i), (0, i), colors.white if sum(C.hex_to_rgb(c["color"])) < 380 else colors.black))
        t.setStyle(TableStyle(sty))
        el.append(KeepTogether([lc_head, t]))
        el.append(Paragraph(_esc("Automatic labels are model output and have not been reviewed unless counted under Manual. " + ""), S))

    # ---- footprints
    reps = ex.footprint_reports()
    if reps:
        head = Paragraph("Proposed footprints (preliminary planning estimate)", H2)
        for k, r in enumerate(reps):
            rep = r["report"]
            rows = [["Measure", "Value"]]
            for k, v in rep.items():
                if k in ("disclaimer", "warnings"):
                    continue
                if isinstance(v, (int, float, str, bool)) and not isinstance(v, bool):
                    rows.append([k.replace("_", " "), f"{v:.3f}" if isinstance(v, float) else str(v)])
                elif isinstance(v, dict):
                    rows.append([k.replace("_", " "), ", ".join(f"{a}: {b:.3f}" if isinstance(b, float) else f"{a}: {b}" for a, b in v.items())[:300]])
            el.append(KeepTogether(([head] if k == 0 else []) + [Paragraph("<b>" + _esc(r["name"]) + "</b>", B), tbl(rows, [2.6 * inch, 4.4 * inch])]
                                   + [Paragraph("Warning: " + _esc(w), S) for w in rep.get("warnings", [])]))
            el.append(Spacer(1, 6))

    # ---- photogrammetry quality
    q = ex.product("odm_quality")
    if q and q["meta"].get("checks"):
        el.append(Paragraph("Photogrammetry quality checks", H2))
        el.append(tbl([["Check", "Status", "Detail"]] + [[c["name"], c["status"], c["message"]] for c in q["meta"]["checks"]], [2.0 * inch, 0.8 * inch, 4.2 * inch]))

    el.append(Paragraph("Limitations", H2))
    for t in ("Photogrammetry sees the top of the vegetation canopy. Ground under dense canopy is inferred from the few points that reach it, or from external data.",
              "Interpolated cells can be wrong by more than the reported cross-validation error, especially in gullies, banks and under dense canopy.",
              "Cut/fill volumes are only reported where evidence supports them and carry the same limitations.",
              "Obtain supplemental survey data (LiDAR, total station or GNSS ground shots) wherever confidence is low or unknown."):
        el.append(Paragraph("&bull; " + _esc(t), B))
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                            title=f"{p['name']} - terrain report", author="TerraScope")
    doc.build(el, onFirstPage=_footer, onLaterPages=_footer)
