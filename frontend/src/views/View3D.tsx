import { useEffect, useMemo, useRef, useState } from 'react';
import { Project, api, url } from '../api';
import { Check, Chip, Empty, Field, Kv, Notice, Progress, RightPanel, Section, Slider, SurfaceBadge } from '../ui';
import { Cam, Frame, GLPane, Heightfield, LineSet, PointData, RenderOpts, V3, fromScene, hfBounds, makeHeightfield, sampleHF, toScene } from '../gl3d';
import { MaskLayer, OrthoLayer } from '../tiles';
import { Viewport } from '../viewport';
import { RAMP_NAMES, rampCss } from '../ramps';
import { bump, setApp, toast, useApp, useProject } from '../store';
import { boundaryRings, elevLabel, errMsg, fmtNum, lengthLabel, toMetres, usePoll } from '../util';
import { DISCLAIMER, downloadBlob } from './terrainPanels';
import { MeshView } from './MeshView';

type ColorBy = 'ortho' | 'classes' | 'elevation' | 'confidence' | 'evidence';
const CLS_NAMES: Record<number, string> = { 0: 'never classified', 1: 'unclassified', 2: 'ground', 3: 'low vegetation', 4: 'medium vegetation', 5: 'high vegetation', 6: 'building', 7: 'noise / removed' };
const CLS_RGB: V3[] = [[0.55, 0.55, 0.55], [0.75, 0.75, 0.75], [0.62, 0.42, 0.2], [0.6, 0.85, 0.35], [0.25, 0.65, 0.25], [0.06, 0.35, 0.12], [0.85, 0.2, 0.2], [0.6, 0.2, 0.75]];
const EDIT_CLASSES: [number, string][] = [[2, 'ground'], [1, 'unclassified'], [3, 'low vegetation'], [4, 'medium vegetation'], [5, 'high vegetation'], [6, 'building'], [7, 'noise']];
const CLS_RGB_EDIT: V3[] = Array.from({ length: 32 }, (_, i) => (i === 31 ? [1, 0.88, 0.1] : CLS_RGB[i] ?? [0.6, 0.6, 0.6]));
const inPoly = (x: number, y: number, p: [number, number][]) => { let c = false; for (let i = 0, j = p.length - 1; i < p.length; j = i++) { if ((p[i][1] > y) !== (p[j][1] > y) && x < ((p[j][0] - p[i][0]) * (y - p[i][1])) / (p[j][1] - p[i][1]) + p[i][0]) c = !c; } return c; };
interface MPoint { x: number; y: number; z: number; surf: 'observed' | 'estimated' }
interface PaneReg { pane: GLPane; overlay: HTMLCanvasElement }

async function fetchHF(pid: string, layer: string, maxdim: number): Promise<Heightfield | null> {
  const r = await fetch(url(`/projects/${pid}/terrain/heightmap/${layer}?maxdim=${maxdim}`));
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`${layer} heightmap: HTTP ${r.status}`);
  return makeHeightfield(JSON.parse(r.headers.get('X-Grid') ?? '{}'), await r.arrayBuffer());
}
async function fetchCloud(pid: string, max: number): Promise<{ meta: any; raw: { n: number; origin: number[]; x: Float32Array; y: Float32Array; z: Float32Array; cls: Uint8Array; rgb: Uint8Array | null } } | null> {
  const r = await fetch(url(`/projects/${pid}/pointcloud/data?max=${max}`), { cache: 'no-store' });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`point cloud: HTTP ${r.status}`);
  const m = JSON.parse(r.headers.get('X-Meta') ?? '{}'), buf = await r.arrayBuffer(), n = m.n as number, pad = (4 - (n % 4)) % 4;
  const x = new Float32Array(buf, 0, n), y = new Float32Array(buf, 4 * n, n), z = new Float32Array(buf, 8 * n, n), cls = new Uint8Array(buf, 12 * n, n);
  const rgb = m.has_rgb ? new Uint8Array(buf, 12 * n + n + pad, n * 3) : null;
  return { meta: m, raw: { n, origin: m.origin, x, y, z, cls, rgb } };
}
const loadImg = (u: string) => new Promise<HTMLImageElement | null>((res) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => res(null); i.src = u; });

export function View3D() {
  const p = useProject()!;
  const jt = useApp((s) => s.jobsTick);
  const ts = usePoll(() => api.terrain(p.id), 0, [p.id, jt]);
  const cm = usePoll(() => api.cloudMeta(p.id), 0, [p.id, jt]);
  if (ts.loading || cm.loading) return <div className="m-auto text-slate-500">Loading…</div>;
  if (!ts.data) return <div className="m-auto text-warn">{ts.error}</div>;
  if (!ts.data.available && !cm.data?.available) return <Empty title="Nothing to show in 3D yet">Run photogrammetry, then terrain reconstruction (Terrain 2D tab). The 3D view shows the observed surface (DSM), the estimated bare-earth terrain (DTM) and the point cloud.</Empty>;
  return <Scene p={p} ts={ts.data} cm={cm.data} reloadCloud={cm.reload} key={p.id} />;
}

function Scene({ p, ts, cm, reloadCloud }: { p: Project; ts: any; cm: any; reloadCloud: () => void }) {
  const pid = p.id;
  const zexag = useApp((s) => s.zexag), ramp = useApp((s) => s.ramp);
  const panes = useRef<(PaneReg | null)[]>([null, null]);
  const [split, setSplit] = useState(false);
  const [sideOrtho, setSideOrtho] = useState(false);
  const [meshMode, setMeshMode] = useState(false);
  const meshApi = useRef<{ preset: (k: 'top' | 'oblique' | 'reset') => void } | null>(null);
  const jobsTick = useApp((x) => x.jobsTick);
  const sideRef = useRef<HTMLCanvasElement | null>(null), linkRef = useRef<HTMLCanvasElement | null>(null), linkDrag = useRef(false);
  const [detail, setDetail] = useState(400);
  const [hfsRaw, setHfs] = useState<{ dtm: Heightfield | null; dsm: Heightfield | null }>({ dtm: null, dsm: null });
  const [showEdited, setShowEdited] = useState(false);
  const [editedHf, setEditedHf] = useState<Heightfield | null>(null);
  const [surfBusy, setSurfBusy] = useState(false);
  // when the edited-surface preview is on, it takes the observed-surface slot (so meshing, picking and hover all use it)
  const hfs = useMemo(() => (showEdited && editedHf ? { ...hfsRaw, dsm: editedHf } : hfsRaw), [hfsRaw, showEdited, editedHf]);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [show, setShow] = useState({ dtm: true, dsm: false, points: false, lines: true });
  const [alpha, setAlpha] = useState({ dtm: 1, dsm: 0.6 });
  const [colorBy, setColorBy] = useState<ColorBy>('ortho');
  const [shade, setShade] = useState(true);
  const [pointMode, setPointMode] = useState<0 | 1 | 2>(1);
  const [pointSize, setPointSize] = useState(0.35);
  const [clsVis, setClsVis] = useState<boolean[]>(Array.from({ length: 32 }, (_, i) => i !== 7));   // class 7 = removed/noise: hidden by default
  const [cloud, setCloud] = useState<PointData | null>(null);
  const [cloudBusy, setCloudBusy] = useState(false);
  const [cloudJob, setCloudJob] = useState<string | null>(null);
  const [measure, setMeasure] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editTool, setEditTool] = useState<'lasso' | 'box'>('lasso');
  const [selMode, setSelMode] = useState<'new' | 'add' | 'remove'>('new');
  const [selCount, setSelCount] = useState(0);
  const [target, setTarget] = useState(2);
  const [editInfo, setEditInfo] = useState<any>(null);
  const [editBusy, setEditBusy] = useState(false);
  const [liveCounts, setLiveCounts] = useState<Record<string, number> | null>(null);
  const [cloudComplete, setCloudComplete] = useState<boolean | null>(null);
  const selMask = useRef<Uint8Array | null>(null), dispCls = useRef<Uint8Array | null>(null);
  const sketch = useRef<{ tool: string; idx: number; pts: [number, number][] } | null>(null);
  const [mpts, setMpts] = useState<MPoint[]>([]);
  const [hover, setHover] = useState<{ x: number; y: number; dsm: number; dtm: number } | null>(null);
  const [glErr, setGlErr] = useState<string | null>(null);
  const [drapeBusy, setDrapeBusy] = useState(false);
  const cam = useRef<Cam>({ tx: 0, ty: 0, tz: 0, yaw: 0.5, pitch: 0.75, dist: 100 });
  const camInit = useRef(false);
  const raf = useRef(0);

  // ------------------------------------------------------------ frame from data
  const ref = hfs.dtm ?? hfs.dsm;
  const mh = isFinite(toMetres(p.horizontal_units)) ? toMetres(p.horizontal_units) : 1, mv = isFinite(toMetres(p.vertical_units)) ? toMetres(p.vertical_units) : 1;
  const bounds = useMemo(() => (ref ? hfBounds(ref) : null), [ref]);
  const frame: Frame | null = useMemo(() => (ref && bounds ? { cx: (bounds[0] + bounds[2]) / 2, cy: (bounds[1] + bounds[3]) / 2, zref: (ref.zmin + ref.zmax) / 2, mh, mv } : null), [ref, bounds, mh, mv]);
  const radius = bounds ? Math.hypot((bounds[2] - bounds[0]) * mh, (bounds[3] - bounds[1]) * mh) / 2 : 100;
  const vmin = Math.min(...[hfs.dtm?.p2, hfs.dsm?.p2].filter((v) => v != null) as number[]), vmax = Math.max(...[hfs.dtm?.p98, hfs.dsm?.p98].filter((v) => v != null) as number[]);
  const rng = isFinite(vmin) && isFinite(vmax) ? { min: vmin, max: vmax } : { min: 0, max: 1 };

  // the latest of everything, for the render loop
  const st = useRef<any>({}); st.current = { show, alpha, colorBy, shade, pointMode, pointSize, clsVis, split, zexag, ramp, frame, hfs, radius, mpts, measure, rng, mh, mv, bounds, hover, showEdited, editing, editTool, selMode, target, meshMode };
  const schedule = () => { if (!raf.current) raf.current = requestAnimationFrame(renderAll); };

  const optsFor = (idx: number): RenderOpts => {
    const s = st.current; const f: Frame = s.frame;
    const sh = s.split ? (idx === 0 ? { dtm: false, dsm: !!s.hfs.dsm, points: s.show.points, lines: s.show.lines } : { dtm: true, dsm: false, points: s.show.points, lines: s.show.lines }) : { dtm: s.show.dtm, dsm: s.show.dsm, points: s.show.points, lines: s.show.lines };
    return { cam: cam.current, zexag: s.zexag, radius: s.radius, zref: f?.zref ?? 0, mv: s.mv, vmin: s.rng.min, vmax: s.rng.max, ramp: s.ramp, show: sh, alpha: s.split ? { dtm: 1, dsm: 1 } : s.alpha,
      colorBy: s.colorBy === 'elevation' ? 'ramp' : 'tex', shade: s.shade, pointMode: s.pointMode, pointSize: s.pointSize, clsVis: s.clsVis, clsCol: CLS_RGB_EDIT, bg: [0.86, 0.89, 0.92] };
  };
  useEffect(() => { schedule(); /* eslint-disable-next-line */ }, [meshMode]);
  const surfaceFor = (idx: number): 'dsm' | 'dtm' => { const s = st.current; return s.split ? (idx === 0 ? 'dsm' : 'dtm') : (s.show.dsm && s.hfs.dsm && (!s.show.dtm || s.alpha.dsm > 0.5) ? 'dsm' : s.hfs.dtm ? 'dtm' : 'dsm'); };

  // Linked view: marker + approximate footprint of the 3D camera on the flat orthomosaic panel (and the reverse: click/drag there to move the 3D camera).
  const drawLink = () => {
    const lc = linkRef.current, s = st.current, f: Frame | null = s.frame, bb = s.bounds;
    if (!lc || !f || !bb) return; const W = lc.width, H = lc.height, c = lc.getContext('2d')!; c.clearRect(0, 0, W, H);
    if (s.meshMode) return;                       // the photo model has its own camera; the footprint box would describe the hidden terrain view
    const P = (E: number, N: number): [number, number] => [((E - bb[0]) / (bb[2] - bb[0])) * W, ((bb[3] - N) / (bb[3] - bb[1])) * H];
    const k = cam.current, g = panes.current[0]?.pane.canvas, asp = g && g.height ? g.width / g.height : 1.4;
    const tE = k.tx / f.mh + f.cx, tN = k.ty / f.mh + f.cy, hh = (k.dist * Math.tan(0.4)) / f.mh / Math.max(0.25, Math.sin(k.pitch)), hw = (k.dist * Math.tan(0.4) * asp) / f.mh;
    const fx = -Math.sin(k.yaw), fy = Math.cos(k.yaw), rx = Math.cos(k.yaw), ry = Math.sin(k.yaw);
    const corner = (a: number, b: number) => P(tE + rx * hw * a + fx * hh * b, tN + ry * hw * a + fy * hh * b);
    const q = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)], sc = W / 1000;
    c.save(); c.lineWidth = 3 * sc; c.strokeStyle = '#fff'; c.fillStyle = 'rgba(11,92,171,.14)'; c.beginPath(); q.forEach((a, i) => (i ? c.lineTo(a[0], a[1]) : c.moveTo(a[0], a[1]))); c.closePath(); c.fill(); c.stroke();
    c.lineWidth = 1.5 * sc; c.strokeStyle = '#0b5cab'; c.stroke();
    const [cx, cy] = P(tE, tN), r = 9 * sc; c.lineWidth = 2.5 * sc; c.strokeStyle = '#d9480f'; c.beginPath(); c.moveTo(cx - r, cy); c.lineTo(cx + r, cy); c.moveTo(cx, cy - r); c.lineTo(cx, cy + r); c.stroke();
    c.restore();
  };
  const linkMove = (e: { clientX: number; clientY: number }) => {
    const lc = linkRef.current, s = st.current, f: Frame | null = s.frame, bb = s.bounds; if (!lc || !f || !bb) return;
    const r = lc.getBoundingClientRect(), sc = Math.min(r.width / lc.width, r.height / lc.height), w = lc.width * sc, h = lc.height * sc, ox = r.left + (r.width - w) / 2, oy = r.top + (r.height - h) / 2;
    const u = (e.clientX - ox) / w, v = (e.clientY - oy) / h; if (u < 0 || u > 1 || v < 0 || v > 1) return;
    const E = bb[0] + u * (bb[2] - bb[0]), N = bb[3] - v * (bb[3] - bb[1]); cam.current.tx = (E - f.cx) * f.mh; cam.current.ty = (N - f.cy) * f.mh; schedule();
  };
  const renderAll = () => {
    raf.current = 0; const s = st.current; if (!s.frame) return;
    panes.current.forEach((pr, idx) => {
      if (!pr || (idx === 1 && !s.split)) return;
      try { pr.pane.render(optsFor(idx)); } catch (e) { setGlErr(errMsg(e)); return; }
      drawOverlay(idx, pr);
    });
    drawLink();
  };

  const drawOverlay = (idx: number, pr: PaneReg) => {
    const s = st.current, f: Frame = s.frame, cv = pr.overlay, dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight; if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const c = cv.getContext('2d')!; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
    const surf = surfaceFor(idx), hf: Heightfield | null = s.hfs[surf];
    const pts: { sx: number; sy: number }[] = [];
    for (const q of s.mpts as MPoint[]) {
      const z = hf ? sampleHF(hf, q.x, q.y) : q.z; const sc = toScene(f, q.x, q.y, isFinite(z) ? z : q.z); const pj = pr.pane.project(sc, s.zexag); if (pj) pts.push({ sx: pj[0], sy: pj[1] });
    }
    if (pts.length) {
      c.strokeStyle = '#d9480f'; c.lineWidth = 2; c.beginPath(); pts.forEach((q, i) => (i ? c.lineTo(q.sx, q.sy) : c.moveTo(q.sx, q.sy))); c.stroke();
      pts.forEach((q, i) => { c.fillStyle = '#fff'; c.strokeStyle = '#d9480f'; c.lineWidth = 2; c.beginPath(); c.arc(q.sx, q.sy, 5, 0, Math.PI * 2); c.fill(); c.stroke(); c.fillStyle = '#7c2d12'; c.font = 'bold 11px system-ui'; c.fillText(String(i + 1), q.sx + 8, q.sy - 6); });
    }
    // labels: what this pane shows
    const lab = (t: string, col: string, x: number, y: number) => { c.font = 'bold 11px system-ui'; const wd = c.measureText(t).width + 12; c.fillStyle = col; c.fillRect(x, y, wd, 20); c.fillStyle = '#fff'; c.fillText(t, x + 6, y + 14); return wd; };
    const o = optsFor(idx); let x = 8;
    if (o.show.dsm && s.showEdited) x += lab('EDITED surface (preview, removed features filled)', '#6d28d9', x, 8) + 6;
    else if (o.show.dsm) x += lab(`OBSERVED surface (DSM)${!s.split && o.alpha.dsm < 1 ? ` · ${Math.round(o.alpha.dsm * 100)}%` : ''}`, '#0b5cab', x, 8) + 6;
    if (o.show.dtm) x += lab(`ESTIMATED bare earth (DTM)${!s.split && o.alpha.dtm < 1 ? ` · ${Math.round(o.alpha.dtm * 100)}%` : ''}`, '#b45309', x, 8) + 6;
    if (o.show.points) x += lab('Point cloud (observed returns)', '#475569', x, 8) + 6;
    if (s.editing) lab('EDIT MODE: selected points are yellow', '#a16207', x, 8);
    const sk = sketch.current;
    if (sk && sk.idx === idx && sk.pts.length > 1) {
      c.save(); c.strokeStyle = '#a16207'; c.fillStyle = 'rgba(250,204,21,.18)'; c.lineWidth = 1.5; c.setLineDash([5, 3]); c.beginPath();
      if (sk.tool === 'box') { const [a, b] = [sk.pts[0], sk.pts[sk.pts.length - 1]]; c.rect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])); }
      else { sk.pts.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.closePath(); }
      c.fill(); c.stroke(); c.restore();
    }
    c.fillStyle = 'rgba(255,255,255,.85)'; c.fillRect(8, h - 24, 250, 18); c.fillStyle = '#111'; c.font = '11px system-ui';
    c.fillText(`Vertical exaggeration ${s.zexag}× · preliminary planning estimate`, 12, h - 11);
    // hover marker
    if (s.hover && !s.measure) { const z = hf ? sampleHF(hf, s.hover.x, s.hover.y) : NaN; if (isFinite(z)) { const pj = pr.pane.project(toScene(f, s.hover.x, s.hover.y, z), s.zexag); if (pj) { c.strokeStyle = '#111'; c.lineWidth = 1.5; c.beginPath(); c.arc(pj[0], pj[1], 6, 0, Math.PI * 2); c.stroke(); } } }
  };

  // ------------------------------------------------------------ data loading
  useEffect(() => {
    let alive = true; setLoading(true); setLoadErr(null);
    (async () => {
      try {
        const [dtm, dsm] = await Promise.all([ts.layers.dtm ? fetchHF(pid, 'dtm', detail) : null, ts.layers.dsm ? fetchHF(pid, 'dsm', detail) : null]);
        if (alive) { setHfs({ dtm, dsm }); if (!dtm && dsm) setShow((s) => ({ ...s, dtm: false, dsm: true })); }
      } catch (e) { if (alive) setLoadErr(errMsg(e)); } finally { if (alive) setLoading(false); }
    })();
    return () => { alive = false; };
  }, [pid, detail, ts]);

  const cloudPts = useRef<any>(null);
  useEffect(() => {
    if (!show.points || cloudPts.current || !cm?.available) return;
    let alive = true; setCloudBusy(true);
    fetchCloud(pid, 2_500_000).then((r) => { if (alive && r) { cloudPts.current = r; dispCls.current = new Uint8Array(r.raw.cls); selMask.current = new Uint8Array(r.raw.n); setCloudComplete(!!r.meta.complete); setCloud({ n: r.raw.n, pos: new Float32Array(0), cls: r.raw.cls, rgb: r.raw.rgb }); } }).catch((e) => toast('error', errMsg(e))).finally(() => alive && setCloudBusy(false));
    return () => { alive = false; };
  }, [show.points, cm?.available, pid]);
  const cloudJobQ = usePoll(() => (cloudJob ? api.job(cloudJob) : Promise.resolve(null)), 1500, [cloudJob]);
  useEffect(() => { if (cloudJobQ.data?.status === 'succeeded') { setCloudJob(null); reloadCloud(); bump('jobsTick'); } /* eslint-disable-next-line */ }, [cloudJobQ.data?.status]);

  // camera framing once data exists
  useEffect(() => {
    if (!frame || !ref || camInit.current) return; camInit.current = true;
    cam.current = { tx: 0, ty: 0, tz: ((ref.p2 + ref.p98) / 2 - frame.zref) * mv, yaw: 0.55, pitch: 0.7, dist: Math.max(30, radius * 1.9) };
    schedule();
  }, [frame]);

  // class counts / names
  const clsCounts: Record<string, number> = liveCounts ?? cm?.classes ?? {};

  // ------------------------------------------------------------ drape texture (imagery, classes, confidence/evidence)
  const orthoQ = usePoll(() => api.ortho(pid), 0, [pid]);
  const cinfoQ = usePoll(() => (orthoQ.data?.available ? api.classInfo(pid) : Promise.resolve(null)), 0, [pid, orthoQ.data?.available]);
  const drapeCanvas = useRef<HTMLCanvasElement | null>(null);
  const rebuildTimer = useRef<any>(0), rebuildCount = useRef(0);
  const orthoLayer = useMemo(() => (orthoQ.data?.available && orthoQ.data.pyramid ? new OrthoLayer(pid, orthoQ.data, () => rebuildSoon()) : null), [pid, orthoQ.data]);
  const mask = useMemo(() => {
    if (!orthoLayer || !cinfoQ.data) return null;
    const m = new MaskLayer(pid, cinfoQ.data.width, cinfoQ.data.height, orthoLayer.grid, () => rebuildSoon());
    m.setVersion(cinfoQ.data.version); m.setClasses(cinfoQ.data.classes, new Set(cinfoQ.data.classes.filter((c) => c.visible === false).map((c) => c.id))); return m;
  }, [orthoLayer, cinfoQ.data]);
  const rebuildRef = useRef<() => void>(() => {});
  const rebuildSoon = () => { clearTimeout(rebuildTimer.current); if (rebuildCount.current > 40) return; rebuildTimer.current = setTimeout(() => rebuildRef.current(), 250); };
  const applyDrape = () => { const cv = drapeCanvas.current; panes.current.forEach((pr) => { if (pr && cv) pr.pane.setDrape(cv); }); schedule(); };
  rebuildRef.current = async () => {
    if (bounds && sideOrtho && orthoLayer && sideRef.current) { // flat imagery panel beside the 3D view (imagery only: no classes, no elevation colouring)
      const bb = bounds, asp = (bb[2] - bb[0]) / (bb[3] - bb[1]), W = asp >= 1 ? 1600 : Math.max(256, Math.round(1600 * asp)), H = asp >= 1 ? Math.max(256, Math.round(1600 / asp)) : 1600;
      if (linkRef.current) { linkRef.current.width = W; linkRef.current.height = H; }
      const sc = sideRef.current; sc.width = W; sc.height = H; const sx = sc.getContext('2d')!; const v2 = new Viewport(); v2.w = W; v2.h = H; v2.fit(bb, 1);
      sx.fillStyle = '#8a8f87'; sx.fillRect(0, 0, W, H); orthoLayer.draw(sx, v2, 1); schedule();
    }
    if (!bounds || colorBy === 'elevation') return; rebuildCount.current++;
    const bb = bounds, asp = (bb[2] - bb[0]) / (bb[3] - bb[1]), W = asp >= 1 ? 2048 : Math.max(256, Math.round(2048 * asp)), H = asp >= 1 ? Math.max(256, Math.round(2048 / asp)) : 2048;
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H; const c = cv.getContext('2d')!;
    const vp = new Viewport(); vp.w = W; vp.h = H; vp.fit(bb, 1);
    c.fillStyle = '#8a8f87'; c.fillRect(0, 0, W, H);
    orthoLayer?.draw(c, vp, 1);
    if (colorBy === 'classes' && mask) mask.draw(c, vp, 0.62);
    if (colorBy === 'confidence' || colorBy === 'evidence') {
      const im = await loadImg(url(`/projects/${pid}/terrain/render/${colorBy}?bbox=${bb.join(',')}&w=${W >> 1}&h=${H >> 1}`));
      if (im) { c.imageSmoothingEnabled = false; c.drawImage(im, 0, 0, W, H); }
    }
    drapeCanvas.current = cv; applyDrape();
  };
  useEffect(() => { rebuildCount.current = 0; setDrapeBusy(true); rebuildRef.current(); const t = setTimeout(() => setDrapeBusy(false), 1200); return () => clearTimeout(t); /* eslint-disable-next-line */ }, [colorBy, orthoLayer, mask, bounds, sideOrtho]);
  useEffect(() => { if (colorBy !== 'elevation') return; panes.current.forEach((pr) => pr?.pane.setDrape(null)); schedule(); }, [colorBy]);

  // ------------------------------------------------------------ push data to panes
  const pushData = (pr: PaneReg) => {
    const f = st.current.frame as Frame | null; if (!f || !bounds) return;
    pr.pane.setHeightfield('dtm', hfs.dtm, f, bounds); pr.pane.setHeightfield('dsm', hfs.dsm, f, bounds);
    if (drapeCanvas.current && st.current.colorBy !== 'elevation') pr.pane.setDrape(drapeCanvas.current);
    pr.pane.setPoints(pointData.current); pr.pane.setLines(lineSet.current);
  };
  const pointData = useRef<PointData | null>(null), lineSet = useRef<LineSet | null>(null);

  // points -> scene coordinates
  useEffect(() => {
    const r = cloudPts.current?.raw, f = frame; if (!r || !f) return;
    const pos = new Float32Array(r.n * 3), o = r.origin;
    for (let i = 0; i < r.n; i++) { pos[i * 3] = (o[0] + r.x[i] - f.cx) * f.mh; pos[i * 3 + 1] = (o[1] + r.y[i] - f.cy) * f.mh; pos[i * 3 + 2] = (o[2] + r.z[i] - f.zref) * f.mv; }
    pointData.current = { n: r.n, pos, cls: dispCls.current ?? r.cls, rgb: r.rgb };
    panes.current.forEach((pr) => pr?.pane.setPoints(pointData.current)); schedule();
  }, [cloud, frame]);

  // boundary + saved features draped on the terrain
  const featsQ = usePoll(() => api.features(pid), 0, [pid]);
  useEffect(() => {
    const f = frame; if (!f || !ref) return; const hf = hfs.dtm ?? hfs.dsm!; const lines: Float32Array[] = [];
    const drape = (ring: number[][], close: boolean) => {
      const pts: number[] = []; const rr = close ? [...ring, ring[0]] : ring;
      for (let i = 0; i < rr.length - 1; i++) {
        const a = rr[i], b = rr[i + 1], n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / (hf.res * 0.6)));
        for (let k = 0; k < n; k++) { const x = a[0] + ((b[0] - a[0]) * k) / n, y = a[1] + ((b[1] - a[1]) * k) / n, z = sampleHF(hf, x, y); if (isFinite(z)) pts.push(...toScene(f, x, y, z)); }
      }
      const e = rr[rr.length - 1], ze = sampleHF(hf, e[0], e[1]); if (isFinite(ze)) pts.push(...toScene(f, e[0], e[1], ze));
      if (pts.length >= 6) lines.push(new Float32Array(pts));
    };
    const rings = boundaryRings(p.boundary, p.epsg); if (rings) for (const r of rings) drape(r, true);
    for (const ft of featsQ.data ?? []) { const g = ft.geometry; if (!g || ft.kind === 'boundary') continue; if (g.type === 'Polygon') drape(g.coordinates[0], true); else if (g.type === 'LineString') drape(g.coordinates, false); }
    lineSet.current = { color: [0.95, 0.55, 0.05, 1], lines };
    panes.current.forEach((pr) => pr?.pane.setLines(lineSet.current)); schedule();
  }, [frame, hfs, p.boundary, p.epsg, featsQ.data]);

  // heightfields changed -> re-upload
  useEffect(() => { panes.current.forEach((pr) => pr && pushData(pr)); schedule(); /* eslint-disable-next-line */ }, [hfs, frame, bounds]);
  useEffect(() => { schedule(); });

  // ------------------------------------------------------------ interaction
  const onReady = (idx: number, pr: PaneReg | null) => { panes.current[idx] = pr; if (pr) { pushData(pr); schedule(); } };
  const pickAt = (idx: number, sx: number, sy: number): MPoint | null => {
    const s = st.current, pr = panes.current[idx]; if (!pr || !s.frame) return null;
    const surf = surfaceFor(idx), hf: Heightfield | null = s.hfs[surf]; if (!hf) return null;
    const hit = pr.pane.pick(sx, sy, hf, s.frame, s.zexag, cam.current.dist * 5 + s.radius * 3); if (!hit) return null;
    const m = fromScene(s.frame, hit); return { x: m[0], y: m[1], z: m[2], surf: surf === 'dsm' ? 'observed' : 'estimated' };
  };
  const drag = useRef<{ x: number; y: number; moved: number; mode: 'orbit' | 'pan' | 'select' } | null>(null);
  const hoverTimer = useRef(0);
  const handlers = (idx: number) => ({
    down: (e: any, el: HTMLElement) => {
      el.setPointerCapture(e.pointerId);
      const pick = st.current.editing && e.button === 0 && !e.altKey && !e.shiftKey;
      drag.current = { x: e.clientX, y: e.clientY, moved: 0, mode: pick ? 'select' : e.button === 2 || e.button === 1 || e.shiftKey ? 'pan' : 'orbit' };
      if (pick) { const r = el.getBoundingClientRect(); sketch.current = { tool: st.current.editTool, idx, pts: [[e.clientX - r.left, e.clientY - r.top]] }; }
    },
    move: (e: any, el: HTMLElement) => {
      const r = el.getBoundingClientRect(), d = drag.current;
      if (d) {
        if (d.mode === 'select' && sketch.current) {
          const q: [number, number] = [e.clientX - r.left, e.clientY - r.top], sk = sketch.current; d.moved += 1;
          if (sk.tool === 'box') sk.pts = [sk.pts[0], q]; else sk.pts.push(q);
          schedule(); return;
        }
        const dx = e.clientX - d.x, dy = e.clientY - d.y; d.x = e.clientX; d.y = e.clientY; d.moved += Math.abs(dx) + Math.abs(dy); const c = cam.current;
        if (d.mode === 'orbit') { c.yaw -= dx * 0.0065; c.pitch = Math.min(1.5, Math.max(0.06, c.pitch + dy * 0.0055)); }
        else { const k = (c.dist * Math.tan(0.4) * 2) / Math.max(1, r.height), sc = 1 / Math.max(0.25, Math.sin(c.pitch)); c.tx += (-Math.cos(c.yaw) * dx - Math.sin(c.yaw) * dy * sc) * k; c.ty += (-Math.sin(c.yaw) * dx + Math.cos(c.yaw) * dy * sc) * k; }
        schedule(); return;
      }
      const t = performance.now(); if (t - hoverTimer.current < 50) return; hoverTimer.current = t;
      const hit = pickAt(idx, e.clientX - r.left, e.clientY - r.top);
      if (hit) { const s = st.current; setHover({ x: hit.x, y: hit.y, dsm: s.hfs.dsm ? sampleHF(s.hfs.dsm, hit.x, hit.y) : NaN, dtm: s.hfs.dtm ? sampleHF(s.hfs.dtm, hit.x, hit.y) : NaN }); } else setHover(null);
      schedule();
    },
    up: (e: any, el: HTMLElement) => {
      const d = drag.current; drag.current = null; try { el.releasePointerCapture(e.pointerId); } catch { /* not captured */ }
      if (d?.mode === 'select') { const sk = sketch.current; sketch.current = null; if (sk) finishSelect(sk); schedule(); return; }
      if (d && d.moved < 5 && st.current.measure && e.button === 0) { const r = el.getBoundingClientRect(); const hit = pickAt(idx, e.clientX - r.left, e.clientY - r.top); if (hit) setMpts((a) => [...a, hit]); }
    },
    wheel: (e: WheelEvent) => { e.preventDefault(); cam.current.dist = Math.min(radius * 12, Math.max(2, cam.current.dist * Math.exp(Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 120) * 0.0014))); schedule(); },
    leave: () => { setHover(null); schedule(); },
    resize: () => schedule(),
  });

  // ------------------------------------------------------------ point editing (display sample only)
  const fetchSurface = async () => {
    setSurfBusy(true);
    try {
      const r = await fetch(url(`/projects/${pid}/pointcloud/edit/surface?maxdim=${detail}`), { cache: 'no-store' });
      if (!r.ok) { const j = await r.json().catch(() => null); throw new Error(j?.error?.message ?? `surface: HTTP ${r.status}`); }
      setEditedHf(makeHeightfield(JSON.parse(r.headers.get('X-Grid') ?? '{}'), await r.arrayBuffer()));
    } catch (e) { toast('error', errMsg(e)); setShowEdited(false); } finally { setSurfBusy(false); }
  };
  useEffect(() => { if (showEdited) fetchSurface(); /* eslint-disable-next-line */ }, [showEdited, editInfo?.head, editInfo?.ops, detail]);
  const refreshHighlight = () => {
    const r = cloudPts.current?.raw, d = dispCls.current, sm = selMask.current; if (!r || !d || !sm) return;
    for (let i = 0; i < r.n; i++) d[i] = sm[i] ? 31 : r.cls[i];
    panes.current.forEach((pr) => pr?.pane.setPoints(pointData.current)); schedule();
  };
  const finishSelect = (sk: { tool: string; idx: number; pts: [number, number][] }) => {
    const r = cloudPts.current?.raw, pd = pointData.current, pr = panes.current[sk.idx], sm = selMask.current; if (!r || !pd || !pr || !sm) return;
    const box = sk.tool === 'box', pts = sk.pts;
    if (box ? pts.length < 2 || Math.abs(pts[0][0] - pts[1][0]) < 3 || Math.abs(pts[0][1] - pts[1][1]) < 3 : pts.length < 3) return;
    const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const scr = new Float32Array(r.n * 2); pr.pane.projectBuffer(pd.pos, r.n, st.current.zexag, scr);
    const vis: boolean[] = st.current.clsVis, mode = st.current.selMode as string;
    if (mode === 'new') sm.fill(0);
    const val = mode === 'remove' ? 0 : 1; let count = 0;
    for (let i = 0; i < r.n; i++) {
      const sx = scr[i * 2], sy = scr[i * 2 + 1];
      if (!(sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1) || vis[r.cls[i]] === false) continue;      // hidden classes are protected from selection
      if (!box && !inPoly(sx, sy, pts)) continue;
      sm[i] = val;
    }
    for (let i = 0; i < r.n; i++) if (sm[i]) count++;
    setSelCount(count); refreshHighlight();
  };
  const recount = () => { const r = cloudPts.current?.raw; if (!r) return; const c: Record<string, number> = {}; for (let i = 0; i < r.n; i++) c[r.cls[i]] = (c[r.cls[i]] ?? 0) + 1; setLiveCounts(c); };
  const clearSel = () => { selMask.current?.fill(0); setSelCount(0); refreshHighlight(); };
  const refreshEditInfo = async () => { try { setEditInfo(await api.cloudEditInfo(pid)); } catch { /* shown as unavailable */ } };
  useEffect(() => { if (cm?.available) refreshEditInfo(); /* eslint-disable-next-line */ }, [cm?.available, pid]);
  const applyEdit = async (cls: number) => {
    const r = cloudPts.current?.raw, sm = selMask.current; if (!r || !sm || editBusy) return;
    const ids: number[] = []; for (let i = 0; i < r.n; i++) if (sm[i] && r.cls[i] !== cls) ids.push(i);
    if (!ids.length) { toast('info', 'Nothing to change: the selected points already have that class.'); return; }
    setEditBusy(true);
    try {
      const res = await api.cloudEdit(pid, Uint32Array.from(ids), cls);
      for (const i of ids) r.cls[i] = cls;
      sm.fill(0); setSelCount(0); refreshHighlight(); setEditInfo((o: any) => ({ ...o, ...res })); recount();
      toast('ok', `${res.changed.toLocaleString()} points set to ${CLS_NAMES[cls]}`);
    } catch (e) { toast('error', errMsg(e)); } finally { setEditBusy(false); }
  };
  const stepEdit = async (dir: 'undo' | 'redo' | 'reset') => {
    if (editBusy) return;
    if (dir === 'reset' && !window.confirm('Discard ALL point-class edits and return to the classes produced by the ground filter?')) return;
    setEditBusy(true);
    try {
      const res = await (dir === 'undo' ? api.cloudEditUndo(pid) : dir === 'redo' ? api.cloudEditRedo(pid) : api.cloudEditReset(pid));
      const fresh = await fetchCloud(pid, 2_500_000), r = cloudPts.current?.raw;
      if (fresh && r && fresh.raw.n === r.n) { r.cls.set(fresh.raw.cls); selMask.current?.fill(0); setSelCount(0); refreshHighlight(); }
      setEditInfo((o: any) => ({ ...o, ...res })); recount();
    } catch (e) { toast('error', errMsg(e)); } finally { setEditBusy(false); }
  };
  const toggleEdit = () => {
    if (!editing) { setMeasure(false); setShow((o) => ({ ...o, points: true })); setPointMode(1); setEditing(true); }
    else { setEditing(false); clearSel(); sketch.current = null; }
  };
  const editRef = useRef<any>({}); editRef.current = { editing, stepEdit, clearSel };
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const t = (e.target as HTMLElement)?.tagName; if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || !editRef.current.editing) return;
      if (e.key === 'Escape') { editRef.current.clearSel(); return; }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); editRef.current.stepEdit(e.shiftKey ? 'redo' : 'undo'); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); editRef.current.stepEdit('redo'); }
    };
    window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k);
  }, []);

  const preset = (k: 'top' | 'oblique' | 'north' | 'reset') => {
    const c = cam.current; if (k === 'top') { c.pitch = 1.5; c.yaw = 0; } else if (k === 'oblique') { c.pitch = 0.62; c.yaw = 0.55; } else if (k === 'north') { c.pitch = 0.4; c.yaw = 0; }
    else if (ref && frame) { cam.current = { tx: 0, ty: 0, tz: ((ref.p2 + ref.p98) / 2 - frame.zref) * mv, yaw: 0.55, pitch: 0.7, dist: Math.max(30, radius * 1.9) }; }
    schedule();
  };
  const savePng = () => {
    const pr = panes.current[0]; if (!pr) return; renderAll();
    const out = document.createElement('canvas'); const n = split && panes.current[1] ? 2 : 1; const w = pr.pane.canvas.width, h = pr.pane.canvas.height; out.width = w * n; out.height = h;
    const c = out.getContext('2d')!; for (let i = 0; i < n; i++) { const q = panes.current[i]!; c.drawImage(q.pane.canvas, i * w, 0, w, h); c.drawImage(q.overlay, i * w, 0, w, h); }
    out.toBlob((b) => b && downloadBlob(b, `brooksbuild_3d_${new Date().toISOString().slice(0, 10)}.png`));
  };

  // ------------------------------------------------------------ measurement math
  const calc = useMemo(() => {
    const segs = mpts.slice(1).map((b, i) => {
      const a = mpts[i], hd = Math.hypot(b.x - a.x, b.y - a.y) * mh, dz = (b.z - a.z) * mv, d3 = Math.hypot(hd, dz);
      return { hd, dz, d3, slope: hd > 0 ? (dz / hd) * 100 : 0 };
    });
    return { segs, hd: segs.reduce((s, q) => s + q.hd, 0), d3: segs.reduce((s, q) => s + q.d3, 0) };
  }, [mpts, mh, mv]);

  if (glErr) return <div className="m-auto max-w-[520px] p-4"><Notice kind="warn"><b>3D view unavailable.</b> {glErr}. The 2D terrain views still work; WebGL2 is required here.</Notice></div>;
  const hasDsm = !!ts.layers.dsm, hasDtm = !!ts.layers.dtm;
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-1 border-b border-line bg-white px-2 py-1">
        <button className={`btn ${meshMode ? 'btn-on' : ''}`} onClick={() => { setMeshMode(!meshMode); if (!meshMode && editing) toggleEdit(); }}
          title="Orbit the photo-textured model from OpenDroneMap (observed imagery on the observed 3D surface, like a Polycam capture). Needs a Balanced or High photogrammetry run.">📷 Photo model</button>
        <span className="mx-1 h-5 w-px bg-line" />
        <button className="btn" onClick={() => (meshMode ? meshApi.current?.preset('oblique') : preset('oblique'))}>Oblique</button><button className="btn" onClick={() => (meshMode ? meshApi.current?.preset('top') : preset('top'))}>Top</button>
        {!meshMode && <button className="btn" onClick={() => preset('north')}>Low angle</button>}<button className="btn" onClick={() => (meshMode ? meshApi.current?.preset('reset') : preset('reset'))}>Reset</button>
        <span className="mx-1 h-5 w-px bg-line" />
        <button className={`btn ${editing ? 'btn-on' : ''}`} disabled={meshMode || !cm?.available || cloudComplete === false} onClick={toggleEdit}
          title={cm?.available ? 'Select points with a lasso or box and change their class (edits the display sample only)' : 'Build the point-cloud display sample first'}>✎ Edit points</button>
        <button className={`btn ${measure ? 'btn-on' : ''}`} disabled={meshMode} onClick={() => { setMeasure(!measure); if (!measure && editing) toggleEdit(); }} title="Click points on the surface to measure 3D distance and slope">⟷ Measure</button>
        {mpts.length > 0 && <button className="btn" onClick={() => setMpts([])}>Clear {mpts.length} pt</button>}
        <Check label="Split-screen: observed | estimated" checked={split} onChange={(v) => { setSplit(v); }} title="Left = DSM (what the camera saw). Right = DTM (estimated bare earth). Cameras are linked." />
        <Check label="Orthomosaic beside 3D" checked={sideOrtho} onChange={(v) => { rebuildCount.current = 0; setSideOrtho(v); }} title="Shows the stitched orthomosaic (flat, top-down imagery) next to the 3D view for comparison. Imagery only; it is not elevation evidence." />
        <div className="flex-1" />
        {loading && <span className="text-[12px] text-slate-500">Loading terrain…</span>}{drapeBusy && !loading && <span className="text-[12px] text-slate-500">Draping imagery…</span>}
        <button className="btn" onClick={savePng}>Save view (PNG)</button>
      </div>
      <div className="relative flex min-h-0 flex-1 bg-slate-300">
        {sideOrtho && (
          <div className="relative flex min-w-0 flex-1 items-center justify-center overflow-hidden border-r-2 border-white bg-slate-400">
            {orthoLayer ? <div className="relative h-full w-full"><canvas ref={sideRef} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              <canvas ref={linkRef} className="absolute inset-0" style={{ width: '100%', height: '100%', objectFit: 'contain', cursor: 'crosshair', touchAction: 'none' }}
                onPointerDown={(e) => { if (meshMode) return; linkDrag.current = true; (e.target as HTMLElement).setPointerCapture(e.pointerId); linkMove(e); }} onPointerMove={(e) => { if (linkDrag.current) linkMove(e); }} onPointerUp={() => { linkDrag.current = false; }} /></div> : <div className="p-3 text-[12px] text-slate-700">No orthomosaic yet. Build the 3D terrain model on the Upload tab (or upload an orthomosaic) and it will appear here.</div>}
            <span className="pointer-events-none absolute left-2 top-2 rounded bg-black/65 px-1.5 py-0.5 text-[11px] font-bold text-white">ORTHOMOSAIC · flat imagery{meshMode ? '' : ' · blue box = approx. 3D view area; click or drag to move the 3D view'}</span>
          </div>)}
        <div className="min-w-0 flex-1" style={{ display: meshMode ? 'none' : 'flex' }}>
          <PaneView idx={0} h={handlers(0)} onReady={onReady} onErr={setGlErr} cursor={editing ? "crosshair" : undefined} />
          {split && <PaneView idx={1} h={handlers(1)} onReady={onReady} onErr={setGlErr} cursor={editing ? "crosshair" : undefined} />}
        </div>
        {meshMode && <MeshView pid={pid} jobsTick={jobsTick} onApi={(a) => { meshApi.current = a; }} />}
        {loadErr && <div className="absolute left-2 top-10 z-10"><Notice kind="warn">{loadErr}</Notice></div>}
        {hover && !measure && (
          <div className="pointer-events-none absolute right-2 top-9 w-[230px] rounded border border-line bg-white/95 p-2 text-[12px] shadow">
            <Kv k="E, N" v={`${fmtNum(hover.x, 2)}, ${fmtNum(hover.y, 2)}`} mono />
            <Kv k={<span className="flex items-center gap-1">Surface <Chip kind="observed">OBSERVED</Chip></span>} v={elevLabel(p, hover.dsm)} mono />
            <Kv k={<span className="flex items-center gap-1">Ground <Chip kind="estimated">ESTIMATED</Chip></span>} v={elevLabel(p, hover.dtm)} mono />
            {isFinite(hover.dsm) && isFinite(hover.dtm) && <Kv k="Object height" v={elevLabel(p, hover.dsm - hover.dtm)} mono />}
          </div>)}
        <div style={{ display: meshMode ? "none" : undefined }} className="pointer-events-none absolute bottom-8 right-2 w-[210px] rounded border border-line bg-white/92 p-2 text-[11px] shadow">
          {colorBy === 'elevation' || !hasDtm ? <><div className="mb-0.5 font-semibold">Elevation ({p.vertical_units})</div><div className="h-2.5 rounded-sm border border-slate-400" style={{ background: rampCss(ramp) }} />
            <div className="flex justify-between font-mono"><span>{fmtNum(rng.min, 1)}</span><span>{fmtNum((rng.min + rng.max) / 2, 1)}</span><span>{fmtNum(rng.max, 1)}</span></div></> :
            <div>Colour: <b>{colorBy === 'ortho' ? 'imagery' : colorBy === 'classes' ? 'visual land-cover classes' : colorBy === 'confidence' ? 'terrain confidence' : 'elevation evidence'}</b></div>}
          <div className="mt-1 text-slate-500">Drag = orbit · right-drag / Shift = pan · wheel = zoom</div>
        </div>
      </div>

      <RightPanel>
        <Notice kind="estimate"><b>Two different surfaces.</b> The <b>observed surface (DSM)</b> is what the camera saw: treetops, roofs. The <b>estimated bare earth (DTM)</b> is inferred under vegetation and is <i>not</i> directly observed. Imagery colours are visual only; they are not elevation evidence.</Notice>
        <Section title="Surfaces">
          <Check label={<span className="flex items-center gap-1">Estimated bare earth mesh <SurfaceBadge kind="estimated" /></span>} checked={show.dtm && hasDtm && !split} onChange={(v) => setShow({ ...show, dtm: v })} />
          {hasDtm && show.dtm && !split && <Slider label="DTM opacity" value={alpha.dtm} min={0.1} max={1} step={0.05} onChange={(v) => setAlpha({ ...alpha, dtm: v })} fmt={(v) => `${Math.round(v * 100)}%`} />}
          <Check label={<span className="flex items-center gap-1">Observed surface mesh <SurfaceBadge kind="observed" /></span>} checked={show.dsm && hasDsm && !split} onChange={(v) => setShow({ ...show, dsm: v })} />
          {hasDsm && show.dsm && !split && <Slider label="DSM opacity" value={alpha.dsm} min={0.1} max={1} step={0.05} onChange={(v) => setAlpha({ ...alpha, dsm: v })} fmt={(v) => `${Math.round(v * 100)}%`} />}
          {!hasDsm && <div className="text-[11px] text-slate-500">No DSM product.</div>}
          {split && <div className="text-[11px] text-slate-500">Split view: left DSM (observed), right DTM (estimated).</div>}
          {cm?.available && <>
            <Check label={<span className="flex items-center gap-1">Surface without removed features <Chip kind="estimated">PREVIEW</Chip></span>} checked={showEdited}
              onChange={(v) => { setShowEdited(v); if (v) setShow((o) => ({ ...o, dsm: true, dtm: false })); }} />
            {showEdited && <div className="mb-1 rounded bg-violet-50 p-1.5 text-[11px] text-violet-900">
              {surfBusy ? 'Rebuilding…' : <>Highest remaining point per cell from the thinned display sample; points you removed are left out.
                {editedHf && ` Holes where features were removed are filled by interpolation (estimated).`} This is a visual planning aid, not a survey surface, and it does not change the real DSM/DTM.</>}</div>}
          </>}
          <Check label="Point cloud" checked={show.points} onChange={(v) => setShow({ ...show, points: v })} />
          <Check label="Boundary and saved features" checked={show.lines} onChange={(v) => setShow({ ...show, lines: v })} />
          <Slider label="Vertical exaggeration" value={zexag} min={1} max={10} step={0.5} onChange={(v) => setApp({ zexag: v })} fmt={(v) => `${v}×`} />
          <Field label="Mesh colour"><select className="input w-full" value={colorBy} onChange={(e) => setColorBy(e.target.value as ColorBy)}>
            <option value="ortho">Orthomosaic imagery</option><option value="classes" disabled={!mask}>Visual classes over imagery</option><option value="elevation">Elevation ramp</option>
            <option value="confidence" disabled={!ts.layers.confidence}>Terrain confidence</option><option value="evidence" disabled={!ts.layers.evidence}>Elevation evidence</option></select></Field>
          {(colorBy === 'confidence' || colorBy === 'evidence') && <div className="text-[11px] text-slate-500">Describes the estimated ground, so it is shown on both meshes only as a reference overlay.</div>}
          {colorBy === 'elevation' && <Field label="Ramp"><select className="input w-full" value={ramp} onChange={(e) => setApp({ ramp: e.target.value })}>{RAMP_NAMES.map((r) => <option key={r}>{r}</option>)}</select></Field>}
          <Check label="Shading" checked={shade} onChange={setShade} />
          <Field label="Mesh detail"><select className="input w-full" value={detail} onChange={(e) => setDetail(Number(e.target.value))}><option value={200}>Coarse (200)</option><option value={400}>Medium (400)</option><option value={800}>Fine (800)</option><option value={1200}>Very fine (1200)</option></select></Field>
          {ref && <div className="text-[10.5px] text-slate-500">{ref.w}×{ref.h} grid, {fmtNum(ref.res * mh, 2)} m cells. {hfs.dsm && `DSM has ${fmtNum((hfs.dsm.data.reduce((a, v) => a + (isFinite(v) ? 1 : 0), 0) / hfs.dsm.data.length) * 100, 0)}% valid cells; gaps are left empty, not filled.`}</div>}
        </Section>

        <Section key={`pc-${show.points}`} title="Point cloud" open={show.points}>
          {!cm?.available && <div className="text-[12px]">{cm?.has_pointcloud ? <>
            <div className="mb-1 text-slate-600">A thinned display sample must be built first (the full cloud is never sent to the browser).</div>
            <button className="btn btn-primary" onClick={async () => { try { const r = await api.cloudBuild(pid); setCloudJob(r.job_id); bump('jobsTick'); } catch (e) { toast('error', errMsg(e)); } }}>Build display sample</button>
            {cloudJobQ.data && <div className="mt-1"><div className="text-[11px]">{cloudJobQ.data.message ?? cloudJobQ.data.status}</div><Progress value={cloudJobQ.data.progress} /></div>}</> : <span className="text-slate-500">{cm?.hint ?? 'No point cloud.'}</span>}</div>}
          {cm?.available && <>
            <Notice kind="info">{cm.note} {cm.subsampled ? `Showing ${cm.points.toLocaleString()} of ${cm.total_points.toLocaleString()} points.` : `${cm.points.toLocaleString()} points.`}</Notice>
            <Field label="Colour points by"><select className="input w-full" value={pointMode} onChange={(e) => setPointMode(Number(e.target.value) as 0 | 1 | 2)}><option value={0} disabled={!cm.has_rgb}>Imagery colour (RGB)</option><option value={1}>Point class</option><option value={2}>Elevation</option></select></Field>
            <Slider label="Point size (m)" value={pointSize} min={0.05} max={1.5} step={0.05} onChange={setPointSize} />
            <div className="text-[11px] font-semibold text-slate-500">Classes</div>
            {Object.entries(clsCounts).map(([k, n]) => { const i = Number(k); return (
              <label key={k} className="flex cursor-pointer items-center gap-1.5 py-0.5 text-[12px]"><input type="checkbox" checked={clsVis[i] !== false} onChange={(e) => setClsVis(clsVis.map((v, j) => (j === i ? e.target.checked : v)))} />
                <span className="inline-block h-3 w-3 rounded-sm border border-slate-400" style={{ background: `rgb(${CLS_RGB[i]?.map((v) => Math.round(v * 255)).join(',') ?? '150,150,150'})` }} />
                <span className="flex-1">{i === 7 ? CLS_NAMES[7] : cm.class_names?.[k] ?? CLS_NAMES[i] ?? `class ${i}`}</span><span className="font-mono text-[10.5px] text-slate-500">{Number(n).toLocaleString()}</span></label>); })}
            <div className="mt-1 text-[10.5px] text-slate-500">Ground = points the filter judged to be bare earth. Vegetation classes are by height above the <i>estimated</i> ground.</div>
            {cloudBusy && <div className="text-[11px]">Loading points…</div>}
            <div className="mt-2 border-t border-line pt-2">
              <div className="mb-1 flex items-center justify-between"><span className="text-[11px] font-semibold text-slate-500">EDIT POINTS</span>
                <button className={`btn !py-0 ${editing ? 'btn-on' : ''}`} disabled={cloudComplete === false} onClick={toggleEdit}>{editing ? 'Editing: click to stop' : '✎ Start editing'}</button></div>
              {cloudComplete === false && <Notice kind="warn">The browser received only part of the sample, so editing is disabled to keep indices exact.</Notice>}
              <Notice kind="estimate">Edits change point classes in this <b>display sample</b> ({cm.points.toLocaleString()}{cm.subsampled ? ` of ${cm.total_points.toLocaleString()}` : ''} points). They do <b>not</b> change the DSM/DTM, confidence or the full cloud; re-running terrain starts from the full cloud again. Save them with Export → “Edited point cloud (sample)”.</Notice>
              {editing && <>
                <div className="mb-1 flex flex-wrap gap-1">
                  <button className={`btn ${editTool === 'lasso' ? 'btn-on' : ''}`} onClick={() => setEditTool('lasso')}>Lasso</button>
                  <button className={`btn ${editTool === 'box' ? 'btn-on' : ''}`} onClick={() => setEditTool('box')}>Box</button>
                  <span className="mx-1 w-px bg-line" />
                  {(['new', 'add', 'remove'] as const).map((m) => <button key={m} className={`btn ${selMode === m ? 'btn-on' : ''}`} onClick={() => setSelMode(m)}>{m === 'new' ? 'Replace' : m === 'add' ? 'Add' : 'Subtract'}</button>)}
                </div>
                <div className="mb-1 text-[11px] text-slate-500">Drag on the view to select. <kbd>Alt</kbd>+drag orbits, <kbd>Shift</kbd>/right-drag pans, wheel zooms. Selection goes through the whole cloud along the view direction (not only the nearest points); un-tick a class above to protect it from selection. Tip: use Top for areas, Low angle for trees vs ground.</div>
                <div className="mb-1 rounded bg-slate-50 p-1 text-[12px]"><b>{selCount.toLocaleString()}</b> selected <button className="btn !py-0 ml-1" disabled={!selCount} onClick={clearSel}>Clear (Esc)</button></div>
                <Field label="Set selected points to"><select className="input w-full" value={target} onChange={(e) => setTarget(Number(e.target.value))}>{EDIT_CLASSES.map(([i, n]) => <option key={i} value={i}>{i === 7 ? 'noise / removed' : n}</option>)}</select></Field>
                <div className="mb-1 flex flex-wrap gap-1">
                  <button className="btn btn-primary" disabled={!selCount || editBusy} onClick={() => applyEdit(target)}>{editBusy ? 'Working…' : `Apply to ${selCount.toLocaleString()} points`}</button>
                  <button className="btn" disabled={!selCount || editBusy} title="Remove these points (e.g. a tree or a pile): they are hidden and left out of the edited surface. Reversible with Undo; nothing is deleted from the server's data." onClick={() => applyEdit(7)}>Remove selected</button>
                </div>
                <div className="flex flex-wrap items-center gap-1">
                  <button className="btn" disabled={!editInfo?.can_undo || editBusy} onClick={() => stepEdit('undo')}>Undo</button>
                  <button className="btn" disabled={!editInfo?.can_redo || editBusy} onClick={() => stepEdit('redo')}>Redo</button>
                  <button className="btn" disabled={!editInfo?.edited_points || editBusy} onClick={() => stepEdit('reset')}>Discard all edits</button>
                  <span className="text-[11px] text-slate-500">{(editInfo?.edited_points ?? 0).toLocaleString()} points edited · saved on the server</span>
                </div>
              </>}
            </div>
          </>}
        </Section>

        <Section title="3D measurement" actions={mpts.length ? <button className="btn !py-0" onClick={() => setMpts([])}>Clear</button> : undefined}>
          {!measure && mpts.length === 0 && <div className="text-[12px] text-slate-500">Turn on <b>Measure</b> in the toolbar, then click points on the surface. Points land on the {split ? 'surface under the pane you click' : 'visible surface'}; each point records whether it sits on the observed or estimated surface.</div>}
          {mpts.map((q, i) => <div key={i} className="mb-0.5 flex items-center justify-between text-[11.5px]"><span><b>{i + 1}</b> <Chip kind={q.surf}>{q.surf === 'observed' ? 'OBS' : 'EST'}</Chip></span>
            <span className="font-mono">{fmtNum(q.x, 1)}, {fmtNum(q.y, 1)}, {fmtNum(q.z, 2)}</span></div>)}
          {calc.segs.length > 0 && <div className="mt-1 rounded bg-slate-50 p-1">
            {calc.segs.map((s, i) => <div key={i} className="flex justify-between text-[11.5px]"><span>{i + 1}→{i + 2}</span><span className="font-mono">{lengthLabel(p, s.d3)} · Δz {s.dz >= 0 ? '+' : ''}{fmtNum(s.dz / mv, 2)} · {fmtNum(s.slope, 1)}%</span></div>)}
            <Kv k="Horizontal length" v={lengthLabel(p, calc.hd)} /><Kv k="Slope length (3D)" v={lengthLabel(p, calc.d3)} />
            {new Set(mpts.map((q) => q.surf)).size > 1 && <Notice kind="estimate">These points mix observed and estimated surfaces; the elevation difference between them is not a ground slope.</Notice>}
            {mpts.some((q) => q.surf === 'estimated') && <div className="text-[10.5px] text-amber-800">Estimated-surface values inherit the DTM uncertainty (see Terrain 2D summary).</div>}</div>}
        </Section>
        <div className="p-2 text-[10.5px] text-slate-500">{DISCLAIMER}</div>
      </RightPanel>
    </div>
  );
}

function PaneView({ idx, h, onReady, onErr, cursor }: { idx: number; h: { down: (e: any, el: HTMLElement) => void; move: (e: any, el: HTMLElement) => void; up: (e: any, el: HTMLElement) => void; wheel: (e: WheelEvent) => void; leave: () => void; resize: () => void }; onReady: (i: number, p: PaneReg | null) => void; onErr: (m: string) => void; cursor?: string }) {
  const gl = useRef<HTMLCanvasElement | null>(null), ov = useRef<HTMLCanvasElement | null>(null);
  const hh = useRef(h); hh.current = h;
  useEffect(() => {
    let pane: GLPane;
    try { pane = new GLPane(gl.current!); } catch (e) { onErr(errMsg(e)); return; }
    onReady(idx, { pane, overlay: ov.current! });
    const o = ov.current!; const wheel = (e: WheelEvent) => hh.current.wheel(e); o.addEventListener('wheel', wheel, { passive: false });
    const ro = new ResizeObserver(() => hh.current.resize()); ro.observe(o);
    return () => { ro.disconnect(); o.removeEventListener('wheel', wheel); onReady(idx, null); pane.dispose(); };
    // eslint-disable-next-line
  }, []);
  return (
    <div className={`relative min-w-0 flex-1 ${idx === 1 ? 'border-l-2 border-white' : ''}`}>
      <canvas ref={gl} className="absolute inset-0 h-full w-full" />
      <canvas ref={ov} className="absolute inset-0 h-full w-full" style={{ cursor: cursor ?? 'grab', touchAction: 'none' }}
        onPointerDown={(e) => hh.current.down(e, ov.current!)} onPointerMove={(e) => hh.current.move(e, ov.current!)} onPointerUp={(e) => hh.current.up(e, ov.current!)}
        onPointerLeave={() => hh.current.leave()} onContextMenu={(e) => e.preventDefault()} />
    </div>
  );
}
