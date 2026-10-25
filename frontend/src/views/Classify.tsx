import { useEffect, useMemo, useRef, useState } from 'react';
import { ApiError, CClass, ClassInfo, EditBody, OrthoInfo, api, url } from '../api';
import { MapApi, MapCanvas, Ptr, drawGeoJSON, drawNorth, drawRing, drawScaleBar } from '../MapCanvas';
import { Check, Chip, Empty, Field, Kv, Modal, Notice, Progress, RightPanel, Section, Slider, Table } from '../ui';
import { MaskLayer, MaskMode, OrthoLayer } from '../tiles';
import { bump, getState, setApp, setStatus, toast, useApp, useProject } from '../store';
import { Viewport } from '../viewport';
import { boundaryRings, clamp, errMsg, fmtNum, toMetres, useLocal, usePoll } from '../util';

type Tool = 'hand' | 'sample' | 'brush' | 'polygon' | 'rect' | 'lasso' | 'fill' | 'similar';
const TOOLS: { id: Tool; label: string; key: string; icon: string; tip: string }[] = [
  { id: 'hand', label: 'Pan', key: 'G', icon: '✥', tip: 'Pan the view (also: hold Space or drag the middle button)' },
  { id: 'sample', label: 'Sample', key: 'S', icon: '⌖', tip: 'Inspect a pixel: colour, class, provenance, confidence, similar-colour region' },
  { id: 'brush', label: 'Brush', key: 'B', icon: '●', tip: 'Paint the active class' },
  { id: 'polygon', label: 'Polygon', key: 'P', icon: '⬠', tip: 'Click vertices; Enter or double-click closes' },
  { id: 'rect', label: 'Rectangle', key: 'R', icon: '▭', tip: 'Drag a rectangle' },
  { id: 'lasso', label: 'Lasso', key: 'L', icon: '➰', tip: 'Draw a freehand outline' },
  { id: 'fill', label: 'Flood fill', key: 'F', icon: '▨', tip: 'Fill the connected area of similar colour (tolerance); preview before applying' },
  { id: 'similar', label: 'Select similar', key: 'M', icon: '≈', tip: 'Select every pixel in the search window with a similar colour; preview before applying' },
];

interface Preview { seed: [number, number]; kind: 'fill' | 'similar'; win: number[]; canvas: HTMLCanvasElement; count: number; truncated: boolean }
interface Ghost { id: number; kind: string; pts: number[][]; radius: number; classId: number; done?: boolean }

async function previewFetch(pid: string, body: EditBody): Promise<{ win: number[]; blob: Blob; count: number; truncated: boolean } | null> {
  const r = await fetch(url(`/projects/${pid}/classification/select-preview`), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (r.status === 204) return null;
  if (!r.ok) { const j = await r.json().catch(() => null); throw new ApiError(r.status, j?.error?.code ?? 'error', j?.error?.message ?? `HTTP ${r.status}`); }
  return { win: (r.headers.get('X-Window') ?? '0,0,0,0').split(',').map(Number), blob: await r.blob(), count: Number(r.headers.get('X-Selected-Pixels') ?? 0), truncated: r.headers.get('X-Truncated') === '1' };
}
async function blobToSelCanvas(b: Blob, w: number, h: number): Promise<HTMLCanvasElement> {
  const bm = await createImageBitmap(b); const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const c = cv.getContext('2d', { willReadFrequently: true })!; c.drawImage(bm, 0, 0);
  const id = c.getImageData(0, 0, w, h), d = id.data;
  for (let i = 0; i < d.length; i += 4) { const on = d[i] > 127; d[i] = 0; d[i + 1] = 210; d[i + 2] = 255; d[i + 3] = on ? 150 : 0; }
  c.putImageData(id, 0, 0); return cv;
}

export function Classify() {
  const p = useProject()!;
  const jt = useApp((s) => s.jobsTick);
  const ortho = usePoll(() => api.ortho(p.id), 0, [p.id, jt]);
  const info = usePoll(() => (ortho.data?.available ? api.classInfo(p.id) : Promise.resolve(null)), 0, [p.id, ortho.data?.available, ortho.data?.created_at]);
  if (ortho.loading) return <div className="m-auto text-slate-500">Loading…</div>;
  if (!ortho.data?.available) return <Empty title="No orthomosaic yet">Classification works on the orthomosaic. Run photogrammetry on the Dashboard, or import an existing orthomosaic GeoTIFF on the Upload tab.<div className="mt-2"><button className="btn btn-primary" onClick={() => setApp({ view: 'upload' })}>Go to Upload</button></div></Empty>;
  if (!info.data) return <div className="m-auto text-slate-500">{info.error ? <span className="text-warn">{info.error}</span> : 'Loading classification…'}</div>;
  return <Editor ortho={ortho.data} info={info.data} reloadInfo={info.reload} key={p.id + ':' + ortho.data.created_at} />;
}

function Editor({ ortho, info, reloadInfo }: { ortho: OrthoInfo; info: ClassInfo; reloadInfo: () => void }) {
  const p = useProject()!;
  const opacity = useApp((s) => s.opacity.classes);
  const layers = useApp((s) => s.layers);
  const pid = p.id;
  const classes = info.classes;
  const [tool, setTool] = useState<Tool>('brush');
  const [classId, setClassId] = useState(() => (classes.find((c) => c.id !== 0)?.id ?? 1));
  const [radius, setRadius] = useLocal('ts.brush', 10);
  const [tol, setTol] = useLocal('ts.tol', 22);
  const [conn, setConn] = useLocal<4 | 8>('ts.conn', 4);
  const [win, setWin] = useLocal('ts.win', 1024);
  const [minArea, setMinArea] = useLocal('ts.minarea', 0);
  const [holes, setHoles] = useLocal('ts.holes', false);
  const [smooth, setSmooth] = useLocal('ts.smooth', 0);
  const [only, setOnly] = useState<number[]>([]);
  const [protect, setProtect] = useState<number[]>([]);
  const [hidden, setHidden] = useLocal<number[]>('ts.hidden.' + pid, []);
  const [mode, setMode] = useLocal<MaskMode>('ts.maskmode', 'class');
  const [split, setSplit] = useState(false);
  const [peek, setPeek] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [sampled, setSampled] = useState<any>(null);
  const [poly, setPoly] = useState<number[][]>([]);
  const [stats, setStats] = useState<any>(null);
  const [editClasses, setEditClasses] = useState(false);
  const mapA = useRef<MapApi | null>(null), mapB = useRef<MapApi | null>(null);
  const draft = useRef<{ pts: number[][]; rect?: number[] } | null>(null);
  const hover = useRef<[number, number] | null>(null);
  const ghosts = useRef<Ghost[]>([]);
  const queue = useRef<Promise<void>>(Promise.resolve());
  const pending = useRef(0);
  const gid = useRef(1);

  const redraw = () => { mapA.current?.redraw(); mapB.current?.redraw(); };
  const orthoLayer = useMemo(() => new OrthoLayer(pid, ortho, () => redraw()), [pid, ortho]);
  const mask = useMemo(() => new MaskLayer(pid, info.width, info.height, orthoLayer.grid, () => redraw()), [pid, info.width, info.height, orthoLayer]);
  useEffect(() => { mask.setVersion(info.version); mask.setClasses(classes, new Set(hidden)); mask.setMode(mode); mask.setClasses(classes, new Set(hidden)); redraw(); }, [mask, classes, hidden, mode]);
  const g = orthoLayer.grid;
  const toPx = (x: number, y: number): [number, number] => [(x - g.x0) / g.res, (g.y0 - y) / g.res];
  const toMap = (px: number, py: number): [number, number] => [g.x0 + px * g.res, g.y0 - py * g.res];
  const cls = (id: number) => classes.find((c) => c.id === id);
  const active = cls(classId);

  // -------------------------------------------------------------- saving / undo
  const setSaved = (s: 'saved' | 'saving' | 'error', msg?: string) => setApp((st) => ({ saved: { state: s, version: st.saved.version, msg } }));
  const applyState = (st: { version: number; can_undo: boolean; can_redo: boolean }) => {
    mask.setVersion(st.version); setApp({ saved: { state: pending.current > 1 ? 'saving' : 'saved', version: st.version }, undoRedo: { ...getState().undoRedo, canUndo: st.can_undo, canRedo: st.can_redo } });
    redraw(); scheduleStats();
  };
  const run = (body: EditBody, ghost?: Omit<Ghost, 'id'>) => {
    const gh = ghost ? { ...ghost, id: gid.current++ } : null; if (gh) ghosts.current.push(gh);
    pending.current++; setSaved('saving');
    queue.current = queue.current.then(async () => {
      try { const st = await api.classEdit(pid, body); applyState(st); if (st.changed_pixels === 0) setStatus({ mid: 'No pixels changed (already that class, or restricted by “only/protect”).' }); if (st.message) toast('info', st.message); }
      catch (e) { toast('error', errMsg(e)); setSaved('error', errMsg(e)); }
      finally { pending.current--; if (gh) setTimeout(() => { ghosts.current = ghosts.current.filter((x) => x.id !== gh.id); redraw(); }, 450); if (!pending.current && getState().saved.state === 'saving') setApp((s) => ({ saved: { ...s.saved, state: 'saved' } })); }
    });
  };
  const doUndo = async () => { await queue.current; try { applyState(await api.classUndo(pid)); } catch (e) { toast('error', errMsg(e)); } };
  const doRedo = async () => { await queue.current; try { applyState(await api.classRedo(pid)); } catch (e) { toast('error', errMsg(e)); } };
  useEffect(() => {
    setApp({ undoRedo: { canUndo: info.can_undo, canRedo: info.can_redo, undo: doUndo, redo: doRedo }, saved: { state: 'saved', version: info.version } });
    return () => setApp({ undoRedo: { canUndo: false, canRedo: false, undo: null, redo: null } });
    // eslint-disable-next-line
  }, [pid]);
  useEffect(() => { const f = (e: BeforeUnloadEvent) => { if (pending.current) { e.preventDefault(); e.returnValue = ''; } }; window.addEventListener('beforeunload', f); return () => window.removeEventListener('beforeunload', f); }, []);

  const statsTimer = useRef<any>(0);
  const scheduleStats = () => { clearTimeout(statsTimer.current); statsTimer.current = setTimeout(async () => { try { setStats(await api.classStats(pid)); } catch { /* ignore */ } }, 700); };
  useEffect(() => { scheduleStats(); return () => clearTimeout(statsTimer.current); /* eslint-disable-next-line */ }, [pid]);

  // -------------------------------------------------------------- body builders
  const common = (): Partial<EditBody> => ({ only_classes: only.length ? only : null, protect_classes: protect.length ? protect : null, min_area: minArea, fill_holes: holes, smooth_radius: smooth });
  const winFor = (x: number, y: number) => [Math.floor(x - win), Math.floor(y - win), Math.ceil(x + win), Math.ceil(y + win)];
  const bodyFor = (kind: 'fill' | 'similar', seed: [number, number]): EditBody => ({ type: kind, class_id: classId, points: [seed], tolerance: tol, connectivity: conn, analysis_window: winFor(seed[0], seed[1]), ...common() });

  const doPreview = async (kind: 'fill' | 'similar', seed: [number, number]) => {
    try {
      const r = await previewFetch(pid, bodyFor(kind, seed));
      if (!r) { setPreview(null); setStatus({ mid: 'Nothing selected at that point.' }); return; }
      const w = r.win[2] - r.win[0], h = r.win[3] - r.win[1];
      setPreview({ seed, kind, win: r.win, canvas: await blobToSelCanvas(r.blob, w, h), count: r.count, truncated: r.truncated }); redraw();
    } catch (e) { toast('error', errMsg(e)); setPreview(null); }
  };
  const previewTimer = useRef<any>(0);
  useEffect(() => { // live re-preview when tolerance/options change
    if (!preview) return; clearTimeout(previewTimer.current);
    previewTimer.current = setTimeout(() => doPreview(preview.kind, preview.seed), 250);
    return () => clearTimeout(previewTimer.current);
    // eslint-disable-next-line
  }, [tol, conn, win, minArea, holes, smooth, only, protect]);
  const applyPreview = () => { if (!preview) return; run({ ...bodyFor(preview.kind, preview.seed), label: `${preview.kind} -> ${active?.name}` }); setPreview(null); };

  const finishPoly = (pts: number[][]) => {
    if (pts.length < 3) { toast('info', 'A polygon needs at least 3 vertices.'); return; }
    run({ type: 'polygon', class_id: classId, points: pts, ...common() }, { kind: 'polygon', pts, radius: 0, classId });
    setPoly([]); draft.current = null;
  };

  // -------------------------------------------------------------- pointer handling
  const onDown = (pt: Ptr): boolean => {
    const [px, py] = toPx(pt.x, pt.y);
    if (tool === 'hand') return false;
    if (px < 0 || py < 0 || px >= info.width || py >= info.height) { if (tool !== 'polygon') return false; }
    if (tool === 'sample') {
      api.classSample(pid, Math.floor(px), Math.floor(py), tol, true).then((r: any) => { setSampled(r); }).catch((e) => toast('error', errMsg(e)));
      return false;
    }
    if (tool === 'fill' || tool === 'similar') { doPreview(tool, [px, py]); return false; }
    if (tool === 'polygon') { setPoly((a) => [...a, [px, py]]); return false; }
    if (tool === 'brush' || tool === 'lasso') { draft.current = { pts: [[px, py]] }; redraw(); return true; }
    if (tool === 'rect') { draft.current = { pts: [], rect: [px, py, px, py] }; return true; }
    return false;
  };
  const onMove = (pt: Ptr) => {
    const [px, py] = toPx(pt.x, pt.y); hover.current = [px, py];
    const d = draft.current;
    if (d && pt.e.buttons & 1) {
      if (tool === 'brush' || tool === 'lasso') {
        const last = d.pts[d.pts.length - 1]; const step = tool === 'brush' ? Math.max(1, radius / 3) : 1.5 / (pt.sx ? mapA.current!.vp.ppu * g.res : 1);
        if (Math.hypot(px - last[0], py - last[1]) >= step && d.pts.length < 190000) d.pts.push([px, py]);
      } else if (tool === 'rect' && d.rect) { d.rect[2] = px; d.rect[3] = py; }
    }
    const inside = px >= 0 && py >= 0 && px < info.width && py < info.height;
    const pr = inside ? mask.probe(px, py) : null;
    const c = pr ? cls(pr.cls) : null;
    const ll = `x ${fmtNum(pt.x, 2)}  y ${fmtNum(pt.y, 2)}`;
    setStatus({ left: inside ? `px ${Math.floor(px)}, ${Math.floor(py)}   ${p.epsg ? '' : ''}${ll}` : ll,
      mid: inside ? (pr ? `${c?.name ?? 'class ' + pr.cls}${pr.prov === 1 ? ' · manual' : pr.prov === 2 ? ` · automatic (confidence ${pr.conf}%)` : ''}` : '') : '' });
    redraw();
  };
  const onUp = (pt: Ptr) => {
    const d = draft.current; draft.current = null; if (!d) return;
    if (tool === 'brush' && d.pts.length) run({ type: 'brush', class_id: classId, points: d.pts, radius, ...common() }, { kind: 'brush', pts: d.pts, radius, classId });
    else if (tool === 'lasso' && d.pts.length >= 3) run({ type: 'lasso', class_id: classId, points: d.pts, ...common() }, { kind: 'lasso', pts: d.pts, radius: 0, classId });
    else if (tool === 'rect' && d.rect && (Math.abs(d.rect[2] - d.rect[0]) > 0.5 && Math.abs(d.rect[3] - d.rect[1]) > 0.5)) run({ type: 'rect', class_id: classId, rect: d.rect, ...common() }, { kind: 'rect', pts: [[d.rect[0], d.rect[1]], [d.rect[2], d.rect[3]]], radius: 0, classId });
    redraw();
  };

  // -------------------------------------------------------------- keyboard
  useEffect(() => {
    const kd = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName; if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); e.shiftKey ? doRedo() : doUndo(); return; }
      if ((e.ctrlKey || e.metaKey) && k === 'y') { e.preventDefault(); doRedo(); return; }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const t = TOOLS.find((x) => x.key.toLowerCase() === k); if (t) { setTool(t.id); setPoly([]); setPreview(null); return; }
      if (k === 'e') { setClassId(0); return; }
      if (k === 'h') { setApp((s) => ({ layers: { ...s.layers, classes: !s.layers.classes } })); return; }
      if (k === '\\') { setPeek(true); return; }
      if (k === '[') setRadius(clamp(Math.round(radius / 1.25), 1, 300)); if (k === ']') setRadius(clamp(Math.round(radius * 1.25) + 1, 1, 300));
      if (/^[0-9]$/.test(k) && cls(Number(k))) setClassId(Number(k));
      if (k === 'enter') { if (preview) applyPreview(); else if (tool === 'polygon' && poly.length) finishPoly(poly); }
      if (k === 'escape') { setPoly([]); setPreview(null); draft.current = null; redraw(); }
      if (k === 'backspace' && poly.length) { e.preventDefault(); setPoly((a) => a.slice(0, -1)); }
    };
    const ku = (e: KeyboardEvent) => { if (e.key === '\\') setPeek(false); };
    window.addEventListener('keydown', kd); window.addEventListener('keyup', ku);
    return () => { window.removeEventListener('keydown', kd); window.removeEventListener('keyup', ku); };
  });
  useEffect(() => { redraw(); }, [poly, preview, peek, layers, opacity, split, ghosts.current.length]);

  // -------------------------------------------------------------- drawing
  const drawBase = (c: CanvasRenderingContext2D, vp: Viewport, withClasses: boolean) => {
    c.fillStyle = '#20262d'; c.fillRect(0, 0, vp.w, vp.h);
    if (layers.ortho) orthoLayer.draw(c, vp);
    if (withClasses && layers.classes && !peek) mask.draw(c, vp, opacity);
    if (layers.boundary) { const rings = boundaryRings(p.boundary, p.epsg); rings?.forEach((r) => drawRing(c, vp, r, { stroke: '#ffb300', width: 2, dash: [8, 4] })); }
  };
  const drawEdit = (c: CanvasRenderingContext2D, vp: Viewport) => {
    const col = (id: number) => (id === 0 ? 'rgba(255,255,255,0.5)' : cls(id)?.color ?? '#fff');
    for (const gh of ghosts.current) {
      c.save(); c.globalAlpha = opacity; c.fillStyle = col(gh.classId); c.strokeStyle = col(gh.classId);
      if (gh.kind === 'brush') { c.lineCap = 'round'; c.lineJoin = 'round'; c.lineWidth = gh.radius * 2 * g.res * vp.ppu; c.beginPath(); gh.pts.forEach((q, i) => { const [sx, sy] = vp.toScreen(...toMap(q[0], q[1])); i ? c.lineTo(sx, sy) : c.moveTo(sx, sy); }); if (gh.pts.length === 1) { const [sx, sy] = vp.toScreen(...toMap(gh.pts[0][0], gh.pts[0][1])); c.arc(sx, sy, gh.radius * g.res * vp.ppu, 0, 7); c.fill(); } else c.stroke(); }
      else if (gh.kind === 'rect') { const a = vp.toScreen(...toMap(gh.pts[0][0], gh.pts[0][1])), b = vp.toScreen(...toMap(gh.pts[1][0], gh.pts[1][1])); c.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); }
      else drawRing(c, vp, gh.pts.map((q) => toMap(q[0], q[1])), { fill: col(gh.classId) });
      c.restore();
    }
    if (preview) { const [sx, sy] = vp.toScreen(...toMap(preview.win[0], preview.win[1])); c.imageSmoothingEnabled = false; c.drawImage(preview.canvas, sx, sy, preview.canvas.width * g.res * vp.ppu, preview.canvas.height * g.res * vp.ppu); }
    const d = draft.current; const ac = col(classId);
    if (d && (tool === 'brush')) { c.save(); c.globalAlpha = 0.55; c.strokeStyle = ac; c.fillStyle = ac; c.lineCap = 'round'; c.lineJoin = 'round'; c.lineWidth = radius * 2 * g.res * vp.ppu; c.beginPath(); d.pts.forEach((q, i) => { const [sx, sy] = vp.toScreen(...toMap(q[0], q[1])); i ? c.lineTo(sx, sy) : c.moveTo(sx, sy); }); c.stroke(); c.restore(); }
    if (d && tool === 'lasso') drawRing(c, vp, d.pts.map((q) => toMap(q[0], q[1])), { stroke: '#fff', fill: ac + '55', width: 1.5, dash: [5, 3] });
    if (d?.rect) { const a = vp.toScreen(...toMap(d.rect[0], d.rect[1])), b = vp.toScreen(...toMap(d.rect[2], d.rect[3])); c.fillStyle = ac + '55'; c.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); c.strokeStyle = '#fff'; c.setLineDash([5, 3]); c.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); c.setLineDash([]); }
    if (tool === 'polygon' && poly.length) {
      const pts = poly.map((q) => toMap(q[0], q[1])); if (hover.current) pts.push(toMap(hover.current[0], hover.current[1]));
      drawRing(c, vp, pts, { stroke: '#fff', fill: ac + '40', width: 1.5, dash: [5, 3] });
      poly.forEach((q) => { const [sx, sy] = vp.toScreen(...toMap(q[0], q[1])); c.fillStyle = '#fff'; c.fillRect(sx - 3, sy - 3, 6, 6); });
    }
    if (tool === 'brush' && hover.current) { const [sx, sy] = vp.toScreen(...toMap(hover.current[0], hover.current[1])); c.strokeStyle = '#fff'; c.lineWidth = 1.5; c.beginPath(); c.arc(sx, sy, Math.max(2, radius * g.res * vp.ppu), 0, 7); c.stroke(); c.strokeStyle = ac; c.lineWidth = 1; c.beginPath(); c.arc(sx, sy, Math.max(2, radius * g.res * vp.ppu) + 1.5, 0, 7); c.stroke(); }
    if (sampled && tool === 'sample') { const [sx, sy] = vp.toScreen(...toMap(sampled.x + .5, sampled.y + .5)); c.strokeStyle = '#ff0'; c.lineWidth = 2; c.beginPath(); c.arc(sx, sy, 7, 0, 7); c.moveTo(sx - 12, sy); c.lineTo(sx + 12, sy); c.moveTo(sx, sy - 12); c.lineTo(sx, sy + 12); c.stroke(); if (sampled.segment?.bbox) { const b = sampled.segment.bbox; const a1 = vp.toScreen(...toMap(b[0], b[1])), a2 = vp.toScreen(...toMap(b[2], b[3])); c.strokeStyle = 'rgba(255,255,0,.8)'; c.setLineDash([4, 3]); c.strokeRect(a1[0], a1[1], a2[0] - a1[0], a2[1] - a1[1]); c.setLineDash([]); } }
    drawScaleBar(c, vp, toMetres(p.horizontal_units), p.unit_system === 'imperial'); drawNorth(c, vp);
  };
  const drawA = (c: CanvasRenderingContext2D, vp: Viewport) => { drawBase(c, vp, true); drawEdit(c, vp); if (split) { c.fillStyle = 'rgba(0,0,0,.55)'; c.fillRect(8, 8, 110, 18); c.fillStyle = '#fff'; c.font = '11px system-ui'; c.fillText('With classification', 14, 21); } };
  const drawB = (c: CanvasRenderingContext2D, vp: Viewport) => { drawBase(c, vp, false); c.fillStyle = 'rgba(0,0,0,.55)'; c.fillRect(8, 8, 110, 18); c.fillStyle = '#fff'; c.font = '11px system-ui'; c.fillText('Imagery only', 14, 21); };
  const sync = (from: MapApi | null, to: MapApi | null) => { if (!from || !to) return; to.vp.cx = from.vp.cx; to.vp.cy = from.vp.cy; to.vp.ppu = from.vp.ppu; to.redraw(); };
  const bounds = orthoLayer.bounds();
  const minPpu = 0.02 / g.res * 0.2, maxPpu = 80 / g.res;

  // -------------------------------------------------------------- auto classification, reclassify
  const [model, setModel] = useState(() => info.models.find((m) => m.available)?.id ?? 'heuristic');
  const [overwrite, setOverwrite] = useState<'unclassified' | 'auto' | 'all'>('auto');
  const [regionSel, setRegionSel] = useState<'all' | 'view'>('all');
  const [autoJob, setAutoJob] = useState<string | null>(null);
  const job = usePoll(() => (autoJob ? api.job(autoJob) : Promise.resolve(null)), 1500, [autoJob]);
  const lastDone = useRef<string | null>(null);
  useEffect(() => {
    const j = job.data; if (!j || lastDone.current === j.id) return;
    if (j.status === 'succeeded' || j.status === 'failed' || j.status === 'cancelled') {
      lastDone.current = j.id; bump('jobsTick');
      if (j.status === 'succeeded') { toast('ok', 'Automatic classification finished. Review the result before relying on it.'); api.classInfo(pid).then((ci) => { mask.setVersion(ci.version); applyState(ci as any); reloadInfo(); }); }
      else toast('error', j.error ?? `Job ${j.status}`);
    }
  }, [job.data]);
  const runAuto = async () => {
    try {
      const body: any = { model, overwrite };
      if (regionSel === 'view') { const v = mapA.current!.vp.bbox(); const a = toPx(v[0], v[3]), b = toPx(v[2], v[1]); body.region = { type: 'rect', rect: [clamp(a[0], 0, info.width), clamp(a[1], 0, info.height), clamp(b[0], 0, info.width), clamp(b[1], 0, info.height)] }; }
      const r = await api.classAuto(pid, body); setAutoJob(r.job_id); bump('jobsTick');
    } catch (e) { toast('error', errMsg(e)); }
  };
  const [from, setFrom] = useState(2), [to, setTo] = useState(1);
  const reclass = (whole: boolean) => {
    let rect: number[];
    if (whole) { if (!window.confirm(`Reclassify EVERY “${cls(from)?.name}” pixel in the whole image to “${cls(to)?.name}”? You can undo it.`)) return; rect = [0, 0, info.width, info.height]; }
    else { const v = mapA.current!.vp.bbox(); const a = toPx(v[0], v[3]), b = toPx(v[2], v[1]); rect = [Math.max(0, a[0]), Math.max(0, a[1]), Math.min(info.width, b[0]), Math.min(info.height, b[1])]; }
    run({ type: 'rect', class_id: to, rect, only_classes: [from], label: `reclassify ${cls(from)?.name} -> ${cls(to)?.name}` });
  };

  const total = stats?.total_pixels ?? info.width * info.height;
  const statRow = (id: number) => stats?.classes?.find((r: any) => r.class_id === id);
  const hiddenSet = new Set(hidden);
  const toggleList = (list: number[], set: (l: number[]) => void, id: number) => set(list.includes(id) ? list.filter((x) => x !== id) : [...list, id]);

  return (
    <div className="flex min-h-0 flex-1">
      <div className="flex w-[46px] shrink-0 flex-col gap-1 border-r border-line bg-white p-1">
        {TOOLS.map((t) => <button key={t.id} title={`${t.label} (${t.key}) - ${t.tip}`} className={`btn flex-col !gap-0 !px-0 !py-1 text-center ${tool === t.id ? 'btn-on' : ''}`} onClick={() => { setTool(t.id); setPoly([]); setPreview(null); }}><span className="text-[15px] leading-4">{t.icon}</span><span className="text-[9px]">{t.key}</span></button>)}
        <div className="my-1 border-t border-line" />
        <button className={`btn flex-col !gap-0 !px-0 !py-1 ${split ? 'btn-on' : ''}`} title="Side-by-side: imagery and imagery + classification" onClick={() => setSplit(!split)}><span className="text-[15px] leading-4">◫</span><span className="text-[9px]">Split</span></button>
        <button className={`btn flex-col !gap-0 !px-0 !py-1 ${layers.classes ? 'btn-on' : ''}`} title="Toggle class overlay (H). Hold \ to peek." onClick={() => setApp((s) => ({ layers: { ...s.layers, classes: !s.layers.classes } }))}><span className="text-[15px] leading-4">◐</span><span className="text-[9px]">H</span></button>
        <button className="btn flex-col !gap-0 !px-0 !py-1" title="Fit to view" onClick={() => { mapA.current?.fit(bounds); sync(mapA.current, mapB.current); }}><span className="text-[15px] leading-4">⤢</span><span className="text-[9px]">Fit</span></button>
      </div>
      <div className="relative flex min-w-0 flex-1">
        {split && <div className="w-1/2 border-r-2 border-white"><MapCanvas apiRef={mapB} draw={drawB} initialBounds={bounds} minPpu={minPpu} maxPpu={maxPpu} panAlways cursor="grab" onChange={() => sync(mapB.current, mapA.current)} /></div>}
        <div className={split ? 'w-1/2' : 'w-full'}>
          <MapCanvas apiRef={mapA} draw={drawA} initialBounds={bounds} minPpu={minPpu} maxPpu={maxPpu} onDown={onDown} onMove={onMove} onUp={onUp} onChange={() => sync(mapA.current, mapB.current)}
            onDouble={() => { if (tool === 'polygon' && poly.length >= 3) finishPoly(poly.slice(0, -1).length >= 3 ? poly.slice(0, -1) : poly); }} onLeave={() => { hover.current = null; redraw(); }}
            cursor={tool === 'hand' ? 'grab' : tool === 'brush' ? 'none' : 'crosshair'} panAlways={tool === 'hand'} />
        </div>
        {preview && <div className="absolute left-1/2 top-2 z-10 flex -translate-x-1/2 items-center gap-2 rounded border border-line bg-white px-3 py-1.5 shadow">
          <span>{preview.count.toLocaleString()} px selected{preview.truncated ? ' (reached the search window edge)' : ''}</span>
          <button className="btn btn-primary" onClick={applyPreview}>Apply as {active?.name ?? classId} (Enter)</button><button className="btn" onClick={() => setPreview(null)}>Cancel (Esc)</button></div>}
        {tool === 'polygon' && poly.length > 0 && <div className="absolute left-1/2 top-2 z-10 flex -translate-x-1/2 items-center gap-2 rounded border border-line bg-white px-3 py-1.5 shadow"><span>{poly.length} vertices</span><button className="btn btn-primary" disabled={poly.length < 3} onClick={() => finishPoly(poly)}>Close and apply (Enter)</button><button className="btn" onClick={() => setPoly([])}>Cancel</button></div>}
        {ortho.warnings?.length ? <div className="absolute bottom-8 left-2 right-2 z-10 max-w-[620px]"><Notice kind="estimate">{ortho.warnings[0]}</Notice></div> : null}
      </div>

      <RightPanel>
        <Notice kind="estimate"><b>Visual classification only.</b> {info.notice}</Notice>
        <Section title="Tool options">
          <div className="mb-1.5 flex items-center gap-2"><span className="inline-block h-4 w-4 rounded border border-slate-400" style={{ background: classId === 0 ? 'repeating-linear-gradient(45deg,#fff,#fff 3px,#bbb 3px,#bbb 6px)' : active?.color }} /><span className="font-medium">{classId === 0 ? 'Erase to Unknown' : active?.name}</span></div>
          {tool === 'brush' && <Slider label={`Brush radius (${fmtNum(radius * g.res * toMetres(p.horizontal_units), 2)} m)`} value={radius} min={1} max={300} onChange={setRadius} fmt={(v) => `${v}px`} />}
          {(tool === 'fill' || tool === 'similar' || tool === 'sample') && <>
            <Slider label="Colour tolerance" value={tol} min={0} max={120} onChange={setTol} />
            {tool !== 'sample' && <><Slider label="Search window (half-size)" value={win} min={128} max={4096} step={64} onChange={setWin} fmt={(v) => `${v}px`} />
              {tool === 'fill' && <Field label="Connectivity"><select className="input" value={conn} onChange={(e) => setConn(Number(e.target.value) as 4 | 8)}><option value={4}>4-connected</option><option value={8}>8-connected</option></select></Field>}</>}
          </>}
          {(tool === 'polygon' || tool === 'fill' || tool === 'similar' || tool === 'lasso' || tool === 'rect') && <details className="mb-1"><summary className="cursor-pointer text-[11px] text-slate-600">Tidy selection</summary>
            <Slider label="Drop regions smaller than" value={minArea} min={0} max={500} onChange={setMinArea} fmt={(v) => `${v}px`} /><Check label="Fill holes" checked={holes} onChange={setHoles} /><Slider label="Smooth edges" value={smooth} min={0} max={10} onChange={setSmooth} /></details>}
          <details><summary className="cursor-pointer text-[11px] text-slate-600">Restrict edits {only.length || protect.length ? `(${only.length} only, ${protect.length} protected)` : ''}</summary>
            <div className="mt-1 text-[11px] text-slate-500">Change only pixels currently in:</div><div className="flex flex-wrap gap-1">{classes.map((c) => <button key={c.id} className={`btn !px-1 !py-0 ${only.includes(c.id) ? 'btn-on' : ''}`} onClick={() => toggleList(only, setOnly, c.id)}>{c.name.split(' ')[0]}</button>)}</div>
            <div className="mt-1 text-[11px] text-slate-500">Never change pixels in:</div><div className="flex flex-wrap gap-1">{classes.map((c) => <button key={c.id} className={`btn !px-1 !py-0 ${protect.includes(c.id) ? 'btn-on' : ''}`} onClick={() => toggleList(protect, setProtect, c.id)}>{c.name.split(' ')[0]}</button>)}</div></details>
        </Section>

        <Section title="Classes, legend and statistics" actions={<button className="btn !py-0" onClick={() => setEditClasses(true)}>Edit</button>}>
          {classes.map((c) => { const r = statRow(c.id); const off = hiddenSet.has(c.id); return (
            <div key={c.id} className={`mb-0.5 flex items-center gap-1 rounded border px-1 py-0.5 ${classId === c.id ? 'border-accent bg-sky-50' : 'border-transparent hover:bg-slate-50'}`}>
              <button className="btn !px-1 !py-0" title={off ? 'Show class' : 'Hide class'} onClick={() => setHidden(off ? hidden.filter((x) => x !== c.id) : [...hidden, c.id])}>{off ? '○' : '●'}</button>
              <button className="flex min-w-0 flex-1 items-center gap-1.5 text-left" onClick={() => setClassId(c.id)} title={`${c.name} (key ${c.id < 10 ? c.id : '-'})`}>
                <span className="inline-block h-3.5 w-3.5 shrink-0 rounded-sm border border-slate-400" style={{ background: c.id === 0 ? '#9aa0a6' : c.color }} /><span className="truncate">{c.name}</span></button>
              <span className="w-24 shrink-0 text-right text-[10.5px] text-slate-500">{r ? `${fmtNum(r.percent, 1)}%${r.area_m2 != null ? ' · ' + (p.unit_system === 'imperial' ? fmtNum(r.area_acres, 2) + ' ac' : fmtNum(r.area_m2 / 10000, 2) + ' ha') : ''}` : ''}</span>
            </div>); })}
          {stats && <div className="mt-1 text-[11px] text-slate-500">{fmtNum(stats.classified_pixels / total * 100, 1)}% classified · manual {fmtNum(stats.classes.reduce((a: number, r: any) => a + r.manual_pixels, 0), 0)} px · automatic {fmtNum(stats.classes.reduce((a: number, r: any) => a + r.automatic_pixels, 0), 0)} px</div>}
        </Section>

        <Section title="Display">
          <Slider label="Overlay opacity" value={opacity} min={0.05} max={1} step={0.05} onChange={(v) => setApp((s) => ({ opacity: { ...s.opacity, classes: v } }))} fmt={(v) => `${Math.round(v * 100)}%`} />
          <Field label="Colour overlay by"><select className="input w-full" value={mode} onChange={(e) => setMode(e.target.value as MaskMode)}><option value="class">Class</option><option value="provenance">Provenance (manual / automatic)</option><option value="confidence">Automatic confidence (review)</option></select></Field>
          {mode !== 'class' && <div className="text-[11px] text-slate-500">{mode === 'provenance' ? 'Blue = set by you, orange = automatic, grey = classified without provenance.' : 'Blue = set by you; automatic labels go red (low confidence) to green (high). Review the red areas first.'}</div>}
        </Section>

        {sampled && <Section title="Sampled pixel">
          <div className="mb-1 flex items-center gap-2"><span className="inline-block h-6 w-6 rounded border border-slate-400" style={{ background: `rgb(${sampled.rgb.join(',')})` }} /><span className="font-mono text-[11px]">rgb({sampled.rgb.join(', ')})</span></div>
          <Kv k="Pixel" v={`${sampled.x}, ${sampled.y}`} mono /><Kv k="Class" v={sampled.class_name ?? '-'} />
          <Kv k="Provenance" v={sampled.provenance} />{sampled.confidence != null && <Kv k="Confidence" v={`${Math.round(sampled.confidence * 100)}%`} />}
          {sampled.map && <Kv k="Map (x, y)" v={`${fmtNum(sampled.map.x, 2)}, ${fmtNum(sampled.map.y, 2)}`} mono />}
          {sampled.segment && <Kv k={`Similar region (tol ${tol})`} v={`${sampled.segment.pixels.toLocaleString()} px${sampled.segment.truncated ? '+' : ''}`} />}
          <div className="mt-1 flex gap-1"><button className="btn" onClick={() => setClassId(sampled.class_id)}>Use this class</button><button className="btn" onClick={() => { setTool('similar'); doPreview('similar', [sampled.x + .5, sampled.y + .5]); }}>Select similar…</button></div>
        </Section>}

        <Section title="Reclassify">
          <div className="flex items-center gap-1"><select className="input min-w-0 flex-1" value={from} onChange={(e) => setFrom(Number(e.target.value))}>{classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}</select><span>→</span><select className="input min-w-0 flex-1" value={to} onChange={(e) => setTo(Number(e.target.value))}>{classes.map((c) => <option key={c.id} value={c.id}>{c.id === 0 ? 'Unknown (erase)' : c.name}</option>)}</select></div>
          <div className="mt-1 flex gap-1"><button className="btn" onClick={() => reclass(false)}>In current view</button><button className="btn" onClick={() => reclass(true)}>Whole image…</button></div>
        </Section>

        <Section title="Automatic classification" open={false}>
          <Notice kind="info">Automatic labels are a starting point. They carry a confidence, are marked as <i>automatic</i>, never overwrite your manual edits unless you choose “everything”, and are never used as elevation evidence.</Notice>
          <Field label="Model"><select className="input w-full" value={model} onChange={(e) => setModel(e.target.value)}>{info.models.map((m) => <option key={m.id} value={m.id} disabled={!m.available}>{m.name}{m.available ? '' : ' (unavailable)'}</option>)}</select></Field>
          {info.models.find((m) => m.id === model)?.reason && !info.models.find((m) => m.id === model)?.available && <div className="mb-1 text-[11px] text-warn">{info.models.find((m) => m.id === model)?.reason}</div>}
          {info.models.find((m) => m.id === model)?.description && <div className="mb-1 text-[11px] text-slate-500">{info.models.find((m) => m.id === model)?.description}</div>}
          <Field label="Overwrite"><select className="input w-full" value={overwrite} onChange={(e) => setOverwrite(e.target.value as any)}><option value="unclassified">Only unclassified pixels</option><option value="auto">Unclassified + earlier automatic labels</option><option value="all">Everything (including my manual edits)</option></select></Field>
          <Field label="Region"><select className="input w-full" value={regionSel} onChange={(e) => setRegionSel(e.target.value as any)}><option value="all">Whole orthomosaic</option><option value="view">Current view only</option></select></Field>
          <div className="flex gap-1"><button className="btn btn-primary" disabled={!!autoJob && job.data?.status === 'running'} onClick={runAuto}>Run</button>
            <button className="btn" title="Fit a model on pixels you have labelled by hand, then classify the rest" onClick={async () => { try { const r = await api.classTrain(pid); setAutoJob(r.job_id); bump('jobsTick'); } catch (e) { toast('error', errMsg(e)); } }}>Learn from my corrections</button></div>
          {job.data && <div className="mt-2"><div className="flex justify-between text-[11px]"><span>{job.data.message ?? job.data.status}</span><Chip kind={job.data.status}>{job.data.status}</Chip></div>{(job.data.status === 'running' || job.data.status === 'queued') && <Progress value={job.data.progress} />}
            {job.data.status === 'failed' && <div className="mt-1 text-[11px] text-warn">{job.data.error}</div>}
            {job.data.status === 'succeeded' && job.data.result && <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap rounded bg-slate-50 p-1 text-[10.5px]">{JSON.stringify(job.data.result, null, 1).slice(0, 900)}</pre>}</div>}
          <div className="mt-2 text-[11px] font-semibold text-slate-500">Snapshots (taken before each automatic run)</div>
          {info.snapshots.length === 0 && <div className="text-[11px] text-slate-500">None yet.</div>}
          {info.snapshots.map((s) => <div key={s.id} className="flex items-center justify-between py-0.5 text-[11.5px]"><span className="truncate" title={new Date(s.created_at * 1000).toLocaleString()}>{s.label}</span>
            <button className="btn !py-0" onClick={async () => { if (!window.confirm(`Restore “${s.label}”? Current classification will be replaced (a snapshot of it is NOT kept unless you save one first).`)) return; try { await api.snapshotRestore(pid, s.id); const ci = await api.classInfo(pid); applyState(ci as any); reloadInfo(); toast('ok', 'Snapshot restored'); } catch (e) { toast('error', errMsg(e)); } }}>Restore</button></div>)}
          <button className="btn mt-1" onClick={async () => { try { await api.snapshotCreate(pid, 'manual ' + new Date().toLocaleTimeString()); reloadInfo(); toast('ok', 'Snapshot saved'); } catch (e) { toast('error', errMsg(e)); } }}>Save snapshot now</button>
        </Section>
      </RightPanel>
      {editClasses && <ClassEditor classes={classes} onClose={() => setEditClasses(false)} onSaved={() => { setEditClasses(false); bump('projectsTick'); reloadInfo(); }} pid={pid} />}
    </div>
  );
}

function ClassEditor({ classes, pid, onClose, onSaved }: { classes: CClass[]; pid: string; onClose: () => void; onSaved: () => void }) {
  const [rows, setRows] = useState(classes.map((c) => ({ ...c })));
  const upd = (i: number, k: string, v: any) => setRows((r) => r.map((x, j) => (j === i ? { ...x, [k]: v } : x)));
  const add = () => { const used = new Set(rows.map((r) => r.id)); let id = 10; while (used.has(id)) id++; if (id > 63) { toast('error', 'At most 64 classes (ids 0-63).'); return; } setRows([...rows, { id, key: `class_${id}`, name: `New class ${id}`, color: '#3b82f6', vegetation: false }]); };
  const save = async () => { try { await api.patchProject(pid, { classes: rows.map((r) => ({ id: r.id, key: r.key, name: r.name, color: r.color, vegetation: r.vegetation })) }); toast('ok', 'Classes saved'); onSaved(); } catch (e) { toast('error', errMsg(e)); } };
  return (
    <Modal title="Edit classes" onClose={onClose} wide>
      <div className="mb-2 text-[12px] text-slate-600">Colours and names are per project. “Obscures ground” marks classes (trees, brush, tall grass) that hide the ground in the surface model; the terrain step uses it only to downgrade suspect “ground” points under canopy, never as proof of height. Class 0 (Unknown) is required.</div>
      <Table head={['ID', 'Name', 'Colour', 'Obscures ground']} dense rows={rows.map((r, i) => [r.id, <input className="input w-full" value={r.name} onChange={(e) => upd(i, 'name', e.target.value)} />, <input type="color" value={r.color} onChange={(e) => upd(i, 'color', e.target.value)} />, <input type="checkbox" checked={r.vegetation} disabled={r.id === 0} onChange={(e) => upd(i, 'vegetation', e.target.checked)} />])} />
      <div className="mt-2 flex justify-between"><button className="btn" onClick={add}>+ Add class</button><div className="flex gap-2"><button className="btn" onClick={onClose}>Cancel</button><button className="btn btn-primary" onClick={save}>Save</button></div></div>
    </Modal>
  );
}
