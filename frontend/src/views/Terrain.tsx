import { useEffect, useMemo, useRef, useState } from 'react';
import { Project, TerrainState, api, url } from '../api';
import { MapApi, MapCanvas, Ptr, drawGeoJSON, drawNorth, drawRing, drawScaleBar } from '../MapCanvas';
import { Check, Chip, Empty, Field, Kv, Notice, Progress, RightPanel, Section, SurfaceBadge } from '../ui';
import { MaskLayer, OrthoLayer } from '../tiles';
import { LAYER_KIND, LAYER_NAME, RAMP_NAMES, rampCss } from '../ramps';
import { bump, setApp, setLayer, setOpacity, setStatus, toast, useApp, useProject } from '../store';
import { Viewport } from '../viewport';
import { areaLabel, boundaryRings, elevLabel, errMsg, fmtNum, lengthLabel, pathLength, polygonArea, toMetres, usePoll } from '../util';
import { CONF_COLOR, CutFillPanel, DISCLAIMER, FootprintPanel, Profile, ProfileChart, RunForm, SourcesPanel, SummaryPanel, profileCsv } from './terrainPanels';

type Tool = 'hand' | 'inspect' | 'distance' | 'area' | 'profile' | 'footprint' | 'boundary';
const TOOLS: { id: Tool; label: string; key: string; icon: string; tip: string; planning?: boolean }[] = [
  { id: 'hand', label: 'Pan', key: 'H', icon: '✥', tip: 'Pan the map (or hold Space)' },
  { id: 'inspect', label: 'Inspect', key: 'I', icon: '⌖', tip: 'Hover for elevations; click to pin the readout' },
  { id: 'distance', label: 'Distance', key: 'D', icon: '↔', tip: 'Click points; Enter or double-click to finish. Reports length, vertex elevations and grade' },
  { id: 'area', label: 'Area', key: 'A', icon: '▱', tip: 'Click vertices; Enter or double-click to finish. Reports area and perimeter' },
  { id: 'profile', label: 'Profile', key: 'P', icon: '⌇', tip: 'Draw a line; get a cross-section of observed surface vs estimated ground' },
  { id: 'footprint', label: 'Footprint', key: 'F', icon: '⌂', tip: 'Draw a building footprint; get elevations, slope, confidence, and cut/fill', planning: true },
  { id: 'boundary', label: 'Boundary', key: 'B', icon: '⬡', tip: 'Draw the property boundary (replaces the current one)', planning: true },
];
const ELEV = ['dsm', 'dtm', 'ndsm', 'evidence', 'confidence'] as const;
type LKey = typeof ELEV[number];

// A server-rendered raster window; the previous image stays visible until the next one arrives.
class Rendered {
  img: HTMLImageElement | null = null; bbox: number[] | null = null; want = '';
  constructor(private onLoad: () => void) {}
  request(u: string, bbox: number[]) {
    if (u === this.want) return; this.want = u;
    const im = new Image();
    im.onload = () => { if (this.want === u) { this.img = im; this.bbox = bbox; this.onLoad(); } };
    im.src = u;
  }
  draw(c: CanvasRenderingContext2D, vp: Viewport, alpha: number, smooth: boolean) {
    if (!this.img || !this.bbox) return;
    const [x0, y0] = vp.toScreen(this.bbox[0], this.bbox[3]), [x1, y1] = vp.toScreen(this.bbox[2], this.bbox[1]);
    c.save(); c.globalAlpha = alpha; c.imageSmoothingEnabled = smooth; c.drawImage(this.img, x0, y0, x1 - x0, y1 - y0); c.restore();
  }
}

interface Measure { kind: 'distance' | 'area'; pts: number[][]; length?: number; area?: number; perimeter?: number; z?: { dsm: (number | null)[] | null; dtm: (number | null)[] | null } }

export function TerrainView({ mode }: { mode: 'terrain' | 'planning' }) {
  const p = useProject()!;
  const jt = useApp((s) => s.jobsTick);
  const ts = usePoll(() => api.terrain(p.id), 0, [p.id, jt]);
  if (ts.loading) return <div className="m-auto text-slate-500">Loading terrain…</div>;
  if (!ts.data) return <div className="m-auto text-warn">{ts.error ?? 'Terrain state unavailable'}</div>;
  if (!ts.data.available) return <NoTerrain ts={ts.data} p={p} reload={ts.reload} />;
  return <Workspace mode={mode} ts={ts.data} p={p} reload={ts.reload} key={p.id} />;
}

function NoTerrain({ ts, p, reload }: { ts: TerrainState; p: Project; reload: () => void }) {
  const [jobId, setJobId] = useState<string | null>(null);
  const job = usePoll(() => (jobId ? api.job(jobId) : Promise.resolve(null)), 1500, [jobId]);
  const st = job.data?.status;
  useEffect(() => { if (st === 'succeeded') { bump('jobsTick'); reload(); } /* eslint-disable-next-line */ }, [st]);
  const hasCloud = ts.layers.pointcloud;
  return (
    <>
      <Empty title="No terrain model yet">
        {hasCloud
          ? 'A point cloud is available. Run terrain reconstruction (right panel) to separate ground from canopy and build the surface (DSM), bare-earth estimate (DTM) and confidence layers.'
          : 'Terrain reconstruction needs a photogrammetric point cloud. Run OpenDroneMap from the Dashboard (or import a point cloud on the Upload tab), then return here.'}
      </Empty>
      <RightPanel>
        <Notice kind="estimate">{ts.notice}</Notice>
        <Section title="Terrain reconstruction">
          {hasCloud ? <RunForm pid={p.id} ts={ts} onStarted={setJobId} /> : <div className="text-[12px] text-slate-500">No point cloud yet.</div>}
          {job.data && <div className="mt-2"><div className="flex justify-between text-[11px]"><span className="truncate">{job.data.message ?? job.data.status}</span><Chip kind={job.data.status}>{job.data.status}</Chip></div>
            <Progress value={job.data.progress} />{job.data.status === 'failed' && <div className="mt-1 text-[11px] text-warn">{job.data.error}</div>}</div>}
        </Section>
      </RightPanel>
    </>
  );
}

function Workspace({ mode, ts, p, reload }: { mode: 'terrain' | 'planning'; ts: TerrainState; p: Project; reload: () => void }) {
  const pid = p.id;
  const layers = useApp((s) => s.layers), opacity = useApp((s) => s.opacity), ramp = useApp((s) => s.ramp), zexag = useApp((s) => s.zexag), steepDeg = useApp((s) => s.steepDeg);
  const jt = useApp((s) => s.jobsTick);
  const mapRef = useRef<MapApi | null>(null);
  const redraw = () => mapRef.current?.redraw();
  const mpu = toMetres(p.horizontal_units);

  const ortho = usePoll(() => api.ortho(pid), 0, [pid]);
  const cinfo = usePoll(() => (ortho.data?.available ? api.classInfo(pid) : Promise.resolve(null)), 0, [pid, ortho.data?.available]);
  const stats = usePoll(async () => {
    const out: Record<string, any> = {};
    await Promise.all(ELEV.filter((k) => ts.layers[k]).map(async (k) => { try { out[k] = await api.layerStats(pid, k); } catch { /* layer missing */ } }));
    return out;
  }, 0, [pid, ts]);
  const feats = usePoll(() => api.features(pid), 0, [pid]);
  const cps = usePoll(() => api.control(pid), 0, [pid]);
  const steep = usePoll(() => (layers.steep ? api.steep(pid, steepDeg) : Promise.resolve(null)), 0, [pid, layers.steep, steepDeg, ts]);
  const [minArea, setMinArea] = useState(500);
  const drain = usePoll(() => (layers.drainage ? api.drainage(pid, minArea) : Promise.resolve(null)), 0, [pid, layers.drainage, minArea, ts]);

  const orthoLayer = useMemo(() => (ortho.data?.available && ortho.data.pyramid ? new OrthoLayer(pid, ortho.data, redraw) : null), [pid, ortho.data]);
  const mask = useMemo(() => {
    if (!orthoLayer || !cinfo.data) return null;
    const m = new MaskLayer(pid, cinfo.data.width, cinfo.data.height, orthoLayer.grid, redraw);
    m.setVersion(cinfo.data.version); m.setClasses(cinfo.data.classes, new Set(cinfo.data.classes.filter((c) => c.visible === false).map((c) => c.id))); return m;
  }, [orthoLayer, cinfo.data]);
  const rendered = useMemo(() => Object.fromEntries(ELEV.map((k) => [k, new Rendered(redraw)])) as Record<LKey, Rendered>, [pid]);

  // ------------------------------------------------------------ state
  const [tool, setTool] = useState<Tool>('inspect');
  const [pts, setPts] = useState<number[][]>([]);
  const hover = useRef<[number, number] | null>(null);
  const [insp, setInsp] = useState<any>(null);
  const [pinned, setPinned] = useState<[number, number] | null>(null);
  const [meas, setMeas] = useState<Measure | null>(null);
  const [prof, setProf] = useState<{ line: number[][]; data: Profile } | null>(null);
  const [profHover, setProfHover] = useState<number | null>(null);
  const [fp, setFp] = useState<{ poly: number[][]; rep: any } | null>(null);
  const [swipe, setSwipe] = useState<{ on: boolean; a: string; b: string; pos: number }>({ on: false, a: 'ortho', b: 'dtm', pos: 0.5 });
  const [range, setRange] = useState<{ min: string; max: string }>({ min: '', max: '' });
  const [drainOpts, setDrainOpts] = useState(false);
  const dragSwipe = useRef(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = usePoll(() => (jobId ? api.job(jobId) : Promise.resolve(null)), 1500, [jobId]);
  useEffect(() => { if (job.data?.status === 'succeeded') { bump('jobsTick'); setJobId(null); toast('ok', 'Terrain updated'); reload(); } /* eslint-disable-next-line */ }, [job.data?.status]);

  // first visit: show something useful (estimated bare earth) when no elevation layer is on
  useEffect(() => { const l = layers; if (!ELEV.some((k) => l[k])) setLayer('dtm', true); /* eslint-disable-next-line */ }, []);
  useEffect(() => { if (tool === 'boundary' || tool === 'footprint') setLayer('features', true); }, [tool]);

  // ------------------------------------------------------------ layer ranges and rendering
  const sd = stats.data ?? {};
  const shared = useMemo(() => {
    const lo = [sd.dsm?.stats?.p2, sd.dtm?.stats?.p2].filter((v) => v != null) as number[], hi = [sd.dsm?.stats?.p98, sd.dtm?.stats?.p98].filter((v) => v != null) as number[];
    return lo.length ? { min: Math.min(...lo), max: Math.max(...hi) } : null;
  }, [sd.dsm, sd.dtm]);
  const rngFor = (k: string): { min: number; max: number } | null => {
    if (k === 'ndsm') { const s = sd.ndsm?.stats; return s ? { min: 0, max: Math.max(1, s.p98) } : null; }
    const mn = range.min !== '' && isFinite(Number(range.min)) ? Number(range.min) : shared?.min, mx = range.max !== '' && isFinite(Number(range.max)) ? Number(range.max) : shared?.max;
    return mn != null && mx != null && mx > mn ? { min: mn, max: mx } : shared;
  };
  const cfg = useRef<any>({}); cfg.current = { layers, ramp, zexag, sd, range, shared, ts, swipe };
  const requestAll = () => {
    const m = mapRef.current; if (!m || m.vp.w < 4) return;
    const vp = m.vp, c = cfg.current;
    const [x0, y0, x1, y1] = vp.bbox(), px = (x1 - x0) * 0.12, py = (y1 - y0) * 0.12;
    const bb = [x0 - px, y0 - py, x1 + px, y1 + py];
    const sc = Math.min(1, 1800 / (vp.w * 1.24), 1800 / (vp.h * 1.24)) * (window.devicePixelRatio > 1 ? 1.25 : 1);
    const w = Math.max(16, Math.round(vp.w * 1.24 * sc)), h = Math.max(16, Math.round(vp.h * 1.24 * sc));
    const want = c.swipe.on ? new Set([c.swipe.a, c.swipe.b]) : null;
    for (const k of ELEV) {
      if (!c.ts.layers[k] || !(want ? want.has(k) : c.layers[k])) continue;
      const q = new URLSearchParams({ bbox: bb.map((v) => v.toFixed(4)).join(','), w: String(w), h: String(h), v: String(c.sd[k]?.created_at ?? 0) });
      if (k === 'dsm' || k === 'dtm' || k === 'ndsm') {
        const r = k === 'ndsm' ? (c.sd.ndsm?.stats ? { min: 0, max: Math.max(1, c.sd.ndsm.stats.p98) } : null) : rngFor(k);
        q.set('ramp', c.ramp); if (r) { q.set('vmin', String(r.min)); q.set('vmax', String(r.max)); }
        if (c.layers.hillshade) { q.set('hillshade', '1'); q.set('zexag', String(c.zexag)); }
      }
      rendered[k].request(url(`/projects/${pid}/terrain/render/${k}?${q}`), bb);
    }
    redraw();
  };
  const rkey = JSON.stringify([layers.dsm, layers.dtm, layers.ndsm, layers.confidence, layers.evidence, layers.hillshade, ramp, zexag, range, swipe.on, swipe.a, swipe.b, !!stats.data]);
  useEffect(() => { requestAll(); /* eslint-disable-next-line */ }, [rkey, rendered]);
  useEffect(() => { redraw(); });

  // ------------------------------------------------------------ geometry helpers
  const rings = useMemo(() => boundaryRings(p.boundary, p.epsg), [p.boundary, p.epsg]);
  const bounds = useMemo((): [number, number, number, number] | null => {
    const g = sd.dtm?.grid ?? sd.dsm?.grid ?? orthoLayer?.grid; if (!g) return null;
    return [g.x0, g.y0 - g.height * g.res, g.x0 + g.width * g.res, g.y0];
  }, [sd, orthoLayer]);

  // ------------------------------------------------------------ tools
  const cancel = () => setPts([]);
  const finish = async (list = pts) => {
    const need = tool === 'distance' || tool === 'profile' ? 2 : 3;
    if (list.length < need) { toast('info', `Need at least ${need} points.`); return; }
    setPts([]);
    try {
      if (tool === 'distance' || tool === 'area') {
        const z: any = await api.sampleElev(pid, list);
        const m: Measure = tool === 'distance' ? { kind: 'distance', pts: list, length: pathLength(list) * mpu, z } : { kind: 'area', pts: list, area: polygonArea(list) * mpu * mpu, perimeter: pathLength([...list, list[0]]) * mpu, z };
        setMeas(m);
      } else if (tool === 'profile') {
        const data = await api.profile(pid, list); setProf({ line: list, data }); setProfHover(null);
      } else if (tool === 'footprint') {
        const rep = await api.footprintReport(pid, list, steepDeg); setFp({ poly: list, rep });
      } else if (tool === 'boundary') {
        if (!window.confirm('Replace the project boundary with this polygon?')) return;
        await api.patchProject(pid, { boundary: { type: 'Polygon', coordinates: [[...list, list[0]]] } }); bump('projectsTick'); toast('ok', 'Boundary saved');
      }
    } catch (e) { toast('error', errMsg(e)); }
  };
  const inspectAt = useRef<{ t: any; n: number }>({ t: 0, n: 0 });
  const doInspect = (x: number, y: number, now = false) => {
    clearTimeout(inspectAt.current.t);
    const go = async () => { const n = ++inspectAt.current.n; try { const r = await api.inspect(pid, x, y); if (n === inspectAt.current.n) setInsp(r); } catch { /* outside raster */ } };
    if (now) go(); else inspectAt.current.t = setTimeout(go, 90);
  };

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName; const ty = (e.target as HTMLInputElement)?.type; if ((tag === 'INPUT' && !['checkbox', 'radio', 'range', 'button'].includes(ty)) || tag === 'TEXTAREA' || tag === 'SELECT' || e.ctrlKey || e.metaKey || e.altKey) return;
      const t = TOOLS.find((x) => x.key === e.key.toUpperCase() && (mode === 'planning' || !x.planning));
      if (t) { setTool(t.id); setPts([]); return; }
      if (e.key === 'Enter') { e.preventDefault(); finish(); } else if (e.key === 'Escape') { cancel(); setPinned(null); setMeas(null); } else if (e.key === 'Backspace') { e.preventDefault(); setPts((a) => a.slice(0, -1)); }
    };
    window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k);
  });

  const onDown = (q: Ptr): boolean | void => {
    if (swipe.on && mapRef.current && Math.abs(q.sx - swipe.pos * mapRef.current.vp.w) < 9) { dragSwipe.current = true; return true; }
    return tool !== 'hand';
  };
  const onMove = (q: Ptr) => {
    hover.current = [q.x, q.y];
    if (dragSwipe.current && mapRef.current) { setSwipe((s) => ({ ...s, pos: Math.min(0.98, Math.max(0.02, q.sx / mapRef.current!.vp.w)) })); return; }
    setStatus({ left: `E ${fmtNum(q.x, 2)}   N ${fmtNum(q.y, 2)}${p.epsg ? `   (EPSG:${p.epsg})` : ''}` });
    if (tool === 'inspect' && !pinned) doInspect(q.x, q.y);
    if (pts.length || tool === 'inspect') redraw();
  };
  const onUp = (q: Ptr) => {
    if (dragSwipe.current) { dragSwipe.current = false; return; }
    if (tool === 'inspect') { setPinned([q.x, q.y]); doInspect(q.x, q.y, true); return; }
    if (tool === 'hand') return;
    const v = mapRef.current!.vp, last = pts[pts.length - 1];
    if (last) { const [lx, ly] = v.toScreen(last[0], last[1]); if (Math.hypot(lx - q.sx, ly - q.sy) < 4) return; }
    setPts([...pts, [q.x, q.y]]);
  };
  const onDouble = (q: Ptr) => { if (tool !== 'hand' && tool !== 'inspect') finish(pts); void q; };

  // ------------------------------------------------------------ drawing
  const drawElev = (c: CanvasRenderingContext2D, vp: Viewport, k: string) => {
    if (k === 'ortho') orthoLayer?.draw(c, vp, 1);
    else if (k === 'classes') mask?.draw(c, vp, opacity.classes ?? 0.6);
    else if ((ELEV as readonly string[]).includes(k)) (rendered as any)[k].draw(c, vp, opacity[k] ?? 0.85, vp.ppu * (sd[k]?.grid?.res ?? 1) < 3);
  };
  const draw = (c: CanvasRenderingContext2D, vp: Viewport) => {
    c.fillStyle = '#eef1f4'; c.fillRect(0, 0, vp.w, vp.h);
    if (swipe.on) {
      const x = swipe.pos * vp.w;
      for (const side of [0, 1]) {
        c.save(); c.beginPath(); if (side === 0) c.rect(0, 0, x, vp.h); else c.rect(x, 0, vp.w - x, vp.h); c.clip();
        const k = side === 0 ? swipe.a : swipe.b;
        if (k !== 'ortho') orthoLayer?.draw(c, vp, 1);
        drawElev(c, vp, k); c.restore();
      }
      c.save(); c.strokeStyle = '#fff'; c.lineWidth = 3; c.beginPath(); c.moveTo(x, 0); c.lineTo(x, vp.h); c.stroke();
      c.fillStyle = '#0b5cab'; c.beginPath(); c.arc(x, vp.h / 2, 11, 0, Math.PI * 2); c.fill(); c.fillStyle = '#fff'; c.font = 'bold 12px system-ui'; c.fillText('◂▸', x - 9, vp.h / 2 + 4);
      c.font = 'bold 11px system-ui'; c.fillStyle = 'rgba(0,0,0,.65)'; c.fillRect(6, 6, 190, 18); c.fillRect(vp.w - 196, 6, 190, 18); c.fillStyle = '#fff';
      c.fillText(LAYER_NAME[swipe.a] ?? swipe.a, 10, 19); c.textAlign = 'right'; c.fillText(LAYER_NAME[swipe.b] ?? swipe.b, vp.w - 10, 19); c.restore();
    } else {
      if (layers.ortho) orthoLayer?.draw(c, vp, 1);
      if (layers.classes) mask?.draw(c, vp, opacity.classes ?? 0.6);
      for (const k of ['dsm', 'dtm', 'ndsm', 'evidence', 'confidence'] as LKey[]) if (layers[k] && ts.layers[k]) drawElev(c, vp, k);
    }
    // overlays
    if (layers.steep && steep.data) drawGeoJSON(c, vp, steep.data.polygons, { fill: 'rgba(220,40,30,0.38)', stroke: 'rgba(160,20,10,0.9)', width: 1 });
    if (layers.drainage && drain.data) {
      drawGeoJSON(c, vp, drain.data.depressions, { fill: 'rgba(0,150,255,0.35)', stroke: 'rgba(0,90,200,0.8)', width: 1 });
      for (const f of drain.data.channels.features) { const a = f.properties.upstream_area_m2 ?? 0; drawRing(c, vp, f.geometry.coordinates, { stroke: 'rgba(10,80,220,0.9)', width: Math.min(5, 1 + Math.log10(Math.max(10, a)) - 1), close: false }); }
    }
    if (layers.boundary && rings) for (const r of rings) drawRing(c, vp, r, { stroke: '#e08a00', width: 2.5, dash: [8, 4] });
    if (layers.features) for (const f of feats.data ?? []) {
      const g = f.geometry; if (!g) continue;
      const st = f.kind === 'footprint' ? { stroke: '#0f766e', fill: 'rgba(15,118,110,0.15)', width: 2 } : f.kind === 'boundary' ? { stroke: '#e08a00', width: 2, dash: [8, 4] } : { stroke: '#a21caf', width: 2 };
      if (f.kind === 'boundary') continue;
      drawGeoJSON(c, vp, g, st);
      const pt = g.type === 'Polygon' ? g.coordinates[0][0] : g.type === 'LineString' ? g.coordinates[0] : g.type === 'Point' ? g.coordinates : null;
      if (pt && f.name) { const [sx, sy] = vp.toScreen(pt[0], pt[1]); c.fillStyle = '#111'; c.font = '11px system-ui'; c.fillText(f.name, sx + 5, sy - 5); }
    }
    if (layers.control) for (const q of cps.data ?? []) {
      const [sx, sy] = vp.toScreen(q.x, q.y); c.lineWidth = 2;
      if (q.kind === 'gcp') { c.strokeStyle = '#b3261e'; c.beginPath(); c.moveTo(sx - 6, sy - 6); c.lineTo(sx + 6, sy + 6); c.moveTo(sx + 6, sy - 6); c.lineTo(sx - 6, sy + 6); c.stroke(); }
      else { c.strokeStyle = '#1b7a3a'; c.strokeRect(sx - 5, sy - 5, 10, 10); }
      c.fillStyle = '#111'; c.font = '10px system-ui'; c.fillText(q.label, sx + 8, sy - 6);
    }
    // results
    if (fp) drawRing(c, vp, fp.poly, { stroke: '#0f766e', fill: 'rgba(15,118,110,0.2)', width: 2.5 });
    if (meas) drawRing(c, vp, meas.pts, { stroke: '#111', fill: meas.kind === 'area' ? 'rgba(255,255,255,0.2)' : undefined, width: 2, close: meas.kind === 'area', dash: [6, 3] });
    if (prof) {
      drawRing(c, vp, prof.line, { stroke: '#a21caf', width: 2.5, close: false });
      const cc = prof.data.columns;
      if (profHover != null && cc.x[profHover] != null) { const [sx, sy] = vp.toScreen(cc.x[profHover], cc.y[profHover]); c.fillStyle = '#fff'; c.strokeStyle = '#a21caf'; c.lineWidth = 3; c.beginPath(); c.arc(sx, sy, 6, 0, Math.PI * 2); c.fill(); c.stroke(); }
    }
    if (pinned) { const [sx, sy] = vp.toScreen(pinned[0], pinned[1]); c.strokeStyle = '#111'; c.lineWidth = 2; c.beginPath(); c.arc(sx, sy, 7, 0, Math.PI * 2); c.moveTo(sx - 12, sy); c.lineTo(sx + 12, sy); c.moveTo(sx, sy - 12); c.lineTo(sx, sy + 12); c.stroke(); }
    // sketch in progress
    if (pts.length) {
      const cur: number[][] = hover.current && tool !== 'inspect' ? [...pts, hover.current] : pts;
      const poly = tool === 'area' || tool === 'footprint' || tool === 'boundary';
      drawRing(c, vp, cur, { stroke: '#d9480f', fill: poly ? 'rgba(217,72,15,0.12)' : undefined, width: 2, close: poly && cur.length > 2 });
      c.fillStyle = '#d9480f'; for (const q of pts) { const [sx, sy] = vp.toScreen(q[0], q[1]); c.beginPath(); c.arc(sx, sy, 3.5, 0, Math.PI * 2); c.fill(); }
      const lab = poly ? areaLabel(p, polygonArea(cur) * mpu * mpu) : lengthLabel(p, pathLength(cur) * mpu);
      if (hover.current) { const [sx, sy] = vp.toScreen(hover.current[0], hover.current[1]); c.font = 'bold 11px system-ui'; c.fillStyle = 'rgba(255,255,255,.9)'; c.fillRect(sx + 10, sy + 6, c.measureText(lab).width + 8, 16); c.fillStyle = '#7c2d12'; c.fillText(lab, sx + 14, sy + 18); }
    }
    drawScaleBar(c, vp, mpu, p.unit_system === 'imperial'); drawNorth(c, vp);
  };

  // ------------------------------------------------------------ right panel pieces
  const activeLayers = swipe.on ? [swipe.a, swipe.b] : (['dsm', 'dtm', 'ndsm', 'evidence', 'confidence'] as string[]).filter((k) => layers[k as LKey] && ts.layers[k]);
  const availList = ['ortho', ...(mask ? ['classes'] : []), ...ELEV.filter((k) => ts.layers[k])];
  const polyForCF = fp?.poly ?? (meas?.kind === 'area' ? meas.pts : null);
  const saveFeature = async (kind: string, name: string, geometry: any, props: any = {}) => {
    try { await api.createFeature(pid, { kind, name, geometry, props }); feats.reload(); toast('ok', 'Saved'); setLayer('features', true); } catch (e) { toast('error', errMsg(e)); }
  };
  const legend = (k: string) => {
    const r = k === 'ndsm' ? rngFor('ndsm') : rngFor(k); const kind = LAYER_KIND[k];
    if (k === 'confidence') return (
      <div key={k} className="mb-2"><div className="mb-0.5 flex items-center gap-1 font-semibold">{LAYER_NAME[k]} <SurfaceBadge kind="evidence" /></div>
        {(['high', 'moderate', 'low', 'unknown'] as const).map((c) => <div key={c} className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: CONF_COLOR[c] }} />{c}{c === 'high' ? ' (measured ground nearby)' : c === 'unknown' ? ' (no data)' : ''}</div>)}</div>);
    if (k === 'evidence') return (
      <div key={k} className="mb-2"><div className="mb-0.5 flex items-center gap-1 font-semibold">{LAYER_NAME[k]} <SurfaceBadge kind="evidence" /></div>
        {[['#31a354', 'measured ground point'], ['#fdae61', 'interpolated (estimate)'], ['#4292c6', 'external model (LiDAR/DEM)']].map(([col, t]) => <div key={t} className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: col }} />{t}</div>)}</div>);
    if (k === 'ortho' || k === 'classes') return null;
    return (
      <div key={k} className="mb-2"><div className="mb-0.5 flex items-center gap-1 font-semibold">{LAYER_NAME[k]} <SurfaceBadge kind={kind} /></div>
        <div className="h-2.5 w-full rounded-sm border border-slate-400" style={{ background: rampCss(ramp) }} />
        <div className="flex justify-between font-mono"><span>{r ? fmtNum(r.min, 1) : '-'}</span><span>{r ? fmtNum((r.min + r.max) / 2, 1) : ''}</span><span>{r ? fmtNum(r.max, 1) : '-'}</span></div></div>);
  };

  const toGeo = (kind: 'Polygon' | 'LineString', list: number[][]) => ({ type: kind, coordinates: kind === 'Polygon' ? [[...list, list[0]]] : list });
  const mz = (arr: (number | null)[] | null | undefined) => (arr ? arr.map((v) => (v == null ? '-' : fmtNum(v, 2))) : null);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-1 border-b border-line bg-white px-2 py-1">
        {TOOLS.filter((t) => mode === 'planning' || !t.planning).map((t) => (
          <button key={t.id} className={`btn ${tool === t.id ? 'btn-on' : ''}`} title={`${t.tip} (${t.key})`} onClick={() => { setTool(t.id); setPts([]); }}><span className="mr-1">{t.icon}</span>{t.label}</button>))}
        <span className="mx-1 h-5 w-px bg-line" />
        <Check label="Swipe compare" checked={swipe.on} onChange={(v) => setSwipe((s) => ({ ...s, on: v }))} />
        {swipe.on && <>
          <select className="input" value={swipe.a} onChange={(e) => setSwipe({ ...swipe, a: e.target.value })}>{availList.map((k) => <option key={k} value={k}>{LAYER_NAME[k]}</option>)}</select>
          <span>⇄</span>
          <select className="input" value={swipe.b} onChange={(e) => setSwipe({ ...swipe, b: e.target.value })}>{availList.map((k) => <option key={k} value={k}>{LAYER_NAME[k]}</option>)}</select></>}
        <div className="flex-1" />
        {pts.length > 0 && <><span className="text-[12px] text-slate-600">{pts.length} points</span><button className="btn btn-primary" onClick={() => finish()}>Finish (Enter)</button><button className="btn" onClick={cancel}>Cancel (Esc)</button></>}
        <button className="btn" onClick={() => bounds && mapRef.current?.fit(bounds, 0.92)}>Fit</button>
      </div>
      <div className="relative min-h-0 flex-1">
        <MapCanvas apiRef={mapRef} draw={draw} onDown={onDown} onMove={onMove} onUp={onUp} onDouble={onDouble} onView={requestAll} initialBounds={bounds}
          onLeave={() => { hover.current = null; if (!pinned) setStatus({ left: '' }); }} cursor={tool === 'hand' ? 'grab' : tool === 'inspect' ? 'crosshair' : 'copy'} panAlways={tool === 'hand'} />
        {/* what is on screen, and what it means */}
        <div className="pointer-events-none absolute bottom-9 left-2 max-w-[240px] rounded border border-line bg-white/92 p-2 text-[11px] shadow">
          {activeLayers.map((k) => legend(k))}
          {!activeLayers.some((k) => k !== 'ortho' && k !== 'classes') && <div className="text-slate-500">Turn on an elevation layer in the left panel.</div>}
          {activeLayers.some((k) => k === 'dtm' || k === 'confidence' || k === 'evidence') && <div className="text-amber-800">Bare earth is an <b>estimate</b>; it was not seen under canopy.</div>}
          {activeLayers.some((k) => k === 'dsm') && <div className="text-sky-900">The DSM shows what the camera <b>observed</b> (treetops, roofs).</div>}
        </div>
        {(insp || pinned) && tool === 'inspect' && insp && (
          <div className="absolute right-2 top-2 w-[250px] rounded border border-line bg-white/95 p-2 text-[12px] shadow">
            <div className="mb-1 flex items-center justify-between"><b>{pinned ? 'Pinned point' : 'Under cursor'}</b>{pinned && <button className="btn !py-0" onClick={() => setPinned(null)}>Unpin</button>}</div>
            <Kv k="E, N" v={`${fmtNum(insp.x, 2)}, ${fmtNum(insp.y, 2)}`} mono />
            <Kv k={<span className="flex items-center gap-1">Surface <Chip kind="observed">OBSERVED</Chip></span>} v={elevLabel(p, insp.dsm)} mono />
            <Kv k={<span className="flex items-center gap-1">Ground <Chip kind="estimated">ESTIMATED</Chip></span>} v={elevLabel(p, insp.dtm)} mono />
            <Kv k="Object height" v={elevLabel(p, insp.ndsm)} mono />
            <Kv k="Ground evidence" v={insp.evidence ?? '-'} /><Kv k="Confidence" v={insp.confidence ? <span className="flex items-center justify-end gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: CONF_COLOR[insp.confidence] }} />{insp.confidence}</span> : '-'} />
            <Kv k="Source" v={insp.source ?? '-'} /><Kv k="Land cover (visual)" v={insp.land_cover ?? '-'} />
            {insp.note && <div className={`mt-1 rounded p-1 text-[11px] ${insp.evidence === 'measured ground' ? 'bg-emerald-50 text-emerald-900' : 'bg-amber-50 text-amber-900'}`}>{insp.note}</div>}
          </div>)}
        {tool !== 'hand' && tool !== 'inspect' && pts.length === 0 && <div className="pointer-events-none absolute bottom-9 left-1/2 -translate-x-1/2 rounded bg-slate-900/80 px-3 py-1 text-[12px] text-white">{TOOLS.find((t) => t.id === tool)?.tip}</div>}
      </div>

      <RightPanel>
        <Notice kind="estimate"><b>Preliminary planning estimate.</b> {ts.notice.split('Preliminary')[0]}</Notice>
        <Section title="Terrain layers">
          <Field label="Colour ramp"><select className="input w-full" value={ramp} onChange={(e) => setApp({ ramp: e.target.value })}>{RAMP_NAMES.map((r) => <option key={r}>{r}</option>)}</select></Field>
          <div className="grid grid-cols-2 gap-x-2">
            <Field label="Ramp min"><input className="input w-full" type="number" placeholder={shared ? fmtNum(shared.min, 1) : 'auto'} value={range.min} onChange={(e) => setRange({ ...range, min: e.target.value })} /></Field>
            <Field label="Ramp max"><input className="input w-full" type="number" placeholder={shared ? fmtNum(shared.max, 1) : 'auto'} value={range.max} onChange={(e) => setRange({ ...range, max: e.target.value })} /></Field>
          </div>
          <div className="mb-1 text-[10.5px] text-slate-500">DSM and DTM share one range so equal colours mean equal elevation.</div>
          <div className="mb-1 flex flex-wrap gap-x-3">{ELEV.filter((k) => ts.layers[k]).map((k) => <Check key={k} label={k.toUpperCase()} checked={layers[k]} onChange={(v) => setLayer(k, v)} />)}</div>
          {layers.hillshade && <div className="mb-1"><div className="flex justify-between text-[11px] text-slate-600"><span>Vertical exaggeration (hillshade)</span><span className="font-mono">{zexag}×</span></div>
            <input type="range" className="w-full" min={1} max={8} step={0.5} value={zexag} onChange={(e) => setApp({ zexag: Number(e.target.value) })} /></div>}
          <div className="grid grid-cols-2 gap-x-2">
            <Field label={`Steep slope > ${steepDeg}°`}><input type="range" className="w-full" min={5} max={45} value={steepDeg} onChange={(e) => setApp({ steepDeg: Number(e.target.value) })} /></Field>
            <Field label="Min drainage area (m²)"><input className="input w-full" type="number" value={minArea} onChange={(e) => setMinArea(Math.max(10, Number(e.target.value)))} /></Field>
          </div>
          {steep.data && layers.steep && <div className="text-[11px] text-slate-600">{fmtNum(steep.data.fraction * 100, 1)}% of the site ({areaLabel(p, steep.data.area_m2)}) is steeper than {steepDeg}° on the estimated surface.</div>}
          {drain.data && layers.drainage && <div className="text-[11px] text-slate-600">{drain.data.note}</div>}
          {(steep.error || drain.error) && <div className="text-[11px] text-warn">{steep.error ?? drain.error}</div>}
        </Section>

        {meas && <Section title={meas.kind === 'distance' ? 'Distance' : 'Area measurement'} actions={<button className="btn !py-0" onClick={() => setMeas(null)}>Clear</button>}>
          {meas.kind === 'distance' ? <>
            <Kv k="Length (plan view)" v={lengthLabel(p, meas.length!)} />
            {(() => {
              const zt = meas.z?.dtm, zs = meas.z?.dsm; if (!zt) return null;
              const rows = meas.pts.map((_, i) => i);
              return <div className="mt-1 max-h-40 overflow-auto text-[11px]"><table className="w-full"><thead><tr className="text-left text-slate-500"><th>#</th><th>Ground (est.)</th><th>Surface (obs.)</th><th>Grade</th></tr></thead><tbody>
                {rows.map((i) => { const dl = i ? Math.hypot(meas.pts[i][0] - meas.pts[i - 1][0], meas.pts[i][1] - meas.pts[i - 1][1]) * mpu : 0; const g = i && zt[i] != null && zt[i - 1] != null && dl > 0 ? ((zt[i]! - zt[i - 1]!) * toMetres(p.vertical_units)) / dl * 100 : null;
                  return <tr key={i} className="odd:bg-slate-50"><td>{i + 1}</td><td className="font-mono">{mz(zt)?.[i]}</td><td className="font-mono">{mz(zs)?.[i] ?? '-'}</td><td className="font-mono">{g == null ? '-' : `${g > 0 ? '+' : ''}${fmtNum(g, 1)}%`}</td></tr>; })}</tbody></table></div>;
            })()}
            <div className="mt-1 flex flex-wrap gap-1"><button className="btn" onClick={async () => { setTool('profile'); try { setProf({ line: meas.pts, data: await api.profile(pid, meas.pts) }); } catch (e) { toast('error', errMsg(e)); } }}>Make profile</button>
              <button className="btn" onClick={() => saveFeature('measurement', `Distance ${lengthLabel(p, meas.length!)}`, toGeo('LineString', meas.pts), { length_m: meas.length })}>Save</button></div>
          </> : <>
            <Kv k="Area (plan view)" v={areaLabel(p, meas.area!)} /><Kv k="Perimeter" v={lengthLabel(p, meas.perimeter!)} />
            <div className="mt-1 flex flex-wrap gap-1">{mode === 'planning' && <button className="btn" onClick={async () => { try { setFp({ poly: meas.pts, rep: await api.footprintReport(pid, meas.pts, steepDeg) }); } catch (e) { toast('error', errMsg(e)); } }}>Terrain report for this area</button>}
              <button className="btn" onClick={() => saveFeature('measurement', `Area ${areaLabel(p, meas.area!)}`, toGeo('Polygon', meas.pts), { area_m2: meas.area })}>Save</button></div>
          </>}
          <div className="mt-1 text-[10.5px] text-slate-500">Elevations are from the DTM (estimated) and DSM (observed). Lengths and areas are horizontal.</div>
        </Section>}

        {prof && <Section title="Elevation profile" actions={<button className="btn !py-0" onClick={() => setProf(null)}>Clear</button>}>
          <ProfileChart prof={prof.data} p={p} onHover={(i) => { setProfHover(i); }} />
          <div className="mt-1 flex flex-wrap gap-1"><button className="btn" onClick={() => profileCsv(pid, prof.line).catch((e) => toast('error', errMsg(e)))}>Download CSV</button>
            <button className="btn" onClick={() => saveFeature('profile', `Profile ${lengthLabel(p, prof.data.total_length)}`, toGeo('LineString', prof.line), { max_grade_percent: prof.data.max_grade_percent })}>Save line</button></div>
        </Section>}

        {mode === 'planning' && <>
          <Section title="Footprint terrain report" actions={fp ? <button className="btn !py-0" onClick={() => setFp(null)}>Clear</button> : undefined}>
            {!fp && <div className="text-[12px] text-slate-500">Choose the <b>Footprint</b> tool (F) and draw the building outline. You get ground elevations, best-fit slope, steep fraction, and how much of the footprint rests on measured vs estimated ground.</div>}
            {fp && <><FootprintPanel rep={fp.rep} p={p} />
              <div className="mt-1 flex gap-1"><button className="btn" onClick={() => saveFeature('footprint', `Footprint ${areaLabel(p, polygonArea(fp.poly) * mpu * mpu)}`, toGeo('Polygon', fp.poly), { report: { mean_z: fp.rep.mean_z, mean_slope_percent: fp.rep.mean_slope_percent } })}>Save footprint</button></div></>}
          </Section>
          <Section title="Cut / fill (proposed grading)" open={!!fp}>
            <CutFillPanel pid={pid} p={p} polygon={polyForCF} defaultZ={fp?.rep?.mean_z ?? null} planeFrom={fp?.rep?.best_fit_plane ? polyForCF?.[0] ?? null : null} />
          </Section>
        </>}

        <Section title="Saved features" open={mode === 'planning'}>
          {(feats.data ?? []).length === 0 && <div className="text-[12px] text-slate-500">Nothing saved. Footprints, lines and measurements you save appear here and in the GeoJSON export.</div>}
          {(feats.data ?? []).map((f: any) => <div key={f.id} className="flex items-center justify-between gap-1 py-0.5 text-[12px]"><span className="truncate" title={f.name ?? ''}><Chip>{f.kind}</Chip> {f.name ?? ''}</span>
            <button className="btn !py-0" onClick={async () => { try { await api.deleteFeature(pid, f.id); feats.reload(); } catch (e) { toast('error', errMsg(e)); } }}>✕</button></div>)}
        </Section>

        <Section title="Terrain summary and coverage"><SummaryPanel ts={ts} p={p} /></Section>
        <Section title="Elevation sources" open={false}><SourcesPanel pid={pid} ts={ts} reload={reload} /></Section>
        <Section title="Rebuild terrain" open={false}>
          <RunForm pid={pid} ts={ts} onStarted={setJobId} />
          {job.data && <div className="mt-2"><div className="flex justify-between text-[11px]"><span className="truncate">{job.data.message ?? job.data.status}</span><Chip kind={job.data.status}>{job.data.status}</Chip></div><Progress value={job.data.progress} />
            {job.data.status === 'failed' && <div className="mt-1 text-[11px] text-warn">{job.data.error}</div>}</div>}
        </Section>
        <div className="p-2 text-[10.5px] text-slate-500">{DISCLAIMER}</div>
      </RightPanel>
    </div>
  );
}
