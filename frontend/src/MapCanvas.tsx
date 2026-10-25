import { SyntheticEvent, useEffect, useRef } from 'react';
import { Viewport } from './viewport';

export interface Ptr { sx: number; sy: number; x: number; y: number; e: SyntheticEvent }
export interface MapApi { vp: Viewport; redraw: () => void; fit: (b: [number, number, number, number], pad?: number) => void; canvas: HTMLCanvasElement | null }
interface Props {
  apiRef: { current: MapApi | null };
  draw: (c: CanvasRenderingContext2D, vp: Viewport) => void;
  onDown?: (p: Ptr) => boolean | void;     // return true to take the drag (no panning)
  onMove?: (p: Ptr) => void;
  onUp?: (p: Ptr) => void;
  onDouble?: (p: Ptr) => void;
  onLeave?: () => void;
  onView?: (vp: Viewport) => void;          // debounced after pan / zoom / resize
  onChange?: (vp: Viewport) => void;        // immediate, on every pan / zoom (used to sync split views)
  cursor?: string;
  minPpu?: number; maxPpu?: number;
  initialBounds?: [number, number, number, number] | null;
  className?: string;
  panAlways?: boolean;                      // left-drag always pans (tool = hand)
}

export function MapCanvas({ apiRef, draw, onDown, onMove, onUp, onDouble, onLeave, onView, onChange, cursor, minPpu = 1e-4, maxPpu = 1e4, initialBounds, className, panAlways }: Props) {
  const wrap = useRef<HTMLDivElement | null>(null);
  const cv = useRef<HTMLCanvasElement | null>(null);
  const vp = useRef(new Viewport());
  const drawRef = useRef(draw); drawRef.current = draw;
  const cb = useRef({ onDown, onMove, onUp, onDouble, onLeave, onView, onChange }); cb.current = { onDown, onMove, onUp, onDouble, onLeave, onView, onChange };
  const lim = useRef({ minPpu, maxPpu }); lim.current = { minPpu, maxPpu };
  const raf = useRef(0);
  const drag = useRef<null | { kind: 'pan' | 'tool'; lx: number; ly: number }>(null);
  const space = useRef(false);
  const fitted = useRef(false);
  const viewTimer = useRef<any>(0);

  const render = () => {
    raf.current = 0;
    const c = cv.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, vp.current.w, vp.current.h);
    drawRef.current(ctx, vp.current);
  };
  const redraw = () => { if (!raf.current) raf.current = requestAnimationFrame(render); };
  const viewChanged = () => { clearTimeout(viewTimer.current); viewTimer.current = setTimeout(() => cb.current.onView?.(vp.current), 120); };

  apiRef.current = { vp: vp.current, redraw, canvas: cv.current, fit: (b, pad) => { vp.current.fit(b, pad); redraw(); viewChanged(); } };

  useEffect(() => {
    const el = wrap.current!, c = cv.current!;
    const size = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight);
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.width = w + 'px'; c.style.height = h + 'px';
      const v = vp.current; v.w = w; v.h = h;
      if (!fitted.current && initialBounds) { v.fit(initialBounds); fitted.current = true; }
      redraw(); viewChanged();
    };
    const ro = new ResizeObserver(size); ro.observe(el); size();
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const f = Math.exp(-Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 120) * 0.0016 * (e.ctrlKey ? 3 : 1));
      vp.current.zoomAt(e.clientX - r.left, e.clientY - r.top, f, lim.current.minPpu, lim.current.maxPpu);
      redraw(); viewChanged(); cb.current.onChange?.(vp.current);
    };
    c.addEventListener('wheel', wheel, { passive: false });
    const kd = (e: KeyboardEvent) => { if (e.code === 'Space' && (e.target as HTMLElement)?.tagName !== 'INPUT') { space.current = true; if (cv.current) cv.current.style.cursor = 'grab'; } };
    const ku = (e: KeyboardEvent) => { if (e.code === 'Space') { space.current = false; if (cv.current) cv.current.style.cursor = ''; } };
    window.addEventListener('keydown', kd); window.addEventListener('keyup', ku);
    return () => { ro.disconnect(); c.removeEventListener('wheel', wheel); window.removeEventListener('keydown', kd); window.removeEventListener('keyup', ku); cancelAnimationFrame(raf.current); clearTimeout(viewTimer.current); };
    // eslint-disable-next-line
  }, []);

  // when bounds arrive after mount (data loaded late)
  useEffect(() => { if (initialBounds && !fitted.current && vp.current.w > 1) { vp.current.fit(initialBounds); fitted.current = true; redraw(); viewChanged(); } });
  useEffect(() => { redraw(); });

  const mk = (e: SyntheticEvent): Ptr => {
    const r = cv.current!.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top; const [x, y] = vp.current.toMap(sx, sy);
    return { sx, sy, x, y, e };
  };
  const down = (e: SyntheticEvent) => {
    const p = mk(e);
    const wantsPan = e.button === 1 || e.button === 2 || space.current || panAlways;
    cv.current!.setPointerCapture(e.pointerId);
    if (!wantsPan && cb.current.onDown?.(p)) { drag.current = { kind: 'tool', lx: p.sx, ly: p.sy }; return; }
    drag.current = { kind: 'pan', lx: p.sx, ly: p.sy };
    if (cv.current) cv.current.style.cursor = 'grabbing';
  };
  const move = (e: SyntheticEvent) => {
    const p = mk(e); const d = drag.current;
    if (d?.kind === 'pan') { vp.current.panPx(p.sx - d.lx, p.sy - d.ly); d.lx = p.sx; d.ly = p.sy; redraw(); viewChanged(); cb.current.onChange?.(vp.current); }
    cb.current.onMove?.(p);
  };
  const up = (e: SyntheticEvent) => {
    const p = mk(e); const d = drag.current; drag.current = null;
    if (cv.current) cv.current.style.cursor = space.current ? 'grab' : '';
    try { cv.current!.releasePointerCapture(e.pointerId); } catch { /* not captured */ }
    if (d?.kind === 'tool') cb.current.onUp?.(p);
  };
  return (
    <div ref={wrap} className={`relative h-full w-full overflow-hidden ${className ?? ''}`}>
      <canvas ref={cv} style={{ cursor: cursor ?? 'default' }} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerLeave={() => cb.current.onLeave?.()}
        onDoubleClick={(e) => cb.current.onDouble?.(mk(e))} onContextMenu={(e) => e.preventDefault()} />
    </div>
  );
}

// ------------------------------------------------------------------ drawing helpers
export function drawRing(c: CanvasRenderingContext2D, vp: Viewport, ring: number[][], opt: { stroke?: string; fill?: string; width?: number; dash?: number[]; close?: boolean } = {}) {
  if (ring.length < 2) return;
  c.beginPath();
  ring.forEach((p, i) => { const [sx, sy] = vp.toScreen(p[0], p[1]); i ? c.lineTo(sx, sy) : c.moveTo(sx, sy); });
  if (opt.close !== false) c.closePath();
  if (opt.fill) { c.fillStyle = opt.fill; c.fill(); }
  if (opt.stroke) { c.strokeStyle = opt.stroke; c.lineWidth = opt.width ?? 1.5; c.setLineDash(opt.dash ?? []); c.stroke(); c.setLineDash([]); }
}
export function drawGeoJSON(c: CanvasRenderingContext2D, vp: Viewport, g: any, style: { stroke?: string; fill?: string; width?: number; dash?: number[] }) {
  if (!g) return;
  if (g.type === 'FeatureCollection') return g.features.forEach((f: any) => drawGeoJSON(c, vp, f.geometry, style));
  if (g.type === 'Feature') return drawGeoJSON(c, vp, g.geometry, style);
  if (g.type === 'Polygon') {
    c.beginPath();
    for (const ring of g.coordinates) ring.forEach((p: number[], i: number) => { const [sx, sy] = vp.toScreen(p[0], p[1]); i ? c.lineTo(sx, sy) : c.moveTo(sx, sy); });
    if (style.fill) { c.fillStyle = style.fill; c.fill('evenodd'); }
    if (style.stroke) { c.strokeStyle = style.stroke; c.lineWidth = style.width ?? 1.5; c.setLineDash(style.dash ?? []); c.stroke(); c.setLineDash([]); }
  } else if (g.type === 'MultiPolygon') g.coordinates.forEach((co: any) => drawGeoJSON(c, vp, { type: 'Polygon', coordinates: co }, style));
  else if (g.type === 'LineString') drawRing(c, vp, g.coordinates, { ...style, close: false });
}
export function drawScaleBar(c: CanvasRenderingContext2D, vp: Viewport, mPerUnit: number, imperial: boolean) {
  const targetPx = 120; const mPerPx = mPerUnit / vp.ppu;
  const unitM = imperial ? 0.3048 : 1; const target = (targetPx * mPerPx) / unitM;
  const mag = Math.pow(10, Math.floor(Math.log10(target))); const n = target / mag; const nice = (n >= 5 ? 5 : n >= 2 ? 2 : 1) * mag;
  const px = (nice * unitM) / mPerPx; const x = 14, y = vp.h - 18;
  c.save(); c.fillStyle = 'rgba(255,255,255,0.8)'; c.fillRect(x - 6, y - 16, px + 12, 28);
  c.strokeStyle = '#111'; c.lineWidth = 2; c.beginPath(); c.moveTo(x, y - 5); c.lineTo(x, y); c.lineTo(x + px, y); c.lineTo(x + px, y - 5); c.stroke();
  c.fillStyle = '#111'; c.font = '11px system-ui'; c.fillText(`${nice} ${imperial ? 'ft' : 'm'}`, x + 2, y - 7); c.restore();
}
export function drawNorth(c: CanvasRenderingContext2D, vp: Viewport) {
  c.save(); c.translate(vp.w - 24, 28); c.fillStyle = 'rgba(255,255,255,0.8)'; c.beginPath(); c.arc(0, 0, 16, 0, Math.PI * 2); c.fill();
  c.fillStyle = '#111'; c.beginPath(); c.moveTo(0, -12); c.lineTo(5, 6); c.lineTo(0, 2); c.lineTo(-5, 6); c.closePath(); c.fill();
  c.font = 'bold 9px system-ui'; c.fillText('N', -3, -14); c.restore();
}
