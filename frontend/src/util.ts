import { useEffect, useRef, useState } from 'react';
import { ApiError, Project } from './api';

export const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '-';
  const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${u[i]}`;
}
export const fmtNum = (v: number | null | undefined, d = 2) => (v == null || !isFinite(v) ? '-' : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 }));
export function fmtTime(ts: number | null | undefined): string { return ts ? new Date(ts * 1000).toLocaleString() : '-'; }
export function fmtDur(s: number): string { if (s < 90) return `${s.toFixed(0)}s`; if (s < 5400) return `${(s / 60).toFixed(0)}m`; return `${(s / 3600).toFixed(1)}h`; }
export const errMsg = (e: unknown) => (e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));

// ---------------------------------------------------------------- units
const TO_M: Record<string, number> = { metre: 1, foot: 0.3048, us_survey_foot: 1200 / 3937, degree: NaN };
export const toMetres = (units: string | null | undefined) => TO_M[units || 'metre'] ?? 1;
export function lengthLabel(p: Pick<Project, 'unit_system'> | null, m: number): string {
  if (!isFinite(m)) return '-';
  if (p?.unit_system === 'imperial') { const ft = m / 0.3048; return ft >= 5280 ? `${(ft / 5280).toFixed(2)} mi` : `${ft.toFixed(ft < 100 ? 2 : 1)} ft`; }
  return m >= 1000 ? `${(m / 1000).toFixed(3)} km` : `${m.toFixed(m < 100 ? 2 : 1)} m`;
}
export function areaLabel(p: Pick<Project, 'unit_system'> | null, m2: number): string {
  if (!isFinite(m2)) return '-';
  if (p?.unit_system === 'imperial') { const ac = m2 / 4046.8564224; return ac >= 0.1 ? `${ac.toFixed(3)} ac` : `${(m2 * 10.763910417).toFixed(0)} ft²`; }
  return m2 >= 10000 ? `${(m2 / 10000).toFixed(3)} ha` : `${m2.toFixed(1)} m²`;
}
export function volumeLabel(p: Pick<Project, 'unit_system'> | null, m3: number): string {
  if (!isFinite(m3)) return '-';
  return p?.unit_system === 'imperial' ? `${(m3 * 1.30795062).toFixed(0)} yd³` : `${m3.toFixed(1)} m³`;
}
export function elevLabel(p: Pick<Project, 'vertical_units'> | null, v: number | null | undefined, d = 2): string {
  if (v == null || !isFinite(v)) return '-';
  const u = p?.vertical_units === 'metre' ? 'm' : p?.vertical_units === 'foot' ? 'ft' : p?.vertical_units === 'us_survey_foot' ? 'ft (US)' : '';
  return `${v.toFixed(d)} ${u}`.trim();
}

// ---------------------------------------------------------------- WGS84 <-> projected (UTM / web mercator), Krueger series (matches backend core/crs.py)
const A = 6378137.0, Fl = 1 / 298.257223563, K0 = 0.9996, N = Fl / (2 - Fl);
const AA = A / (1 + N) * (1 + N ** 2 / 4 + N ** 4 / 64 + N ** 6 / 256);
const ALPHA = [N / 2 - 2 * N ** 2 / 3 + 5 * N ** 3 / 16 + 41 * N ** 4 / 180 - 127 * N ** 5 / 288 + 7891 * N ** 6 / 37800,
  13 * N ** 2 / 48 - 3 * N ** 3 / 5 + 557 * N ** 4 / 1440 + 281 * N ** 5 / 630 - 1983433 * N ** 6 / 1935360,
  61 * N ** 3 / 240 - 103 * N ** 4 / 140 + 15061 * N ** 5 / 26880 + 167603 * N ** 6 / 181440,
  49561 * N ** 4 / 161280 - 179 * N ** 5 / 168 + 6601661 * N ** 6 / 7257600,
  34729 * N ** 5 / 80640 - 3418889 * N ** 6 / 1995840, 212378941 * N ** 6 / 319334400];
const E = Math.sqrt(Fl * (2 - Fl));
const rad = (d: number) => d * Math.PI / 180;
const atanh = (x: number) => 0.5 * Math.log((1 + x) / (1 - x));
export function lonlatToProject(epsg: number | null, lon: number, lat: number): [number, number] | null {
  if (epsg === 4326) return [lon, lat];
  if (epsg === 3857) return [A * rad(lon), A * Math.log(Math.tan(Math.PI / 4 + rad(lat) / 2))];
  const zone = epsg && ((epsg >= 32601 && epsg <= 32660) ? epsg - 32600 : (epsg >= 32701 && epsg <= 32760) ? epsg - 32700 : 0);
  if (!epsg || !zone) return null;
  const north = epsg < 32700;
  const dl = rad(lon) - rad(-183 + 6 * zone), la = rad(lat);
  const t = Math.sinh(atanh(Math.sin(la)) - E * atanh(E * Math.sin(la)));
  const xp = Math.atan2(t, Math.cos(dl)), ep = atanh(Math.sin(dl) / Math.sqrt(1 + t * t));
  let xi = xp, eta = ep;
  ALPHA.forEach((a, i) => { const j = i + 1; xi += a * Math.sin(2 * j * xp) * Math.cosh(2 * j * ep); eta += a * Math.cos(2 * j * xp) * Math.sinh(2 * j * ep); });
  return [500000 + K0 * AA * eta, K0 * AA * xi + (north ? 0 : 1e7)];
}
/** Boundaries may be stored as lon/lat or in the project CRS (the API accepts both). Returns rings in project CRS or null if not convertible. */
export function boundaryRings(b: any, epsg: number | null): number[][][] | null {
  if (!b) return null;
  const polys: number[][][][] = b.type === 'Polygon' ? [b.coordinates] : b.type === 'MultiPolygon' ? b.coordinates : [];
  const all = polys.flat(2);
  if (!all.length) return null;
  const ll = all.every((p) => Math.abs(p[0]) <= 180 && Math.abs(p[1]) <= 90);
  const rings: number[][][] = [];
  for (const poly of polys) for (const ring of poly) {
    if (ll && epsg && epsg !== 4326) {
      const out: number[][] = [];
      for (const p of ring) { const q = lonlatToProject(epsg, p[0], p[1]); if (!q) return null; out.push(q); }
      rings.push(out);
    } else rings.push(ring.map((p) => [p[0], p[1]]));
  }
  return rings;
}

// ---------------------------------------------------------------- geometry
export function polygonArea(pts: number[][]): number {
  let a = 0; for (let i = 0; i < pts.length; i++) { const [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % pts.length]; a += x1 * y2 - x2 * y1; } return Math.abs(a) / 2;
}
export function pathLength(pts: number[][]): number { let l = 0; for (let i = 1; i < pts.length; i++) l += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); return l; }

// ---------------------------------------------------------------- hooks
/** Run `fn` now and every `ms` while mounted; returns latest value, error and a reload function. */
export function usePoll<T>(fn: () => Promise<T>, ms: number, deps: readonly unknown[]): { data: T | null; error: string | null; reload: () => void; loading: boolean } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [n, setN] = useState(0);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true; let timer: any;
    const run = async () => {
      try { const d = await fn(); if (alive.current) { setData(d); setError(null); } } catch (e) { if (alive.current) setError(errMsg(e)); }
      if (alive.current) { setLoading(false); if (ms > 0 && !document.hidden) timer = setTimeout(run, ms); else if (ms > 0) timer = setTimeout(run, ms * 3); }
    };
    setLoading(true); run();
    return () => { alive.current = false; clearTimeout(timer); };
    // eslint-disable-next-line
  }, [...deps, n]);
  return { data, error, loading, reload: () => setN((x) => x + 1) };
}
export function useDebounced<T>(v: T, ms: number): T {
  const [d, setD] = useState(v);
  useEffect(() => { const t = setTimeout(() => setD(v), ms); return () => clearTimeout(t); }, [v, ms]);
  return d;
}
export function useElementSize<T extends HTMLElement>(): [{ current: T | null }, { w: number; h: number }] {
  const ref = useRef<T | null>(null);
  const [s, setS] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const ro = new ResizeObserver(() => setS({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el); setS({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);
  return [ref, s];
}
export function useLocal<T>(key: string, init: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(() => { try { const s = localStorage.getItem(key); return s ? JSON.parse(s) : init; } catch { return init; } });
  return [v, (x: T) => { setV(x); try { localStorage.setItem(key, JSON.stringify(x)); } catch { /* storage unavailable */ } }];
}
