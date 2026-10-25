import { useState } from 'react';
import { Project, TerrainState, api, url } from '../api';
import { bump, toast } from '../store';
import { Check, Chip, Field, Kv, Notice, Section, SurfaceBadge, Table } from '../ui';
import { areaLabel, elevLabel, errMsg, fmtNum, lengthLabel, toMetres, volumeLabel } from '../util';

export const DISCLAIMER = 'Preliminary planning estimate. Not for final foundation design, permitting or construction staking without validation by a licensed surveyor / engineer.';
export const CONF_COLOR: Record<string, string> = { high: '#31a354', moderate: '#fdae61', low: '#d63027', unknown: '#8a8a96' };

export function downloadBlob(blob: Blob, name: string) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}

// ------------------------------------------------------------------ run form
export function RunForm({ pid, ts, onStarted }: { pid: string; ts: TerrainState; onStarted: (jobId: string) => void }) {
  const [f, setF] = useState({
    engine: ts.pdal_available ? 'pdal_smrf' : 'builtin_pmf', res: 0.5, cell: 1, slope: 0.15, window: 18, thr: 0.5, scalar: 1.25, outliers: true,
    method: 'local_plane', k: 16, power: 2, radius: 30, smooth: 0, hi: 2, mo: 6, lo: 15, clear: 1, useClass: true,
  });
  const set = (k: string, v: any) => setF((x) => ({ ...x, [k]: v }));
  const num = (k: string, label: string, step = 0.1, hint?: string) => (
    <Field label={label} hint={hint}><input className="input w-full" type="number" step={step} value={(f as any)[k]} onChange={(e) => set(k, Number(e.target.value))} /></Field>
  );
  const pdalEngine = f.engine.startsWith('pdal');
  const start = async () => {
    try {
      const r = await api.terrainRun(pid, {
        resolution_m: f.res, canopy_clearance_m: f.clear, use_classification: f.useClass,
        ground: { engine: f.engine, smrf_cell: f.cell, smrf_slope: f.slope, smrf_window: f.window, smrf_threshold: f.thr, smrf_scalar: f.scalar, outlier_removal: f.outliers },
        interp: { method: f.method, k: f.k, power: f.power, max_radius_m: f.radius, smoothing: f.smooth },
        confidence: { high_dist_m: f.hi, moderate_dist_m: f.mo, low_dist_m: f.lo },
      });
      toast('ok', 'Terrain job queued'); bump('jobsTick'); onStarted(r.job_id);
    } catch (e) { toast('error', errMsg(e)); }
  };
  return (
    <div>
      {!ts.pdal_available && <Notice kind="warn">PDAL is not installed here, so the PDAL SMRF/PMF ground filters cannot run. The built-in progressive morphological filter is selected instead; it is simpler and less proven on steep or dense terrain.</Notice>}
      <Field label="Ground-point filter">
        <select className="input w-full" value={f.engine} onChange={(e) => set('engine', e.target.value)}>
          <option value="pdal_smrf" disabled={!ts.pdal_available}>PDAL SMRF{ts.pdal_available ? '' : ' (PDAL missing)'}</option>
          <option value="pdal_pmf" disabled={!ts.pdal_available}>PDAL PMF{ts.pdal_available ? '' : ' (PDAL missing)'}</option>
          <option value="builtin_pmf">Built-in progressive morphological filter</option>
        </select>
      </Field>
      <div className="grid grid-cols-2 gap-x-2">
        {num('res', 'Terrain resolution (m)', 0.1, 'Cell size of DSM/DTM')}
        {num('clear', 'Canopy clearance (m)', 0.1, 'Ground within this of canopy top is not trusted')}
      </div>
      {pdalEngine && f.engine === 'pdal_smrf' && <div className="grid grid-cols-2 gap-x-2">
        {num('cell', 'SMRF cell', 0.1)}{num('slope', 'SMRF slope', 0.01)}{num('window', 'SMRF window', 1)}{num('thr', 'SMRF threshold', 0.1)}{num('scalar', 'SMRF scalar', 0.05)}
      </div>}
      <Check label="Remove statistical outliers first" checked={f.outliers} onChange={(v) => set('outliers', v)} />
      <Check label="Use visual classification to demote suspect ground" checked={f.useClass} onChange={(v) => set('useClass', v)} title="Ground points under mapped vegetation but close to the canopy top are treated as canopy. Classification is never used as elevation evidence by itself." />
      <Field label="Interpolation where ground was not seen">
        <select className="input w-full" value={f.method} onChange={(e) => set('method', e.target.value)}>
          <option value="local_plane">Local plane fit</option><option value="idw">Inverse distance (IDW)</option><option value="tin">TIN (linear)</option>
          <option value="tps">Thin-plate spline</option><option value="nearest">Nearest</option>
        </select>
      </Field>
      <div className="grid grid-cols-2 gap-x-2">{num('k', 'Neighbours (k)', 1)}{num('radius', 'Max radius (m)', 1)}{f.method === 'idw' && num('power', 'IDW power', 0.5)}{num('smooth', 'Smoothing', 0.1)}</div>
      <div className="mt-1 text-[11px] font-semibold text-slate-500">Confidence = distance to nearest measured ground</div>
      <div className="grid grid-cols-3 gap-x-1">{num('hi', 'High ≤ (m)', 0.5)}{num('mo', 'Moderate ≤ (m)', 0.5)}{num('lo', 'Low ≤ (m)', 0.5)}</div>
      <button className="btn btn-primary mt-1 w-full" onClick={start}>{ts.available ? 'Re-run terrain reconstruction' : 'Run terrain reconstruction'}</button>
    </div>
  );
}

// ------------------------------------------------------------------ summary + coverage
export function SummaryPanel({ ts, p }: { ts: TerrainState; p: Project }) {
  const s = ts.summary; if (!s) return null;
  const cov = s.coverage ?? {}, conf = s.confidence ?? {}, cv = s.cross_validation;
  const pct = (v: number | undefined) => (v == null ? '-' : `${fmtNum(v * 100, 1)}%`);
  return (
    <div>
      <div className="mb-1 text-[11px] font-semibold text-slate-500">Where the ground elevation comes from</div>
      <div className="mb-2 flex h-3 overflow-hidden rounded border border-line">
        <div title={`Measured ${pct(cov.measured)}`} style={{ width: `${(cov.measured ?? 0) * 100}%`, background: '#31a354' }} />
        <div title={`Interpolated ${pct(cov.interpolated)}`} style={{ width: `${(cov.interpolated ?? 0) * 100}%`, background: '#fdae61' }} />
        <div title={`External ${pct(cov.external)}`} style={{ width: `${(cov.external ?? 0) * 100}%`, background: '#4292c6' }} />
        <div title={`None ${pct(cov.none)}`} style={{ width: `${(cov.none ?? 0) * 100}%`, background: '#bbb' }} />
      </div>
      <Kv k="Measured (photogrammetry ground points)" v={pct(cov.measured)} /><Kv k="Interpolated (estimated)" v={pct(cov.interpolated)} />
      <Kv k="External model (LiDAR / DEM)" v={pct(cov.external)} /><Kv k="No data" v={pct(cov.none)} />
      <div className="mt-2 mb-1 text-[11px] font-semibold text-slate-500">Confidence of the bare-earth estimate</div>
      {(['high', 'moderate', 'low', 'unknown'] as const).map((k) => conf[k] && (
        <div key={k} className="flex items-center justify-between py-0.5 text-[12px]"><span className="flex items-center gap-1.5"><span className="inline-block h-3 w-3 rounded-sm" style={{ background: CONF_COLOR[k] }} />{k}</span>
          <span className="font-mono">{pct(conf[k].fraction)} · {areaLabel(p, conf[k].area_m2)}</span></div>))}
      {ts.dense_canopy_advice && <Notice kind="warn"><b>Supplemental survey advised.</b> {ts.dense_canopy_advice}</Notice>}
      {conf.weak_fraction > 0.2 && !ts.dense_canopy_advice && <Notice kind="warn"><b>{pct(conf.weak_fraction)} of the terrain is low-confidence.</b> Under dense canopy the ground is rarely visible to a camera. Consider a ground survey (total station / GNSS rover), or LiDAR if available, before relying on elevations here.</Notice>}
      {cv && <>
        <div className="mt-2 mb-1 text-[11px] font-semibold text-slate-500">Hold-out check against measured ground</div>
        <Kv k="Overall RMSE" v={cv.overall_rmse_m != null ? `${fmtNum(cv.overall_rmse_m, 2)} m` : '-'} />
        <Table dense head={['Distance to ground pts', 'n', 'RMSE (m)', 'Bias (m)']} rows={(cv.by_distance ?? []).filter((r: any) => r.n > 0).map((r: any) => [`${r.dist_from_m}–${r.dist_to_m} m`, r.n, fmtNum(r.rmse_m, 2), fmtNum(r.bias_m, 2)])} />
        <div className="mt-1 text-[10.5px] text-slate-500">{cv.note}</div>
      </>}
      {s.check_points && s.check_points.length > 0 && <>
        <div className="mt-2 mb-1 text-[11px] font-semibold text-slate-500">Surveyed check points</div>
        <Table dense head={['Label', 'dz (m)']} rows={s.check_points.map((c: any) => [c.label, fmtNum(c.dz, 2)])} />
      </>}
      {(s.warnings ?? []).map((w: string, i: number) => <Notice key={i} kind="estimate">{w}</Notice>)}
      <div className="mt-1 text-[10.5px] text-slate-500">Ground filter: {s.ground_engine}; DSM from {s.dsm_source}; {s.cells?.toLocaleString()} cells at {s.resolution_m} m.</div>
    </div>
  );
}

// ------------------------------------------------------------------ elevation sources
export function SourcesPanel({ pid, ts, reload }: { pid: string; ts: TerrainState; reload: () => void }) {
  const patch = async (id: number, b: any) => { try { await api.sourcePatch(pid, id, b); reload(); toast('info', 'Saved. Re-run terrain to apply source changes.'); } catch (e) { toast('error', errMsg(e)); } };
  if (!ts.sources.length) return <div className="text-[12px] text-slate-500">No external elevation sources. LiDAR / DEM GeoTIFFs imported on the Upload tab appear here; higher priority wins where sources overlap.</div>;
  return (
    <div>
      {ts.sources.map((s) => (
        <div key={s.id} className="mb-1.5 rounded border border-line p-1.5">
          <div className="flex items-center justify-between gap-1"><b className="truncate" title={s.name}>{s.name}</b><Chip kind={s.enabled ? 'ok' : 'queued'}>{s.enabled ? 'on' : 'off'}</Chip></div>
          <div className="text-[10.5px] text-slate-500">{s.kind}</div>
          <div className="mt-1 grid grid-cols-3 gap-1">
            <Field label="Priority"><input className="input w-full" type="number" defaultValue={s.priority} onBlur={(e) => Number(e.target.value) !== s.priority && patch(s.id, { priority: Number(e.target.value) })} /></Field>
            <Field label="Z offset"><input className="input w-full" type="number" step="0.01" defaultValue={s.z_offset} onBlur={(e) => Number(e.target.value) !== s.z_offset && patch(s.id, { z_offset: Number(e.target.value) })} /></Field>
            <Field label="Trust"><select className="input w-full" defaultValue={s.trust} onChange={(e) => patch(s.id, { trust: e.target.value })}><option>high</option><option>moderate</option><option>low</option></select></Field>
          </div>
          <div className="flex justify-between"><Check label="Enabled" checked={s.enabled} onChange={(v) => patch(s.id, { enabled: v })} />
            <button className="btn !py-0" onClick={async () => { if (!window.confirm(`Remove source “${s.name}”?`)) return; try { await api.sourceDelete(pid, s.id); reload(); } catch (e) { toast('error', errMsg(e)); } }}>Remove</button></div>
        </div>))}
    </div>
  );
}

// ------------------------------------------------------------------ profile chart (SVG)
export interface Profile { total_length: number; min_z: number; max_z: number; max_grade_percent: number | null; valid_fraction: number; columns: { distance: number[]; x: number[]; y: number[]; ground_z: (number | null)[]; surface_z?: (number | null)[]; canopy_height?: (number | null)[]; confidence?: string[] } }
export function ProfileChart({ prof, p, onHover }: { prof: Profile; p: Project; onHover?: (i: number | null) => void }) {
  const [hov, setHov] = useState<number | null>(null);
  const W = 262, H = 140, L = 34, R = 6, T = 6, B = 18;
  const c = prof.columns, n = c.distance.length;
  const zs: number[] = [];
  for (const v of c.ground_z) if (v != null) zs.push(v);
  for (const v of c.surface_z ?? []) if (v != null) zs.push(v);
  if (!zs.length) return <div className="text-[12px] text-slate-500">No elevation data along this line.</div>;
  const z0 = Math.min(...zs), z1 = Math.max(...zs), dz = Math.max(1e-6, z1 - z0), d1 = c.distance[n - 1] || 1;
  const X = (d: number) => L + (d / d1) * (W - L - R), Y = (z: number) => T + (1 - (z - z0) / dz) * (H - T - B);
  const line = (a: (number | null)[]) => { let s = '', pen = false; a.forEach((v, i) => { if (v == null) { pen = false; return; } s += `${pen ? 'L' : 'M'}${X(c.distance[i]).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; }); return s; };
  // distance is in metres (server computes it from map units); exaggeration = vertical px per metre / horizontal px per metre
  const exag = ((H - T - B) * d1) / Math.max(1e-9, (W - L - R) * dz * toMetres(p.vertical_units));
  const set = (i: number | null) => { setHov(i); onHover?.(i); };
  const move = (e: any) => { const r = e.currentTarget.getBoundingClientRect(); const px = ((e.clientX - r.left) / r.width) * W; set(Math.max(0, Math.min(n - 1, Math.round(((px - L) / (W - L - R)) * (n - 1))))); };
  const col = (i: number) => CONF_COLOR[c.confidence?.[i] ?? 'unknown'];
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded border border-line bg-white" onMouseMove={move} onMouseLeave={() => set(null)}>
        {[0, 0.5, 1].map((t) => <g key={t}><line x1={L} x2={W - R} y1={T + t * (H - T - B)} y2={T + t * (H - T - B)} stroke="#e5e7eb" /><text x={2} y={T + t * (H - T - B) + 3} fontSize="8" fill="#555">{fmtNum(z1 - t * dz, 1)}</text></g>)}
        <text x={L} y={H - 4} fontSize="8" fill="#555">0</text><text x={W - R} y={H - 4} fontSize="8" fill="#555" textAnchor="end">{lengthLabel(p, d1)}</text>
        {c.surface_z && <path d={line(c.surface_z)} fill="none" stroke="#2b6cb0" strokeWidth="1.2" />}
        <path d={line(c.ground_z)} fill="none" stroke="#b7791f" strokeWidth="1.6" strokeDasharray="4 2" />
        {c.ground_z.map((v, i) => v != null && i % Math.max(1, Math.floor(n / 70)) === 0 ? <circle key={i} cx={X(c.distance[i])} cy={Y(v)} r="1.8" fill={col(i)} /> : null)}
        {hov != null && <line x1={X(c.distance[hov])} x2={X(c.distance[hov])} y1={T} y2={H - B} stroke="#111" strokeWidth=".7" />}
      </svg>
      <div className="mt-0.5 flex flex-wrap items-center gap-x-3 text-[10.5px] text-slate-600">
        <span><span className="inline-block h-0 w-4 border-t-2 border-sky-700 align-middle" /> observed surface</span>
        <span><span className="inline-block h-0 w-4 border-t-2 border-dashed border-amber-600 align-middle" /> estimated ground</span>
        <span>dots = confidence</span><span>vertical exaggeration ≈ {fmtNum(exag, 1)}×</span>
      </div>
      {hov != null && <div className="mt-1 rounded bg-slate-50 p-1 text-[11px]">
        <Kv k="Distance" v={lengthLabel(p, c.distance[hov])} mono /><Kv k={<>Ground <Chip kind="estimated">EST</Chip></>} v={elevLabel(p, c.ground_z[hov])} mono />
        {c.surface_z && <Kv k={<>Surface <Chip kind="observed">OBS</Chip></>} v={elevLabel(p, c.surface_z[hov])} mono />}
        {c.canopy_height && <Kv k="Object height" v={elevLabel(p, c.canopy_height[hov])} mono />}
        {c.confidence && <Kv k="Confidence" v={c.confidence[hov]} />}
      </div>}
      <Kv k="Length" v={lengthLabel(p, prof.total_length)} /><Kv k="Estimated ground range" v={`${elevLabel(p, prof.min_z)} – ${elevLabel(p, prof.max_z)}`} />
      <Kv k="Max grade (estimated)" v={prof.max_grade_percent != null ? `${fmtNum(prof.max_grade_percent, 1)}%` : '-'} />
    </div>
  );
}

export async function profileCsv(pid: string, line: number[][]) {
  const r = await fetch(url(`/projects/${pid}/planning/profile`), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ line, format: 'csv' }) });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  downloadBlob(await r.blob(), 'elevation_profile.csv');
}

// ------------------------------------------------------------------ footprint + cut/fill
export function FootprintPanel({ rep, p }: { rep: any; p: Project }) {
  if (!rep) return null;
  if (rep.status && rep.status !== 'ok') return <Notice kind="warn">{rep.reason ?? rep.message ?? `Report status: ${rep.status}`}</Notice>;
  const c = rep.confidence ?? {};
  return (
    <div>
      <SurfaceBadge kind="estimated" />
      <Kv k="Footprint area (plan view)" v={areaLabel(p, (rep.area_m2 ?? 0) * toMetres(p.horizontal_units) ** 2)} />
      <Kv k="Ground elevation min / mean / max" v={`${fmtNum(rep.min_z, 2)} / ${fmtNum(rep.mean_z, 2)} / ${fmtNum(rep.max_z, 2)}`} mono />
      <Kv k="Relief" v={elevLabel(p, rep.relief)} /><Kv k="5th–95th percentile" v={`${fmtNum(rep.p5_z, 2)} – ${fmtNum(rep.p95_z, 2)}`} mono />
      <Kv k="Mean slope" v={`${fmtNum(rep.mean_slope_percent, 1)}% (${fmtNum(rep.mean_slope_degrees, 1)}°)`} />
      <Kv k="Downslope direction" v={rep.downslope_azimuth_deg != null ? `${fmtNum(rep.downslope_azimuth_deg, 0)}° azimuth` : '-'} />
      <Kv k="Plane-fit residual (RMS)" v={elevLabel(p, rep.plane_residual_rms)} /><Kv k="Steepest cell" v={`${fmtNum(rep.max_cell_slope_degrees, 1)}°`} />
      <Kv k="Area steeper than threshold" v={`${fmtNum((rep.steep_fraction ?? 0) * 100, 1)}%`} />
      <Kv k="Mean object height (DSM−DTM)" v={rep.mean_canopy_height != null ? elevLabel(p, rep.mean_canopy_height) : '-'} />
      <Kv k="Terrain coverage" v={`${fmtNum((rep.coverage_fraction ?? 0) * 100, 0)}%`} />
      <div className="mt-1 text-[11px] font-semibold text-slate-500">Confidence under the footprint</div>
      <div className="flex h-3 overflow-hidden rounded border border-line">{(['high', 'moderate', 'low', 'unknown'] as const).map((k) => <div key={k} title={`${k} ${fmtNum((c[k] ?? 0) * 100, 0)}%`} style={{ width: `${(c[k] ?? 0) * 100}%`, background: CONF_COLOR[k] }} />)}</div>
      <div className="text-[10.5px] text-slate-500">high {fmtNum((c.high ?? 0) * 100, 0)}% · moderate {fmtNum((c.moderate ?? 0) * 100, 0)}% · low {fmtNum((c.low ?? 0) * 100, 0)}% · unknown {fmtNum((c.unknown ?? 0) * 100, 0)}%</div>
      {rep.vertex_elevations && <details className="mt-1"><summary className="cursor-pointer text-[11px] text-slate-600">Corner elevations (estimated)</summary>
        <Table dense head={['#', 'x', 'y', 'z']} rows={rep.vertex_elevations.map((v: any, i: number) => Array.isArray(v) ? [i + 1, fmtNum(v[0], 2), fmtNum(v[1], 2), fmtNum(v[2], 2)] : [i + 1, fmtNum(v.x, 2), fmtNum(v.y, 2), fmtNum(v.z, 2)])} /></details>}
      {(rep.warnings ?? []).map((w: string, i: number) => <Notice key={i} kind="estimate">{w}</Notice>)}
    </div>
  );
}

export function CutFillPanel({ pid, p, polygon, defaultZ, planeFrom }: { pid: string; p: Project; polygon: number[][] | null; defaultZ: number | null; planeFrom: number[] | null }) {
  const [type, setType] = useState<'flat' | 'plane' | 'three_points'>('flat');
  const [z, setZ] = useState<string>('');
  const [sx, setSx] = useState('0'), [sy, setSy] = useState('0');
  const [pts, setPts] = useState<string>('');
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    if (!polygon) { toast('error', 'Draw a footprint first.'); return; }
    const zz = z !== '' ? Number(z) : defaultZ;
    let surface: any;
    if (type === 'flat') surface = { type, z: zz };
    else if (type === 'plane') { const ref = planeFrom ?? polygon[0]; surface = { type, z: zz, x_ref: ref[0], y_ref: ref[1], dzdx: Number(sx) / 100, dzdy: Number(sy) / 100 }; }
    else {
      const nums = pts.split(/[\s,;]+/).filter(Boolean).map(Number);
      if (nums.length !== 9 || nums.some((v) => !isFinite(v))) { toast('error', 'Enter three points as x y z, nine numbers in all.'); return; }
      surface = { type, points: [nums.slice(0, 3), nums.slice(3, 6), nums.slice(6, 9)] };
    }
    setBusy(true);
    try { setRes(await api.cutFill(pid, polygon, surface)); } catch (e) { toast('error', errMsg(e)); setRes(null); } finally { setBusy(false); }
  };
  const vl = (m3: number) => volumeLabel(p, m3);
  return (
    <div>
      <Notice kind="estimate">Volumes use the <b>estimated</b> bare-earth surface. They are given only where the terrain evidence is strong enough to defend; otherwise the tool says why it refuses.</Notice>
      <Field label="Proposed surface"><select className="input w-full" value={type} onChange={(e) => setType(e.target.value as any)}><option value="flat">Flat pad (one elevation)</option><option value="plane">Sloped plane</option><option value="three_points">Plane through three points</option></select></Field>
      {type !== 'three_points' && <Field label={`Finished elevation (${p.vertical_units})`} hint={defaultZ != null ? `Blank = mean existing ground ${fmtNum(defaultZ, 2)}` : undefined}><input className="input w-full" type="number" step="0.05" value={z} placeholder={defaultZ != null ? String(Math.round(defaultZ * 100) / 100) : ''} onChange={(e) => setZ(e.target.value)} /></Field>}
      {type === 'plane' && <div className="grid grid-cols-2 gap-x-2"><Field label="Slope toward east (%)"><input className="input w-full" type="number" value={sx} onChange={(e) => setSx(e.target.value)} /></Field><Field label="Slope toward north (%)"><input className="input w-full" type="number" value={sy} onChange={(e) => setSy(e.target.value)} /></Field></div>}
      {type === 'three_points' && <Field label="Three points: x y z ×3" hint="Nine numbers, in project CRS units"><textarea className="input w-full" rows={3} value={pts} onChange={(e) => setPts(e.target.value)} /></Field>}
      <button className="btn btn-primary w-full" disabled={!polygon || busy} onClick={run}>{polygon ? 'Compute cut / fill' : 'Draw a footprint first'}</button>
      {res?.status === 'refused' && <Notice kind="warn"><b>Not computed.</b> {res.reason}</Notice>}
      {res?.status === 'ok' && <div className="mt-2">
        <Kv k="Cut (remove)" v={vl(res.cut)} mono /><Kv k="Fill (add)" v={vl(res.fill)} mono /><Kv k="Net" v={`${vl(res.net)} ${res.net > 0 ? 'fill' : 'cut'}`} mono />
        <Kv k="Mean depth of change" v={elevLabel(p, res.mean_depth_of_change)} /><Kv k="Area" v={areaLabel(p, res.area)} />
        <Kv k="Worst-case uncertainty" v={`± ${vl(res.volume_uncertainty_worst_case)}`} mono />
        <Kv k="Weak-confidence share" v={`${fmtNum((res.weak_fraction ?? 0) * 100, 0)}%`} /><Kv k="Coverage" v={`${fmtNum((res.coverage_fraction ?? 0) * 100, 0)}%`} />
        <div className="mt-1 text-[11px] text-slate-600">{res.uncertainty_note}</div>
        <Notice kind="estimate">{DISCLAIMER}</Notice>
      </div>}
    </div>
  );
}

export function PlanningSection({ title, children, open }: { title: string; children?: any; open?: boolean }) { return <Section title={title} open={open}>{children}</Section>; }
