import { useMemo, useRef, useState } from 'react';
import { api } from '../api';
import { Check, Chip, Empty, Field, Kv, Notice, Progress, Section, Table } from '../ui';
import { JobList } from '../jobs';
import { MapApi, MapCanvas, drawGeoJSON, drawNorth, drawRing, drawScaleBar } from '../MapCanvas';
import { bump, selectProject, setApp, toast, useApp, useProject } from '../store';
import { Viewport } from '../viewport';
import { areaLabel, boundaryRings, errMsg, fmtBytes, fmtNum, fmtTime, lonlatToProject, toMetres, usePoll } from '../util';

function useMapper(epsg: number | null, fp: any) {
  return useMemo(() => {
    const feats: any[] = fp?.features ?? [];
    const supported = !!(epsg && lonlatToProject(epsg, 0, 0));
    if (supported) return { supported, map: (lon: number, lat: number) => lonlatToProject(epsg, lon, lat)! };
    const pts = feats.filter((f) => f.geometry?.type === 'Point').map((f) => f.geometry.coordinates);
    const lon0 = pts.length ? pts.reduce((a: number, p: number[]) => a + p[0], 0) / pts.length : 0, lat0 = pts.length ? pts.reduce((a: number, p: number[]) => a + p[1], 0) / pts.length : 0;
    return { supported: false, map: (lon: number, lat: number): [number, number] => [(lon - lon0) * Math.cos(lat0 * Math.PI / 180) * 111320, (lat - lat0) * 110574] };
  }, [epsg, fp]);
}

function SiteMap() {
  const p = useProject()!;
  const layers = useApp((s) => s.layers);
  const tick = useApp((s) => s.jobsTick);
  const fpq = usePoll(() => api.footprints(p.id), 10000, [p.id, p.image_count, tick]);
  const cpq = usePoll(() => api.control(p.id), 15000, [p.id]);
  const api_ = useRef<MapApi | null>(null);
  const m = useMapper(p.epsg, fpq.data);
  const proj = useMemo(() => {
    const feats: any[] = fpq.data?.features ?? [];
    const conv = (g: any): any => g.type === 'Point' ? { ...g, coordinates: m.map(g.coordinates[0], g.coordinates[1]) }
      : g.type === 'Polygon' ? { ...g, coordinates: g.coordinates.map((r: number[][]) => r.map((q) => m.map(q[0], q[1]))) } : g;
    const cams = feats.filter((f) => f.properties.kind === 'camera').map((f) => ({ ...f, geometry: conv(f.geometry) }));
    const fps = feats.filter((f) => f.properties.kind === 'footprint').map((f) => ({ ...f, geometry: conv(f.geometry) }));
    const rings = m.supported ? boundaryRings(p.boundary, p.epsg) : null;
    const xs: number[] = [], ys: number[] = [];
    [...cams, ...fps].forEach((f) => (f.geometry.type === 'Point' ? [f.geometry.coordinates] : f.geometry.coordinates[0]).forEach((q: number[]) => { xs.push(q[0]); ys.push(q[1]); }));
    (rings ?? []).flat().forEach((q) => { xs.push(q[0]); ys.push(q[1]); });
    const bounds: [number, number, number, number] | null = xs.length ? [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] : null;
    return { cams, fps, rings, bounds };
  }, [fpq.data, m, p.boundary, p.epsg]);
  const cps = m.supported ? (cpq.data ?? []) : [];

  const draw = (c: CanvasRenderingContext2D, vp: Viewport) => {
    c.fillStyle = '#eef1f4'; c.fillRect(0, 0, vp.w, vp.h);
    if (layers.footprints) for (const f of proj.fps) {
      const poor = f.properties.poor_overlap;
      drawRing(c, vp, f.geometry.coordinates[0], { fill: poor ? 'rgba(220,60,40,0.12)' : 'rgba(40,120,170,0.06)', stroke: poor ? 'rgba(200,50,30,0.7)' : 'rgba(40,120,170,0.35)', width: 1 });
    }
    if (layers.boundary && proj.rings) for (const r of proj.rings) drawRing(c, vp, r, { stroke: '#e08a00', width: 2.5, dash: [8, 4] });
    if (layers.cameras || !layers.footprints) for (const f of proj.cams) {
      const [sx, sy] = vp.toScreen(f.geometry.coordinates[0], f.geometry.coordinates[1]);
      c.fillStyle = f.properties.poor_overlap ? '#c0392b' : '#1f5f8b'; c.beginPath(); c.arc(sx, sy, 2.5, 0, Math.PI * 2); c.fill();
    }
    if (layers.control) for (const q of cps) {
      const [sx, sy] = vp.toScreen(q.x, q.y); c.lineWidth = 2;
      if (q.kind === 'gcp') { c.strokeStyle = '#b3261e'; c.beginPath(); c.moveTo(sx - 6, sy - 6); c.lineTo(sx + 6, sy + 6); c.moveTo(sx + 6, sy - 6); c.lineTo(sx - 6, sy + 6); c.stroke(); }
      else { c.strokeStyle = '#1b7a3a'; c.strokeRect(sx - 5, sy - 5, 10, 10); }
      c.fillStyle = '#222'; c.font = '10px system-ui'; c.fillText(q.label, sx + 8, sy - 6);
    }
    drawScaleBar(c, vp, m.supported ? toMetres(p.horizontal_units) : 1, p.unit_system === 'imperial'); drawNorth(c, vp);
  };
  return (
    <div className="relative h-[340px] overflow-hidden rounded border border-line bg-white">
      <MapCanvas apiRef={api_} draw={draw} initialBounds={proj.bounds ? padded(proj.bounds) : null} cursor="grab" panAlways />
      {!proj.bounds && <div className="absolute inset-0 flex items-center justify-center text-slate-500">No image positions yet. Upload GPS-tagged images to see the flight footprints.</div>}
      <div className="absolute left-2 top-2 flex flex-col gap-1 rounded bg-white/90 px-2 py-1 text-[11px]">
        <span><span className="mr-1 inline-block h-2 w-3 border border-sky-700 bg-sky-100" />footprint</span>
        <span><span className="mr-1 inline-block h-2 w-3 border border-red-600 bg-red-100" />poor estimated overlap</span>
        <span><span className="mr-1 inline-block h-0 w-3 border-t-2 border-dashed border-amber-500" />boundary</span>
      </div>
      {!m.supported && proj.bounds && <div className="absolute bottom-2 left-2 max-w-[60%] rounded bg-amber-50/95 px-2 py-1 text-[11px] text-amber-900">Project CRS {p.epsg ? `EPSG:${p.epsg}` : '(not set)'} cannot be transformed in the browser, so this map is a local approximation around the mean camera position. Boundary and control points are hidden here.</div>}
    </div>
  );
}
const padded = (b: [number, number, number, number]): [number, number, number, number] => { const w = Math.max(b[2] - b[0], 10), h = Math.max(b[3] - b[1], 10); return [b[0] - w * 0.05, b[1] - h * 0.05, b[2] + w * 0.05, b[3] + h * 0.05]; };

function CrsPanel() {
  const p = useProject()!;
  const [epsg, setEpsg] = useState(p.epsg ? String(p.epsg) : '');
  const [units, setUnits] = useState(p.horizontal_units);
  const [vunits, setVunits] = useState(p.vertical_units);
  const [vepsg, setVepsg] = useState(p.vertical_epsg ? String(p.vertical_epsg) : '');
  const [sys, setSys] = useState(p.unit_system);
  const locked = p.products.length > 0;
  const save = async () => {
    try {
      await api.patchProject(p.id, { epsg: epsg ? Number(epsg) : null, vertical_epsg: vepsg ? Number(vepsg) : null, horizontal_units: units, vertical_units: vunits, unit_system: sys });
      toast('ok', 'Coordinate settings saved'); bump('projectsTick');
    } catch (e) { toast('error', errMsg(e)); }
  };
  return (
    <Section title="Coordinate system and units">
      {locked && <Notice kind="info">This project already has processed products, so the horizontal CRS is locked: relabelling would not reproject them.</Notice>}
      <div className="grid grid-cols-2 gap-2">
        <Field label="Horizontal EPSG"><input className="input w-full" value={epsg} disabled={locked} inputMode="numeric" onChange={(e) => setEpsg(e.target.value.replace(/\D/g, ''))} /></Field>
        <Field label="Vertical EPSG (datum)"><input className="input w-full" value={vepsg} inputMode="numeric" placeholder="e.g. 5703 (NAVD88)" onChange={(e) => setVepsg(e.target.value.replace(/\D/g, ''))} /></Field>
        <Field label="Horizontal units"><select className="input w-full" value={units} disabled={locked} onChange={(e) => setUnits(e.target.value)}><option value="metre">metre</option><option value="foot">foot</option><option value="us_survey_foot">US survey foot</option></select></Field>
        <Field label="Vertical units"><select className="input w-full" value={vunits} onChange={(e) => setVunits(e.target.value)}><option value="metre">metre</option><option value="foot">foot</option><option value="us_survey_foot">US survey foot</option></select></Field>
      </div>
      <Field label="Display units"><select className="input w-full" value={sys} onChange={(e) => setSys(e.target.value as any)}><option value="metric">Metric (m, ha)</option><option value="imperial">Imperial (ft, acres)</option></select></Field>
      {!p.epsg && p.suggested_crs && <Notice kind="info">Suggested from image GPS: <b>EPSG:{p.suggested_crs.epsg}</b> ({p.suggested_crs.basis}). <button className="btn ml-1 !py-0" onClick={() => setEpsg(String(p.suggested_crs!.epsg))}>Use</button></Notice>}
      {!p.vertical_epsg && <div className="mb-1 text-[11px] text-slate-500">Vertical datum is not specified. Elevations are only comparable to external data if the datum matches.</div>}
      <button className="btn btn-primary" onClick={save}>Save</button>
    </Section>
  );
}

function OdmPanel() {
  const p = useProject()!;
  const tick = useApp((s) => s.jobsTick);
  const { data, error, reload } = usePoll(() => api.odmState(p.id), 4000, [p.id, tick]);
  const [preset, setPreset] = useState('balanced');
  const [ungeo, setUngeo] = useState(false);
  const start = async () => {
    try { await api.odmStart(p.id, { preset, allow_ungeoreferenced: ungeo }); toast('ok', 'Photogrammetry job queued'); bump('jobsTick'); reload(); } catch (e) { toast('error', errMsg(e)); }
  };
  const q = data?.quality;
  return (
    <Section title="Photogrammetry (OpenDroneMap)">
      {error && <div className="text-warn">{error}</div>}
      {data && (
        <>
          <Kv k="Images" v={`${data.images} (${data.images_with_gps} with GPS)`} />
          <Kv k="GCP list" v={data.gcp_list ? 'present' : 'none'} />
          <Kv k="Docker" v={data.docker.available ? <Chip kind="ok">found</Chip> : <Chip kind="warn">not found</Chip>} />
          {!data.docker.available && <Notice kind="warn">{data.docker.hint} Orthomosaic, point cloud and DSM cannot be produced here until Docker and the ODM image are available. You can still import an existing orthomosaic, DEM or LiDAR on the Upload tab.</Notice>}
          {data.blocking.map((b: string) => <Notice key={b} kind="estimate">{b}</Notice>)}
          <div className="flex items-end gap-2">
            <Field label="Quality preset"><select className="input" value={preset} onChange={(e) => setPreset(e.target.value)}><option value="draft">Draft (fast)</option><option value="balanced">Balanced</option><option value="high">High (slow, memory hungry)</option></select></Field>
            <button className="btn btn-primary mb-1.5" disabled={data.blocking.length > 0 && !ungeo && data.images < 3} onClick={start}>Run</button>
          </div>
          <Check label="Allow non-georeferenced result (no GPS / GCP)" checked={ungeo} onChange={setUngeo} />
          <div className="mt-1 text-[11px] text-slate-500">{data.resume_note}</div>
          {q && <div className="mt-2"><div className="mb-1 text-[11px] font-semibold text-slate-500">Last quality checks</div>
            {q.checks.map((c: any) => <div key={c.name} className="flex items-start gap-1 py-0.5 text-[12px]"><Chip kind={c.status === 'pass' ? 'ok' : c.status === 'fail' ? 'failed' : 'estimated'}>{c.status}</Chip><span><b>{c.name}</b>: {c.message}</span></div>)}</div>}
        </>
      )}
    </Section>
  );
}

function ControlPanel() {
  const p = useProject()!;
  const { data, reload } = usePoll(() => api.control(p.id), 0, [p.id]);
  const [text, setText] = useState('');
  const [kind, setKind] = useState<'gcp' | 'check'>('check');
  const imp = async () => { try { const r: any = await api.importControl(p.id, text, kind, false); toast('ok', `Imported ${r.imported} points`); setText(''); reload(); } catch (e) { toast('error', errMsg(e)); } };
  return (
    <Section title="GCPs and check points" badge={<span className="ml-1 font-normal normal-case text-slate-400">{data?.length ?? 0}</span>} open={false}>
      <div className="mb-1 text-[11px] text-slate-500">GCPs help georeference the model. <b>Check points</b> are held out and used only to measure the terrain's vertical error. Provide X, Y in the project CRS and Z as the ground elevation.</div>
      {(data ?? []).length > 0 && <div className="mb-2 max-h-40 overflow-auto"><Table head={['Label', 'Kind', 'X', 'Y', 'Z', 'dz']} dense rows={(data ?? []).map((c: any) => [c.label, c.kind, fmtNum(c.x, 2), fmtNum(c.y, 2), fmtNum(c.z, 3), c.residual?.status === 'ok' ? <span title={`${c.residual.confidence} / ${c.residual.evidence}`}>{fmtNum(c.residual.dz, 3)}</span> : (c.residual?.status ?? '-')])} /></div>}
      <textarea className="input h-20 w-full font-mono" placeholder={'label,x,y,z\nCP1,571012.3,4402044.8,251.62'} value={text} onChange={(e) => setText(e.target.value)} />
      <div className="mt-1 flex items-center gap-2"><select className="input" value={kind} onChange={(e) => setKind(e.target.value as any)}><option value="check">check points</option><option value="gcp">GCPs</option></select><button className="btn" disabled={!text.trim()} onClick={imp}>Import CSV</button></div>
    </Section>
  );
}

export function Dashboard() {
  const p = useProject()!;
  const tick = useApp((s) => s.jobsTick);
  const flights = usePoll(() => api.flights(p.id), 8000, [p.id, tick]);
  const prods = usePoll(() => api.products(p.id), 8000, [p.id, tick]);
  const storage = usePoll(() => api.storage(), 15000, []);
  const ortho = usePoll(() => api.ortho(p.id), 10000, [p.id, tick]);
  const pstore = storage.data?.projects?.find((x: any) => x.id === p.id)?.bytes;
  const del = async () => {
    const n = window.prompt(`Delete project "${p.name}" and ALL its images, products and edits? Type the project name to confirm.`);
    if (n == null) return;
    try { await api.deleteProject(p.id, n); toast('ok', 'Project deleted'); selectProject(null); bump('projectsTick'); } catch (e) { toast('error', errMsg(e)); }
  };
  const fs = flights.data?.summary;
  return (
    <div className="flex min-h-0 flex-1 overflow-auto">
      <div className="grid w-full grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-3 p-3 [&>*]:min-w-0 max-[1250px]:grid-cols-1">
        <div className="col-span-full flex flex-wrap items-end justify-between gap-2 rounded border border-line bg-white px-3 py-2">
          <div><div className="text-[17px] font-semibold">{p.name}</div><div className="text-slate-500">{p.address || 'No address'} · {p.image_count} images ({fmtBytes(p.image_bytes)})</div></div>
          <div className="flex flex-wrap items-center gap-2">
            <Chip kind={p.epsg ? 'ok' : 'warn'}>{p.epsg ? `EPSG:${p.epsg}` : 'CRS not set'}</Chip><Chip>{p.horizontal_units} / {p.vertical_units}</Chip>
            {p.is_sample && <Chip kind="warn">Synthetic sample data</Chip>}
            <button className="btn btn-danger" onClick={del}>Delete project…</button>
          </div>
        </div>
        {p.is_sample && <div className="col-span-full"><Notice kind="warn"><b>SYNTHETIC SAMPLE.</b> This lot was generated by the development-mode sample builder with a known terrain so the algorithms can be exercised. It is not a real site and must not be used for any decision.</Notice></div>}
        <div className="space-y-3">
          <div className="rounded border border-line bg-white"><Section title="Flights, footprints and overlap"><SiteMap />
            <div className="mt-2 grid grid-cols-4 gap-2 text-center">
              {[['Images', p.image_count], ['With GPS', p.image_count - p.images_without_gps], ['Flights', flights.data?.groups?.filter((g: any) => g.group > 0).length ?? '-'], ['Poor overlap', fs?.poor_overlap ?? '-']].map(([k, v]) => <div key={k as string} className="rounded border border-line bg-panel py-1"><div className="text-[16px] font-semibold">{v as any}</div><div className="text-[10.5px] text-slate-500">{k}</div></div>)}
            </div>
            {fs && <div className="mt-2 text-[11.5px] text-slate-600">{fs.notes ? (Array.isArray(fs.notes) ? fs.notes.join(' ') : fs.notes) : ''}{fs.mean_gsd_cm ? ` Mean GSD ≈ ${fmtNum(fs.mean_gsd_cm, 1)} cm/px.` : ''}{fs.estimated_forward_overlap ? ` Est. forward overlap ${fmtNum(fs.estimated_forward_overlap * 100, 0)}%.` : ''}</div>}
            {(flights.data?.groups ?? []).length > 0 && <div className="mt-2"><Table head={['Flight', 'Images', 'No GPS', 'Poor overlap', 'Start', 'Alt (m)']} dense rows={flights.data.groups.map((g: any) => [g.label, g.images, g.images_without_gps, g.poor_overlap, fmtTime(g.start), fmtNum(g.mean_relative_altitude_m, 0)])} /></div>}
          </Section></div>
          <div className="rounded border border-line bg-white"><ControlPanel /></div>
          <div className="rounded border border-line bg-white"><Section title="Jobs, controls and logs"><JobList limit={25} /></Section></div>
        </div>
        <div className="space-y-3">
          <div className="rounded border border-line bg-white"><CrsPanel /></div>
          <div className="rounded border border-line bg-white"><OdmPanel /></div>
          <div className="rounded border border-line bg-white"><Section title="Products">
            {ortho.data?.warnings?.map((w) => <Notice key={w} kind="estimate">{w}</Notice>)}
            {(prods.data ?? []).filter((x) => x.exists).length === 0 && <Empty title="No products yet">Run photogrammetry, or import an orthomosaic / DEM / LiDAR on the Upload tab.</Empty>}
            <Table head={['Product', 'Kind', 'Size', 'Created']} dense rows={(prods.data ?? []).filter((x) => x.exists).map((x) => [<span title={x.surface_text ?? ''}>{x.name}</span>, x.surface_kind ? <Chip kind={x.surface_kind}>{x.surface_kind}</Chip> : x.kind, fmtBytes(x.size), fmtTime(x.created_at)])} />
          </Section></div>
          <div className="rounded border border-line bg-white"><Section title="Storage">
            {storage.data && <>
              <Kv k="This project" v={fmtBytes(pstore ?? p.storage_bytes)} /><Kv k="Disk free" v={`${fmtBytes(storage.data.disk_free)} of ${fmtBytes(storage.data.disk_total)}`} />
              <Progress value={1 - storage.data.disk_free / storage.data.disk_total} tone={storage.data.low_space ? 'warn' : 'accent'} />
              {storage.data.low_space && <Notice kind="warn">Free disk space is low. New uploads and processing may be refused until space is freed.</Notice>}
              <Kv k="Pending upload data" v={fmtBytes(storage.data.pending_upload_bytes)} />
              <Kv k="Image count limit" v="none" /><Kv k="Per-file limit" v={storage.data.limits.max_file_bytes ? fmtBytes(storage.data.limits.max_file_bytes) : 'none'} />
            </>}
          </Section></div>
        </div>
      </div>
    </div>
  );
}
