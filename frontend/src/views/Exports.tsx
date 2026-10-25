import { useEffect, useState } from 'react';
import { api, url } from '../api';
import { Check, Chip, Empty, Field, Kv, Notice, Progress, Section } from '../ui';
import { bump, toast, useApp, useProject } from '../store';
import { errMsg, fmtBytes, fmtTime, usePoll } from '../util';

const INFO: Record<string, { title: string; desc: string; kind?: 'observed' | 'estimated' | 'evidence' }> = {
  originals: { title: 'Original images', desc: 'Byte-identical zip of every uploaded photo. Can be many GB; off by default.' },
  orthophoto: { title: 'Orthomosaic (GeoTIFF)', desc: 'RGB imagery in the project CRS. Visual only.' },
  classification: { title: 'Classification (GeoTIFF + provenance + class table)', desc: 'Visual land-cover class ids; provenance raster says manual vs automatic. Not elevation evidence.' },
  class_stats: { title: 'Class statistics (CSV)', desc: 'Pixels and area per class.' },
  class_polygons: { title: 'Land-cover polygons (GeoJSON)', desc: 'Vectorised classes in the project CRS.' },
  dsm: { title: 'DSM (GeoTIFF): observed surface', desc: 'Canopy and structure tops as seen by the photogrammetry.', kind: 'observed' },
  dtm: { title: 'DTM (GeoTIFF): estimated bare earth', desc: 'Inferred ground. Interpolated where ground was not seen.', kind: 'estimated' },
  confidence: { title: 'Terrain confidence (GeoTIFF)', desc: 'High / moderate / low / unknown, by distance to measured ground.', kind: 'evidence' },
  evidence: { title: 'Elevation evidence (GeoTIFF)', desc: 'Per cell: measured, interpolated, or external model.', kind: 'evidence' },
  pointcloud: { title: 'Point cloud (LAS)', desc: 'Photogrammetric points, as produced.', kind: 'observed' },
  classified_cloud: { title: 'Ground-classified point cloud (LAS)', desc: 'Ground / vegetation classes from the ground filter.' },
  edited_cloud: { title: 'Edited point cloud (LAS, display SAMPLE)', desc: 'The thinned 3D-viewer sample with your manual class edits. Not the full cloud; does not change the DTM.', kind: 'observed' },
  image_classes: { title: 'Photo classification statistics (CSV)', desc: 'Land-cover fractions per classified drone photo. Per photo pixels; not georeferenced.' },
  features: { title: 'Boundary, footprints, lines (GeoJSON)', desc: 'Project CRS and WGS84 copies.' },
  profiles: { title: 'Elevation profiles (CSV)', desc: 'Saved profile lines: estimated ground vs observed surface, with confidence.' },
  footprints: { title: 'Footprint terrain reports (CSV/JSON)', desc: 'Elevations, slope and confidence for saved footprints.' },
  cad_dxf: { title: 'CAD drawing (DXF)', desc: 'Boundary, footprints, profile lines, control points and estimated-ground contours on separate layers. Preliminary.' },
  report: { title: 'Site report (PDF)', desc: 'Summary, CRS and units, layer meanings, confidence, limitations.' },
};

export function Exports() {
  const p = useProject()!;
  const jt = useApp((s) => s.jobsTick);
  const cat = usePoll(() => api.exportCatalog(p.id), 0, [p.id, jt]);
  const list = usePoll(() => api.exports(p.id), 0, [p.id]);
  const [sel, setSel] = useState<Record<string, boolean> | null>(null);
  const [laz, setLaz] = useState(false);
  const [label, setLabel] = useState('');
  const [active, setActive] = useState<{ job: string; eid: string } | null>(null);
  const job = usePoll(() => (active ? api.job(active.job) : Promise.resolve(null)), 1200, [active?.job]);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => { if (cat.data && !sel) setSel(Object.fromEntries(cat.data.items.map((i: any) => [i.item, !!i.default && i.available]))); }, [cat.data]);
  const st = job.data?.status;
  useEffect(() => {
    if (st === 'succeeded' && active) { toast('ok', 'Export ready'); setOpen(active.eid); setActive(null); list.reload(); bump('jobsTick'); }
    if (st === 'failed' || st === 'cancelled') { list.reload(); }
    /* eslint-disable-next-line */
  }, [st]);

  if (cat.loading) return <div className="m-auto text-slate-500">Loading…</div>;
  if (!cat.data || !sel) return <div className="m-auto text-warn">{cat.error ?? 'Export catalog unavailable'}</div>;
  const items: any[] = cat.data.items;
  const chosen = items.filter((i) => sel[i.item] && i.available).map((i) => i.item);
  const anyAvail = items.some((i) => i.available);
  if (!anyAvail) return <Empty title="Nothing to export yet">Exports are built from what the project has produced: imagery, classification, terrain and features. Process some data first.</Empty>;

  const start = async () => {
    try {
      const r = await api.exportCreate(p.id, { items: chosen, laz, label: label.trim() || null });
      setActive({ job: r.job_id, eid: r.export_id }); bump('jobsTick'); toast('info', 'Export queued');
    } catch (e) { toast('error', errMsg(e)); }
  };
  const busy = !!active && (st === 'queued' || st === 'running' || st === undefined);
  const dl = (eid: string, name: string) => url(`/projects/${p.id}/exports/${eid}/files/${encodeURI(name)}`);

  return (
    <div className="min-h-0 flex-1 overflow-auto p-3">
      <div className="mx-auto grid max-w-[1100px] grid-cols-1 gap-3 lg:grid-cols-2">
        <div className="rounded border border-line bg-white">
          <Section title="Create an export">
            <Notice kind="estimate"><b>Preliminary planning estimate.</b> {cat.data.disclaimer} Every file records its CRS and units; elevation files say whether they are <i>observed</i> or <i>estimated</i>.</Notice>
            <div className="mb-2 flex gap-1">
              <button className="btn" onClick={() => setSel(Object.fromEntries(items.map((i) => [i.item, !!i.default && i.available])))}>Defaults</button>
              <button className="btn" onClick={() => setSel(Object.fromEntries(items.map((i) => [i.item, i.available])))}>Everything available</button>
              <button className="btn" onClick={() => setSel(Object.fromEntries(items.map((i) => [i.item, false])))}>None</button>
            </div>
            {items.map((i) => { const inf = INFO[i.item] ?? { title: i.item, desc: '' };
              return (
                <div key={i.item} className={`mb-1 rounded border px-2 py-1 ${i.available ? 'border-line' : 'border-dashed border-slate-300 bg-slate-50 text-slate-400'}`}>
                  <Check label={<span className="flex flex-wrap items-center gap-1.5"><b className="font-medium">{inf.title}</b>{inf.kind === 'observed' && <Chip kind="observed">OBSERVED</Chip>}{inf.kind === 'estimated' && <Chip kind="estimated">ESTIMATED</Chip>}</span>}
                    checked={!!sel[i.item] && i.available} onChange={(v) => i.available && setSel({ ...sel, [i.item]: v })} />
                  <div className="ml-5 text-[11px] text-slate-500">{i.available ? inf.desc : `Not available: ${i.reason}`}{i.item === 'originals' && i.available && p.image_bytes ? ` (${fmtBytes(p.image_bytes)})` : ''}</div>
                </div>); })}
            <Check label="Compress point clouds to LAZ (needs PDAL)" checked={laz} onChange={setLaz} />
            <Field label="Label (optional)"><input className="input w-full" value={label} maxLength={120} onChange={(e) => setLabel(e.target.value)} placeholder="e.g. Pre-grading survey" /></Field>
            <button className="btn btn-primary w-full" disabled={!chosen.length || busy} onClick={start}>{busy ? 'Exporting…' : `Export ${chosen.length} item${chosen.length === 1 ? '' : 's'}`}</button>
            {active && job.data && <div className="mt-2"><div className="flex justify-between text-[11px]"><span className="truncate">{job.data.message ?? job.data.status}</span><Chip kind={job.data.status}>{job.data.status}</Chip></div><Progress value={job.data.progress} />
              {job.data.status === 'failed' && <Notice kind="warn">{job.data.error}</Notice>}</div>}
            {st === 'failed' && !active && null}
          </Section>
        </div>

        <div className="rounded border border-line bg-white">
          <Section title={`Previous exports (${list.data?.length ?? 0})`}>
            {(list.data ?? []).length === 0 && <div className="text-[12px] text-slate-500">None yet.</div>}
            {(list.data ?? []).map((e: any) => (
              <div key={e.id} className="mb-2 rounded border border-line">
                <div className="flex items-center justify-between gap-2 bg-panel px-2 py-1">
                  <button className="min-w-0 flex-1 text-left" onClick={() => setOpen(open === e.id ? null : e.id)}><div className="truncate font-medium">{e.label || 'Export'} <span className="font-normal text-slate-500">{fmtTime(e.created_at)}</span></div>
                    <div className="text-[11px] text-slate-500">{e.files} files · {fmtBytes(e.bytes)}{e.skipped?.length ? ` · ${e.skipped.length} skipped` : ''}</div></button>
                  <button className="btn !py-0" title="Delete this export from the server" onClick={async () => { if (!window.confirm('Delete this export and its files?')) return; try { await api.exportDelete(p.id, e.id); list.reload(); } catch (er) { toast('error', errMsg(er)); } }}>Delete</button>
                </div>
                {open === e.id && <ExportDetail pid={p.id} eid={e.id} dl={dl} />}
              </div>))}
          </Section>
        </div>
      </div>
    </div>
  );
}

function ExportDetail({ pid, eid, dl }: { pid: string; eid: string; dl: (eid: string, name: string) => string }) {
  const d = usePoll(() => api.exportGet(pid, eid), 0, [pid, eid]);
  if (!d.data) return <div className="p-2 text-[12px] text-slate-500">{d.error ?? 'Loading…'}</div>;
  const files: any[] = d.data.files;
  const bundle = files.find((f) => f.item === 'bundle');
  return (
    <div className="p-2">
      {bundle && <a className="btn btn-primary mb-2 inline-block" href={dl(eid, bundle.name)} download>Download everything (.zip, {fmtBytes(bundle.bytes)})</a>}
      <table className="w-full text-[11.5px]"><tbody>
        {files.filter((f) => f.item !== 'bundle').map((f) => (
          <tr key={f.name} className="border-t border-slate-100 align-top"><td className="py-1 pr-2"><a className="text-accent-dark underline" href={dl(eid, f.name)} download>{f.name}</a>
            <div className="text-[10.5px] text-slate-500">{f.description}{f.epsg ? ` · EPSG:${f.epsg}` : ''}{f.georeferenced === false ? ' · NOT georeferenced' : ''}</div></td>
            <td className="whitespace-nowrap py-1 text-right font-mono text-[10.5px] text-slate-500">{fmtBytes(f.bytes)}</td></tr>))}
      </tbody></table>
      {(d.data.skipped ?? []).length > 0 && <div className="mt-2"><div className="text-[11px] font-semibold text-slate-500">Not produced</div>{d.data.skipped.map((s: any) => <Kv key={s.item} k={s.item} v={s.reason} />)}</div>}
      <div className="mt-1 text-[10.5px] text-slate-500">Each file's SHA-256 is in metadata.json inside the zip. {d.data.disclaimer}</div>
    </div>
  );
}
