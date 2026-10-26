import { useEffect, useState } from 'react';
import { api, Project } from './api';
import { Chip, Check, Field, Modal, Section, Slider } from './ui';
import { JobList } from './jobs';
import { bump, selectProject, setApp, setLayer, setOpacity, toast, useApp, LayerState, View } from './store';
import { errMsg, fmtBytes, usePoll } from './util';

function NewProject({ onClose }: { onClose: () => void }) {
  const [f, setF] = useState({ name: '', address: '', epsg: '', units: 'metre', system: 'imperial' });
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true);
    try {
      const p = await api.createProject({ name: f.name, address: f.address || null, epsg: f.epsg ? Number(f.epsg) : null, horizontal_units: f.units, vertical_units: f.units, unit_system: f.system });
      bump('projectsTick'); selectProject(p.id); setApp({ view: 'upload' }); onClose();
    } catch (e) { toast('error', errMsg(e)); } finally { setBusy(false); }
  };
  return (
    <Modal title="New project" onClose={onClose}>
      <Field label="Project name"><input className="input w-full" autoFocus value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder="e.g. Lot 7, Maple Ridge" /></Field>
      <Field label="Address (optional)"><input className="input w-full" value={f.address} onChange={(e) => setF({ ...f, address: e.target.value })} /></Field>
      <div className="grid grid-cols-2 gap-2">
        <Field label="Horizontal CRS (EPSG)" hint="Leave blank to take it from the orthomosaic or image GPS. Example: 32616 = WGS84 / UTM 16N."><input className="input w-full" inputMode="numeric" value={f.epsg} onChange={(e) => setF({ ...f, epsg: e.target.value.replace(/\D/g, '') })} /></Field>
        <Field label="Linear units"><select className="input w-full" value={f.units} onChange={(e) => setF({ ...f, units: e.target.value })}><option value="metre">metre</option><option value="foot">international foot</option><option value="us_survey_foot">US survey foot</option></select></Field>
      </div>
      <div className="mt-2 flex justify-end gap-2"><button className="btn" onClick={onClose}>Cancel</button><button className="btn btn-primary" disabled={!f.name.trim() || busy} onClick={submit}>Create</button></div>
    </Modal>
  );
}

function Projects() {
  const pid = useApp((s) => s.projectId);
  const tick = useApp((s) => s.projectsTick);
  const { data, error } = usePoll(() => api.projects(), 6000, [tick]);
  const [open, setOpen] = useState(false);
  const sample = async () => { try { const p = await api.sample(); bump('projectsTick'); selectProject(p.id); setApp({ view: 'dashboard' }); toast('ok', 'Synthetic sample project created (not real survey data).'); } catch (e) { toast('error', errMsg(e)); } };
  return (
    <Section title="Projects" actions={<><button className="btn !py-0" onClick={() => setOpen(true)}>+ New</button></>}>
      {error && <div className="text-warn">{error}</div>}
      {data && data.length === 0 && <div className="text-slate-600">No projects yet. Create one, or add the synthetic sample to explore the tools.</div>}
      <div className="max-h-48 overflow-auto">
        {(data ?? []).map((p: Project) => (
          <button key={p.id} onClick={() => selectProject(p.id)} className={`mb-1 block w-full rounded border px-2 py-1 text-left ${p.id === pid ? 'border-accent bg-sky-50' : 'border-line bg-white hover:bg-slate-50'}`}>
            <div className="flex items-center justify-between gap-1"><span className="truncate font-medium">{p.name}</span>{p.is_sample && <Chip kind="warn" title="Synthetic data">sample</Chip>}</div>
            <div className="text-[11px] text-slate-500">{p.image_count} images · {fmtBytes(p.image_bytes)}{p.jobs.running ? ` · ${p.jobs.running} running` : ''}{p.jobs.failed ? ` · ${p.jobs.failed} failed` : ''}</div>
          </button>
        ))}
      </div>
      <button className="btn mt-1 w-full justify-center" onClick={sample} title="Development mode: generates a synthetic wooded lot with a known terrain">Add synthetic sample project</button>
      {open && <NewProject onClose={() => setOpen(false)} />}
    </Section>
  );
}

function BatchesFlights() {
  const pid = useApp((s) => s.projectId);
  const tick = useApp((s) => s.jobsTick);
  const b = usePoll(() => (pid ? api.batches(pid) : Promise.resolve([])), 5000, [pid, tick]);
  const f = usePoll(() => (pid ? api.flights(pid) : Promise.resolve(null)), 8000, [pid, tick]);
  if (!pid) return null;
  return (
    <Section title="Batches and flights" open={false}>
      {(b.data ?? []).length === 0 && <div className="text-slate-500">No upload batches yet.</div>}
      {(b.data ?? []).map((x: any) => <div key={x.id} className="flex justify-between py-0.5 text-[12px]"><span className="truncate">{x.label}</span><span className="text-slate-500">{x.images} img{x.problems ? <span className="text-warn"> · {x.problems} problem</span> : null}</span></div>)}
      <div className="mt-2 border-t border-slate-100 pt-1 text-[11px] font-semibold text-slate-500">Flights (grouped by capture time gaps)</div>
      {(f.data?.groups ?? []).map((g: any) => <div key={g.group} className="flex justify-between py-0.5 text-[12px]"><span>{g.label}</span><span className="text-slate-500">{g.images} img{g.poor_overlap ? <span className="text-warn"> · {g.poor_overlap} poor overlap</span> : null}</span></div>)}
    </Section>
  );
}

const LAYER_DEFS: { key: keyof LayerState; label: string; views: View[]; opacity?: string }[] = [
  { key: 'ortho', label: 'Orthomosaic (imagery)', views: ['terrain', 'planning'] },
  { key: 'classes', label: 'Land-cover classes (visual)', views: ['terrain', 'planning'], opacity: 'classes' },
  { key: 'dsm', label: 'Observed surface (DSM)', views: ['terrain', 'planning'], opacity: 'dsm' },
  { key: 'dtm', label: 'Estimated bare earth (DTM)', views: ['terrain', 'planning'], opacity: 'dtm' },
  { key: 'ndsm', label: 'Object height (DSM − DTM)', views: ['terrain', 'planning'], opacity: 'ndsm' },
  { key: 'confidence', label: 'Terrain confidence', views: ['terrain', 'planning'], opacity: 'confidence' },
  { key: 'evidence', label: 'Elevation evidence', views: ['terrain', 'planning'], opacity: 'evidence' },
  { key: 'hillshade', label: 'Hillshade on elevation', views: ['terrain', 'planning'] },
  { key: 'steep', label: 'Steep slopes', views: ['terrain', 'planning'] },
  { key: 'drainage', label: 'Drainage paths', views: ['terrain', 'planning'] },
  { key: 'boundary', label: 'Project boundary', views: ['terrain', 'planning', 'dashboard'] },
  { key: 'features', label: 'Footprints, lines, notes', views: ['terrain', 'planning'] },
  { key: 'control', label: 'GCP / check points', views: ['terrain', 'planning', 'dashboard'] },
  { key: 'footprints', label: 'Image footprints', views: ['dashboard'] },
  { key: 'cameras', label: 'Camera positions', views: ['dashboard'] },
];

function Layers() {
  const view = useApp((s) => s.view);
  const layers = useApp((s) => s.layers);
  const opacity = useApp((s) => s.opacity);
  const defs = LAYER_DEFS.filter((d) => d.views.includes(view));
  if (!defs.length) return <Section title="Layers"><div className="text-slate-500">This view has no layers.</div></Section>;
  return (
    <Section title="Layers">
      {defs.map((d) => (
        <div key={d.key}>
          <Check label={d.label} checked={layers[d.key]} onChange={(v) => setLayer(d.key, v)} />
          {d.opacity && layers[d.key] && <div className="ml-5"><Slider label="Opacity" value={opacity[d.opacity]} min={0.05} max={1} step={0.05} onChange={(v) => setOpacity(d.opacity!, v)} fmt={(v) => `${Math.round(v * 100)}%`} /></div>}
        </div>
      ))}
    </Section>
  );
}

function Queue() {
  const { data } = usePoll(() => api.queue(), 4000, []);
  if (!data) return null;
  return (
    <div className="flex items-center justify-between border-b border-line bg-white px-2 py-1 text-[11px] text-slate-600">
      <span>Queue: {data.by_status?.running ?? 0} running · {data.by_status?.queued ?? 0} queued{data.paused ? ' · PAUSED' : ''}</span>
      <button className="btn !px-1.5 !py-0" onClick={async () => { try { data.paused ? await api.resumeQueue() : await api.pauseQueue(); } catch (e) { toast('error', errMsg(e)); } }}>{data.paused ? 'Resume' : 'Pause'}</button>
    </div>
  );
}

export function LeftSidebar() {
  const open = useApp((s) => s.leftOpen);
  useEffect(() => { /* keep hook order stable */ }, []);
  if (!open) return null;
  return (
    <aside className="flex w-[264px] shrink-0 flex-col overflow-y-auto border-r border-line bg-white">
      <Projects />
      <BatchesFlights />
      <Layers />
      <Section title="Jobs"><Queue /><JobList limit={10} compact /></Section>
    </aside>
  );
}
