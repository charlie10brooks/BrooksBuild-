import { useEffect, useState } from 'react';
import { Project, api } from '../api';
import { Chip, Field, Kv, Notice, Progress, Section } from '../ui';
import { bump, setApp, toast, useApp } from '../store';
import { errMsg, usePoll } from '../util';

const STAGES: { kind: string; label: string }[] = [
  { kind: 'odm', label: '1. Photogrammetry (OpenDroneMap): point cloud, surface model, orthomosaic' },
  { kind: 'terrain', label: '2. Terrain reconstruction: observed surface, estimated bare earth, confidence' },
];

/** One-click path from uploaded (and optionally classified) photos to an editable 3D terrain model. */
export function Build3D({ p }: { p: Project }) {
  const jt = useApp((s) => s.jobsTick);
  const st = usePoll(() => api.odmState(p.id), 0, [p.id, jt, p.image_count]);
  const jobs = usePoll(() => api.jobs(p.id, 'limit=20'), 3000, [p.id, jt]);
  const cls = usePoll(() => api.imgClassState(p.id), 0, [p.id, jt, p.image_count]);
  const classified: number = cls.data?.classified ?? 0;
  const terr = usePoll(() => api.terrain(p.id), 0, [p.id, jt]);
  const [preset, setPreset] = useState<'draft' | 'balanced' | 'high'>('draft');
  const [ungeo, setUngeo] = useState(false);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  useEffect(() => { const t = setInterval(() => bump('jobsTick'), 10000); return () => clearInterval(t); }, []);
  if (!st.data || p.image_count === 0) return null;
  const s = st.data;
  const latest = (kind: string) => (jobs.data ?? []).filter((j) => j.kind === kind).sort((a, b) => b.created_at - a.created_at)[0];
  const odm = latest('odm'), terrain = latest('terrain');
  const running = [odm, terrain].some((j) => j && (j.status === 'queued' || j.status === 'running'));
  const modelReady = !!terr.data?.available && !!terr.data?.layers?.dtm;
  const blocked: string[] = [...(s.blocking ?? [])].filter((m: string) => !(ungeo && m.startsWith('No image has GPS')));
  if (!s.docker.available) blocked.push('Docker is not available to the server, and OpenDroneMap runs in Docker. ' + (s.docker.hint ?? ''));
  const go = async () => {
    setBusy(true);
    try {
      const r: any = await api.odmStart(p.id, { preset, allow_ungeoreferenced: ungeo, build_terrain: true });
      setNote(r.note ?? null); bump('jobsTick'); toast('info', 'Building the 3D terrain model: photogrammetry first, then terrain reconstruction');
    } catch (e) { toast('error', errMsg(e)); } finally { setBusy(false); }
  };
  return (
    <div className={`rounded border bg-white ${classified > 0 && !modelReady ? 'border-accent ring-2 ring-sky-100' : 'border-line'}`}>
      <Section title="Build the 3D terrain model" badge={modelReady ? <Chip kind="ok">ready</Chip> : undefined}>
        <div className="mb-2 text-[12px] text-slate-600">
          {classified > 0 ? <>Your photos are classified. </> : null}
          Turns these {s.images} photos into an orthomosaic, a point cloud and a terrain model you can open in <b>Terrain 3D</b>, where you can select
          trees or other features, <b>remove</b> them, and preview the surface without them.
        </div>
        {classified > 0 && <Notice kind="info">The per-photo classes above describe each photo on its own; they don't feed the 3D model. You classify the finished orthomosaic in the Classify tab.</Notice>}
        <Kv k="Photos" v={`${s.images} (${s.images_with_gps} with GPS)`} />
        {STAGES.map((g) => { const j = g.kind === 'odm' ? odm : terrain; return (
          <div key={g.kind} className="mt-1.5">
            <div className="flex items-center justify-between gap-2 text-[12px]"><span>{g.label}</span>{j ? <Chip kind={j.status}>{j.status}</Chip> : <Chip kind="queued">not started</Chip>}</div>
            {j && (j.status === 'running' || j.status === 'queued') && <><Progress value={j.progress} /><div className="truncate text-[11px] text-slate-500">{j.message}</div></>}
            {j?.status === 'failed' && <Notice kind="warn">{j.error ?? 'failed'} Open the job log in the Jobs panel for details.</Notice>}
          </div>); })}
        {!running && !modelReady && <>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <Field label="Quality"><select className="input w-full" value={preset} onChange={(e) => setPreset(e.target.value as any)}>
              <option value="draft">Draft (fast, try this first)</option><option value="balanced">Balanced</option><option value="high">High (slow, memory-hungry)</option></select></Field>
          </div>
          {preset === 'draft' ? <Notice kind="info">Draft is quick but skips the <b>photo-textured 3D model</b> (the orbitable, Polycam-style model). Choose Balanced or High to get it.</Notice>
            : <div className="mb-1 text-[11px] text-slate-500">Balanced and High also build the photo-textured 3D model, opened with <b>📷 Photo model</b> in Terrain 3D.</div>}
          {(s.blocking ?? []).some((m: string) => m.startsWith('No image has GPS')) && <label className="mb-1 flex cursor-pointer items-center gap-1.5 text-[12px]"><input type="checkbox" checked={ungeo} onChange={(e) => setUngeo(e.target.checked)} />Allow processing without GPS (result will not be georeferenced)</label>}
          {blocked.map((m, i) => <Notice key={i} kind="warn">{m}</Notice>)}
          <button className="btn btn-primary w-full" disabled={busy || blocked.length > 0} onClick={go}>{busy ? 'Starting…' : 'Build 3D terrain model'}</button>
        </>}
        {note && <div className="mt-1"><Notice kind="estimate">{note}</Notice></div>}
        {modelReady && <div className="mt-2 flex gap-2"><button className="btn btn-primary flex-1" onClick={() => setApp({ view: 'view3d' })}>Open the 3D model</button>
          <button className="btn" onClick={() => setApp({ view: 'classify' })}>Classify the orthomosaic</button></div>}
      </Section>
    </div>
  );
}
