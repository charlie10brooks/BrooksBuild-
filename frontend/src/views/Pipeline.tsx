import { useEffect, useState } from 'react';
import { Project, api } from '../api';
import { Chip, Field, Kv, Notice, Progress, Section } from '../ui';
import { bump, setApp, toast, useApp } from '../store';
import { errMsg, usePoll } from '../util';

/** The whole path in one place: photos -> 3D model (OpenDroneMap) -> labels on the model -> terrain. One button. */
const STEPS: { kinds: string[]; title: string; hint: string }[] = [
  { kinds: ['odm'], title: 'Build the 3D model from your photos', hint: 'OpenDroneMap matches the photos and builds the 3D model. This is the long step.' },
  { kinds: ['mesh_prep'], title: 'Get the model ready to view', hint: 'Prepares the textured model for the browser.' },
  { kinds: ['terrain'], title: 'Work out the ground', hint: 'Finds the bare earth under the trees for the Terrain pages.' },
  { kinds: ['classify_mesh'], title: 'Label what is in the model', hint: 'Trees, grass, buildings and so on, from the model\u2019s own photos.' },
];

export function Pipeline({ p }: { p: Project }) {
  const jt = useApp((s) => s.jobsTick);
  const st = usePoll(() => api.odmState(p.id), 0, [p.id, jt, p.image_count]);
  const jobs = usePoll(() => api.jobs(p.id, 'limit=40'), 3000, [p.id, jt]);
  const mesh = usePoll(() => api.meshState(p.id), 0, [p.id, jt]);
  const lab = usePoll(() => api.meshLabels(p.id), 0, [p.id, jt]);
  const [preset, setPreset] = useState<'balanced' | 'high'>('balanced');
  const [ungeo, setUngeo] = useState(false);
  const [busy, setBusy] = useState(false);
  const [again, setAgain] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  useEffect(() => { const t = setInterval(() => bump('jobsTick'), 8000); return () => clearInterval(t); }, []);
  const s = st.data;
  const latest = (kind: string) => (jobs.data ?? []).filter((j) => j.kind === kind).sort((a, b) => b.created_at - a.created_at)[0];
  const active = (jobs.data ?? []).some((j) => ['odm', 'mesh_prep', 'terrain', 'classify_mesh'].includes(j.kind) && (j.status === 'queued' || j.status === 'running'));
  const modelReady = !!mesh.data?.available, labelled = !!lab.data?.has_labels;
  const odm = latest('odm');

  if (p.image_count === 0 && !modelReady) return (
    <div className="col-span-full rounded border border-accent bg-white ring-2 ring-sky-100">
      <Section title="Start here">
        <div className="text-[13px] text-slate-700">Add your drone photos and BrooksBuild turns them into a 3D model you can look around, label and plan on.</div>
        <ol className="my-2 list-decimal space-y-0.5 pl-5 text-[13px] text-slate-700">
          <li><b>Add photos</b>: JPEGs straight from the drone (they need GPS in them).</li><li><b>Process</b>: one button builds the 3D model and labels it.</li><li><b>Check and fix</b> the labels, then use the terrain and planning tools.</li>
        </ol>
        <button className="btn btn-primary" onClick={() => setApp({ view: 'upload' })}>Add photos</button>
      </Section>
    </div>);
  if (!s) return null;
  const nPhotos = s.images;

  const blocked: string[] = [...(s.blocking ?? [])].filter((m: string) => !(ungeo && m.startsWith('No image has GPS')));
  if (!s.docker.available) blocked.push('The 3D model builder (OpenDroneMap) runs in Docker, and Docker is not available to this server. ' + (s.docker.hint ?? ''));
  const go = async () => {
    setBusy(true);
    try {
      const r: any = await api.odmStart(p.id, { preset, allow_ungeoreferenced: ungeo, build_terrain: true, auto_label: true });
      setNote(r.note ?? null); bump('jobsTick'); toast('info', 'Started. You can leave this page; it keeps going.');
    } catch (e) { toast('error', errMsg(e)); } finally { setBusy(false); }
  };
  const relabel = async () => { try { await api.meshLabelsAuto(p.id); bump('jobsTick'); toast('info', 'Labelling the model'); } catch (e) { toast('error', errMsg(e)); } };
  const next = active ? 'Working on it. You can leave this page; the steps below update on their own.'
    : !modelReady ? (odm?.status === 'failed' ? 'The last run failed. Read the message below, fix it and run again.' : 'Ready to process your photos.')
    : !labelled ? 'Your 3D model is ready. Label it automatically, or open Label to do it by hand.' : 'Done. Look around the model below, then check and fix the labels.';
  return (
    <div className={`col-span-full rounded border bg-white ${!modelReady || active ? 'border-accent ring-2 ring-sky-100' : 'border-line'}`}>
      <Section title="Photos to 3D model" badge={modelReady && !active ? <Chip kind="ok">ready</Chip> : active ? <Chip kind="running">working</Chip> : undefined}>
        <div className="mb-2 text-[13px] font-medium text-slate-800">{next}</div>
        <div className="mb-2 grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))' }}>
          {STEPS.map((g, i) => { const j = latest(g.kinds[0]); const done = j?.status === 'succeeded'; return (
            <div key={g.kinds[0]} className={`rounded border p-2 ${j?.status === 'running' ? 'border-accent bg-sky-50' : 'border-line'}`}>
              <div className="flex items-center justify-between gap-1 text-[12px]"><b>{i + 1}. {g.title}</b>{j ? <Chip kind={j.status}>{done ? 'done' : j.status}</Chip> : <Chip kind="queued">waiting</Chip>}</div>
              <div className="mt-0.5 text-[11px] text-slate-500">{g.hint}</div>
              {j && (j.status === 'running' || j.status === 'queued') && <><Progress value={j.progress} /><div className="truncate text-[11px] text-slate-500">{j.message}</div></>}
              {j?.status === 'failed' && <div className="mt-1 text-[11px] text-warn">{j.error ?? 'failed'}</div>}
            </div>); })}
        </div>
        {!active && (!modelReady || again) && <>
          <div className="mb-1 flex flex-wrap items-end gap-3">
            <Field label="Quality"><select className="input" value={preset} onChange={(e) => setPreset(e.target.value as any)}>
              <option value="balanced">Standard (recommended)</option><option value="high">Detailed (slower, needs a lot of memory)</option></select></Field>
            <div className="mb-1.5 text-[12px] text-slate-500"><Kv k="Photos" v={`${s.images} (${s.images_with_gps} with GPS)`} /></div>
          </div>
          {(s.blocking ?? []).some((m: string) => m.startsWith('No image has GPS')) && <label className="mb-1 flex cursor-pointer items-center gap-1.5 text-[12px]"><input type="checkbox" checked={ungeo} onChange={(e) => setUngeo(e.target.checked)} />My photos have no GPS. Continue anyway (the model will not be placed on the map).</label>}
          {blocked.map((m, i) => <Notice key={i} kind="warn">{m}</Notice>)}
          <div className="flex flex-wrap gap-2">
            <button className="btn btn-primary" disabled={busy || blocked.length > 0} onClick={go}>{busy ? 'Starting…' : modelReady ? 'Process again' : 'Process photos into a 3D model'}</button>
            {modelReady && <button className="btn" onClick={() => setAgain(false)}>Cancel</button>}
          </div>
          <div className="mt-1 text-[11px] text-slate-500">Standard is the right first try. It can take from minutes to hours depending on the number of photos and your computer.</div>
        </>}
        {!active && modelReady && !again && <div className="flex flex-wrap gap-2">
          <button className="btn btn-primary" onClick={() => setApp({ view: 'classify' })}>Check and fix labels</button>
          <button className="btn" onClick={relabel}>{labelled ? 'Re-run automatic labelling' : 'Label the model automatically'}</button>
          <button className="btn" onClick={() => setApp({ view: 'view3d' })}>Terrain 3D</button>
          <button className="btn" onClick={() => setAgain(true)} title="Run the photogrammetry again, for example with different quality">Process photos again…</button>
        </div>}
        {note && <div className="mt-1"><Notice kind="estimate">{note}</Notice></div>}
        {labelled && <div className="mt-1 text-[11px] text-slate-500">Labels are automatic guesses from colour and height. Check them in <b>Label</b> before relying on them.</div>}
      </Section>
    </div>
  );
}
