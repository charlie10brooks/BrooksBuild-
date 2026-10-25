import { useEffect, useMemo, useState } from 'react';
import { CClass, Project, api, url } from '../api';
import { Chip, Field, Kv, Modal, Notice, Progress, Section, Slider } from '../ui';
import { bump, toast, useApp } from '../store';
import { errMsg, usePoll } from '../util';

const DEFAULT_COLORS: Record<string, string> = { '1': '#1b5e20', '2': '#7cb342', '3': '#d4e157', '4': '#8d6e63', '5': '#b0a99f', '6': '#d32f2f', '7': '#455a64', '8': '#1e88e5', '9': '#ab47bc' };
const useClasses = (p: Project): CClass[] => useMemo(() => p.classes ?? [], [p.classes]);
const pct = (v: number) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;

/** Option to classify the uploaded drone photos (per-photo land cover). Shown in the Upload view. */
export function PhotoClassify({ p, justUploaded }: { p: Project; justUploaded: boolean }) {
  const jt = useApp((s) => s.jobsTick);
  const st = usePoll(() => api.imgClassState(p.id), 0, [p.id, jt, p.image_count]);
  const batches = usePoll(() => api.batches(p.id), 0, [p.id, p.image_count]);
  const classes = useClasses(p);
  const [scope, setScope] = useState<'all' | 'batch'>('all');
  const [batch, setBatch] = useState('');
  const [model, setModel] = useState('heuristic');
  const [redo, setRedo] = useState(false);
  const [job, setJob] = useState<string | null>(null);
  const jq = usePoll(() => (job ? api.job(job) : Promise.resolve(null)), 1200, [job]);
  const status = jq.data?.status;
  useEffect(() => {
    if (status === 'succeeded' || status === 'failed' || status === 'cancelled') {
      if (status === 'succeeded') toast('ok', 'Photos classified'); else if (status === 'failed') toast('error', jq.data?.error ?? 'Photo classification failed');
      setJob(null); st.reload(); bump('jobsTick');
    }
    /* eslint-disable-next-line */
  }, [status]);
  useEffect(() => { if (!batch && batches.data?.length) setBatch(batches.data[batches.data.length - 1].id); }, [batches.data]);
  if (!st.data) return null;
  const s = st.data, models: any[] = s.models ?? [];
  const todo = s.images - s.classified;
  const busy = !!job && status !== 'succeeded';
  const start = async () => {
    try {
      const r = await api.imgClassStart(p.id, { scope, batch_id: scope === 'batch' ? batch : null, model, overwrite: redo });
      setJob(r.job_id); bump('jobsTick'); toast('info', `Classifying ${r.photos} photo(s)`);
    } catch (e) { toast('error', errMsg(e)); }
  };
  if (s.images === 0) return (
    <div className="rounded border border-line bg-white"><Section title="Classify drone photos"><div className="text-[12px] text-slate-500">Upload drone images first. Once they are ingested you can classify them here.</div></Section></div>
  );
  const nameOf = (k: string) => classes.find((c) => String(c.id) === k)?.name ?? `class ${k}`;
  const colorOf = (k: string) => classes.find((c) => String(c.id) === k)?.color ?? DEFAULT_COLORS[k] ?? '#999';
  return (
    <div className={`rounded border bg-white ${justUploaded && todo > 0 ? 'border-accent ring-2 ring-sky-100' : 'border-line'}`}>
      <Section title="Classify drone photos" badge={<span className="ml-1 font-normal normal-case text-slate-400">{s.classified}/{s.images} done</span>}>
        {justUploaded && todo > 0 && <div className="mb-2 rounded bg-sky-50 px-2 py-1 text-[12px] text-sky-900">Your photos are in. Classify them to see land cover (trees, brush, grass, soil, rock, road, water) in every photo.</div>}
        <div className="mb-2 grid grid-cols-2 gap-2">
          <Field label="Which photos"><select className="input w-full" value={scope} onChange={(e) => setScope(e.target.value as any)}><option value="all">All photos</option><option value="batch">One batch</option></select></Field>
          {scope === 'batch' ? <Field label="Batch"><select className="input w-full" value={batch} onChange={(e) => setBatch(e.target.value)}>{(batches.data ?? []).map((b: any) => <option key={b.id} value={b.id}>{b.label} ({b.images})</option>)}</select></Field> : <div />}
        </div>
        <Field label="Method">
          <select className="input w-full" value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => <option key={m.id} value={m.id} disabled={!m.available}>{m.name}{m.available ? '' : ' (not available)'}</option>)}
          </select>
        </Field>
        <div className="mb-1 text-[11px] text-slate-500">{models.find((m) => m.id === model)?.description}</div>
        <label className="mb-2 flex cursor-pointer items-center gap-1.5 text-[12px]"><input type="checkbox" checked={redo} onChange={(e) => setRedo(e.target.checked)} />Redo photos that already have labels</label>
        <button className="btn btn-primary w-full" disabled={busy || (todo === 0 && !redo)} onClick={start}>{busy ? 'Classifying…' : todo === 0 && !redo ? 'All photos are classified' : `Classify ${redo ? 'photos' : `${todo} photo${todo === 1 ? '' : 's'}`}`}</button>
        {job && jq.data && <div className="mt-2"><div className="flex justify-between text-[11px]"><span className="truncate">{jq.data.message ?? jq.data.status}</span><Chip kind={jq.data.status}>{jq.data.status}</Chip></div><Progress value={jq.data.progress} /></div>}
        {s.classified > 0 && <div className="mt-3">
          <div className="mb-1 text-[11px] font-semibold text-slate-500">LAND COVER ACROSS CLASSIFIED PHOTOS</div>
          <div className="flex h-3 overflow-hidden rounded border border-line">{Object.entries(s.class_fractions as Record<string, number>).filter(([k]) => k !== '0').map(([k, v]) => <div key={k} title={`${nameOf(k)} ${pct(v)}`} style={{ width: `${v * 100}%`, background: colorOf(k) }} />)}
            <div title={`Unclassified ${pct(s.class_fractions['0'] ?? 0)}`} style={{ flex: 1, background: '#e5e7eb' }} /></div>
          {Object.entries(s.class_fractions as Record<string, number>).map(([k, v]) => <div key={k} className="flex items-center gap-1.5 text-[11.5px]"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: k === '0' ? '#e5e7eb' : colorOf(k) }} /><span className="flex-1">{k === '0' ? 'Left unclassified (uncertain)' : nameOf(k)}</span><span className="font-mono">{pct(v)}</span></div>)}
          <Kv k="Method(s)" v={Object.entries(s.by_model).map(([m, n]) => `${m} × ${n}`).join(', ')} />
          <button className="btn mt-1" onClick={async () => { if (!window.confirm('Remove the labels of all classified photos? The photos themselves are untouched.')) return; try { await api.imgClassClear(p.id); st.reload(); } catch (e) { toast('error', errMsg(e)); } }}>Remove all photo labels</button>
        </div>}
        <div className="mt-2"><Notice kind="estimate">{s.disclaimer}</Notice></div>
      </Section>
    </div>
  );
}

/** Thumbnail grid with the class overlay; click for a larger view with opacity and class toggles. */
export function PhotoGallery({ p }: { p: Project }) {
  const jt = useApp((s) => s.jobsTick);
  const [state, setState] = useState<'classified' | 'all'>('classified');
  const [page, setPage] = useState(0);
  const PER = 24;
  const q = usePoll(() => api.imgClassItems(p.id, `state=${state}&limit=${PER}&offset=${page * PER}`), 0, [p.id, state, page, jt]);
  const classes = useClasses(p);
  const [open, setOpen] = useState<any | null>(null);
  const items: any[] = q.data?.items ?? [];
  const total = q.data?.total ?? 0;
  const top = (f: Record<string, number> | null) => !f ? '' : Object.entries(f).filter(([k]) => k !== '0').sort((a, b) => b[1] - a[1]).slice(0, 2).map(([k, v]) => `${classes.find((c) => String(c.id) === k)?.name ?? k} ${pct(v)}`).join(' · ');
  if (!q.data) return null;
  if (state === 'classified' && total === 0 && !q.loading) return null;
  return (
    <div className="rounded border border-line bg-white">
      <Section title="Photo classification" badge={<span className="ml-1 font-normal normal-case text-slate-400">{total} photos</span>}
        actions={<><button className={`btn !py-0 ${state === 'classified' ? 'btn-on' : ''}`} onClick={() => { setState('classified'); setPage(0); }}>Classified</button>
          <button className={`btn !py-0 ${state === 'all' ? 'btn-on' : ''}`} onClick={() => { setState('all'); setPage(0); }}>All</button></>}>
        <div className="gap-2" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))' }}>
          {items.map((it) => (
            <button key={it.id} className="overflow-hidden rounded border border-line bg-panel text-left hover:border-accent" onClick={() => it.classified && setOpen(it)} title={it.filename}>
              <div className="relative aspect-[4/3] bg-slate-200">
                <img src={url(`/projects/${p.id}/images/${it.id}/thumbnail`)} className="absolute inset-0 h-full w-full object-cover" loading="lazy" alt="" />
                {it.classified && <img src={url(`/projects/${p.id}/images/${it.id}/classification/overlay.png?opacity=0.55`)} className="absolute inset-0 h-full w-full object-cover" style={{ imageRendering: 'pixelated' }} loading="lazy" alt="" />}
              </div>
              <div className="px-1.5 py-1"><div className="truncate text-[11px] font-medium">{it.filename}</div>
                <div className="truncate text-[10.5px] text-slate-500">{it.classified ? top(it.fractions) || 'nothing recognised' : 'not classified'}</div></div>
            </button>))}
        </div>
        {total > PER && <div className="mt-2 flex items-center gap-2 text-[12px]"><button className="btn" disabled={page === 0} onClick={() => setPage(page - 1)}>Prev</button><span>{page * PER + 1}–{Math.min(total, (page + 1) * PER)} of {total}</span><button className="btn" disabled={(page + 1) * PER >= total} onClick={() => setPage(page + 1)}>Next</button></div>}
      </Section>
      {open && <PhotoModal p={p} item={open} classes={classes} onClose={() => setOpen(null)} />}
    </div>
  );
}

function PhotoModal({ p, item, classes, onClose }: { p: Project; item: any; classes: CClass[]; onClose: () => void }) {
  const [opacity, setOpacity] = useState(0.55);
  const [hidden, setHidden] = useState<number[]>([]);
  const [showOv, setShowOv] = useState(true);
  const d = usePoll(() => api.imgClassDetail(p.id, item.id), 0, [item.id]);
  const total = d.data ? Object.values(d.data.counts as Record<string, number>).reduce((a, b) => a + b, 0) : 0;
  return (
    <Modal title={item.filename} onClose={onClose} wide>
      <div className="gap-3" style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 240px' }}>
        <div className="relative self-start bg-slate-200" style={{ justifySelf: 'start' }}>
          <img src={url(`/projects/${p.id}/images/${item.id}/thumbnail`)} className="block" style={{ maxWidth: '100%', maxHeight: '66vh' }} alt="" />
          {showOv && <img src={url(`/projects/${p.id}/images/${item.id}/classification/overlay.png?opacity=${opacity}&hide=${hidden.join(',')}`)} className="absolute inset-0 h-full w-full" style={{ imageRendering: 'pixelated' }} alt="" />}
        </div>
        <div className="text-[12px]">
          <label className="mb-1 flex items-center gap-1.5"><input type="checkbox" checked={showOv} onChange={(e) => setShowOv(e.target.checked)} />Show classes</label>
          <Slider label="Overlay opacity" value={opacity} min={0.1} max={1} step={0.05} onChange={setOpacity} fmt={(v) => `${Math.round(v * 100)}%`} />
          {d.data && Object.entries(d.data.counts as Record<string, number>).sort((a, b) => b[1] - a[1]).map(([k, n]) => {
            const id = Number(k), c = classes.find((x) => x.id === id);
            return <label key={k} className="flex cursor-pointer items-center gap-1.5 py-0.5"><input type="checkbox" disabled={id === 0} checked={!hidden.includes(id)} onChange={(e) => setHidden(e.target.checked ? hidden.filter((h) => h !== id) : [...hidden, id])} />
              <span className="inline-block h-3 w-3 rounded-sm border border-slate-400" style={{ background: id === 0 ? 'transparent' : c?.color }} /><span className="flex-1">{id === 0 ? 'Unclassified' : c?.name ?? `class ${k}`}</span><span className="font-mono text-[10.5px]">{pct(n / total)}</span></label>;
          })}
          {d.data && <div className="mt-2 text-[11px] text-slate-500">Method: {d.data.model} · mean confidence {d.data.mean_confidence != null ? pct(d.data.mean_confidence) : 'n/a'} · working size {d.data.width}×{d.data.height}px</div>}
          <div className="mt-2"><Notice kind="estimate">{d.data?.note ?? 'Labels are per photo; not georeferenced and not elevation evidence.'}</Notice></div>
        </div>
      </div>
    </Modal>
  );
}
