import { useCallback, useEffect, useRef, useState } from 'react';
import { CClass, api } from '../api';
import { Chip, Kv, Notice, Section, Slider } from '../ui';
import { bump, toast, useApp } from '../store';
import { errMsg } from '../util';
import { MeshApi, MeshTool, MeshView } from './MeshView';

type SelMode = 'replace' | 'add' | 'subtract';

/** Classify tab, 3D mode: label the stitched photo model directly. Labels are stored on the model (per vertex); the orthomosaic classification is untouched. */
export function MeshClassify({ pid }: { pid: string }) {
  const jt = useApp((s) => s.jobsTick);
  const [info, setInfo] = useState<any>(null);
  const [labels, setLabels] = useState<Uint8Array | null>(null);
  const [sel, setSel] = useState<Uint8Array | null>(null);
  const [selCount, setSelCount] = useState(0);
  const [tool, setTool] = useState<MeshTool>('lasso');
  const [selMode, setSelMode] = useState<SelMode>('replace');
  const [active, setActive] = useState(1);
  const [show, setShow] = useState(true);
  const [opacity, setOpacity] = useState(0.6);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [busy, setBusy] = useState(false);
  const api3 = useRef<MeshApi | null>(null);
  const selRef = useRef<Uint8Array | null>(null);

  const refresh = useCallback(async () => {
    const i = await api.meshLabels(pid); setInfo(i);
    if (i.available && i.has_labels) setLabels(await api.meshLabelsData(pid)); else setLabels(null);
    return i;
  }, [pid]);
  useEffect(() => { refresh().catch((e) => toast('error', errMsg(e))); setSel(null); selRef.current = null; setSelCount(0); }, [pid, jt, refresh]);

  const classes: CClass[] = (info?.classes ?? []).map((c: CClass) => ({ ...c, visible: !hidden.has(c.id) }));
  const n = info?.vertices ?? 0;
  const setSelection = (a: Uint8Array | null) => { selRef.current = a; setSel(a); setSelCount(a ? a.reduce((s, v) => s + v, 0) : 0); };
  const onSelect = (idx: Uint32Array) => {
    const cur = selRef.current && selRef.current.length === n ? selRef.current : new Uint8Array(n);
    const next = selMode === 'replace' ? new Uint8Array(n) : cur.slice();
    for (let i = 0; i < idx.length; i++) next[idx[i]] = selMode === 'subtract' ? 0 : 1;
    setSelection(next);
  };
  const run = async (f: () => Promise<any>, ok?: string) => {
    setBusy(true);
    try { await f(); await refresh(); if (ok) toast('ok', ok); } catch (e) { toast('error', errMsg(e)); } finally { setBusy(false); }
  };
  const apply = (cls: number) => {
    const s = selRef.current; if (!s) return; const idx: number[] = []; for (let i = 0; i < s.length; i++) if (s[i]) idx.push(i);
    if (!idx.length) return;
    run(async () => { await api.meshLabelsEdit(pid, cls, Uint32Array.from(idx)); setSelection(null); }, `Labelled ${idx.length.toLocaleString()} points`);
  };
  const cnt = (id: number) => (info?.counts?.[String(id)] ?? 0) as number;

  if (!info) return <div className="m-auto text-slate-500">Loading…</div>;
  if (!info.available) return (
    <div className="m-auto max-w-[520px] p-4"><Notice kind="info"><b>There is no 3D model yet.</b> Add your photos and press <b>Process</b> on the <b>Image</b> tab first.</Notice>
      <MeshView pid={pid} jobsTick={jt} /></div>);
  return (
    <div className="flex min-h-0 flex-1">
      <div className="relative flex min-w-0 flex-1">
        <MeshView pid={pid} jobsTick={jt} onApi={(a) => { api3.current = a; }}
          cls={{ classes, show, opacity, labels, selection: sel, tool, onSelect }} />
      </div>
      <div className="w-[300px] shrink-0 overflow-auto border-l border-line bg-white">
        <Section title="Label the 3D model">
          <div className="mb-2 text-[12px] text-slate-600">Select part of the model, then give it a class. Colours show on the stitched photo model so it is easier to see what is what. Hidden-behind surfaces are not selected.</div>
          <div className="mb-1 grid grid-cols-3 gap-1">
            {([['orbit', 'Orbit'], ['lasso', 'Lasso'], ['box', 'Box']] as [MeshTool, string][]).map(([t, l]) => <button key={t} className={`btn ${tool === t ? 'btn-on' : ''}`} onClick={() => setTool(t)}>{l}</button>)}
          </div>
          <div className="mb-2 grid grid-cols-3 gap-1">
            {([['replace', 'New'], ['add', 'Add'], ['subtract', 'Remove']] as [SelMode, string][]).map(([t, l]) => <button key={t} className={`btn ${selMode === t ? 'btn-on' : ''}`} onClick={() => setSelMode(t)} title="How a new selection combines with the current one">{l}</button>)}
          </div>
          <div className="mb-2 flex items-center justify-between text-[12px]"><span><b>{selCount.toLocaleString()}</b> points selected</span>{selCount > 0 && <button className="btn !py-0" onClick={() => setSelection(null)}>Clear selection</button>}</div>
          <div className="mb-1 text-[12px] font-semibold">Class</div>
          <div className="mb-2 space-y-0.5">
            {[{ id: 0, name: 'Unlabelled (erase)', color: '#9aa0a6' } as CClass, ...classes.filter((c) => c.id !== 0)].map((c) => (
              <div key={c.id} className={`flex items-center gap-1.5 rounded px-1 py-0.5 text-[12px] ${active === c.id ? 'bg-sky-50 ring-1 ring-accent' : ''}`}>
                <input type="radio" name="mcls" checked={active === c.id} onChange={() => setActive(c.id)} />
                <span className="inline-block h-3 w-3 shrink-0 rounded-sm border border-slate-400" style={{ background: c.color }} />
                <button className="min-w-0 flex-1 truncate text-left" onClick={() => setActive(c.id)}>{c.name}</button>
                <span className="text-[11px] text-slate-500">{c.id === 0 ? '' : cnt(c.id).toLocaleString()}</span>
                {c.id !== 0 && <input type="checkbox" title="Show this class on the model" checked={!hidden.has(c.id)} onChange={() => { const h = new Set(hidden); h.has(c.id) ? h.delete(c.id) : h.add(c.id); setHidden(h); }} />}
              </div>))}
          </div>
          <button className="btn btn-primary w-full" disabled={busy || selCount === 0} onClick={() => apply(active)}>{busy ? 'Working…' : `Apply to ${selCount.toLocaleString()} selected`}</button>
          <div className="mt-2 grid grid-cols-2 gap-1">
            <button className="btn" disabled={busy || !info.can_undo} onClick={() => run(() => api.meshLabelsUndo(pid))}>↶ Undo</button>
            <button className="btn" disabled={busy || !info.can_redo} onClick={() => run(() => api.meshLabelsRedo(pid))}>↷ Redo</button>
          </div>
        </Section>
        <Section title="Show on the model">
          <label className="mb-1 flex cursor-pointer items-center gap-1.5 text-[12px]"><input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} />Show class colours</label>
          <Slider label="Colour strength" value={opacity} min={0.1} max={1} step={0.05} onChange={setOpacity} fmt={(v) => `${Math.round(v * 100)}%`} />
          <div className="mt-1 grid grid-cols-3 gap-1"><button className="btn" onClick={() => api3.current?.preset('oblique')}>Oblique</button><button className="btn" onClick={() => api3.current?.preset('top')}>Top</button><button className="btn" onClick={() => api3.current?.preset('reset')}>Reset</button></div>
        </Section>
        <Section title="Automatic labels">
          <div className="mb-1 text-[12px] text-slate-600">Labels every part of the model from what its own photos look like, and splits vegetation into grass, brush and trees by height above the ground. They are guesses, so check them and fix what is wrong.</div>
          <button className="btn btn-primary w-full" disabled={busy} onClick={() => run(async () => { await api.meshLabelsAuto(pid); bump('jobsTick'); }, 'Labelling started. It will appear here when finished.')}>{info.has_labels ? 'Run automatic labelling again' : 'Label the model automatically'}</button>
          <div className="mt-1"><Kv k="Labels" v={info.has_labels ? info.source : 'none yet'} /></div>
          <button className="btn mt-1 w-full" disabled={busy || !info.has_labels} onClick={() => { if (window.confirm('Remove all labels from the model? You can undo this.')) run(() => api.meshLabelsClear(pid)); }}>Clear all labels</button>
        </Section>
        <Section title="Advanced: copy from the flat map" open={false}>
          <div className="mb-1 text-[12px] text-slate-600">If you already classified the flat orthomosaic, copy it onto the model. Each point takes the class of the map position it sits above, so tree sides and walls follow the ground under them.</div>
          <button className="btn w-full" disabled={busy || !info.can_seed} onClick={() => { if (info.has_labels && !window.confirm('Replace the current labels with the orthomosaic classification? You can undo this.')) return; run(() => api.meshLabelsSeed(pid), 'Copied the orthomosaic classification onto the model'); }}>Copy orthomosaic classes onto the model</button>
          {!info.can_seed && <div className="mt-1"><Notice kind="warn">{info.seed_reason}</Notice></div>}
        </Section>
        <div className="p-2"><Notice kind="estimate"><Chip kind="estimated">LABELS ON THE 3D MODEL</Chip> These labels live on the photo model only. They do not change the orthomosaic classification, the point-cloud classes or any elevation result, and they are not elevation evidence.</Notice></div>
      </div>
    </div>
  );
}
