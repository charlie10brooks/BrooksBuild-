import { useEffect, useState } from 'react';
import { api } from './api';
import { LeftSidebar } from './Sidebar';
import { Modal, Chip } from './ui';
import { setApp, useApp, useProject, View, selectProject } from './store';
import { Dashboard } from './views/Dashboard';
import { Upload } from './views/Upload';
import { Classify } from './views/Classify';
import { TerrainView } from './views/Terrain';
import { View3D } from './views/View3D';
import { Exports } from './views/Exports';

const TABS: { id: View; label: string; key: string }[] = [
  { id: 'dashboard', label: 'Image', key: '1' }, { id: 'upload', label: 'Add photos', key: '2' }, { id: 'classify', label: 'Label', key: '3' },
  { id: 'terrain', label: 'Terrain map', key: '4' }, { id: 'view3d', label: 'Terrain 3D', key: '5' }, { id: 'planning', label: 'Plan', key: '6' }, { id: 'exports', label: 'Export', key: '7' },
];

const SHORTCUTS: [string, string][] = [
  ['Alt + 1 … 7', 'Switch view (Image, Add photos, Label, Terrain map, Terrain 3D, Plan, Export)'], ['Ctrl/Cmd + Z', 'Undo last classification edit'], ['Ctrl/Cmd + Shift + Z or Ctrl + Y', 'Redo'],
  ['S', 'Sample tool (inspect pixel)'], ['B', 'Brush'], ['P', 'Polygon'], ['R', 'Rectangle'], ['L', 'Lasso'], ['F', 'Flood fill'], ['M', 'Select similar colour'], ['E', 'Erase to Unknown'],
  ['[ and ]', 'Smaller / larger brush'], ['0 … 9', 'Choose class by id'], ['Enter / Esc', 'Apply / cancel the current polygon or selection preview'], ['Backspace', 'Remove the last polygon vertex'],
  ['Space + drag, or middle-drag', 'Pan'], ['Wheel', 'Zoom at cursor'], ['H', 'Toggle class overlay'], ['\\', 'Hold to peek at imagery without overlay'], ['?', 'This help'],
];

function Status() {
  const st = useApp((s) => s.status);
  const saved = useApp((s) => s.saved);
  const p = useProject();
  return (
    <footer className="flex h-6 shrink-0 items-center justify-between gap-4 border-t border-line bg-slate-800 px-3 font-mono text-[11px] text-slate-200">
      <span className="min-w-[220px] truncate">{st.left || ' '}</span>
      <span className="flex-1 truncate text-slate-300">{st.mid}</span>
      <span className="flex items-center gap-3 whitespace-nowrap">
        {st.right && <span>{st.right}</span>}
        {p && <span title="Horizontal CRS / linear units">{p.epsg ? `EPSG:${p.epsg}` : 'CRS not set'} · {p.horizontal_units}</span>}
        <span className={saved.state === 'error' ? 'text-red-300' : saved.state === 'saving' ? 'text-amber-300' : 'text-emerald-300'}>
          {saved.state === 'saved' ? `✓ edits saved (v${saved.version})` : saved.state === 'saving' ? 'saving…' : `⚠ ${saved.msg ?? 'save failed'}`}
        </span>
      </span>
    </footer>
  );
}

function Toasts() {
  const t = useApp((s) => s.toasts);
  return (
    <div className="pointer-events-none fixed bottom-9 left-1/2 z-[60] flex -translate-x-1/2 flex-col items-center gap-1">
      {t.map((x) => <div key={x.id} className={`pointer-events-auto max-w-[620px] rounded border px-3 py-1.5 text-[12px] shadow ${x.kind === 'error' ? 'border-red-300 bg-red-50 text-red-900' : x.kind === 'ok' ? 'border-emerald-300 bg-emerald-50 text-emerald-900' : 'border-slate-300 bg-white'}`}>{x.text}</div>)}
    </div>
  );
}

export function App() {
  const view = useApp((s) => s.view);
  const pid = useApp((s) => s.projectId);
  const ptick = useApp((s) => s.projectsTick);
  const ur = useApp((s) => s.undoRedo);
  const rightOpen = useApp((s) => s.rightOpen);
  const project = useProject();
  const [help, setHelp] = useState(false);

  useEffect(() => {
    if (!pid) { setApp({ project: null }); return; }
    let alive = true, t: any;
    const load = async () => {
      try { const p = await api.project(pid); if (alive) setApp({ project: p }); }
      catch (e: any) { if (alive && e?.status === 404) { selectProject(null); } }
      if (alive) t = setTimeout(load, 7000);
    };
    load();
    return () => { alive = false; clearTimeout(t); };
  }, [pid, ptick]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === '?') { setHelp((h) => !h); return; }
      if (e.altKey && !e.ctrlKey && !e.metaKey) { const t = TABS.find((x) => x.key === e.code.replace('Digit', '')); if (t) { e.preventDefault(); setApp({ view: t.id }); } }
    };
    window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k);
  }, []);

  const body = !pid ? (
    <div className="m-auto max-w-[560px] p-6 text-slate-600">
      <div className="mb-1 text-[22px] font-semibold text-slate-800">Welcome to Brooks<span className="text-sky-600">Build</span></div>
      <div className="mb-3 text-[14px]">Turn drone photos into a 3D model of your site, label what is on it, and plan around the terrain.</div>
      <ol className="mb-4 list-decimal space-y-1 pl-5 text-[14px]">
        <li>Press <b>+ New</b> in the left panel to start a project.</li><li>Open <b>Add photos</b> and upload your drone photos.</li><li>On <b>Image</b>, press <b>Process</b>. BrooksBuild builds and labels the 3D model.</li>
      </ol>
      <div className="text-[13px]">Just want a look around first? <button className="btn btn-primary ml-1" onClick={async () => { try { const q = await api.sample(); selectProject(q.id); setApp({ view: 'dashboard' }); } catch { /* the sidebar shows the error */ } }}>Open the sample project</button></div>
      <div className="mt-1 text-[11px] text-slate-500">The sample is made-up data for trying the tools. It is not a real site.</div>
    </div>
  ) : !project ? <div className="m-auto text-slate-500">Loading project…</div> :
    view === 'dashboard' ? <Dashboard /> : view === 'upload' ? <Upload /> : view === 'classify' ? <Classify /> :
    view === 'terrain' ? <TerrainView mode="terrain" /> : view === 'planning' ? <TerrainView mode="planning" /> : view === 'view3d' ? <View3D /> : <Exports />;

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-10 shrink-0 items-center gap-2 border-b border-line bg-slate-900 px-2 text-slate-100">
        <button className="rounded px-1.5 py-1 hover:bg-slate-700" title="Toggle left sidebar" onClick={() => setApp((s) => ({ leftOpen: !s.leftOpen }))}>☰</button>
        <div className="mr-2 font-semibold tracking-wide">Brooks<span className="text-sky-300">Build</span></div>
        {project && <div className="mr-2 flex items-center gap-2 truncate text-[12px] text-slate-300"><span className="max-w-[220px] truncate font-medium text-white">{project.name}</span>
          {project.is_sample && <span className="rounded border border-red-400 px-1 text-[10px] text-red-300">SYNTHETIC</span>}</div>}
        <nav className="flex gap-0.5">
          {TABS.map((t) => <button key={t.id} title={`Alt+${t.key}`} disabled={!pid} onClick={() => setApp({ view: t.id })} className={`rounded px-2.5 py-1 text-[12px] disabled:opacity-40 ${view === t.id ? 'bg-sky-600 text-white' : 'text-slate-300 hover:bg-slate-700'}`}>{t.label}</button>)}
        </nav>
        <div className="flex-1" />
        <div className="flex items-center gap-1">
          <button className="btn !border-slate-600 !bg-slate-800 !text-slate-100" disabled={!ur.canUndo} title="Undo (Ctrl+Z)" onClick={() => ur.undo?.()}>↶ Undo</button>
          <button className="btn !border-slate-600 !bg-slate-800 !text-slate-100" disabled={!ur.canRedo} title="Redo (Ctrl+Shift+Z)" onClick={() => ur.redo?.()}>↷ Redo</button>
        </div>
        <span className="ml-2 hidden rounded border border-amber-400 bg-amber-900/40 px-1.5 py-0.5 text-[10.5px] text-amber-200 xl:inline" title="All elevation, slope, volume and drainage outputs">Preliminary planning estimates only</span>
        <button className="rounded px-1.5 py-1 hover:bg-slate-700" onClick={() => setHelp(true)} title="Keyboard shortcuts (?)">?</button>
        <button className="rounded px-1.5 py-1 hover:bg-slate-700" title="Toggle right sidebar" onClick={() => setApp((s) => ({ rightOpen: !s.rightOpen }))}>▤</button>
      </header>
      <div className="flex min-h-0 flex-1">
        <LeftSidebar />
        <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-slate-200">{body}</main>
        <aside id="right-slot" className={`${rightOpen ? 'flex' : 'hidden'} w-[286px] shrink-0 flex-col overflow-y-auto border-l border-line bg-white`} />
      </div>
      <Status />
      <Toasts />
      {help && <Modal title="Keyboard shortcuts" onClose={() => setHelp(false)}><table className="w-full text-[12px]"><tbody>{SHORTCUTS.map(([k, d]) => <tr key={k}><td className="w-52 py-0.5 pr-2 align-top"><kbd>{k}</kbd></td><td className="py-0.5">{d}</td></tr>)}</tbody></table></Modal>}
    </div>
  );
}
