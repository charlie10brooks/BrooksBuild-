import { createPortal } from 'react-dom';
import { ReactNode, useEffect, useState } from 'react';
import { clamp } from './util';

export function Section({ title, children, actions, open = true, badge }: { title: string; children?: ReactNode; actions?: ReactNode; open?: boolean; badge?: ReactNode }) {
  const [o, setO] = useState(open);
  return (
    <section className="border-b border-line">
      <div className="flex items-center justify-between bg-panel">
        <button className="sec-title flex flex-1 items-center gap-1 text-left" onClick={() => setO(!o)}>
          <span className="inline-block w-3 text-[9px]">{o ? '▼' : '▶'}</span>{title}{badge}
        </button>
        <div className="flex gap-1 pr-1">{actions}</div>
      </div>
      {o && <div className="p-2">{children}</div>}
    </section>
  );
}

export function Field({ label, children, hint }: { label: string; children?: ReactNode; hint?: string }) {
  return (
    <label className="mb-1.5 block">
      <span className="mb-0.5 block text-[11px] text-slate-600">{label}</span>
      {children}
      {hint && <span className="mt-0.5 block text-[10px] text-slate-500">{hint}</span>}
    </label>
  );
}

export function Slider({ label, value, min, max, step = 1, onChange, fmt }: { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <div className="mb-1.5">
      <div className="flex justify-between text-[11px] text-slate-600"><span>{label}</span><span className="font-mono">{fmt ? fmt(value) : value}</span></div>
      <input type="range" className="w-full" min={min} max={max} step={step} value={value} onInput={(e) => onChange(clamp(Number(e.target.value), min, max))} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export function Check({ label, checked, onChange, title }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void; title?: string }) {
  return (
    <label className="flex cursor-pointer items-center gap-1.5 py-0.5 text-[12px]" title={title}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />{label}
    </label>
  );
}

export function Progress({ value, tone = 'accent' }: { value: number; tone?: 'accent' | 'warn' | 'ok' }) {
  const c = tone === 'warn' ? 'bg-amber-500' : tone === 'ok' ? 'bg-emerald-600' : 'bg-accent';
  return <div className="h-1.5 w-full overflow-hidden rounded bg-slate-200"><div className={`h-full ${c}`} style={{ width: `${clamp(value, 0, 1) * 100}%` }} /></div>;
}

const CHIP: Record<string, string> = {
  observed: 'border-sky-300 bg-sky-50 text-observed', estimated: 'border-amber-300 bg-amber-50 text-estimated', evidence: 'border-slate-300 bg-slate-100 text-slate-700',
  ok: 'border-emerald-300 bg-emerald-50 text-emerald-800', warn: 'border-red-300 bg-red-50 text-warn', info: 'border-slate-300 bg-white text-slate-700',
  queued: 'border-slate-300 bg-slate-100 text-slate-700', running: 'border-sky-300 bg-sky-50 text-accent-dark', succeeded: 'border-emerald-300 bg-emerald-50 text-emerald-800',
  failed: 'border-red-300 bg-red-50 text-warn', cancelled: 'border-slate-300 bg-slate-100 text-slate-500', cancelling: 'border-amber-300 bg-amber-50 text-estimated',
};
export const Chip = ({ kind = 'info', children, title }: { kind?: string; children?: ReactNode; title?: string }) => <span className={`chip ${CHIP[kind] ?? CHIP.info}`} title={title}>{children}</span>;

export function SurfaceBadge({ kind }: { kind: string | null | undefined }) {
  if (kind === 'observed') return <Chip kind="observed" title="Seen by the photogrammetry: tops of canopy and structures">OBSERVED surface</Chip>;
  if (kind === 'estimated') return <Chip kind="estimated" title="Bare-earth estimate; interpolated where ground was not seen">ESTIMATED bare earth</Chip>;
  if (kind === 'evidence') return <Chip kind="evidence">EVIDENCE layer</Chip>;
  if (kind === 'derived') return <Chip kind="info">DERIVED</Chip>;
  return null;
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="m-4 rounded border border-dashed border-line bg-white p-4 text-center text-slate-600"><div className="mb-1 font-semibold text-slate-800">{title}</div><div className="text-[12px]">{children}</div></div>;
}

export function Notice({ kind = 'info', children }: { kind?: 'info' | 'warn' | 'estimate'; children?: ReactNode }) {
  const c = kind === 'warn' ? 'border-red-300 bg-red-50 text-red-900' : kind === 'estimate' ? 'border-amber-300 bg-amber-50 text-amber-900' : 'border-slate-300 bg-slate-50 text-slate-700';
  return <div className={`mb-2 rounded border px-2 py-1.5 text-[11.5px] leading-4 ${c}`}>{children}</div>;
}

export function Modal({ title, onClose, children, wide }: { title: string; onClose: () => void; children?: ReactNode; wide?: boolean }) {
  useEffect(() => { const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose(); window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k); }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div className={`max-h-[85vh] ${wide ? 'w-[820px]' : 'w-[460px]'} max-w-[95vw] overflow-auto rounded-md border border-line bg-white shadow-xl`} onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-line bg-panel px-3 py-2"><div className="font-semibold">{title}</div><button className="btn" onClick={onClose}>✕</button></div>
        <div className="p-3">{children}</div>
      </div>
    </div>
  );
}

/** Renders children into the right sidebar (App provides <aside id="right-slot">). */
export function RightPanel({ children }: { children?: ReactNode }) {
  const [el, setEl] = useState<Element | null>(null);
  useEffect(() => { setEl(document.getElementById('right-slot')); }, []);
  return el ? createPortal(children, el) : null;
}

export function Kv({ k, v, mono }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return <div className="flex justify-between gap-2 py-0.5 text-[12px]"><span className="text-slate-500">{k}</span><span className={`text-right ${mono ? 'font-mono' : ''}`}>{v}</span></div>;
}

export function Table({ head, rows, dense }: { head: ReactNode[]; rows: ReactNode[][]; dense?: boolean }) {
  return (
    <table className="w-full border-collapse text-[12px]">
      <thead><tr>{head.map((h, i) => <th key={i} className="border-b border-line bg-panel px-1.5 py-1 text-left font-medium text-slate-600">{h}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i} className="odd:bg-white even:bg-slate-50">{r.map((c, j) => <td key={j} className={`border-b border-slate-100 px-1.5 ${dense ? 'py-0.5' : 'py-1'} align-top`}>{c}</td>)}</tr>)}</tbody>
    </table>
  );
}
