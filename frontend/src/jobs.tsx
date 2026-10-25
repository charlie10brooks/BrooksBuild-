import { useEffect, useState } from 'react';
import { api, Job } from './api';
import { Chip, Modal, Progress } from './ui';
import { bump, toast, useApp } from './store';
import { errMsg, fmtDur, fmtTime, usePoll } from './util';

export const KIND_LABEL: Record<string, string> = {
  odm: 'Photogrammetry (ODM)', terrain: 'Terrain reconstruction', classify: 'Auto-classify', train_classifier: 'Learn from corrections', prepare_ortho: 'Prepare orthomosaic',
  export: 'Export', viewer_cloud: 'Point-cloud preview', import_asset: 'Import data', analyze_flights: 'Flight analysis', ingest_image: 'Ingest image',
};

export function JobLogs({ job, onClose }: { job: Job; onClose: () => void }) {
  const [lines, setLines] = useState<{ id: number; ts: number; level: string; message: string }[]>([]);
  useEffect(() => {
    let alive = true, after = 0, t: any;
    const tick = async () => {
      try {
        const rows = await api.jobLogs(job.id, after);
        if (!alive) return;
        if (rows.length) { after = rows[rows.length - 1].id; setLines((l) => [...l, ...rows].slice(-2000)); }
      } catch { /* retry */ }
      t = setTimeout(tick, 1500);
    };
    tick();
    return () => { alive = false; clearTimeout(t); };
  }, [job.id]);
  return (
    <Modal title={`${KIND_LABEL[job.kind] ?? job.kind} - logs`} onClose={onClose} wide>
      <div className="mb-2 flex flex-wrap items-center gap-2 text-[12px]"><Chip kind={job.status}>{job.status}</Chip><span className="font-mono text-slate-500">{job.id}</span>{job.error && <span className="text-warn">{job.error}</span>}</div>
      <div className="h-[55vh] overflow-auto rounded border border-line bg-slate-950 p-2 font-mono text-[11px] leading-4 text-slate-200">
        {lines.length === 0 && <div className="text-slate-500">No log lines yet.</div>}
        {lines.map((l) => <div key={l.id} className={l.level === 'error' ? 'text-red-300' : l.level === 'warning' ? 'text-amber-300' : ''}><span className="text-slate-500">{new Date(l.ts * 1000).toLocaleTimeString()} </span>{l.message}</div>)}
      </div>
    </Modal>
  );
}

export function JobRow({ j, compact, onLogs }: { j: Job; compact?: boolean; onLogs: (j: Job) => void }) {
  const act = async (f: () => Promise<any>, ok: string) => { try { await f(); toast('ok', ok); bump('jobsTick'); } catch (e) { toast('error', errMsg(e)); } };
  const active = j.status === 'running' || j.status === 'queued' || j.status === 'cancelling';
  const dur = j.started_at ? (j.finished_at ?? Date.now() / 1000) - j.started_at : 0;
  return (
    <div className="border-b border-slate-100 px-2 py-1.5">
      <div className="flex items-center justify-between gap-1">
        <div className="truncate font-medium" title={j.id}>{KIND_LABEL[j.kind] ?? j.kind}</div>
        <Chip kind={j.status}>{j.status}</Chip>
      </div>
      {active && <div className="my-1"><Progress value={j.progress} /></div>}
      {j.message && active && <div className="truncate text-[11px] text-slate-600" title={j.message}>{j.message}</div>}
      {j.status === 'failed' && <div className="mt-0.5 text-[11px] text-warn" title={j.error ?? ''}>{(j.error ?? 'failed').slice(0, compact ? 110 : 400)}{j.error_code && <span className="text-slate-500"> [{j.error_code}]</span>}</div>}
      <div className="mt-1 flex items-center justify-between text-[10.5px] text-slate-500">
        <span>{dur ? fmtDur(dur) : fmtTime(j.created_at)}{j.attempts > 1 ? ` · attempt ${j.attempts}/${j.max_attempts}` : ''}</span>
        <span className="flex gap-1">
          <button className="btn !px-1.5 !py-0" onClick={() => onLogs(j)}>Logs</button>
          {active && <button className="btn btn-danger !px-1.5 !py-0" onClick={() => act(() => api.cancelJob(j.id), 'Cancel requested')}>Cancel</button>}
          {(j.status === 'failed' || j.status === 'cancelled') && <button className="btn !px-1.5 !py-0" onClick={() => act(() => api.retryJob(j.id), 'Job re-queued')}>Retry</button>}
        </span>
      </div>
    </div>
  );
}

export function JobList({ limit = 12, compact }: { limit?: number; compact?: boolean }) {
  const pid = useApp((s) => s.projectId);
  const tick = useApp((s) => s.jobsTick);
  const [logs, setLogs] = useState<Job | null>(null);
  const { data, error } = usePoll(() => (pid ? api.jobs(pid, `limit=${limit}`) : Promise.resolve([] as Job[])), 2500, [pid, tick, limit]);
  if (!pid) return <div className="p-2 text-slate-500">Select a project.</div>;
  return (
    <div>
      {error && <div className="p-2 text-warn">{error}</div>}
      {data && data.length === 0 && <div className="p-2 text-slate-500">No jobs yet.</div>}
      {(data ?? []).map((j) => <JobRow key={j.id} j={j} compact={compact} onLogs={setLogs} />)}
      {logs && <JobLogs job={logs} onClose={() => setLogs(null)} />}
    </div>
  );
}
