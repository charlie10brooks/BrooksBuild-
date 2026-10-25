import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { api } from '../api';
import { Chip, Field, Kv, Notice, Progress, Section, Table } from '../ui';
import { bump, setApp, toast, useProject } from '../store';
import { Purpose, UItem, filesFromDrop, filesFromInput, uploads } from '../uploader';
import { errMsg, fmtBytes, fmtDur, usePoll } from '../util';
import { PhotoClassify, PhotoGallery } from './PhotoClassify';

const PURPOSES: { id: Purpose; label: string; hint: string; accept: string }[] = [
  { id: 'image', label: 'Drone images', hint: 'JPEG or TIFF with EXIF GPS. No limit on the number of images. Duplicates (identical content) are detected and skipped.', accept: '.jpg,.jpeg,.tif,.tiff' },
  { id: 'orthophoto', label: 'Orthomosaic', hint: 'A GeoTIFF you already produced (RGB). Must be in the project CRS; a mismatch is refused, never silently reprojected.', accept: '.tif,.tiff' },
  { id: 'dem', label: 'DEM / DTM', hint: 'External bare-earth elevation GeoTIFF (e.g. state LiDAR DEM). Becomes an elevation source with a priority and datum offset you can adjust under Terrain.', accept: '.tif,.tiff' },
  { id: 'lidar', label: 'LiDAR (LAS/LAZ)', hint: 'Ground-classified or raw LiDAR points. LAZ needs PDAL on the server.', accept: '.las,.laz' },
  { id: 'survey_points', label: 'Survey points', hint: 'CSV: x,y,z (project CRS) for measured ground shots. Treated as the highest-trust source.', accept: '.csv,.txt' },
  { id: 'gcp_list', label: 'GCP list', hint: 'ODM gcp_list.txt format (CRS line, then x y z im_x im_y image_name).', accept: '.txt' },
];
const ST: Record<string, string> = { queued: 'queued', uploading: 'uploading', done: 'sent', duplicate: 'duplicate', failed: 'failed', paused: 'paused' };

function Rows({ items }: { items: UItem[] }) {
  const box = useRef<HTMLDivElement | null>(null);
  const [top, setTop] = useState(0);
  const H = 22, VIS = 16;
  const start = Math.max(0, Math.floor(top / H) - 4);
  return (
    <div ref={box} className="h-[360px] overflow-auto rounded border border-line bg-white" onScroll={(e) => setTop(e.currentTarget.scrollTop)}>
      <div style={{ height: items.length * H, position: 'relative' }}>
        {items.slice(start, start + VIS + 10).map((i, k) => (
          <div key={i.id} className="absolute left-0 right-0 flex items-center gap-2 px-2 text-[11.5px]" style={{ top: (start + k) * H, height: H }}>
            <span className="w-20 shrink-0"><Chip kind={i.status === 'done' ? 'ok' : i.status === 'failed' ? 'warn' : i.status === 'uploading' ? 'running' : 'queued'}>{ST[i.status]}{i.resumed && i.status === 'uploading' ? ' (resumed)' : ''}</Chip></span>
            <span className="min-w-0 flex-1 truncate" title={i.path}>{i.path}</span>
            <span className="w-16 shrink-0 text-right text-slate-500">{fmtBytes(i.size)}</span>
            <span className="w-24 shrink-0">{i.status === 'uploading' ? <Progress value={i.sent / i.size} /> : i.status === 'failed' ? <span className="text-warn" title={i.error}>{(i.error ?? '').slice(0, 18)}</span> : null}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Upload() {
  const p = useProject()!;
  const [purpose, setPurpose] = useState<Purpose>('image');
  const [batch, setBatch] = useState<string>('new');
  const [label, setLabel] = useState('');
  const [drag, setDrag] = useState(false);
  const [filter, setFilter] = useState<'all' | 'failed' | 'active'>('all');
  useSyncExternalStore((cb) => uploads.subscribe(cb), () => uploads.version);
  const batches = usePoll(() => api.batches(p.id), 6000, [p.id]);
  const summary = usePoll(() => api.uploadSummary(p.id), 3000, [p.id]);
  const fileIn = useRef<HTMLInputElement | null>(null);
  const dirIn = useRef<HTMLInputElement | null>(null);
  const pi = PURPOSES.find((x) => x.id === purpose)!;
  useEffect(() => { dirIn.current?.setAttribute('webkitdirectory', ''); }, []);

  const enqueue = async (files: { file: File; path: string }[]) => {
    if (!files.length) return;
    try {
      let bid: string | null = null;
      if (purpose === 'image') {
        if (batch === 'new') { const b = await api.createBatch(p.id, label.trim() || undefined); bid = b.id; setBatch(b.id); bump('projectsTick'); batches.reload(); }
        else bid = batch;
      }
      const r = uploads.add(files, purpose, p.id, bid);
      if (r.ignored.length) toast('info', `${r.ignored.length} file(s) ignored: not valid for “${pi.label}” (${r.ignored.slice(0, 3).join(', ')}${r.ignored.length > 3 ? '…' : ''})`);
      if (r.added) toast('ok', `${r.added} file(s) queued`);
    } catch (e) { toast('error', errMsg(e)); }
  };
  const onDrop = async (e: any) => { e.preventDefault(); setDrag(false); try { await enqueue(await filesFromDrop(e.dataTransfer)); } catch (er) { toast('error', errMsg(er)); } };
  const t = uploads.totals();
  const shown = uploads.items.filter((i) => filter === 'all' ? true : filter === 'failed' ? i.status === 'failed' : i.status === 'uploading' || i.status === 'queued');
  const sv = summary.data;

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-auto p-3">
      <div className="mx-auto w-full max-w-[1200px] gap-3" style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 340px' }}>
        <div className="min-w-0 space-y-3">
          <div className="rounded border border-line bg-white">
            <Section title="Add data">
              <div className="mb-2 flex flex-wrap gap-1">{PURPOSES.map((x) => <button key={x.id} className={`btn ${x.id === purpose ? 'btn-on' : ''}`} onClick={() => setPurpose(x.id)}>{x.label}</button>)}</div>
              <div className="mb-2 text-[12px] text-slate-600">{pi.hint}</div>
              {purpose === 'image' && (
                <div className="mb-2 grid grid-cols-2 gap-2">
                  <Field label="Batch"><select className="input w-full" value={batch} onChange={(e) => setBatch(e.target.value)}><option value="new">New batch for this upload</option>{(batches.data ?? []).map((b: any) => <option key={b.id} value={b.id}>{b.label} ({b.images})</option>)}</select></Field>
                  {batch === 'new' && <Field label="Batch label (e.g. flight date)"><input className="input w-full" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Morning flight" /></Field>}
                </div>
              )}
              <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
                className={`flex h-32 flex-col items-center justify-center rounded border-2 border-dashed ${drag ? 'border-accent bg-sky-50' : 'border-line bg-panel'}`}>
                <div className="mb-2 text-slate-600">Drop files or whole folders here</div>
                <div className="flex gap-2">
                  <button className="btn btn-primary" onClick={() => fileIn.current?.click()}>Choose files…</button>
                  <button className="btn" onClick={() => dirIn.current?.click()}>Choose folder…</button>
                </div>
                <input ref={fileIn} type="file" multiple accept={pi.accept} className="hidden" onChange={(e) => { enqueue(filesFromInput(e.target.files)); e.target.value = ''; }} />
                <input ref={dirIn} type="file" multiple className="hidden" onChange={(e) => { enqueue(filesFromInput(e.target.files)); e.target.value = ''; }} />
              </div>
              <div className="mt-2 text-[11px] text-slate-500">Uploads are sent in 8 MB chunks and survive network drops: if a transfer is interrupted (or you reload the page), add the same files again and only the missing chunks are sent. Checksums are verified on the server.</div>
            </Section>
          </div>
          <div className="rounded border border-line bg-white">
            <Section title="Upload queue" badge={<span className="ml-1 font-normal normal-case text-slate-400">{t.files} files</span>}
              actions={<>
                <button className="btn !py-0" onClick={() => (uploads.running ? uploads.pause() : uploads.resume())}>{uploads.running ? 'Pause' : 'Resume'}</button>
                <button className="btn !py-0" disabled={!t.failed} onClick={() => uploads.retryFailed()}>Retry failed</button>
                <button className="btn !py-0" onClick={() => uploads.clearFinished()}>Clear finished</button></>}>
              {t.files === 0 ? <div className="text-slate-500">Nothing queued.</div> : <>
                <Progress value={t.bytes ? t.sent / t.bytes : 0} tone={t.failed ? 'warn' : 'accent'} />
                <div className="my-1 flex flex-wrap gap-3 text-[12px] text-slate-600">
                  <span>{fmtBytes(t.sent)} / {fmtBytes(t.bytes)}</span><span>{fmtBytes(t.speed)}/s</span>{t.eta != null && <span>ETA {fmtDur(t.eta)}</span>}
                  <span>sent {t.done}</span><span>duplicates {t.duplicate}</span><span className={t.failed ? 'text-warn' : ''}>failed {t.failed}</span><span>waiting {t.queued + t.paused}</span>
                </div>
                <div className="mb-1 flex gap-1">{(['all', 'active', 'failed'] as const).map((f) => <button key={f} className={`btn !py-0 ${filter === f ? 'btn-on' : ''}`} onClick={() => setFilter(f)}>{f}</button>)}</div>
                <Rows items={shown} />
              </>}
            </Section>
          </div>
          <PhotoGallery p={p} />
        </div>
        <div className="min-w-0 space-y-3">
          <PhotoClassify p={p} justUploaded={t.files > 0 && t.queued + t.active + t.paused === 0 && summary.data?.needs_attention === 0} />
          {p.image_count > 0 && <div className="rounded border border-accent bg-white ring-2 ring-sky-100"><Section title="Next: build the 3D model">
            <div className="mb-2 text-[13px] text-slate-700">When your photos are uploaded, go to the <b>Image</b> tab and press <b>Process</b>. It builds the 3D model and labels it for you.</div>
            <button className="btn btn-primary" onClick={() => setApp({ view: 'dashboard' })}>Go to Image</button></Section></div>}
          <div className="rounded border border-line bg-white">
            <Section title="Server-side processing">
              <div className="mb-1 text-[11px] text-slate-500">After upload, each file is assembled, checksummed, de-duplicated, validated and read for EXIF/GPS by background workers.</div>
              {sv && Object.entries(sv.by_status).map(([k, v]: any) => <Kv key={k} k={k} v={`${v.files} files · ${fmtBytes(v.bytes)}`} />)}
              {sv && sv.needs_attention > 0 && <Notice kind="warn">{sv.needs_attention} file(s) need attention (failed or rejected). Rejected files are kept in quarantine with the reason in the project record.</Notice>}
              <Kv k="Images ingested" v={p.image_count} /><Kv k="Without GPS" v={p.images_without_gps} />
            </Section>
          </div>
          <div className="rounded border border-line bg-white">
            <Section title="Batches" open={false}>
              <Table head={['Batch', 'Images', 'Size']} dense rows={(batches.data ?? []).map((b: any) => [b.label, b.images, fmtBytes(b.bytes)])} />
            </Section>
          </div>
        </div>
      </div>
    </div>
  );
}
