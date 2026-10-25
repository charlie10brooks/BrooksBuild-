// Resumable, chunked uploader. Many thousands of files are supported: files are queued as lightweight records and only
// `concurrency` of them are in flight. An interrupted file is resumed on the server by its client_key (name|size|mtime): add the same
// files again and only the missing chunks are sent.
import { ApiError, req } from './api';

export type Purpose = 'image' | 'orthophoto' | 'dem' | 'lidar' | 'survey_points' | 'gcp_list';
export const PURPOSE_EXT: Record<Purpose, RegExp> = {
  image: /\.(jpe?g|tiff?)$/i, orthophoto: /\.tiff?$/i, dem: /\.tiff?$/i, lidar: /\.la[sz]$/i, survey_points: /\.(csv|txt)$/i, gcp_list: /\.txt$/i,
};
export type UStatus = 'queued' | 'uploading' | 'done' | 'duplicate' | 'failed' | 'paused';
export interface UItem {
  id: number; file: File; path: string; purpose: Purpose; projectId: string; batchId: string | null; status: UStatus; sent: number; size: number;
  error?: string; attempts: number; ctl?: AbortController; resumed?: boolean;
}
export interface Totals { files: number; bytes: number; sent: number; done: number; duplicate: number; failed: number; active: number; queued: number; paused: number; speed: number; eta: number | null }

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class UploadManager {
  items: UItem[] = [];
  concurrency = 3;
  running = true;
  private nextId = 1;
  private subs = new Set<() => void>();
  private timer: any = 0;
  private samples: { t: number; b: number }[] = [];
  private cursor = 0;
  private activeN = 0;
  version = 0;

  subscribe(f: () => void) { this.subs.add(f); return () => { this.subs.delete(f); }; }
  private notify() { if (this.timer) return; this.timer = setTimeout(() => { this.timer = 0; this.version++; this.subs.forEach((f) => f()); }, 200); }

  add(files: { file: File; path: string }[], purpose: Purpose, projectId: string, batchId: string | null): { added: number; ignored: string[] } {
    const ignored: string[] = []; let added = 0;
    for (const f of files) {
      if (!PURPOSE_EXT[purpose].test(f.file.name)) { ignored.push(f.path); continue; }
      this.items.push({ id: this.nextId++, file: f.file, path: f.path, purpose, projectId, batchId, status: 'queued', sent: 0, size: f.file.size, attempts: 0 });
      added++;
    }
    this.notify(); this.pump();
    return { added, ignored };
  }
  pause() { this.running = false; this.items.forEach((i) => { if (i.status === 'uploading') i.ctl?.abort(); }); this.notify(); }
  resume() { this.running = true; this.items.forEach((i) => { if (i.status === 'paused') i.status = 'queued'; }); this.cursor = 0; this.notify(); this.pump(); }
  retryFailed() { this.items.forEach((i) => { if (i.status === 'failed') { i.status = 'queued'; i.error = undefined; i.attempts = 0; } }); this.cursor = 0; this.notify(); this.pump(); }
  clearFinished() { this.items = this.items.filter((i) => i.status !== 'done' && i.status !== 'duplicate'); this.cursor = 0; this.notify(); }
  totals(): Totals {
    const t: Totals = { files: this.items.length, bytes: 0, sent: 0, done: 0, duplicate: 0, failed: 0, active: 0, queued: 0, paused: 0, speed: 0, eta: null };
    for (const i of this.items) {
      t.bytes += i.size; t.sent += i.status === 'done' || i.status === 'duplicate' ? i.size : i.sent;
      if (i.status === 'done') t.done++; else if (i.status === 'duplicate') t.duplicate++; else if (i.status === 'failed') t.failed++; else if (i.status === 'uploading') t.active++; else if (i.status === 'paused') t.paused++; else t.queued++;
    }
    const now = performance.now(); this.samples.push({ t: now, b: t.sent }); this.samples = this.samples.filter((s) => now - s.t < 8000);
    if (this.samples.length > 1) { const a = this.samples[0]; t.speed = Math.max(0, (t.sent - a.b) / ((now - a.t) / 1000)); if (t.speed > 1) t.eta = (t.bytes - t.sent) / t.speed; }
    return t;
  }

  private pump() {
    while (this.running && this.activeN < this.concurrency) {
      while (this.cursor < this.items.length && this.items[this.cursor].status !== 'queued') this.cursor++;
      if (this.cursor >= this.items.length) return;
      const it = this.items[this.cursor++]; this.activeN++; it.status = 'uploading';
      this.run(it).finally(() => { this.activeN--; this.notify(); this.pump(); });
    }
  }

  private async run(it: UItem) {
    it.ctl = new AbortController(); const sig = it.ctl.signal;
    const base = `/projects/${it.projectId}/uploads`;
    try {
      const st: any = await this.retry(() => req('POST', base, { filename: it.file.name, size: it.size, chunk_size: 8 * 1024 * 1024, batch_id: it.batchId, purpose: it.purpose,
        client_key: `${it.file.name}|${it.size}|${it.file.lastModified}|${it.purpose}` }, { signal: sig }), sig);
      if (st.status === 'duplicate') { it.status = 'duplicate'; it.sent = it.size; return; }
      it.resumed = !!st.resumed;
      const cs: number = st.chunk_size; const missing: number[] = st.missing_chunks;
      it.sent = it.size - missing.reduce((a, i) => a + Math.min(cs, it.size - i * cs), 0);
      for (const i of missing) {
        if (sig.aborted) throw new DOMException('aborted', 'AbortError');
        const blob = it.file.slice(i * cs, Math.min(it.size, (i + 1) * cs));
        await this.retry(() => req('PUT', `${base}/${st.upload_id}/chunks/${i}`, blob, { signal: sig, headers: { 'Content-Type': 'application/octet-stream' } }), sig);
        it.sent += blob.size; this.notify();
      }
      await this.retry(() => req('POST', `${base}/${st.upload_id}/complete`, {}, { signal: sig }), sig);
      it.status = 'done'; it.sent = it.size;
    } catch (e: any) {
      if (e?.name === 'AbortError') { it.status = this.running ? 'queued' : 'paused'; if (this.running) this.cursor = 0; }
      else { it.status = 'failed'; it.error = e instanceof ApiError ? `${e.message}` : String(e?.message ?? e); }
    } finally { it.ctl = undefined; }
  }

  /** Retry transient failures (network, 5xx, 429) with exponential backoff; client errors are final. */
  private async retry<T>(fn: () => Promise<T>, sig: AbortSignal): Promise<T> {
    let delay = 800;
    for (let n = 0; ; n++) {
      try { return await fn(); } catch (e: any) {
        if (e?.name === 'AbortError') throw e;
        const transient = e instanceof ApiError && (e.status === 0 || e.status >= 500 || e.status === 429) && e.status !== 507;
        if (!transient || n >= 5) throw e;
        await sleep(delay); delay = Math.min(delay * 2, 15000);
        if (sig.aborted) throw new DOMException('aborted', 'AbortError');
      }
    }
  }
}
export const uploads = new UploadManager();

// ------------------------------------------------------------------ folder drag and drop
export async function filesFromDrop(dt: DataTransfer): Promise<{ file: File; path: string }[]> {
  const out: { file: File; path: string }[] = [];
  const entries = Array.from(dt.items || []).map((i: any) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null)).filter(Boolean);
  if (!entries.length) return Array.from(dt.files).map((f) => ({ file: f, path: f.name }));
  const walk = async (en: any, prefix: string): Promise<void> => {
    if (en.isFile) { const f: File = await new Promise((res, rej) => en.file(res, rej)); out.push({ file: f, path: prefix + f.name }); return; }
    if (en.isDirectory) {
      const rd = en.createReader();
      for (;;) { // readEntries returns batches (max ~100); keep reading until empty
        const batch: any[] = await new Promise((res, rej) => rd.readEntries(res, rej));
        if (!batch.length) break;
        for (const c of batch) await walk(c, prefix + en.name + '/');
      }
    }
  };
  for (const en of entries) await walk(en, '');
  return out;
}
export const filesFromInput = (list: FileList | null) => Array.from(list ?? []).map((f) => ({ file: f, path: (f as any).webkitRelativePath || f.name }));
