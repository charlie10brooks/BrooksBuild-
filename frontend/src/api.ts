// Typed client for the TerraScope HTTP API. Every call either returns data or throws ApiError (never a silent fallback).
export class ApiError extends Error {
  status: number; code: string; details: any;
  constructor(status: number, code: string, message: string, details?: any) { super(message); this.status = status; this.code = code; this.details = details; }
}
const BASE: string = (window as any).TERRASCOPE_API ?? '/api';
export const url = (p: string) => BASE + p;

export async function req<T = any>(method: string, path: string, body?: any, init: RequestInit = {}): Promise<T> {
  let r: Response;
  try {
    r = await fetch(BASE + path, { method, headers: body !== undefined && !(body instanceof Blob) ? { 'Content-Type': 'application/json' } : undefined,
      body: body === undefined ? undefined : body instanceof Blob ? body : JSON.stringify(body), ...init });
  } catch (e: any) {
    if (e?.name === 'AbortError') throw e;
    throw new ApiError(0, 'network', 'Cannot reach the TerraScope server. Is the API running?');
  }
  if (r.status === 204) return undefined as T;
  const ct = r.headers.get('content-type') || '';
  const data = ct.includes('json') ? await r.json().catch(() => null) : null;
  if (!r.ok) {
    const e = data?.error;
    throw new ApiError(r.status, e?.code ?? 'error', e?.message ?? `HTTP ${r.status}`, e?.details);
  }
  return (data as T);
}
export const get = <T = any>(p: string, init?: RequestInit) => req<T>('GET', p, undefined, init);
export const post = <T = any>(p: string, b?: any) => req<T>('POST', p, b ?? {});
export const patch = <T = any>(p: string, b: any) => req<T>('PATCH', p, b);
export const put = <T = any>(p: string, b: any) => req<T>('PUT', p, b);
export const del = <T = any>(p: string) => req<T>('DELETE', p);

// ------------------------------------------------------------------ types
export interface CClass { id: number; key: string; name: string; color: string; vegetation: boolean; visible?: boolean }
export interface Grid { x0: number; y0: number; res: number; width: number; height: number }
export interface Project {
  id: string; name: string; address: string | null; lat: number | null; lon: number | null; epsg: number | null; vertical_epsg: number | null;
  horizontal_units: string; vertical_units: string; unit_system: 'metric' | 'imperial'; boundary: any | null; classes: CClass[] | null;
  settings: any; is_sample: boolean; image_count: number; image_bytes: number; images_without_gps: number;
  jobs: { running: number; queued: number; failed: number }; products: string[]; suggested_crs?: { epsg: number; basis: string } | null; storage_bytes?: number;
}
export interface Job {
  id: string; project_id: string | null; kind: string; status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'cancelling';
  progress: number; message: string | null; error: string | null; error_code: string | null; attempts: number; max_attempts: number;
  params: any; result?: any; created_at: number; started_at: number | null; finished_at: number | null; cancel_requested?: number;
}
export interface OrthoInfo {
  available: boolean; width?: number; height?: number; epsg?: number | null; georeferenced?: boolean; grid?: Grid | null; horizontal_units?: string | null;
  pyramid?: { tile: number; levels: number; ext: string; width: number; height: number; alpha: boolean }; warnings?: string[]; created_at?: number;
}
export interface ClassInfo {
  width: number; height: number; tile: number; version: number; can_undo: boolean; can_redo: boolean; classes: CClass[]; epsg: number | null;
  georeferenced: boolean; grid: Grid | null; pixel_area_m2: number | null; horizontal_units: string; notice: string;
  models: { id: string; name: string; available: boolean; description?: string; reason?: string; [k: string]: any }[]; snapshots: { id: string; label: string; created_at: number; [k: string]: any }[];
}
export interface EditBody {
  type: 'brush' | 'polygon' | 'lasso' | 'rect' | 'fill' | 'similar'; class_id: number; points?: number[][]; radius?: number; tolerance?: number;
  connectivity?: 4 | 8; rect?: number[]; analysis_window?: number[]; only_classes?: number[] | null; protect_classes?: number[] | null;
  min_area?: number; fill_holes?: boolean; smooth_radius?: number; label?: string;
}
export interface EditState { version: number; can_undo: boolean; can_redo: boolean; window: number[] | null; changed_pixels?: number; truncated?: boolean; message?: string }
export interface Product { name: string; kind: string; exists: boolean; meta: any; created_at: number; size: number | null; surface_kind?: string | null; surface_text?: string | null }
export interface TerrainState {
  available: boolean; layers: Record<string, boolean>; summary: any | null; sources: any[]; epsg: number | null; horizontal_units: string; vertical_units: string;
  pdal_available: boolean; notice: string; dense_canopy_advice: string | null;
}

export const api = {
  system: () => get('/system'),
  storage: () => get('/storage'),
  projects: () => get<Project[]>('/projects'),
  project: (id: string) => get<Project>(`/projects/${id}`),
  createProject: (b: any) => post<Project>('/projects', b),
  patchProject: (id: string, b: any) => patch<Project>(`/projects/${id}`, b),
  deleteProject: (id: string, name: string) => del(`/projects/${id}?confirm=${encodeURIComponent(name)}`),
  sample: () => post<Project>('/dev/sample-project'),
  flights: (id: string) => get(`/projects/${id}/flights`),
  footprints: (id: string) => get(`/projects/${id}/footprints`),
  reanalyze: (id: string) => post(`/projects/${id}/flights/analyze`),
  images: (id: string, q: string) => get(`/projects/${id}/images?${q}`),
  control: (id: string) => get<any[]>(`/projects/${id}/control-points`),
  importControl: (id: string, csv: string, kind: 'gcp' | 'check', replace: boolean) => post(`/projects/${id}/control-points`, { csv, kind, replace }),
  delControl: (id: string, cid: string) => del(`/projects/${id}/control-points/${cid}`),
  batches: (id: string) => get<any[]>(`/projects/${id}/batches`),
  createBatch: (id: string, label?: string) => post<{ id: string; label: string }>(`/projects/${id}/batches`, { label }),
  uploadSummary: (id: string) => get(`/projects/${id}/uploads/summary`),
  jobs: (id: string | null, q = '') => get<Job[]>(`${id ? `/projects/${id}` : ''}/jobs?${q}`),
  job: (id: string) => get<Job>(`/jobs/${id}`),
  jobLogs: (id: string, after = 0) => get<{ id: number; ts: number; level: string; message: string }[]>(`/jobs/${id}/logs?after=${after}`),
  cancelJob: (id: string) => post(`/jobs/${id}/cancel`),
  retryJob: (id: string) => post(`/jobs/${id}/retry`),
  queue: () => get('/queue'),
  pauseQueue: () => post('/queue/pause'),
  resumeQueue: () => post('/queue/resume'),
  ortho: (id: string) => get<OrthoInfo>(`/projects/${id}/ortho`),
  classInfo: (id: string) => get<ClassInfo>(`/projects/${id}/classification`),
  classEdit: (id: string, b: EditBody) => post<EditState>(`/projects/${id}/classification/edit`, b),
  classUndo: (id: string) => post<EditState>(`/projects/${id}/classification/undo`),
  classRedo: (id: string) => post<EditState>(`/projects/${id}/classification/redo`),
  classSample: (id: string, x: number, y: number, tol: number, seg: boolean) => get(`/projects/${id}/classification/sample?x=${x}&y=${y}&tolerance=${tol}${seg ? '&segment=1' : ''}`),
  classStats: (id: string) => get(`/projects/${id}/classification/stats`),
  classAuto: (id: string, b: any) => post<{ job_id: string }>(`/projects/${id}/classification/auto`, b),
  classTrain: (id: string) => post<{ job_id: string }>(`/projects/${id}/classification/train`),
  snapshotCreate: (id: string, label: string) => post(`/projects/${id}/classification/snapshots`, { label }),
  snapshotRestore: (id: string, sid: string) => post(`/projects/${id}/classification/snapshots/${sid}/restore`),
  products: (id: string) => get<Product[]>(`/projects/${id}/products`),
  terrain: (id: string) => get<TerrainState>(`/projects/${id}/terrain`),
  terrainRun: (id: string, b: any) => post<{ job_id: string }>(`/projects/${id}/terrain/run`, b),
  sourcePatch: (id: string, sid: number, b: any) => patch(`/projects/${id}/elevation-sources/${sid}`, b),
  sourceDelete: (id: string, sid: number) => del(`/projects/${id}/elevation-sources/${sid}`),
  layerStats: (id: string, layer: string) => get(`/projects/${id}/terrain/stats/${layer}`),
  inspect: (id: string, x: number, y: number) => get(`/projects/${id}/terrain/inspect?x=${x}&y=${y}`),
  sampleElev: (id: string, points: number[][]) => post(`/projects/${id}/terrain/sample`, { points }),
  profile: (id: string, line: number[][], spacing?: number) => post(`/projects/${id}/planning/profile`, { line, spacing }),
  footprintReport: (id: string, polygon: number[][], steep_deg: number) => post(`/projects/${id}/planning/footprint`, { polygon, steep_deg }),
  cutFill: (id: string, polygon: number[][], surface: any) => post(`/projects/${id}/planning/cut-fill`, { polygon, surface }),
  drainage: (id: string, minArea: number) => get(`/projects/${id}/planning/drainage?min_area=${minArea}`),
  steep: (id: string, deg: number) => get(`/projects/${id}/planning/steep?degrees=${deg}`),
  features: (id: string) => get<any[]>(`/projects/${id}/features`),
  createFeature: (id: string, b: any) => post(`/projects/${id}/features`, b),
  deleteFeature: (id: string, fid: string) => del(`/projects/${id}/features/${fid}`),
  cloudMeta: (id: string) => get(`/projects/${id}/pointcloud`),
  cloudBuild: (id: string) => post<{ job_id: string }>(`/projects/${id}/pointcloud/build`),
  cloudEditInfo: (id: string) => get(`/projects/${id}/pointcloud/edit`),
  cloudEdit: (id: string, idx: Uint32Array, cls: number) => req<any>('POST', `/projects/${id}/pointcloud/edit?cls=${cls}`, new Blob([idx.buffer as ArrayBuffer])),
  cloudEditUndo: (id: string) => post<any>(`/projects/${id}/pointcloud/edit/undo`),
  cloudEditRedo: (id: string) => post<any>(`/projects/${id}/pointcloud/edit/redo`),
  cloudEditReset: (id: string) => post<any>(`/projects/${id}/pointcloud/edit/reset`),
  imgClassState: (id: string) => get<any>(`/projects/${id}/image-classification`),
  imgClassStart: (id: string, b: any) => post<{ job_id: string; photos: number }>(`/projects/${id}/image-classification`, b),
  imgClassClear: (id: string) => del(`/projects/${id}/image-classification`),
  imgClassItems: (id: string, q: string) => get<any>(`/projects/${id}/image-classification/items?${q}`),
  imgClassDetail: (id: string, iid: string) => get<any>(`/projects/${id}/images/${iid}/classification`),
  exportCatalog: (id: string) => get(`/projects/${id}/exports/catalog`),
  exportCreate: (id: string, b: any) => post<{ job_id: string; export_id: string }>(`/projects/${id}/exports`, b),
  exports: (id: string) => get<any[]>(`/projects/${id}/exports`),
  exportGet: (id: string, eid: string) => get(`/projects/${id}/exports/${eid}`),
  exportDelete: (id: string, eid: string) => del(`/projects/${id}/exports/${eid}`),
  odmState: (id: string) => get(`/projects/${id}/photogrammetry`),
  odmStart: (id: string, b: any) => post<{ job_id: string }>(`/projects/${id}/photogrammetry`, b),
};
