// Tiny global store (no dependency). Selectors must return stable references (state fields), not fresh objects.
import { useSyncExternalStore } from 'react';
import type { Project } from './api';

export type View = 'dashboard' | 'upload' | 'classify' | 'terrain' | 'view3d' | 'planning' | 'exports';
export interface Toast { id: number; kind: 'info' | 'error' | 'ok'; text: string }
export interface LayerState {
  ortho: boolean; classes: boolean; dsm: boolean; dtm: boolean; ndsm: boolean; confidence: boolean; evidence: boolean;
  boundary: boolean; footprints: boolean; features: boolean; steep: boolean; drainage: boolean; cameras: boolean; control: boolean; hillshade: boolean;
}
export interface Status { left: string; mid: string; right: string }
export interface AppState {
  projectId: string | null; view: View; toasts: Toast[]; layers: LayerState; opacity: Record<string, number>;
  ramp: string; zexag: number; status: Status; leftOpen: boolean; rightOpen: boolean; queue: any | null;
  saved: { state: 'saved' | 'saving' | 'error'; version: number; msg?: string };
  undoRedo: { canUndo: boolean; canRedo: boolean; undo: (() => void) | null; redo: (() => void) | null };
  jobsTick: number; projectsTick: number; steepDeg: number; project: Project | null;
}

const defaultLayers: LayerState = { ortho: true, classes: true, dsm: false, dtm: false, ndsm: false, confidence: false, evidence: false,
  boundary: true, footprints: true, features: true, steep: false, drainage: false, cameras: false, control: true, hillshade: true };

let state: AppState = {
  projectId: (() => { try { return localStorage.getItem('ts.project'); } catch { return null; } })(),
  view: 'dashboard', toasts: [], layers: defaultLayers, opacity: { classes: 0.6, dsm: 0.85, dtm: 0.85, ndsm: 0.85, confidence: 0.6, evidence: 0.6 },
  ramp: 'terrain', zexag: 1, status: { left: '', mid: '', right: '' }, leftOpen: true, rightOpen: true, queue: null,
  saved: { state: 'saved', version: 0 }, undoRedo: { canUndo: false, canRedo: false, undo: null, redo: null }, jobsTick: 0, projectsTick: 0, steepDeg: 15, project: null,
};
const subs = new Set<() => void>();
export const getState = () => state;
export function setApp(patch: Partial<AppState> | ((s: AppState) => Partial<AppState>)) {
  state = { ...state, ...(typeof patch === 'function' ? patch(state) : patch) };
  subs.forEach((f) => f());
}
export function useApp<T>(sel: (s: AppState) => T): T {
  return useSyncExternalStore((cb) => { subs.add(cb); return () => { subs.delete(cb); }; }, () => sel(state));
}
export function setLayer(k: keyof LayerState, v: boolean) { setApp((s) => ({ layers: { ...s.layers, [k]: v } })); }
export function setOpacity(k: string, v: number) { setApp((s) => ({ opacity: { ...s.opacity, [k]: v } })); }
export function setStatus(p: Partial<Status>) {
  const s = state.status;
  if ((p.left === undefined || p.left === s.left) && (p.mid === undefined || p.mid === s.mid) && (p.right === undefined || p.right === s.right)) return;
  setApp({ status: { ...s, ...p } });
}
export function selectProject(id: string | null) {
  try { if (id) localStorage.setItem('ts.project', id); else localStorage.removeItem('ts.project'); } catch { /* storage unavailable */ }
  setApp({ projectId: id, project: null, status: { left: '', mid: '', right: '' } });
}
let tid = 1;
export function toast(kind: Toast['kind'], text: string, ms = 5000) {
  const id = tid++;
  setApp((s) => ({ toasts: [...s.toasts, { id, kind, text }] }));
  setTimeout(() => setApp((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })), kind === 'error' ? ms * 2 : ms);
}
export const bump = (k: 'jobsTick' | 'projectsTick') => setApp((s) => ({ [k]: s[k] + 1 } as any));

export const useProject = () => useApp((s) => s.project);
