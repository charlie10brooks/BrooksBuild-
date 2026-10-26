import { useEffect, useRef, useState } from 'react';
import { CClass, api, url } from '../api';
import { Cam, camMatrices, compile } from '../gl3d';
import { Notice } from '../ui';
import { bump, toast } from '../store';
import { errMsg } from '../util';

const VS = `#version 300 es
in vec3 aPos; in vec2 aUv; in float aCls; in float aSel; uniform mat4 uVP; out vec2 vUv; out vec3 vP; flat out float vCls; flat out float vSel;
void main(){ vUv = aUv; vP = aPos; vCls = aCls; vSel = aSel; gl_Position = uVP * vec4(aPos, 1.0); }`;
const FS = `#version 300 es
precision highp float; in vec2 vUv; in vec3 vP; flat in float vCls; flat in float vSel; uniform sampler2D uTex; uniform vec3 uColor; uniform float uHasTex;
uniform vec4 uPal[64]; uniform float uShowCls; uniform float uOp; out vec4 o;
void main(){ vec3 base = uHasTex > 0.5 ? texture(uTex, vUv).rgb : uColor; vec3 col = base;
  int c = int(vCls + 0.5);
  if (uShowCls > 0.5 && c > 0 && c < 64) { vec4 p = uPal[c]; if (p.a > 0.5) col = mix(base, p.rgb, uOp); }
  vec3 n = normalize(cross(dFdx(vP), dFdy(vP))); if (n.z < 0.0) n = -n; float l = 0.88 + 0.12 * dot(n, normalize(vec3(0.35, -0.45, 0.82)));
  col *= l; if (vSel > 0.5) col = mix(col, vec3(1.0, 0.9, 0.1), 0.7);
  o = vec4(col, 1.0); }`;

interface Meta { available: boolean; source_available?: boolean; reason?: string; vertices: number; triangles: number; origin: number[]; bbox_local: number[][]; groups: { texture: string | null; color: number[]; start: number; count: number }[]; aligned: boolean; alignment_note: string; label: string; created_at: number }

export type MeshTool = 'orbit' | 'lasso' | 'box';
export interface MeshCls {
  classes: CClass[]; show: boolean; opacity: number; labels: Uint8Array | null; selection: Uint8Array | null; tool: MeshTool;
  onSelect?: (idx: Uint32Array) => void;
}
export interface MeshApi { preset: (k: 'top' | 'oblique' | 'reset') => void }

const hex = (h: string): [number, number, number] => { const v = parseInt(h.replace('#', ''), 16); return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255]; };
const inPoly = (x: number, y: number, p: [number, number][]) => { let c = false; for (let i = 0, j = p.length - 1; i < p.length; j = i++) if ((p[i][1] > y) !== (p[j][1] > y) && x < ((p[j][0] - p[i][0]) * (y - p[i][1])) / (p[j][1] - p[i][1]) + p[i][0]) c = !c; return c; };

/** Orbit viewer for the photo-textured model OpenDroneMap produces. Observed imagery on observed geometry: no bare earth.
 *  Optionally tints and edits per-vertex land-cover labels (labels live on the model only). */
export function MeshView({ pid, jobsTick, onApi, cls }: { pid: string; jobsTick: number; onApi?: (a: MeshApi | null) => void; cls?: MeshCls }) {
  const cv = useRef<HTMLCanvasElement | null>(null), ov = useRef<HTMLCanvasElement | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const clsRef = useRef<MeshCls | undefined>(cls); clsRef.current = cls;
  const sync = useRef<() => void>(() => {});          // re-uploads labels/selection/palette and redraws (set by the GL effect)
  useEffect(() => { let dead = false; api.meshState(pid).then((m: any) => { if (!dead) setMeta(m); }).catch((e) => !dead && setErr(errMsg(e))); return () => { dead = true; }; }, [pid, jobsTick]);
  const key = meta?.available ? `${meta.created_at}` : '';
  useEffect(() => { sync.current(); }, [cls?.labels, cls?.selection, cls?.show, cls?.opacity, cls?.classes]);

  useEffect(() => {
    if (!meta?.available || !cv.current) return;
    let dead = false, raf = 0, ready = false; const canvas = cv.current, overlay = ov.current!;
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, preserveDrawingBuffer: true });
    if (!gl) { setErr('WebGL2 is not available in this browser'); return; }
    let prog: WebGLProgram; try { prog = compile(gl, VS, FS); } catch (e) { setErr(errMsg(e)); return; }
    const bb = meta.bbox_local, ext = [bb[1][0] - bb[0][0], bb[1][1] - bb[0][1], bb[1][2] - bb[0][2]], radius = Math.max(1, Math.hypot(ext[0], ext[1], ext[2]) / 2);
    const target = { tx: (bb[0][0] + bb[1][0]) / 2, ty: (bb[0][1] + bb[1][1]) / 2, tz: (bb[0][2] + bb[1][2]) / 2 };
    const home = (pitch: number, yaw: number): Cam => ({ ...target, yaw, pitch, dist: radius * 2.2 });
    const cam: { c: Cam } = { c: home(0.7, 0.55) };
    const textures: (WebGLTexture | null)[] = [];
    const n = meta.vertices; let pos: Float32Array | null = null;
    const vao = gl.createVertexArray()!, vb = gl.createBuffer()!, ub = gl.createBuffer()!, ib = gl.createBuffer()!, cb = gl.createBuffer()!, sb = gl.createBuffer()!;
    const lc = (name: string) => gl.getAttribLocation(prog, name);
    const palette = new Float32Array(64 * 4);
    let sketch: { tool: MeshTool; pts: [number, number][] } | null = null;
    const drawSketch = () => {
      const c = overlay.getContext('2d')!, dpr = window.devicePixelRatio || 1; overlay.width = Math.round(overlay.clientWidth * dpr); overlay.height = Math.round(overlay.clientHeight * dpr);
      c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight);
      if (!sketch || sketch.pts.length < 2) return; const p = sketch.pts; c.lineWidth = 2; c.strokeStyle = '#d9480f'; c.fillStyle = 'rgba(255,200,0,.16)'; c.beginPath();
      if (sketch.tool === 'box') c.rect(Math.min(p[0][0], p[p.length - 1][0]), Math.min(p[0][1], p[p.length - 1][1]), Math.abs(p[0][0] - p[p.length - 1][0]), Math.abs(p[0][1] - p[p.length - 1][1]));
      else { p.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.closePath(); }
      c.fill(); c.stroke();
    };
    const draw = () => {
      raf = 0; if (dead || !ready) return;
      const dpr = window.devicePixelRatio || 1, w = Math.max(2, Math.round(canvas.clientWidth * dpr)), h = Math.max(2, Math.round(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      gl.viewport(0, 0, w, h); gl.clearColor(0.86, 0.89, 0.92, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
      const { vp } = camMatrices(cam.c, 1, w / h, radius); gl.useProgram(prog); gl.bindVertexArray(vao);
      const k = clsRef.current;
      gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uVP'), false, vp);
      gl.uniform4fv(gl.getUniformLocation(prog, 'uPal'), palette); gl.uniform1f(gl.getUniformLocation(prog, 'uShowCls'), k?.show ? 1 : 0); gl.uniform1f(gl.getUniformLocation(prog, 'uOp'), k?.opacity ?? 0.6);
      meta.groups.forEach((g, i) => {
        const t = textures[i]; gl.uniform1f(gl.getUniformLocation(prog, 'uHasTex'), t ? 1 : 0); gl.uniform3f(gl.getUniformLocation(prog, 'uColor'), g.color[0], g.color[1], g.color[2]);
        if (t) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(gl.getUniformLocation(prog, 'uTex'), 0); }
        gl.drawElements(gl.TRIANGLES, g.count * 3, gl.UNSIGNED_INT, g.start * 3 * 4);
      });
    };
    const schedule = () => { if (!raf) raf = requestAnimationFrame(draw); };
    const upload = (buf: WebGLBuffer, attr: string, data: Float32Array) => {
      gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW); gl.enableVertexAttribArray(lc(attr)); gl.vertexAttribPointer(lc(attr), 1, gl.FLOAT, false, 0, 0);
    };
    sync.current = () => {
      if (!ready) return; const k = clsRef.current;
      palette.fill(0); (k?.classes ?? []).forEach((c) => { if (c.id > 0 && c.id < 64) { const [r, g, b] = hex(c.color); palette.set([r, g, b, c.visible === false ? 0 : 1], c.id * 4); } });
      const cl = new Float32Array(n), sl = new Float32Array(n); if (k?.labels && k.labels.length === n) for (let i = 0; i < n; i++) cl[i] = k.labels[i]; if (k?.selection && k.selection.length === n) for (let i = 0; i < n; i++) sl[i] = k.selection[i];
      upload(cb, 'aCls', cl); upload(sb, 'aSel', sl); schedule();
    };
    // CPU picking: project every vertex, keep those inside the sketch that are not hidden behind nearer surface (coarse z-buffer of projected vertices).
    const pick = (tool: MeshTool, pts: [number, number][]): Uint32Array => {
      if (!pos) return new Uint32Array(0);
      const W = canvas.clientWidth, H = canvas.clientHeight, { vp } = camMatrices(cam.c, 1, W / H, radius), CELL = 3, gw = Math.ceil(W / CELL), gh = Math.ceil(H / CELL);
      const sx = new Float32Array(n), sy = new Float32Array(n), dz = new Float32Array(n), zb = new Float32Array(gw * gh).fill(Infinity);
      for (let i = 0; i < n; i++) {
        const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
        const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12], cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13], cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
        dz[i] = cw; if (cw <= 0) { sx[i] = -1e9; continue; }
        sx[i] = (cx / cw * 0.5 + 0.5) * W; sy[i] = (0.5 - cy / cw * 0.5) * H;
        if (sx[i] >= 0 && sx[i] < W && sy[i] >= 0 && sy[i] < H) { const g = Math.floor(sy[i] / CELL) * gw + Math.floor(sx[i] / CELL); if (cw < zb[g]) zb[g] = cw; }
      }
      const tol = Math.max(cam.c.dist * 0.012, radius * 0.01), out: number[] = [];
      let x0 = 0, x1 = 0, y0 = 0, y1 = 0; if (tool === 'box') { x0 = Math.min(pts[0][0], pts[pts.length - 1][0]); x1 = Math.max(pts[0][0], pts[pts.length - 1][0]); y0 = Math.min(pts[0][1], pts[pts.length - 1][1]); y1 = Math.max(pts[0][1], pts[pts.length - 1][1]); }
      for (let i = 0; i < n; i++) {
        const X = sx[i], Y = sy[i]; if (X < 0 || X >= W || Y < 0 || Y >= H) continue;
        if (tool === 'box' ? !(X >= x0 && X <= x1 && Y >= y0 && Y <= y1) : !inPoly(X, Y, pts)) continue;
        if (dz[i] > zb[Math.floor(Y / CELL) * gw + Math.floor(X / CELL)] + tol) continue;   // hidden behind nearer geometry
        out.push(i);
      }
      return Uint32Array.from(out);
    };
    onApi?.({ preset: (k) => { cam.c = k === 'top' ? home(1.5, 0) : k === 'oblique' ? home(0.62, 0.55) : home(0.7, 0.55); schedule(); } });
    (async () => {
      setLoading(true);
      try {
        const buf = await (await fetch(url(`/projects/${pid}/mesh/data`), { cache: 'no-store' })).arrayBuffer(); if (dead) return;
        const t = meta.triangles; pos = new Float32Array(buf, 0, n * 3); const uv = new Float32Array(buf, n * 12, n * 2), idx = new Uint32Array(buf, n * 20, t * 3);
        gl.bindVertexArray(vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, vb); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW); gl.enableVertexAttribArray(lc('aPos')); gl.vertexAttribPointer(lc('aPos'), 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, ub); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW); gl.enableVertexAttribArray(lc('aUv')); gl.vertexAttribPointer(lc('aUv'), 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
        const cache: Record<string, WebGLTexture> = {};
        for (const g of meta.groups) {
          if (!g.texture) { textures.push(null); continue; }
          if (!cache[g.texture]) {
            const im = new Image(); im.src = url(`/projects/${pid}/mesh/texture/${g.texture}?v=${meta.created_at}`); await im.decode(); if (dead) return;
            const tx = gl.createTexture()!; gl.bindTexture(gl.TEXTURE_2D, tx); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, im); gl.generateMipmap(gl.TEXTURE_2D);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            cache[g.texture] = tx;
          }
          textures.push(cache[g.texture]);
        }
        ready = true; sync.current();
      } catch (e) { if (!dead) setErr(errMsg(e)); } finally { if (!dead) setLoading(false); }
    })();
    // interaction. Orbit tool: drag = orbit. Lasso/box tool: drag = select, Alt-drag = orbit. Right-drag or Shift-drag = pan always. Wheel = zoom.
    let d: { x: number; y: number; mode: 'orbit' | 'pan' | 'select' } | null = null;
    const rel = (e: PointerEvent): [number, number] => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const down = (e: PointerEvent) => {
      const tool = clsRef.current?.tool ?? 'orbit';
      const mode = e.button === 2 || e.shiftKey ? 'pan' : tool !== 'orbit' && !e.altKey ? 'select' : 'orbit';
      d = { x: e.clientX, y: e.clientY, mode }; canvas.setPointerCapture(e.pointerId);
      if (mode === 'select') { sketch = { tool: tool as MeshTool, pts: [rel(e)] }; drawSketch(); }
    };
    const move = (e: PointerEvent) => {
      if (!d) return;
      if (d.mode === 'select' && sketch) { const q = rel(e); if (sketch.tool === 'box') sketch.pts = [sketch.pts[0], q]; else sketch.pts.push(q); drawSketch(); return; }
      const dx = e.clientX - d.x, dy = e.clientY - d.y; d.x = e.clientX; d.y = e.clientY; const c = cam.c;
      if (d.mode === 'orbit') { c.yaw -= dx * 0.0065; c.pitch = Math.min(1.5, Math.max(0.06, c.pitch + dy * 0.0055)); }
      else { const k = (c.dist * Math.tan(0.4) * 2) / Math.max(1, canvas.clientHeight), sc = 1 / Math.max(0.25, Math.sin(c.pitch)); c.tx += (-Math.cos(c.yaw) * dx - Math.sin(c.yaw) * dy * sc) * k; c.ty += (-Math.sin(c.yaw) * dx + Math.cos(c.yaw) * dy * sc) * k; }
      schedule();
    };
    const up = () => {
      if (d?.mode === 'select' && sketch) {
        const s = sketch; sketch = null; drawSketch();
        const big = s.tool === 'box' ? s.pts.length > 1 && Math.abs(s.pts[0][0] - s.pts[1][0]) > 3 && Math.abs(s.pts[0][1] - s.pts[1][1]) > 3 : s.pts.length > 2;
        if (big) clsRef.current?.onSelect?.(pick(s.tool, s.pts));
      }
      d = null;
    };
    const wheel = (e: WheelEvent) => { e.preventDefault(); cam.c.dist = Math.min(radius * 12, Math.max(radius * 0.02, cam.c.dist * Math.exp(Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 120) * 0.0014))); schedule(); };
    canvas.addEventListener('pointerdown', down); canvas.addEventListener('pointermove', move); canvas.addEventListener('pointerup', up); canvas.addEventListener('wheel', wheel, { passive: false });
    const ro = new ResizeObserver(schedule); ro.observe(canvas);
    return () => {
      dead = true; sync.current = () => {}; if (raf) cancelAnimationFrame(raf); ro.disconnect(); onApi?.(null);
      canvas.removeEventListener('pointerdown', down); canvas.removeEventListener('pointermove', move); canvas.removeEventListener('pointerup', up); canvas.removeEventListener('wheel', wheel);
      textures.forEach((t) => t && gl.deleteTexture(t)); [vb, ub, ib, cb, sb].forEach((b) => gl.deleteBuffer(b)); gl.deleteVertexArray(vao); gl.deleteProgram(prog);
    };
    // eslint-disable-next-line
  }, [pid, key]);

  const prepare = async () => { setBusy(true); try { await api.meshBuild(pid); bump('jobsTick'); toast('info', 'Preparing the textured model for the viewer'); } catch (e) { toast('error', errMsg(e)); } finally { setBusy(false); } };
  if (err) return <div className="m-auto max-w-[520px] p-4"><Notice kind="warn"><b>Photo model unavailable.</b> {err}</Notice></div>;
  if (!meta) return <div className="m-auto p-4 text-[12px] text-slate-600">Checking for a photo-textured model…</div>;
  if (!meta.available) return (
    <div className="m-auto max-w-[520px] p-4">
      <Notice kind="info"><b>No 3D model yet.</b> {meta.source_available ? 'The model from your photos is ready to be prepared for viewing.' : 'Add your photos and press Process on the Image tab.'}</Notice>
      {meta.source_available && <button className="btn btn-primary mt-2" disabled={busy} onClick={prepare}>{busy ? 'Starting…' : 'Prepare the 3D model'}</button>}
    </div>);
  const selecting = (cls?.tool ?? 'orbit') !== 'orbit';
  return (
    <div className="relative min-w-0 flex-1">
      <canvas ref={cv} className="absolute inset-0 h-full w-full" style={{ cursor: selecting ? 'crosshair' : 'grab', touchAction: 'none' }} onContextMenu={(e) => e.preventDefault()} />
      <canvas ref={ov} className="pointer-events-none absolute inset-0 h-full w-full" />
      <span className="pointer-events-none absolute left-2 top-2 rounded bg-sky-800 px-1.5 py-0.5 text-[11px] font-bold text-white">PHOTO MODEL (observed) · {meta.triangles.toLocaleString()} triangles{cls?.show ? ' · land-cover labels' : ''}</span>
      {loading && <span className="pointer-events-none absolute left-2 top-9 rounded bg-white/90 px-1.5 py-0.5 text-[11px]">Loading model and textures…</span>}
      {!meta.aligned && <div className="absolute bottom-8 left-2 right-2 z-10"><Notice kind="warn">{meta.alignment_note}</Notice></div>}
      <div className="pointer-events-none absolute bottom-2 left-2 rounded bg-white/90 px-1.5 py-0.5 text-[11px] text-slate-600">{selecting ? 'Drag = select · Alt-drag = orbit · right-drag / Shift = pan · wheel = zoom' : 'Drag = orbit · right-drag / Shift = pan · wheel = zoom'} · observed model, not bare earth</div>
    </div>
  );
}
