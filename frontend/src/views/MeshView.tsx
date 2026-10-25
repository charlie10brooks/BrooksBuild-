import { useEffect, useRef, useState } from 'react';
import { api, url } from '../api';
import { Cam, camMatrices, compile } from '../gl3d';
import { Notice } from '../ui';
import { bump, toast } from '../store';
import { errMsg } from '../util';

const VS = `#version 300 es
in vec3 aPos; in vec2 aUv; uniform mat4 uVP; out vec2 vUv; out vec3 vP; void main(){ vUv = aUv; vP = aPos; gl_Position = uVP * vec4(aPos, 1.0); }`;
const FS = `#version 300 es
precision highp float; in vec2 vUv; in vec3 vP; uniform sampler2D uTex; uniform vec3 uColor; uniform float uHasTex; out vec4 o;
void main(){ vec3 base = uHasTex > 0.5 ? texture(uTex, vUv).rgb : uColor;
  vec3 n = normalize(cross(dFdx(vP), dFdy(vP))); if (n.z < 0.0) n = -n; float l = 0.88 + 0.12 * dot(n, normalize(vec3(0.35, -0.45, 0.82)));
  o = vec4(base * l, 1.0); }`;

interface Meta { available: boolean; source_available?: boolean; reason?: string; vertices: number; triangles: number; origin: number[]; bbox_local: number[][]; groups: { texture: string | null; color: number[]; start: number; count: number }[]; aligned: boolean; alignment_note: string; label: string; created_at: number }

/** Orbit viewer for the photo-textured model OpenDroneMap produces. Observed imagery on observed geometry: no bare earth, no editing. */
export function MeshView({ pid, jobsTick, setCamApi }: { pid: string; jobsTick: number; setCamApi?: (a: { preset: (k: 'top' | 'oblique' | 'reset') => void } | null) => void }) {
  const cv = useRef<HTMLCanvasElement | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  useEffect(() => { let dead = false; api.meshState(pid).then((m: any) => { if (!dead) setMeta(m); }).catch((e) => !dead && setErr(errMsg(e))); return () => { dead = true; }; }, [pid, jobsTick]);
  const key = meta?.available ? `${meta.created_at}` : '';

  useEffect(() => {
    if (!meta?.available || !cv.current) return;
    let dead = false, raf = 0; const canvas = cv.current;
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, preserveDrawingBuffer: true });
    if (!gl) { setErr('WebGL2 is not available in this browser'); return; }
    let prog: WebGLProgram; try { prog = compile(gl, VS, FS); } catch (e) { setErr(errMsg(e)); return; }
    const bb = meta.bbox_local, ext = [bb[1][0] - bb[0][0], bb[1][1] - bb[0][1], bb[1][2] - bb[0][2]], radius = Math.max(1, Math.hypot(ext[0], ext[1], ext[2]) / 2);
    const target = { tx: (bb[0][0] + bb[1][0]) / 2, ty: (bb[0][1] + bb[1][1]) / 2, tz: (bb[0][2] + bb[1][2]) / 2 };
    const home = (pitch: number, yaw: number): Cam => ({ ...target, yaw, pitch, dist: radius * 2.2 });
    const cam: { c: Cam } = { c: home(0.7, 0.55) };
    const textures: (WebGLTexture | null)[] = [];
    let ready = false;
    const vao = gl.createVertexArray()!, vb = gl.createBuffer()!, ub = gl.createBuffer()!, ib = gl.createBuffer()!;
    const draw = () => {
      raf = 0; if (dead || !ready) return;
      const dpr = window.devicePixelRatio || 1, w = Math.max(2, Math.round(canvas.clientWidth * dpr)), h = Math.max(2, Math.round(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      gl.viewport(0, 0, w, h); gl.clearColor(0.86, 0.89, 0.92, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
      const { vp } = camMatrices(cam.c, 1, w / h, radius); gl.useProgram(prog); gl.bindVertexArray(vao);
      gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uVP'), false, vp);
      meta.groups.forEach((g, i) => {
        const t = textures[i]; gl.uniform1f(gl.getUniformLocation(prog, 'uHasTex'), t ? 1 : 0); gl.uniform3f(gl.getUniformLocation(prog, 'uColor'), g.color[0], g.color[1], g.color[2]);
        if (t) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(gl.getUniformLocation(prog, 'uTex'), 0); }
        gl.drawElements(gl.TRIANGLES, g.count * 3, gl.UNSIGNED_INT, g.start * 3 * 4);
      });
    };
    const schedule = () => { if (!raf) raf = requestAnimationFrame(draw); };
    setCamApi?.({ preset: (k) => { cam.c = k === 'top' ? home(1.5, 0) : k === 'oblique' ? home(0.62, 0.55) : home(0.7, 0.55); schedule(); } });
    (async () => {
      setLoading(true);
      try {
        const buf = await (await fetch(url(`/projects/${pid}/mesh/data`), { cache: 'no-store' })).arrayBuffer(); if (dead) return;
        const n = meta.vertices, t = meta.triangles;
        const pos = new Float32Array(buf, 0, n * 3), uv = new Float32Array(buf, n * 12, n * 2), idx = new Uint32Array(buf, n * 20, t * 3);
        gl.bindVertexArray(vao);
        const loc = (name: string) => gl.getAttribLocation(prog, name);
        gl.bindBuffer(gl.ARRAY_BUFFER, vb); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW); gl.enableVertexAttribArray(loc('aPos')); gl.vertexAttribPointer(loc('aPos'), 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, ub); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW); gl.enableVertexAttribArray(loc('aUv')); gl.vertexAttribPointer(loc('aUv'), 2, gl.FLOAT, false, 0, 0);
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
        ready = true; schedule();
      } catch (e) { if (!dead) setErr(errMsg(e)); } finally { if (!dead) setLoading(false); }
    })();
    // interaction: drag = orbit, right-drag or Shift-drag = pan, wheel = zoom
    let d: { x: number; y: number; pan: boolean } | null = null;
    const down = (e: PointerEvent) => { d = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; canvas.setPointerCapture(e.pointerId); };
    const move = (e: PointerEvent) => {
      if (!d) return; const dx = e.clientX - d.x, dy = e.clientY - d.y; d.x = e.clientX; d.y = e.clientY; const c = cam.c;
      if (!d.pan) { c.yaw -= dx * 0.0065; c.pitch = Math.min(1.5, Math.max(0.06, c.pitch + dy * 0.0055)); }
      else { const k = (c.dist * Math.tan(0.4) * 2) / Math.max(1, canvas.clientHeight), sc = 1 / Math.max(0.25, Math.sin(c.pitch)); c.tx += (-Math.cos(c.yaw) * dx - Math.sin(c.yaw) * dy * sc) * k; c.ty += (-Math.sin(c.yaw) * dx + Math.cos(c.yaw) * dy * sc) * k; }
      schedule();
    };
    const up = () => { d = null; };
    const wheel = (e: WheelEvent) => { e.preventDefault(); cam.c.dist = Math.min(radius * 12, Math.max(radius * 0.02, cam.c.dist * Math.exp(Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 120) * 0.0014))); schedule(); };
    canvas.addEventListener('pointerdown', down); canvas.addEventListener('pointermove', move); canvas.addEventListener('pointerup', up); canvas.addEventListener('wheel', wheel, { passive: false });
    const ro = new ResizeObserver(schedule); ro.observe(canvas);
    return () => {
      dead = true; if (raf) cancelAnimationFrame(raf); ro.disconnect(); setCamApi?.(null);
      canvas.removeEventListener('pointerdown', down); canvas.removeEventListener('pointermove', move); canvas.removeEventListener('pointerup', up); canvas.removeEventListener('wheel', wheel);
      textures.forEach((t) => t && gl.deleteTexture(t)); gl.deleteBuffer(vb); gl.deleteBuffer(ub); gl.deleteBuffer(ib); gl.deleteVertexArray(vao); gl.deleteProgram(prog);
    };
    // eslint-disable-next-line
  }, [pid, key]);

  const prepare = async () => { setBusy(true); try { await api.meshBuild(pid); bump('jobsTick'); toast('info', 'Preparing the textured model for the viewer'); } catch (e) { toast('error', errMsg(e)); } finally { setBusy(false); } };
  if (err) return <div className="m-auto max-w-[520px] p-4"><Notice kind="warn"><b>Photo model unavailable.</b> {err}</Notice></div>;
  if (!meta) return <div className="m-auto p-4 text-[12px] text-slate-600">Checking for a photo-textured model…</div>;
  if (!meta.available) return (
    <div className="m-auto max-w-[520px] p-4"><Notice kind="info"><b>No photo-textured model yet.</b> {meta.reason}</Notice>
      {meta.source_available && <button className="btn btn-primary mt-2" disabled={busy} onClick={prepare}>{busy ? 'Starting…' : 'Prepare the textured model'}</button>}
      <div className="mt-2 text-[12px] text-slate-600">This is the model you can orbit like a Polycam capture. It comes from OpenDroneMap&apos;s texturing step, which needs the <b>Balanced</b> or <b>High</b> quality setting on the Upload tab.</div></div>);
  return (
    <div className="relative min-w-0 flex-1">
      <canvas ref={cv} className="absolute inset-0 h-full w-full" style={{ cursor: 'grab', touchAction: 'none' }} onContextMenu={(e) => e.preventDefault()} />
      <span className="pointer-events-none absolute left-2 top-2 rounded bg-sky-800 px-1.5 py-0.5 text-[11px] font-bold text-white">PHOTO MODEL (observed) · {meta.triangles.toLocaleString()} triangles</span>
      {loading && <span className="pointer-events-none absolute left-2 top-9 rounded bg-white/90 px-1.5 py-0.5 text-[11px]">Loading model and textures…</span>}
      {!meta.aligned && <div className="absolute bottom-8 left-2 right-2 z-10"><Notice kind="warn">{meta.alignment_note}</Notice></div>}
      <div className="pointer-events-none absolute bottom-2 left-2 rounded bg-white/90 px-1.5 py-0.5 text-[11px] text-slate-600">Drag = orbit · right-drag / Shift = pan · wheel = zoom · observed model, not bare earth</div>
    </div>
  );
}
