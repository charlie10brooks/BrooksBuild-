// Small hand-written WebGL2 engine for the terrain viewer: orbit camera, heightfield meshes (textured or ramp-coloured),
// point clouds, draped line sets, and CPU ray picking against the heightfields. No dependencies.
import { RAMPS, rampColor } from './ramps';

export type V3 = [number, number, number];
export type M4 = Float32Array;
const mat = () => new Float32Array(16);
export function mul(a: M4, b: M4): M4 { const o = mat(); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; } return o; }
export function perspective(fovy: number, aspect: number, n: number, f: number): M4 {
  const t = 1 / Math.tan(fovy / 2), o = mat(); o[0] = t / aspect; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = (2 * f * n) / (n - f); return o;
}
export function lookAt(e: V3, c: V3, up: V3): M4 {
  let zx = e[0] - c[0], zy = e[1] - c[1], zz = e[2] - c[2]; const zl = Math.hypot(zx, zy, zz) || 1; zx /= zl; zy /= zl; zz /= zl;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx; const xl = Math.hypot(xx, xy, xz) || 1; xx /= xl; xy /= xl; xz /= xl;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  const o = mat(); o[0] = xx; o[1] = yx; o[2] = zx; o[4] = xy; o[5] = yy; o[6] = zy; o[8] = xz; o[9] = yz; o[10] = zz;
  o[12] = -(xx * e[0] + xy * e[1] + xz * e[2]); o[13] = -(yx * e[0] + yy * e[1] + yz * e[2]); o[14] = -(zx * e[0] + zy * e[1] + zz * e[2]); o[15] = 1; return o;
}
export function invert(m: M4): M4 | null {
  const a = Array.from(m), inv = new Array(16);
  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
  const det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  if (!det) return null;
  const o = mat(); for (let i = 0; i < 16; i++) o[i] = inv[i] / det; return o;
}

// ------------------------------------------------------------------ scene frame (map units -> metres around a centre)
export interface Frame { cx: number; cy: number; zref: number; mh: number; mv: number }
export const toScene = (f: Frame, x: number, y: number, z: number): V3 => [(x - f.cx) * f.mh, (y - f.cy) * f.mh, (z - f.zref) * f.mv];
export const fromScene = (f: Frame, p: V3): V3 => [p[0] / f.mh + f.cx, p[1] / f.mh + f.cy, p[2] / f.mv + f.zref];

// ------------------------------------------------------------------ heightfield (CPU side; raw vertical values, NaN = no data)
export interface Heightfield { w: number; h: number; x0: number; y0: number; res: number; native: number; data: Float32Array; kind: string; label: string; zmin: number; zmax: number; p2: number; p98: number }
export function makeHeightfield(meta: any, buf: ArrayBuffer): Heightfield {
  const data = new Float32Array(buf); const v: number[] = [];
  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < data.length; i++) { const z = data[i]; if (isFinite(z)) { if (z < zmin) zmin = z; if (z > zmax) zmax = z; if (i % 7 === 0) v.push(z); } }
  v.sort((a, b) => a - b);
  return { w: meta.width, h: meta.height, x0: meta.x0, y0: meta.y0, res: meta.res, native: meta.native_res ?? meta.res, data, kind: meta.kind, label: meta.label, zmin, zmax,
    p2: v.length ? v[Math.floor(v.length * 0.02)] : zmin, p98: v.length ? v[Math.min(v.length - 1, Math.floor(v.length * 0.98))] : zmax };
}
/** map-coordinate extent covered by the heightfield */
export const hfBounds = (h: Heightfield): [number, number, number, number] => [h.x0, h.y0 - h.h * h.res, h.x0 + h.w * h.res, h.y0];
/** bilinear sample at a map coordinate (raw vertical units) or NaN */
export function sampleHF(h: Heightfield, x: number, y: number): number {
  const fc = (x - h.x0) / h.res - 0.5, fr = (h.y0 - y) / h.res - 0.5;
  const c0 = Math.floor(fc), r0 = Math.floor(fr); if (c0 < 0 || r0 < 0 || c0 + 1 >= h.w || r0 + 1 >= h.h) { const c = Math.round(fc), r = Math.round(fr); return c >= 0 && r >= 0 && c < h.w && r < h.h ? h.data[r * h.w + c] : NaN; }
  const tx = fc - c0, ty = fr - r0, a = h.data[r0 * h.w + c0], b = h.data[r0 * h.w + c0 + 1], c = h.data[(r0 + 1) * h.w + c0], d = h.data[(r0 + 1) * h.w + c0 + 1];
  if (!isFinite(a) || !isFinite(b) || !isFinite(c) || !isFinite(d)) { const n = tx < 0.5 ? (ty < 0.5 ? a : c) : (ty < 0.5 ? b : d); return n; }
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

// ------------------------------------------------------------------ camera
export interface Cam { tx: number; ty: number; tz: number; yaw: number; pitch: number; dist: number }
export const FOV = 0.8;
export function camMatrices(cam: Cam, zexag: number, aspect: number, radius: number): { view: M4; proj: M4; vp: M4; eye: V3 } {
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const t: V3 = [cam.tx, cam.ty, cam.tz * zexag];
  const eye: V3 = [t[0] + cam.dist * Math.sin(cam.yaw) * cp, t[1] - cam.dist * Math.cos(cam.yaw) * cp, t[2] + cam.dist * sp];
  const view = lookAt(eye, t, [0, 0, 1]);
  const proj = perspective(FOV, aspect, Math.max(0.05, cam.dist * 0.01), Math.max(cam.dist * 4, radius * 6));
  return { view, proj, vp: mul(proj, view), eye };
}

// ------------------------------------------------------------------ shaders
const TERRAIN_VS = `#version 300 es
in vec3 aPos; in vec2 aSlope; in vec2 aUV;
uniform mat4 uVP; uniform float uZ, uZref, uMv;
out vec2 vUV; out float vH; out vec3 vN;
void main(){ gl_Position = uVP * vec4(aPos.xy, aPos.z*uZ, 1.0); vUV = aUV; vH = aPos.z / uMv + uZref; vN = normalize(vec3(-aSlope.x*uZ, -aSlope.y*uZ, 1.0)); }`;
const TERRAIN_FS = `#version 300 es
precision highp float;
in vec2 vUV; in float vH; in vec3 vN; out vec4 o;
uniform sampler2D uTex, uRamp; uniform int uMode; uniform float uAlpha, uShade, uVmin, uVmax; uniform vec3 uLight;
void main(){
  vec4 c;
  if(uMode==0){ c = texture(uTex, vUV); }
  else { float t = clamp((vH-uVmin)/max(uVmax-uVmin,1e-6),0.,1.); c = vec4(texture(uRamp, vec2(t,.5)).rgb, 1.); }
  float l = max(dot(normalize(vN), normalize(uLight)), 0.);
  c.rgb *= mix(1.0, 0.42 + 0.68*l, uShade);
  o = vec4(c.rgb, c.a*uAlpha);
}`;
const POINT_VS = `#version 300 es
in vec3 aPos; in float aCls; in vec3 aRgb;
uniform mat4 uVP; uniform float uZ, uZref, uMv, uSize, uScale, uVmin, uVmax; uniform int uMode; uniform sampler2D uRamp;
uniform vec3 uClsCol[32]; uniform float uClsVis[32];
out vec3 vC;
void main(){
  vec4 p = uVP * vec4(aPos.xy, aPos.z*uZ, 1.0); gl_Position = p;
  int k = int(aCls + 0.5); if(k<0||k>31) k=0;
  if(uClsVis[k] < 0.5){ gl_Position = vec4(2.,2.,2.,1.); }
  if(uMode==0) vC = aRgb; else if(uMode==1) vC = uClsCol[k];
  else { float t = clamp((aPos.z/uMv+uZref-uVmin)/max(uVmax-uVmin,1e-6),0.,1.); vC = texture(uRamp, vec2(t,.5)).rgb; }
  gl_PointSize = clamp(uSize * uScale / max(p.w, 0.1), 1.0, 14.0);
}`;
const POINT_FS = `#version 300 es
precision mediump float; in vec3 vC; out vec4 o; void main(){ vec2 d = gl_PointCoord-0.5; if(dot(d,d)>0.25) discard; o = vec4(vC,1.); }`;
const LINE_VS = `#version 300 es
in vec3 aPos; uniform mat4 uVP; uniform float uZ, uLift; void main(){ gl_Position = uVP * vec4(aPos.xy, aPos.z*uZ + uLift, 1.0); }`;
const LINE_FS = `#version 300 es
precision mediump float; uniform vec4 uColor; out vec4 o; void main(){ o = uColor; }`;

export function compile(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const mk = (t: number, s: string) => { const sh = gl.createShader(t)!; gl.shaderSource(sh, s); gl.compileShader(sh); if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(sh)); return sh; };
  const p = gl.createProgram()!; gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(p)); return p;
}

export interface PointData { n: number; pos: Float32Array; cls: Uint8Array; rgb: Uint8Array | null }
export interface LineSet { color: [number, number, number, number]; lines: Float32Array[] } // each Float32Array = x,y,z triples in SCENE coordinates (z raw)
export interface RenderOpts {
  cam: Cam; zexag: number; radius: number; zref: number; mv: number; vmin: number; vmax: number; ramp: string;
  show: { dtm: boolean; dsm: boolean; points: boolean; lines: boolean };
  alpha: { dtm: number; dsm: number }; colorBy: 'tex' | 'ramp'; shade: boolean; pointMode: 0 | 1 | 2; pointSize: number; clsVis: boolean[]; clsCol: V3[]; bg: [number, number, number];
}
interface MeshGL { vao: WebGLVertexArrayObject; vbo: WebGLBuffer; ibo: WebGLBuffer; count: number; tex: WebGLTexture | null }

export class GLPane {
  gl: WebGL2RenderingContext; private pT: WebGLProgram; private pP: WebGLProgram; private pL: WebGLProgram;
  private meshes: Record<string, MeshGL | null> = { dtm: null, dsm: null };
  hf: Record<string, Heightfield | null> = { dtm: null, dsm: null };
  private rampTex: WebGLTexture; private rampName = '';
  private pts: { vao: WebGLVertexArrayObject; n: number } | null = null;
  private lineVaos: { vao: WebGLVertexArrayObject; n: number }[] = []; private lineColor: [number, number, number, number] = [1, 0.6, 0, 1];
  private dummy: WebGLTexture;
  vp: M4 = mat(); eye: V3 = [0, 0, 0]; w = 1; h = 1;
  constructor(public canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is not available in this browser');
    this.gl = gl; this.pT = compile(gl, TERRAIN_VS, TERRAIN_FS); this.pP = compile(gl, POINT_VS, POINT_FS); this.pL = compile(gl, LINE_VS, LINE_FS);
    this.rampTex = gl.createTexture()!; this.dummy = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.dummy); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 128, 255]));
  }
  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1), w = Math.max(1, Math.round(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.w = this.canvas.clientWidth; this.h = this.canvas.clientHeight;
  }
  private setRamp(name: string) {
    if (name === this.rampName) return; this.rampName = name; const gl = this.gl; const d = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) { const c = rampColor(name in RAMPS ? name : 'terrain', i / 255); d[i * 4] = c[0]; d[i * 4 + 1] = c[1]; d[i * 4 + 2] = c[2]; d[i * 4 + 3] = 255; }
    gl.bindTexture(gl.TEXTURE_2D, this.rampTex); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, d);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  /** Upload a heightfield as a mesh. `frame` converts map units to scene metres; uv are relative to `uvBounds` (map extent of the drape texture). */
  setHeightfield(kind: 'dtm' | 'dsm', hf: Heightfield | null, f: Frame, uvBounds: [number, number, number, number]) {
    const gl = this.gl, old = this.meshes[kind];
    if (old) { gl.deleteVertexArray(old.vao); gl.deleteBuffer(old.vbo); gl.deleteBuffer(old.ibo); if (old.tex) gl.deleteTexture(old.tex); }
    this.meshes[kind] = null; this.hf[kind] = hf; if (!hf) return;
    const { w, h, data } = hf, nv = w * h, V = new Float32Array(nv * 7), cs = hf.res * f.mh;
    const at = (r: number, c: number) => { const v = data[Math.min(h - 1, Math.max(0, r)) * w + Math.min(w - 1, Math.max(0, c))]; return isFinite(v) ? (v - f.zref) * f.mv : NaN; };
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const i = r * w + c, x = hf.x0 + (c + 0.5) * hf.res, y = hf.y0 - (r + 0.5) * hf.res, z = data[i];
      const zc = isFinite(z) ? (z - f.zref) * f.mv : 0;
      const zl = at(r, c - 1), zr = at(r, c + 1), zu = at(r - 1, c), zd = at(r + 1, c);
      const sx = isFinite(zl) && isFinite(zr) ? (zr - zl) / (2 * cs) : 0, sy = isFinite(zu) && isFinite(zd) ? (zu - zd) / (2 * cs) : 0;
      V.set([(x - f.cx) * f.mh, (y - f.cy) * f.mh, zc, sx, sy, (x - uvBounds[0]) / (uvBounds[2] - uvBounds[0]), 1 - (y - uvBounds[1]) / (uvBounds[3] - uvBounds[1])], i * 7);
    }
    const idx: number[] = [];
    for (let r = 0; r < h - 1; r++) for (let c = 0; c < w - 1; c++) {
      const a = r * w + c, b = a + 1, d = a + w, e = d + 1;
      if (isFinite(data[a]) && isFinite(data[b]) && isFinite(data[d]) && isFinite(data[e])) idx.push(a, d, b, b, d, e);
    }
    const vao = gl.createVertexArray()!, vbo = gl.createBuffer()!, ibo = gl.createBuffer()!;
    gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, vbo); gl.bufferData(gl.ARRAY_BUFFER, V, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(idx), gl.STATIC_DRAW);
    const loc = (n: string) => gl.getAttribLocation(this.pT, n);
    gl.enableVertexAttribArray(loc('aPos')); gl.vertexAttribPointer(loc('aPos'), 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(loc('aSlope')); gl.vertexAttribPointer(loc('aSlope'), 2, gl.FLOAT, false, 28, 12);
    gl.enableVertexAttribArray(loc('aUV')); gl.vertexAttribPointer(loc('aUV'), 2, gl.FLOAT, false, 28, 20);
    gl.bindVertexArray(null);
    this.meshes[kind] = { vao, vbo, ibo, count: idx.length, tex: null };
  }
  meshTriangles(kind: string) { return (this.meshes[kind]?.count ?? 0) / 3; }
  /** Texture used by both meshes (draped imagery). Pass null to clear. */
  setDrape(src: TexImageSource | null) {
    const gl = this.gl; const prev = (this as any).drapeTex as WebGLTexture | undefined; if (prev) gl.deleteTexture(prev);
    if (!src) { (this as any).drapeTex = undefined; return; }
    const t = gl.createTexture()!; gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src); gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    (this as any).drapeTex = t;
  }
  setPoints(p: PointData | null) {
    const gl = this.gl; if (this.pts) { gl.deleteVertexArray(this.pts.vao); this.pts = null; } if (!p) return;
    const vao = gl.createVertexArray()!; gl.bindVertexArray(vao);
    const mk = (data: ArrayBufferView, name: string, size: number, type: number, norm: boolean) => {
      const b = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); const l = gl.getAttribLocation(this.pP, name);
      gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, type, norm, 0, 0);
    };
    mk(p.pos, 'aPos', 3, gl.FLOAT, false); mk(p.cls, 'aCls', 1, gl.UNSIGNED_BYTE, false);
    mk(p.rgb ?? new Uint8Array(p.n * 3).fill(200), 'aRgb', 3, gl.UNSIGNED_BYTE, true);
    gl.bindVertexArray(null); this.pts = { vao, n: p.n };
  }
  setLines(ls: LineSet | null) {
    const gl = this.gl; for (const l of this.lineVaos) gl.deleteVertexArray(l.vao); this.lineVaos = []; if (!ls) return; this.lineColor = ls.color;
    for (const a of ls.lines) {
      const vao = gl.createVertexArray()!; gl.bindVertexArray(vao); const b = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, a, gl.STATIC_DRAW);
      const l = gl.getAttribLocation(this.pL, 'aPos'); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 3, gl.FLOAT, false, 0, 0); gl.bindVertexArray(null);
      this.lineVaos.push({ vao, n: a.length / 3 });
    }
  }

  render(o: RenderOpts) {
    const gl = this.gl; this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height); gl.clearColor(o.bg[0], o.bg[1], o.bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const m = camMatrices(o.cam, o.zexag, this.canvas.width / this.canvas.height, o.radius); this.vp = m.vp; this.eye = m.eye;
    this.setRamp(o.ramp); gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    const drape = (this as any).drapeTex as WebGLTexture | undefined;
    gl.useProgram(this.pT);
    const U = (n: string) => gl.getUniformLocation(this.pT, n);
    gl.uniformMatrix4fv(U('uVP'), false, m.vp); gl.uniform1f(U('uZ'), o.zexag); gl.uniform1f(U('uZref'), o.zref); gl.uniform1f(U('uMv'), o.mv);
    gl.uniform1i(U('uMode'), o.colorBy === 'ramp' ? 1 : 0); gl.uniform1f(U('uShade'), o.shade ? 1 : 0); gl.uniform1f(U('uVmin'), o.vmin); gl.uniform1f(U('uVmax'), o.vmax);
    gl.uniform3f(U('uLight'), -0.45, 0.55, 0.75);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, drape ?? this.dummy); gl.uniform1i(U('uTex'), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.rampTex); gl.uniform1i(U('uRamp'), 1);
    for (const k of ['dtm', 'dsm'] as const) {
      const me = this.meshes[k]; if (!me || !o.show[k]) continue; const a = o.alpha[k];
      if (a < 0.999) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); } else { gl.disable(gl.BLEND); gl.depthMask(true); }
      gl.uniform1f(U('uAlpha'), a); gl.bindVertexArray(me.vao); gl.drawElements(gl.TRIANGLES, me.count, gl.UNSIGNED_INT, 0);
    }
    gl.depthMask(true); gl.disable(gl.BLEND);
    if (o.show.points && this.pts) {
      gl.useProgram(this.pP); const P = (n: string) => gl.getUniformLocation(this.pP, n);
      gl.uniformMatrix4fv(P('uVP'), false, m.vp); gl.uniform1f(P('uZ'), o.zexag); gl.uniform1f(P('uZref'), o.zref); gl.uniform1f(P('uMv'), o.mv); gl.uniform1i(P('uMode'), o.pointMode);
      gl.uniform1f(P('uSize'), o.pointSize); gl.uniform1f(P('uScale'), this.canvas.height / (2 * Math.tan(FOV / 2)));
      gl.uniform1f(P('uVmin'), o.vmin); gl.uniform1f(P('uVmax'), o.vmax);
      const col = new Float32Array(96), vis = new Float32Array(32).fill(1);
      for (let i = 0; i < 32; i++) { const c = o.clsCol[i] ?? [0.6, 0.6, 0.6]; col.set(c, i * 3); vis[i] = o.clsVis[i] === false ? 0 : 1; }
      gl.uniform3fv(P('uClsCol'), col); gl.uniform1fv(P('uClsVis'), vis);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.rampTex); gl.uniform1i(P('uRamp'), 1);
      gl.bindVertexArray(this.pts.vao); gl.drawArrays(gl.POINTS, 0, this.pts.n);
    }
    if (o.show.lines && this.lineVaos.length) {
      gl.useProgram(this.pL); const L = (n: string) => gl.getUniformLocation(this.pL, n);
      gl.uniformMatrix4fv(L('uVP'), false, m.vp); gl.uniform1f(L('uZ'), o.zexag); gl.uniform1f(L('uLift'), o.cam.dist * 0.004 + 0.1); gl.uniform4fv(L('uColor'), this.lineColor);
      for (const l of this.lineVaos) { gl.bindVertexArray(l.vao); gl.drawArrays(gl.LINE_STRIP, 0, l.n); }
    }
    gl.bindVertexArray(null);
  }

  /** scene point (z raw, exaggerated by zexag) -> CSS pixel coords; null when behind the camera */
  project(p: V3, zexag: number): [number, number, number] | null {
    const m = this.vp, x = p[0], y = p[1], z = p[2] * zexag;
    const cw = m[3] * x + m[7] * y + m[11] * z + m[15]; if (cw <= 1e-6) return null;
    const cx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / cw, cy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw, cz = (m[2] * x + m[6] * y + m[10] * z + m[14]) / cw;
    return [(cx * 0.5 + 0.5) * this.w, (1 - (cy * 0.5 + 0.5)) * this.h, cz];
  }
  /** Project n scene-space points (x,y,z triples) to CSS pixels in one pass; NaN where a point is behind the camera. */
  projectBuffer(pos: Float32Array, n: number, zexag: number, out: Float32Array) {
    const m = this.vp, w = this.w, h = this.h;
    for (let i = 0; i < n; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2] * zexag, cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw <= 1e-6) { out[i * 2] = NaN; out[i * 2 + 1] = NaN; continue; }
      out[i * 2] = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / cw * 0.5 + 0.5) * w;
      out[i * 2 + 1] = (1 - ((m[1] * x + m[5] * y + m[9] * z + m[13]) / cw * 0.5 + 0.5)) * h;
    }
  }
  /** world ray through a CSS pixel (z in exaggerated units) */
  ray(sx: number, sy: number): { o: V3; d: V3 } | null {
    const inv = invert(this.vp); if (!inv) return null;
    const nx = (sx / this.w) * 2 - 1, ny = 1 - (sy / this.h) * 2;
    const un = (z: number): V3 => { const v = [nx, ny, z, 1]; const o = [0, 0, 0, 0]; for (let r = 0; r < 4; r++) o[r] = inv[r] * v[0] + inv[4 + r] * v[1] + inv[8 + r] * v[2] + inv[12 + r] * v[3]; return [o[0] / o[3], o[1] / o[3], o[2] / o[3]]; };
    const a = un(-1), b = un(1), d: V3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]; const l = Math.hypot(...d) || 1;
    return { o: a, d: [d[0] / l, d[1] / l, d[2] / l] };
  }
  /** Intersect a pixel ray with a heightfield. Returns scene coords (raw z) or null. */
  pick(sx: number, sy: number, hf: Heightfield, f: Frame, zexag: number, maxDist: number): V3 | null {
    const r = this.ray(sx, sy); if (!r) return null;
    const step = Math.max(0.05, hf.res * f.mh * 0.5); let prevT = 0, prevAbove = true;
    for (let t = 0; t < maxDist; t += step) {
      const x = r.o[0] + r.d[0] * t, y = r.o[1] + r.d[1] * t, z = (r.o[2] + r.d[2] * t) / zexag;
      const m = fromScene(f, [x, y, z]); const hz = sampleHF(hf, m[0], m[1]);
      if (!isFinite(hz)) { prevAbove = true; prevT = t; continue; }
      const above = m[2] > hz;
      if (!above && prevAbove && t > 0) {
        let lo = prevT, hi = t;
        for (let i = 0; i < 18; i++) {
          const mid = (lo + hi) / 2, xm = r.o[0] + r.d[0] * mid, ym = r.o[1] + r.d[1] * mid, zm = (r.o[2] + r.d[2] * mid) / zexag;
          const mm = fromScene(f, [xm, ym, zm]), hh = sampleHF(hf, mm[0], mm[1]);
          if (isFinite(hh) && mm[2] <= hh) hi = mid; else lo = mid;
        }
        const x2 = r.o[0] + r.d[0] * hi, y2 = r.o[1] + r.d[1] * hi, mm = fromScene(f, [x2, y2, 0]), hz2 = sampleHF(hf, mm[0], mm[1]);
        return isFinite(hz2) ? [x2, y2, (hz2 - f.zref) * f.mv] : null;
      }
      prevAbove = above; prevT = t;
    }
    return null;
  }
  dispose() { const gl = this.gl; for (const k of ['dtm', 'dsm']) this.setHeightfield(k as any, null, { cx: 0, cy: 0, zref: 0, mh: 1, mv: 1 }, [0, 0, 1, 1]); this.setPoints(null); this.setLines(null); this.setDrape(null); gl.getExtension('WEBGL_lose_context')?.loseContext(); }
}
