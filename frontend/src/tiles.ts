import { CClass, Grid, OrthoInfo, url } from './api';
import { Viewport } from './viewport';

// ------------------------------------------------------------------ image cache
interface Entry { img?: HTMLImageElement; state: 'loading' | 'ok' | 'err' }
class ImageCache {
  private m = new Map<string, Entry>();
  constructor(private max: number) {}
  peek(u: string): Entry | undefined { const e = this.m.get(u); if (e) { this.m.delete(u); this.m.set(u, e); } return e; }
  get(u: string, onLoad: () => void): Entry {
    const have = this.peek(u); if (have) return have;
    const e: Entry = { state: 'loading' }; this.m.set(u, e);
    const img = new Image(); img.decoding = 'async';
    img.onload = () => { e.img = img; e.state = 'ok'; onLoad(); };
    img.onerror = () => { e.state = 'err'; };
    img.src = u;
    if (this.m.size > this.max) { const k = this.m.keys().next().value; if (k !== undefined) this.m.delete(k); }
    return e;
  }
}
export const imageCache = new ImageCache(900);
export const fallbackGrid = (w: number, h: number): Grid => ({ x0: 0, y0: h, res: 1, width: w, height: h });

// ------------------------------------------------------------------ orthomosaic pyramid
export class OrthoLayer {
  grid: Grid;
  constructor(public pid: string, public info: OrthoInfo, private onLoad: () => void) {
    this.grid = info.grid ?? fallbackGrid(info.width!, info.height!);
  }
  bounds(): [number, number, number, number] { const g = this.grid; return [g.x0, g.y0 - g.height * g.res, g.x0 + g.width * g.res, g.y0]; }
  private tileUrl(z: number, x: number, y: number) { return url(`/projects/${this.pid}/ortho/tiles/${z}/${x}/${y}?v=${this.info.created_at ?? 0}`); }
  levelFor(vp: Viewport): number { const s0 = vp.ppu * this.grid.res; return Math.max(0, Math.min(this.info.pyramid!.levels - 1, Math.floor(Math.log2(1 / s0)))); }
  draw(c: CanvasRenderingContext2D, vp: Viewport, alpha = 1) {
    const g = this.grid, py = this.info.pyramid!, T = py.tile, L = py.levels;
    const z = this.levelFor(vp), span = T * 2 ** z; // image px per tile at level z
    const [bx0, by0, bx1, by1] = vp.bbox();
    const ix0 = (bx0 - g.x0) / g.res, ix1 = (bx1 - g.x0) / g.res, iy0 = (g.y0 - by1) / g.res, iy1 = (g.y0 - by0) / g.res;
    const tx0 = Math.max(0, Math.floor(ix0 / span)), tx1 = Math.min(Math.ceil(g.width / span) - 1, Math.floor(ix1 / span));
    const ty0 = Math.max(0, Math.floor(iy0 / span)), ty1 = Math.min(Math.ceil(g.height / span) - 1, Math.floor(iy1 / span));
    c.save(); c.globalAlpha = alpha; c.imageSmoothingEnabled = vp.ppu * g.res < 2;
    const sPx = vp.ppu * g.res;
    for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) {
      const e = imageCache.get(this.tileUrl(z, tx, ty), this.onLoad);
      const dx = (g.x0 + tx * span * g.res - vp.cx) * vp.ppu + vp.w / 2, dy = (vp.cy - g.y0 + ty * span * g.res) * vp.ppu + vp.h / 2;
      if (e.state === 'ok' && e.img) {
        c.drawImage(e.img, Math.floor(dx), Math.floor(dy), Math.ceil(e.img.width * 2 ** z * sPx) + 1, Math.ceil(e.img.height * 2 ** z * sPx) + 1);
      } else if (e.state !== 'err') {
        for (let zz = z + 1; zz < L; zz++) { // show a coarser ancestor while this tile loads
          const f = 2 ** (zz - z), pe = imageCache.peek(this.tileUrl(zz, tx >> (zz - z), ty >> (zz - z)));
          if (pe?.state === 'ok' && pe.img) {
            const sub = T / f, sx = (tx % f) * sub, sy = (ty % f) * sub;
            if (sx < pe.img.width && sy < pe.img.height) c.drawImage(pe.img, sx, sy, Math.min(sub, pe.img.width - sx), Math.min(sub, pe.img.height - sy), Math.floor(dx), Math.floor(dy), Math.ceil(Math.min(sub, pe.img.width - sx) * f * 2 ** z * sPx) + 1, Math.ceil(Math.min(sub, pe.img.height - sy) * f * 2 ** z * sPx) + 1);
            break;
          }
        }
      }
    }
    c.restore();
  }
}

// ------------------------------------------------------------------ classification mask (PNG: R=class, G=provenance 0/1/2, B=confidence 0..100)
export type MaskMode = 'class' | 'confidence' | 'provenance';
interface MTile { ver: number; raw: ImageData; canvas: HTMLCanvasElement; styleKey: string }

export class MaskLayer {
  version = 0;
  mode: MaskMode = 'class';
  private lut = new Uint8ClampedArray(256 * 3);
  private vis = new Uint8Array(256);
  private styleKey = '';
  private tiles = new Map<string, MTile>();
  private pending = new Set<string>();
  constructor(public pid: string, public width: number, public height: number, public grid: Grid, private onLoad: () => void) {}

  setClasses(classes: CClass[], hidden: Set<number>) {
    this.lut.fill(0); this.vis.fill(0);
    for (const c of classes) {
      if (c.id === 0 || c.id > 255) continue;
      const h = c.color.replace('#', ''); this.lut[c.id * 3] = parseInt(h.slice(0, 2), 16); this.lut[c.id * 3 + 1] = parseInt(h.slice(2, 4), 16); this.lut[c.id * 3 + 2] = parseInt(h.slice(4, 6), 16);
      this.vis[c.id] = hidden.has(c.id) ? 0 : 1;
    }
    this.styleKey = `${this.mode}|${classes.map((c) => c.id + c.color).join()}|${[...hidden].sort().join()}`;
  }
  setMode(m: MaskMode) { this.mode = m; this.styleKey = this.styleKey.replace(/^[a-z]+\|/, m + '|'); }
  setVersion(v: number) { this.version = v; }
  invalidateAll() { this.tiles.clear(); }

  private colorize(raw: ImageData): HTMLCanvasElement {
    const cv = document.createElement('canvas'); cv.width = raw.width; cv.height = raw.height;
    const out = new ImageData(raw.width, raw.height), s = raw.data, d = out.data;
    for (let i = 0; i < s.length; i += 4) {
      const cls = s[i], prov = s[i + 1], conf = s[i + 2];
      if (cls === 0 || !this.vis[cls]) continue;
      if (this.mode === 'class') { d[i] = this.lut[cls * 3]; d[i + 1] = this.lut[cls * 3 + 1]; d[i + 2] = this.lut[cls * 3 + 2]; d[i + 3] = 255; }
      else if (this.mode === 'provenance') {
        const col = prov === 1 ? [30, 100, 220] : prov === 2 ? [240, 150, 20] : [130, 130, 130]; d[i] = col[0]; d[i + 1] = col[1]; d[i + 2] = col[2]; d[i + 3] = 255;
      } else { // confidence: manual = blue, automatic = red(low) .. green(high), untouched = grey
        if (prov === 1) { d[i] = 30; d[i + 1] = 100; d[i + 2] = 220; }
        else if (prov === 2) { const t = Math.min(1, conf / 100); d[i] = Math.round(220 * (1 - t) + 40 * t); d[i + 1] = Math.round(60 * (1 - t) + 180 * t); d[i + 2] = 50; }
        else { d[i] = d[i + 1] = d[i + 2] = 140; }
        d[i + 3] = 255;
      }
    }
    cv.getContext('2d')!.putImageData(out, 0, 0);
    return cv;
  }
  private load(key: string, u: string, ver: number) {
    if (this.pending.has(key + '@' + ver)) return;
    this.pending.add(key + '@' + ver);
    const img = new Image();
    img.onload = () => {
      this.pending.delete(key + '@' + ver);
      const t = document.createElement('canvas'); t.width = img.width; t.height = img.height;
      const tc = t.getContext('2d', { willReadFrequently: true })!; tc.imageSmoothingEnabled = false; tc.drawImage(img, 0, 0);
      const raw = tc.getImageData(0, 0, img.width, img.height);
      const old = this.tiles.get(key);
      if (!old || old.ver <= ver) this.tiles.set(key, { ver, raw, canvas: this.colorize(raw), styleKey: this.styleKey });
      if (this.tiles.size > 220) { const k = this.tiles.keys().next().value; if (k !== undefined && k !== key) this.tiles.delete(k); }
      this.onLoad();
    };
    img.onerror = () => { this.pending.delete(key + '@' + ver); };
    img.src = u;
  }
  private get(key: string, u: string): MTile | undefined {
    const t = this.tiles.get(key);
    if (!t || t.ver !== this.version) this.load(key, u + (u.includes('?') ? '&' : '?') + 'v=' + this.version, this.version);
    if (t && t.styleKey !== this.styleKey) { t.canvas = this.colorize(t.raw); t.styleKey = this.styleKey; }
    return t;
  }
  draw(c: CanvasRenderingContext2D, vp: Viewport, alpha: number) {
    const g = this.grid, s0 = vp.ppu * g.res, lz = Math.floor(Math.log2(1 / s0));
    c.save(); c.globalAlpha = alpha; c.imageSmoothingEnabled = false;
    if (lz > 2) { // zoomed far out: one overview image
      const t = this.get('ov', url(`/projects/${this.pid}/classification/overview?size=2048`));
      if (t) { const s = Math.ceil(Math.max(this.width, this.height) / 2048); const [dx, dy] = vp.toScreen(g.x0, g.y0); c.drawImage(t.canvas, dx, dy, t.canvas.width * s * g.res * vp.ppu, t.canvas.height * s * g.res * vp.ppu); }
      c.restore(); return;
    }
    const m = Math.max(0, lz), span = 256 * 2 ** m;
    const [bx0, by0, bx1, by1] = vp.bbox();
    const ix0 = (bx0 - g.x0) / g.res, ix1 = (bx1 - g.x0) / g.res, iy0 = (g.y0 - by1) / g.res, iy1 = (g.y0 - by0) / g.res;
    const tx0 = Math.max(0, Math.floor(ix0 / span)), tx1 = Math.min(Math.ceil(this.width / span) - 1, Math.floor(ix1 / span));
    const ty0 = Math.max(0, Math.floor(iy0 / span)), ty1 = Math.min(Math.ceil(this.height / span) - 1, Math.floor(iy1 / span));
    for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) {
      const t = this.get(`${m}/${tx}/${ty}`, url(`/projects/${this.pid}/classification/tiles/${m}/${tx}/${ty}`));
      if (!t) continue;
      const [dx, dy] = vp.toScreen(g.x0 + tx * span * g.res, g.y0 - ty * span * g.res);
      c.drawImage(t.canvas, Math.floor(dx), Math.floor(dy), Math.ceil(t.canvas.width * 2 ** m * g.res * vp.ppu) + 1, Math.ceil(t.canvas.height * 2 ** m * g.res * vp.ppu) + 1);
    }
    c.restore();
  }
  /** Class / provenance / confidence of one raster pixel if its tile is in memory (level-0 tile), else null. */
  probe(px: number, py: number): { cls: number; prov: number; conf: number } | null {
    const t = this.tiles.get(`0/${Math.floor(px / 256)}/${Math.floor(py / 256)}`); if (!t) return null;
    const x = Math.floor(px) % 256, y = Math.floor(py) % 256; const i = (y * t.raw.width + x) * 4;
    if (x >= t.raw.width || y >= t.raw.height) return null;
    return { cls: t.raw.data[i], prov: t.raw.data[i + 1], conf: t.raw.data[i + 2] };
  }
}
