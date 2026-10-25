// 2D viewport in MAP coordinates (x east, y north). `ppu` = screen pixels per map unit.
export class Viewport {
  cx = 0; cy = 0; ppu = 1; w = 1; h = 1;
  clone(): Viewport { const v = new Viewport(); Object.assign(v, this); return v; }
  toScreen(x: number, y: number): [number, number] { return [(x - this.cx) * this.ppu + this.w / 2, (this.cy - y) * this.ppu + this.h / 2]; }
  toMap(sx: number, sy: number): [number, number] { return [this.cx + (sx - this.w / 2) / this.ppu, this.cy - (sy - this.h / 2) / this.ppu]; }
  bbox(): [number, number, number, number] {
    const [x0, y1] = this.toMap(0, 0), [x1, y0] = this.toMap(this.w, this.h); return [x0, y0, x1, y1];
  }
  zoomAt(sx: number, sy: number, f: number, min: number, max: number) {
    const [mx, my] = this.toMap(sx, sy);
    this.ppu = Math.min(max, Math.max(min, this.ppu * f));
    this.cx = mx - (sx - this.w / 2) / this.ppu; this.cy = my + (sy - this.h / 2) / this.ppu;
  }
  panPx(dx: number, dy: number) { this.cx -= dx / this.ppu; this.cy += dy / this.ppu; }
  fit(b: [number, number, number, number], pad = 0.9) {
    this.cx = (b[0] + b[2]) / 2; this.cy = (b[1] + b[3]) / 2;
    this.ppu = Math.min(this.w / Math.max(1e-9, b[2] - b[0]), this.h / Math.max(1e-9, b[3] - b[1])) * pad;
  }
}
