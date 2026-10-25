// Colour ramps. MUST match backend/app/services/render.py _RAMPS so legends agree with server-rendered layers.
export type RGB = [number, number, number];
export const RAMPS: Record<string, [number, RGB][]> = {
  terrain: [[0, [43, 131, 186]], [0.25, [171, 221, 164]], [0.5, [255, 255, 191]], [0.75, [253, 174, 97]], [1, [215, 25, 28]]],
  hypsometric: [[0, [38, 70, 83]], [0.2, [42, 157, 143]], [0.4, [138, 177, 125]], [0.6, [233, 196, 106]], [0.8, [244, 162, 97]], [1, [231, 111, 81]]],
  viridis: [[0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]], [0.75, [94, 201, 98]], [1, [253, 231, 37]]],
  gray: [[0, [20, 20, 20]], [1, [240, 240, 240]]],
  heat: [[0, [255, 255, 204]], [0.5, [253, 141, 60]], [1, [128, 0, 38]]],
};
export const RAMP_NAMES = Object.keys(RAMPS);

export function rampColor(name: string, t: number): RGB {
  const st = RAMPS[name] ?? RAMPS.terrain; const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < st.length; i++) {
    if (x <= st[i][0]) { const [t0, a] = st[i - 1], [t1, b] = st[i]; const f = (x - t0) / Math.max(1e-9, t1 - t0); return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; }
  }
  return st[st.length - 1][1];
}
export function rampCss(name: string, dir = 'to right'): string {
  const st = RAMPS[name] ?? RAMPS.terrain;
  return `linear-gradient(${dir}, ${st.map(([t, c]) => `rgb(${c.join(',')}) ${t * 100}%`).join(', ')})`;
}

export const LAYER_KIND: Record<string, 'observed' | 'estimated' | 'derived' | 'evidence'> = {
  dsm: 'observed', odm_dsm: 'observed', dtm: 'estimated', odm_dtm: 'estimated', ndsm: 'derived', confidence: 'evidence', evidence: 'evidence',
};
export const LAYER_NAME: Record<string, string> = {
  dsm: 'Observed surface (DSM)', dtm: 'Estimated bare earth (DTM)', ndsm: 'Object height (DSM − DTM)', confidence: 'Terrain confidence', evidence: 'Elevation evidence',
  odm_dsm: 'ODM surface (DSM, unfiltered)', odm_dtm: 'ODM terrain (DTM)', ortho: 'Orthomosaic (imagery)', classes: 'Land-cover classes (visual)',
};
