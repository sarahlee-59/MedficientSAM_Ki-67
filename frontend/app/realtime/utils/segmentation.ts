import {
  CANVAS_SIZE,
  SHAPE_DIM_MIN,
  SHAPE_DIM_MAX,
  SUPPORTED_EXTS,
  type CellMaskSource,
  type Point,
} from "../types";

export function imageToRgbArray(image: HTMLImageElement) {
  const w = image.naturalWidth;
  const h = image.naturalHeight;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("Canvas context 생성 실패");
  ctx.drawImage(image, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4) {
    rgb[j++] = rgba[i];
    rgb[j++] = rgba[i + 1];
    rgb[j++] = rgba[i + 2];
  }
  return { rgb, w, h };
}

export function chaikin(
  pts: [number, number][],
  iterations = 2,
): [number, number][] {
  let p = pts;
  for (let i = 0; i < iterations; i++) {
    const next: [number, number][] = [];
    for (let j = 0; j < p.length; j++) {
      const a = p[j],
        b = p[(j + 1) % p.length];
      next.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
      next.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
    }
    p = next;
  }
  return p;
}

export function smoothPolyline(
  pts: [number, number][],
  window = 3,
): [number, number][] {
  const n = pts.length;
  const half = Math.floor(window / 2);
  return pts.map((_, i) => {
    let sumX = 0,
      sumY = 0;
    for (let k = -half; k <= half; k++) {
      const p = pts[(i + k + n) % n];
      sumX += p[0];
      sumY += p[1];
    }
    return [sumX / window, sumY / window] as [number, number];
  });
}

export function maskToPolyline(
  mask: Uint8Array,
  w: number,
  h: number,
): [number, number][] {
  const DX = [1, 1, 0, -1, -1, -1, 0, 1];
  const DY = [0, 1, 1, 1, 0, -1, -1, -1];
  const isFg = (x: number, y: number) =>
    x >= 0 && x < w && y >= 0 && y < h && mask[y * w + x] > 0;

  let sx = -1,
    sy = -1;
  outer: for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] > 0) {
        sx = x;
        sy = y;
        break outer;
      }
    }
  }
  if (sx < 0) return [];

  const contour: [number, number][] = [[sx, sy]];
  let x = sx,
    y = sy,
    backDir = 4;

  for (let step = 0; step < w * h * 2; step++) {
    let moved = false;
    for (let i = 1; i <= 8; i++) {
      const d = (backDir + i) % 8;
      const nx = x + DX[d],
        ny = y + DY[d];
      if (isFg(nx, ny)) {
        x = nx;
        y = ny;
        backDir = (d + 4) % 8;
        moved = true;
        break;
      }
    }
    if (!moved || (x === sx && y === sy)) break;
    contour.push([x, y]);
  }

  if (contour.length < 3) return [];
  const smoothed = smoothPolyline(contour, 3);
  const skip = Math.max(1, Math.floor(smoothed.length / 60));
  const sampled: [number, number][] = [];
  for (let i = 0; i < smoothed.length; i += skip) sampled.push(smoothed[i]);
  return chaikin(sampled, 3);
}

export function buildOccupiedMask(
  cells: CellMaskSource[],
  excludeCellId: number | null,
  w: number,
  h: number,
): Uint8Array {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (!ctx) return new Uint8Array(w * h);
  ctx.fillStyle = "#ffffff";
  for (const cell of cells) {
    if (cell.id === excludeCellId) continue;
    if (cell.pending || cell.polyline.length < 3) continue;
    ctx.beginPath();
    ctx.moveTo(cell.polyline[0][0], cell.polyline[0][1]);
    for (let i = 1; i < cell.polyline.length; i++) {
      ctx.lineTo(cell.polyline[i][0], cell.polyline[i][1]);
    }
    ctx.closePath();
    ctx.fill();
  }
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const mask = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    mask[i] = rgba[p] > 127 ? 1 : 0;
  }
  return mask;
}

export function subtractOccupiedFromMask(
  binary: Uint8Array,
  occupied: Uint8Array,
) {
  for (let i = 0; i < binary.length; i++) {
    if (occupied[i]) binary[i] = 0;
  }
}

export function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

export function ellipseVertices(
  cx: number,
  cy: number,
  n: number,
  w: number,
  h: number,
  rotationDeg: number,
): { x: number; y: number }[] {
  const a = w / 2;
  const b = h / 2;
  const rad = (rotationDeg * Math.PI) / 180;
  const cosR = Math.cos(rad);
  const sinR = Math.sin(rad);
  const out: { x: number; y: number }[] = [];
  const baseOffset = n % 2 === 0 ? Math.PI / n : 0;
  for (let i = 0; i < n; i++) {
    const ang = -Math.PI / 2 + baseOffset + (i / n) * Math.PI * 2;
    const dx0 = Math.cos(ang) * a;
    const dy0 = Math.sin(ang) * b;
    const dx = dx0 * cosR - dy0 * sinR;
    const dy = dx0 * sinR + dy0 * cosR;
    out.push({ x: cx + dx, y: cy + dy });
  }
  return out;
}

export function ellipsePointsSvg(
  width: number,
  height: number,
  n: number,
  padding = 2,
): string {
  const cx = width / 2;
  const cy = height / 2;
  const a = width / 2 - padding;
  const b = height / 2 - padding;
  const baseOffset = n % 2 === 0 ? Math.PI / n : 0;
  return Array.from({ length: n }, (_, i) => {
    const ang = -Math.PI / 2 + baseOffset + (i / n) * Math.PI * 2;
    return `${(cx + Math.cos(ang) * a).toFixed(1)},${(cy + Math.sin(ang) * b).toFixed(1)}`;
  }).join(" ");
}

export function getShapeVertexExtents(
  width: number,
  height: number,
  rotationDeg: number,
  sides: number,
) {
  const verts = ellipseVertices(0, 0, sides, width, height, rotationDeg);
  let maxX = 0,
    maxY = 0;
  for (const v of verts) {
    maxX = Math.max(maxX, Math.abs(v.x));
    maxY = Math.max(maxY, Math.abs(v.y));
  }
  return { maxX: Math.max(maxX, 0.5), maxY: Math.max(maxY, 0.5) };
}

export function clampShapeCenter(
  center: { x: number; y: number },
  width: number,
  height: number,
  rotationDeg: number,
  sides: number,
) {
  const { maxX, maxY } = getShapeVertexExtents(
    width,
    height,
    rotationDeg,
    sides,
  );
  const loX = maxX,
    hiX = CANVAS_SIZE - maxX;
  const loY = maxY,
    hiY = CANVAS_SIZE - maxY;
  return {
    x: clamp(center.x, Math.min(loX, hiX), Math.max(loX, hiX)),
    y: clamp(center.y, Math.min(loY, hiY), Math.max(loY, hiY)),
  };
}

export function recoverShapeFromPoints(points: Point[], naturalW: number) {
  const s = CANVAS_SIZE / naturalW;
  const pts = points.map((p) => ({ x: p.x * s, y: p.y * s }));
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const w = clamp(maxX - minX, SHAPE_DIM_MIN, SHAPE_DIM_MAX);
  const h = clamp(maxY - minY, SHAPE_DIM_MIN, SHAPE_DIM_MAX);
  const sides = (pts.length >= 3 && pts.length <= 6 ? pts.length : 4) as
    | 3
    | 4
    | 5
    | 6;
  const defaultOffsetDeg = sides % 2 === 0 ? 180 / sides : 0;
  const rotationDeg =
    pts.length > 0
      ? (Math.atan2(pts[0].y - cy, pts[0].x - cx) * 180) / Math.PI +
        90 -
        defaultOffsetDeg
      : 0;
  return { cx, cy, w, h, rotationDeg, sides };
}

export function getScreenBoxSize(
  width: number,
  height: number,
  rotationDeg: number,
) {
  const rad = (rotationDeg * Math.PI) / 180;
  const a = Math.abs(Math.cos(rad));
  const b = Math.abs(Math.sin(rad));
  return { boxW: a * width + b * height, boxH: b * width + a * height, a, b };
}

export function solveShapeSizeFromScreenBox(
  targetBoxW: number,
  targetBoxH: number,
  rotationDeg: number,
  startW: number,
  startH: number,
  startBoxW: number,
  startBoxH: number,
) {
  const { a, b } = getScreenBoxSize(1, 1, rotationDeg);
  const denom = a * a - b * b;
  if (Math.abs(denom) < 1e-3) {
    const rw = targetBoxW / Math.max(1, startBoxW);
    const rh = targetBoxH / Math.max(1, startBoxH);
    const scale = (rw + rh) / 2;
    return {
      w: clamp(startW * scale, SHAPE_DIM_MIN, SHAPE_DIM_MAX),
      h: clamp(startH * scale, SHAPE_DIM_MIN, SHAPE_DIM_MAX),
    };
  }
  const w = (a * targetBoxW - b * targetBoxH) / denom;
  const h = (a * targetBoxH - b * targetBoxW) / denom;
  return {
    w: clamp(w, SHAPE_DIM_MIN, SHAPE_DIM_MAX),
    h: clamp(h, SHAPE_DIM_MIN, SHAPE_DIM_MAX),
  };
}

export function isSupportedImage(file: File): boolean {
  const n = file.name.toLowerCase();
  return SUPPORTED_EXTS.some((e) => n.endsWith(e));
}

export function isPointInPolyline(
  px: number,
  py: number,
  poly: [number, number][],
): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}
