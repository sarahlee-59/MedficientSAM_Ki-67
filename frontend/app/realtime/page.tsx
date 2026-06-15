"use client";

/**
 * /realtime — Ki-67 실시간 세그멘테이션 (OpenVINO 서버)
 *
 * ── 인터랙션 모델 ─────────────────────────────────────────────────────────
 *  - 우측에서 라벨(양성/음성)과 프롬프트 도형(△□⬠⬡)을 미리 선택
 *  - 좌클릭 짧게       → 클릭 위치에 N각형 꼭짓점 N개를 prompt 로 즉시 cell 추가
 *  - 좌드래그          → 마우스 방향 atan2 각도로 도형 회전 (임의 각도)
 *  - 우드래그          → 가로/세로 폭 동시 조정 (X=가로, Y=세로, 1px:1px)
 *  - wheel             → 도형 전체 크기 (가로·세로 비율 유지)
 *  - ctrl + wheel      → 줌
 *  - ctrl + 좌드래그    → 화면 이동(팬)
 *  - R                 → 회전 0° 리셋
 *  - Z                 → 마지막 cell undo
 *  - Y                 → undo한 cell redo
 *  - Esc               → 삭제 모드 토글 / ctrl + Esc → 전체 삭제
 *
 * ── 추론 ───────────────────────────────────────────────────────────────────
 *  미리보기·확정·재추론 모두 서버 OpenVINO (FastAPI, 포트 8000)
 *  encode → decode 분리 호출로 첫 응답 최소화
 */

import { useEffect, useMemo, useRef, useState } from "react";

type Point = { x: number; y: number; label: 1 | 0 };

type Cell = {
  id: number;
  points: Point[];
  polyline: [number, number][];
  kiLabel: "positive" | "negative";
  inferenceMs: number;
  pending: boolean;
};

type SegmentResult = {
  polyline: [number, number][];
  inferenceMs: number;
};

type SegmentFn = (image: HTMLImageElement, points: Point[]) => Promise<SegmentResult>;

const MODEL_NAME = "e11_holdout_int8_server";
const INFER_URL = "/api/infer";
const ENCODE_URL = "/api/encode";
const DECODE_URL = "/api/decode";

const CANVAS_SIZE = 768;
const SHAPE_OPTIONS: { sides: 3 | 4 | 5 | 6; glyph: string; name: string }[] = [
  { sides: 3, glyph: "△", name: "삼각형" },
  { sides: 4, glyph: "□", name: "사각형" },
  { sides: 5, glyph: "⬠", name: "오각형" },
  { sides: 6, glyph: "⬡", name: "육각형" },
];
const SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"];

const SAMPLE_IMAGES = [
  { src: "/samples/bench1.png", name: "bench1.png", label: "bench 1" },
  { src: "/samples/bench2.png", name: "bench2.png", label: "bench 2" },
  { src: "/samples/bench3.png", name: "bench3.png", label: "bench 3" },
  { src: "/samples/bench4.png", name: "bench4.png", label: "bench 4" },
  { src: "/samples/bench5.png", name: "bench5.png", label: "bench 5" },
];

const DEFAULT_SHAPE_WIDTH = 64;
const DEFAULT_SHAPE_HEIGHT = 64;
const SHAPE_DIM_MIN = 12;
const SHAPE_DIM_MAX = 240;
const DRAG_THRESHOLD = 8; // 캔버스 좌표 px — 이거 초과해야 드래그로 인정
const PREVIEW_DEBOUNCE_MS = 80;


/** 점 p에서 선분 a-b까지의 수직거리 제곱 (RDP용) */
function perpDistSq(p: [number, number], a: [number, number], b: [number, number]): number {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  if (len2 < 1e-12) {
    const ex = p[0] - a[0], ey = p[1] - a[1];
    return ex * ex + ey * ey;
  }
  const t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2;
  const cx = a[0] + t * dx, cy = a[1] + t * dy;
  const ex = p[0] - cx, ey = p[1] - cy;
  return ex * ex + ey * ey;
}

function rdpRange(ext: [number, number][], eps2: number, lo: number, hi: number, keep: boolean[]) {
  if (hi <= lo + 1) return;
  let idx = -1, maxD = -1;
  for (let i = lo + 1; i < hi; i++) {
    const d = perpDistSq(ext[i], ext[lo], ext[hi]);
    if (d > maxD) { maxD = d; idx = i; }
  }
  if (maxD > eps2) {
    rdpRange(ext, eps2, lo, idx, keep);
    keep[idx] = true;
    rdpRange(ext, eps2, idx, hi, keep);
  }
}

/** 닫힌 윤곽을 Douglas-Peucker로 단순화 — 직선 구간은 접고 굴곡(디테일)은 보존.
 *  스무딩/Chaikin과 달리 점을 이동시키지 않으므로 경계 위치가 그대로 유지된다. */
function simplifyClosed(pts: [number, number][], eps: number): [number, number][] {
  const n = pts.length;
  if (n < 5) return pts;
  // 시작점에서 가장 먼 점을 두 번째 앵커로 → 닫힌 곡선을 두 호로 분할
  let far = 0, fd = -1;
  for (let i = 1; i < n; i++) {
    const dx = pts[i][0] - pts[0][0], dy = pts[i][1] - pts[0][1];
    const d = dx * dx + dy * dy;
    if (d > fd) { fd = d; far = i; }
  }
  const ext = pts.concat([pts[0]]);          // 마지막→시작 래핑 처리
  const keep = new Array(n + 1).fill(false);
  keep[0] = true; keep[far] = true; keep[n] = true;
  const eps2 = eps * eps;
  rdpRange(ext, eps2, 0, far, keep);
  rdpRange(ext, eps2, far, n, keep);
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(pts[i]);
  return out;
}

function polygonArea(pts: [number, number][]): number {
  let a = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    a += (pts[j][0] + pts[i][0]) * (pts[j][1] - pts[i][1]);
  }
  return a / 2;
}

/** iso=0 marching squares — 연속 logit 필드(w×h, row-major)에서 0 등고선을 추출.
 *  격자 변마다 선형보간으로 교차점을 구해 **서브픽셀** 정밀도의 닫힌 루프들을 반환한다. */
function marchingSquares(field: Float32Array, w: number, h: number): [number, number][][] {
  const at = (x: number, y: number) => field[y * w + x];
  const interp = (va: number, vb: number) => {
    const d = vb - va;
    return Math.abs(d) < 1e-9 ? 0.5 : -va / d;   // iso = 0 → t = (0 - va)/(vb - va)
  };
  const pt = new Map<string, [number, number]>();
  const adj = new Map<string, string[]>();
  const push = (e: string, nbr: string) => {
    const list = adj.get(e);
    if (list) list.push(nbr);
    else adj.set(e, [nbr]);
  };
  const connect = (ea: string, pa: [number, number], eb: string, pb: [number, number]) => {
    pt.set(ea, pa); pt.set(eb, pb);
    push(ea, eb); push(eb, ea);
  };

  for (let y = 0; y < h - 1; y++) {
    for (let x = 0; x < w - 1; x++) {
      const tl = at(x, y), tr = at(x + 1, y), br = at(x + 1, y + 1), bl = at(x, y + 1);
      let code = 0;
      if (tl > 0) code |= 8;
      if (tr > 0) code |= 4;
      if (br > 0) code |= 2;
      if (bl > 0) code |= 1;
      if (code === 0 || code === 15) continue;

      // 격자 변 = 인접 셀과 공유되는 고유 ID → 교차점이 자동으로 일치해 루프가 닫힌다
      const T = (): [string, [number, number]] => [`H:${x}:${y}`, [x + interp(tl, tr), y]];
      const B = (): [string, [number, number]] => [`H:${x}:${y + 1}`, [x + interp(bl, br), y + 1]];
      const L = (): [string, [number, number]] => [`V:${x}:${y}`, [x, y + interp(tl, bl)]];
      const R = (): [string, [number, number]] => [`V:${x + 1}:${y}`, [x + 1, y + interp(tr, br)]];
      const link = (
        a: () => [string, [number, number]],
        b: () => [string, [number, number]],
      ) => { const [ea, pa] = a(), [eb, pb] = b(); connect(ea, pa, eb, pb); };

      switch (code) {
        case 1: link(L, B); break;
        case 2: link(B, R); break;
        case 3: link(L, R); break;
        case 4: link(T, R); break;
        case 5: { // saddle (tr, bl 내부)
          if ((tl + tr + br + bl) / 4 > 0) { link(T, R); link(B, L); }
          else { link(T, L); link(B, R); }
          break;
        }
        case 6: link(T, B); break;
        case 7: link(T, L); break;
        case 8: link(T, L); break;
        case 9: link(T, B); break;
        case 10: { // saddle (tl, br 내부)
          if ((tl + tr + br + bl) / 4 > 0) { link(T, L); link(B, R); }
          else { link(T, R); link(B, L); }
          break;
        }
        case 11: link(T, R); break;
        case 12: link(L, R); break;
        case 13: link(R, B); break;
        case 14: link(L, B); break;
      }
    }
  }

  // 차수≤2 그래프 → 단순 순환 추출
  const loops: [number, number][][] = [];
  const visited = new Set<string>();
  for (const start of adj.keys()) {
    if (visited.has(start)) continue;
    const loop: [number, number][] = [];
    let cur: string | undefined = start;
    let prev: string | null = null;
    while (cur && !visited.has(cur)) {
      visited.add(cur);
      loop.push(pt.get(cur)!);
      const ns: string[] = adj.get(cur) ?? [];
      const next: string | undefined =
        ns.find((n) => n !== prev && !visited.has(n)) ?? ns.find((n) => n !== prev);
      prev = cur;
      cur = next;
    }
    if (loop.length >= 3) loops.push(loop);
  }
  return loops;
}

/** 연속 logit 필드 → 세포 윤곽선. 가장 넓은 0-등고선 루프를 선택하고
 *  RDP로 중복점만 제거(디테일 보존, 스무딩 없음). */
function logitsToPolyline(field: Float32Array, w: number, h: number): [number, number][] {
  const loops = marchingSquares(field, w, h);
  if (loops.length === 0) return [];
  let best = loops[0], bestArea = -1;
  for (const lp of loops) {
    const a = Math.abs(polygonArea(lp));
    if (a > bestArea) { bestArea = a; best = lp; }
  }
  return simplifyClosed(best, 0.5);
}

/** 윤곽선 점들을 직선으로 잇는 경로 */
function drawPolylinePath(
  ctx: CanvasRenderingContext2D,
  pts: [number, number][],
  sx: number,
  sy: number
) {
  if (pts.length < 3) return;
  ctx.moveTo(pts[0][0] * sx, pts[0][1] * sy);
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo(pts[i][0] * sx, pts[i][1] * sy);
  }
  ctx.closePath();
}

type CellMaskSource = {
  id: number;
  pending: boolean;
  polyline: [number, number][];
};

/** 확정된 다른 세포 마스크 합집합 (재추론 시 해당 세포는 제외) */
function buildOccupiedMask(
  cells: CellMaskSource[],
  excludeCellId: number | null,
  w: number,
  h: number
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

// ── 유틸 ────────────────────────────────────────────────────────────────────
function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

/** 실제 꼭짓점 기준으로 중심이 이동 가능한 범위 계산 (회전 bbox보다 덜 보수적 → 모서리까지 닿기 쉬움) */
function getShapeVertexExtents(width: number, height: number, rotationDeg: number, sides: number) {
  const verts = ellipseVertices(0, 0, sides, width, height, rotationDeg);
  let maxX = 0;
  let maxY = 0;
  for (const v of verts) {
    maxX = Math.max(maxX, Math.abs(v.x));
    maxY = Math.max(maxY, Math.abs(v.y));
  }
  return { maxX: Math.max(maxX, 0.5), maxY: Math.max(maxY, 0.5) };
}

function clampShapeCenter(
  center: { x: number; y: number },
  width: number,
  height: number,
  rotationDeg: number,
  sides: number
) {
  const { maxX, maxY } = getShapeVertexExtents(width, height, rotationDeg, sides);
  const loX = maxX;
  const hiX = CANVAS_SIZE - maxX;
  const loY = maxY;
  const hiY = CANVAS_SIZE - maxY;
  return {
    x: clamp(center.x, Math.min(loX, hiX), Math.max(loX, hiX)),
    y: clamp(center.y, Math.min(loY, hiY), Math.max(loY, hiY)),
  };
}

/** 저장된 prompt 점들에서 편집용 도형 파라미터 복원 */
function recoverShapeFromPoints(points: Point[], naturalW: number) {
  const s = CANVAS_SIZE / naturalW;
  const pts = points.map((p) => ({ x: p.x * s, y: p.y * s }));
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
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
  const sides = (pts.length >= 3 && pts.length <= 6 ? pts.length : 4) as 3 | 4 | 5 | 6;
  const defaultOffsetDeg = sides % 2 === 0 ? 180 / sides : 0;
  const rotationDeg =
    pts.length > 0 ? (Math.atan2(pts[0].y - cy, pts[0].x - cx) * 180) / Math.PI + 90 - defaultOffsetDeg : 0;
  return { cx, cy, w, h, rotationDeg, sides };
}

/** 회전된 도형의 화면상 bbox 가로/세로(px) */
function getScreenBoxSize(width: number, height: number, rotationDeg: number) {
  const rad = (rotationDeg * Math.PI) / 180;
  const a = Math.abs(Math.cos(rad));
  const b = Math.abs(Math.sin(rad));
  return {
    boxW: a * width + b * height,
    boxH: b * width + a * height,
    a,
    b,
  };
}

/**
 * 화면 bbox 목표 크기(targetBoxW/H)를 만족하는 내부 width/height 계산.
 * 회전 각도가 45° 부근이면 해가 불안정해질 수 있어 비율 스케일 폴백 사용.
 */
function solveShapeSizeFromScreenBox(
  targetBoxW: number,
  targetBoxH: number,
  rotationDeg: number,
  startW: number,
  startH: number,
  startBoxW: number,
  startBoxH: number
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

function isSupportedImage(file: File): boolean {
  const n = file.name.toLowerCase();
  return SUPPORTED_EXTS.some((e) => n.endsWith(e));
}

/**
 * 폭(w) × 높이(h) 타원 위에 N개 꼭짓점을 등각도 배치 + 임의 각도 회전.
 * - 회전 0° = 디폴트(가로 폭 = w, 세로 폭 = h)
 * - 회전각은 도(deg) 단위
 */
function ellipseVertices(
  cx: number,
  cy: number,
  n: number,
  w: number,
  h: number,
  rotationDeg: number
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

/** SVG 컨테이너 내부 polygon points (회전은 CSS transform 으로 처리) */
function ellipsePointsSvg(width: number, height: number, n: number, padding = 2): string {
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

function isPointInPolyline(px: number, py: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}

// ── 컴포넌트 ────────────────────────────────────────────────────────────────
export default function RealtimePage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);
  const canvasAreaRef = useRef<HTMLDivElement>(null);
  const cellsRef = useRef<Cell[]>([]);
  const inferenceExcludeCellIdRef = useRef<number | null>(null);

  const [redoCount, setRedoCount] = useState(0);

  const redoStackRef = useRef<Cell[]>([]);
  const previewAbortRef = useRef<AbortController | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  const previewInferenceRef = useRef<SegmentFn>(async (image, points) => {
    const t0 = performance.now();

    previewAbortRef.current?.abort();
    const controller = new AbortController();
    previewAbortRef.current = controller;

    let json: { logits: number[]; width: number; height: number };
    const sessionId = sessionIdRef.current;

    if (sessionId) {
      // 빠른 경로: 좌표만 JSON으로 전송 (이미지 전송 없음)
      const res = await fetch(DECODE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          points: points.map((p) => [p.x, p.y]),
          labels: points.map((p) => p.label),
        }),
        signal: controller.signal,
      });
      if (res.status === 404) {
        // 세션 만료 → fallback
        sessionIdRef.current = null;
        throw new Error("session expired");
      }
      if (!res.ok) throw new Error(`서버 추론 실패: ${res.status}`);
      json = await res.json();
    } else {
      // fallback: 이미지 포함 전송
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      canvas.getContext("2d")!.drawImage(image, 0, 0);
      const blob = await new Promise<Blob>((resolve) =>
        canvas.toBlob(resolve as BlobCallback, "image/png"),
      );
      const formData = new FormData();
      formData.append("image", blob, "image.png");
      formData.append("points", JSON.stringify(points.map((p) => [p.x, p.y])));
      formData.append("labels", JSON.stringify(points.map((p) => p.label)));
      const res = await fetch(INFER_URL, { method: "POST", body: formData, signal: controller.signal });
      if (!res.ok) throw new Error(`서버 추론 실패: ${res.status}`);
      json = await res.json();
    }

    const { logits, width: W, height: H } = json;
    const field = Float32Array.from(logits as number[]);
    // 이미 확정된 다른 세포가 점유한 픽셀은 logit을 크게 음수로 깎아 경계에서 제외
    // (바이너리 마스크 subtract와 동일 효과 — 0-등고선이 점유 영역을 비껴간다)
    const occupied = buildOccupiedMask(
      cellsRef.current,
      inferenceExcludeCellIdRef.current,
      W,
      H,
    );
    for (let i = 0; i < field.length; i++) if (occupied[i]) field[i] = -1000;

    const polyline = logitsToPolyline(field, W, H);
    return { polyline, inferenceMs: performance.now() - t0 };
  });

  // 좌측 캔버스 영역 크기에 맞춰 캔버스를 CSS scale로 fit (768 내부 좌표계는 유지)
  const [viewportScale, setViewportScale] = useState(1);

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });

  const [shapeSides, setShapeSides] = useState<3 | 4 | 5 | 6>(4);
  const [pendingKiLabel, setPendingKiLabel] = useState<"positive" | "negative" | null>(null);
  const [cells, setCells] = useState<Cell[]>([]);
  const [hoveredCellId, setHoveredCellId] = useState<number | null>(null);
  const [editingCellId, setEditingCellId] = useState<number | null>(null);
  const [dragCellId, setDragCellId] = useState<number | null>(null);
  const [dragOverCellId, setDragOverCellId] = useState<number | null>(null);
  const [reinferPending, setReinferPending] = useState(false);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  // ▼ 도형 모델: 폭/높이/회전(deg)
  const [shapeWidth, setShapeWidth] = useState(DEFAULT_SHAPE_WIDTH);
  const [shapeHeight, setShapeHeight] = useState(DEFAULT_SHAPE_HEIGHT);
  const [shapeRotationDeg, setShapeRotationDeg] = useState(0);

  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);

  // 좌드래그(꾹 누르고 움직임) = 회전
  const [dragOrigin, setDragOrigin] = useState<{ x: number; y: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // 우드래그 = 가로/세로 폭 동시 조정
  const [resizeOrigin, setResizeOrigin] = useState<{
    x: number;
    y: number;
    startW: number;
    startH: number;
    startBoxW: number;
    startBoxH: number;
  } | null>(null);

  const [activeTool, setActiveTool] = useState<"cursor" | "annotate">("annotate");
  const [modeToast, setModeToast] = useState<"annotate" | "cursor" | null>(null);
  const modeToastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [toolbarWidth, setToolbarWidth] = useState(384);
  const isToolbarDraggingRef = useRef(false);
  const toolbarDragStartXRef = useRef(0);
  const toolbarDragStartWidthRef = useRef(0);
  const windowWidthRef = useRef(typeof window !== "undefined" ? window.innerWidth : 1280);
  const [cellFilter, setCellFilter] = useState<"all" | "positive" | "negative">("all");
  const hoveredFromCanvasRef = useRef(false);
  const cellRowRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const [error, setError] = useState<string | null>(null);
  const [latencies, setLatencies] = useState<{ ms: number; cellId: number }[]>([]);
  const [previewPolyline, setPreviewPolyline] = useState<[number, number][] | null>(null);
  const [isPreviewPending, setIsPreviewPending] = useState(false);
  const previewTimerRef = useRef<number | null>(null);
  const previewReqSeqRef = useRef(0);

  const lastEntry = latencies.length ? latencies[latencies.length - 1] : null;
  const stats = useMemo(() => {
    if (latencies.length === 0) return null;
    const s = latencies.reduce((a, b) => a + b.ms, 0);
    const minEntry = latencies.reduce((a, b) => (b.ms < a.ms ? b : a));
    const maxEntry = latencies.reduce((a, b) => (b.ms > a.ms ? b : a));
    return {
      n: latencies.length,
      avg: s / latencies.length,
      minMs: minEntry.ms,
      minCellId: minEntry.cellId,
      maxMs: maxEntry.ms,
      maxCellId: maxEntry.cellId,
    };
  }, [latencies]);

  const pendingCount = cells.filter((c) => c.pending).length;
  const confirmedCount = cells.length - pendingCount;
  const positiveCount = cells.filter((c) => !c.pending && c.kiLabel === "positive").length;
  const negativeCount = cells.filter((c) => !c.pending && c.kiLabel === "negative").length;
  const ki67Rate = confirmedCount > 0 ? (positiveCount / confirmedCount) * 100 : null;

  // 회전 각도 0~360 정규화 (표시용)
  const displayRotation = ((Math.round(shapeRotationDeg) % 360) + 360) % 360;

  // 도형이 그려질 중심: 드래그 중이면 dragOrigin 고정, 아니면 마우스 위치
  const rawCenter = isDragging && dragOrigin ? dragOrigin : cursorPos;
  const shapeCenter = rawCenter
    ? clampShapeCenter(rawCenter, shapeWidth, shapeHeight, shapeRotationDeg, shapeSides)
    : null;

  useEffect(() => {
    cellsRef.current = cells;
  }, [cells]);

  // ── 좌측 영역 크기에 맞춰 캔버스 fit (반응형) ─────────────────────────────
  useEffect(() => {
    const el = canvasAreaRef.current;
    if (!el) return;
    const update = (w: number, h: number) => {
      if (w <= 0 || h <= 0) return;
      // p-6 = 24px 양쪽 → 48px 감산, 실제 캔버스 가용 영역 기준으로 scale 계산
      const s = Math.min(w - 48, h - 48) / CANVAS_SIZE;
      setViewportScale(Math.max(0.3, Math.min(s, 1)));
    };
    update(el.clientWidth, el.clientHeight);
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0].contentRect;
      update(cr.width, cr.height);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── 캔버스 렌더링 ─────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !naturalSize.w) return;

    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const sx = canvas.width / naturalSize.w;
    const sy = canvas.height / naturalSize.h;

    if (previewPolyline && previewPolyline.length >= 3) {
      ctx.beginPath();
      drawPolylinePath(ctx, previewPolyline, sx, sy);
      ctx.fillStyle = "rgba(16,185,129,0.14)";
      ctx.strokeStyle = "rgba(16,185,129,0.9)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
    }

    cells.forEach((cell) => {
      // 재추론 중인 세포는 기존 세그먼트 숨김 → 좌클릭 후 draft만 표시
      if (editingCellId === cell.id) return;

      if (cell.pending || cell.polyline.length < 3) return;
      ctx.beginPath();
      drawPolylinePath(ctx, cell.polyline, sx, sy);
      const fill =
        cell.kiLabel === "positive" ? "rgba(239,68,68,0.22)" : "rgba(59,130,246,0.22)";
      const stroke = cell.kiLabel === "positive" ? "#ef4444" : "#3b82f6";
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    const hovered = hoveredCellId != null ? cells.find((c) => c.id === hoveredCellId) : null;
    if (hovered && hovered.id !== editingCellId) {
      const hiStroke = hovered.kiLabel === "positive" ? "#f87171" : "#60a5fa";
      const hiFill =
        hovered.kiLabel === "positive" ? "rgba(248,113,113,0.35)" : "rgba(96,165,250,0.35)";

      if (!hovered.pending && hovered.polyline.length >= 3) {
        ctx.beginPath();
        drawPolylinePath(ctx, hovered.polyline, sx, sy);
        ctx.fillStyle = hiFill;
        ctx.strokeStyle = hiStroke;
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        ctx.fill();
        ctx.stroke();
      } else if (hovered.points.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(hovered.points[0].x * sx, hovered.points[0].y * sy);
        for (let i = 1; i < hovered.points.length; i++) {
          ctx.lineTo(hovered.points[i].x * sx, hovered.points[i].y * sy);
        }
        ctx.closePath();
        ctx.fillStyle = "rgba(250,204,21,0.2)";
        ctx.strokeStyle = "#facc15";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([4, 2]);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
      }

      const idx = cells.findIndex((c) => c.id === hovered.id);
      const labelPt =
        hovered.polyline.length >= 1
          ? hovered.polyline[0]
          : hovered.points[0]
            ? [hovered.points[0].x, hovered.points[0].y]
            : null;
      if (labelPt && idx >= 0) {
        let lx = labelPt[0] * sx;
        let ly = labelPt[1] * sy;
        if (hovered.polyline.length >= 3) {
          const cx =
            hovered.polyline.reduce((s, pt) => s + pt[0], 0) / hovered.polyline.length;
          const cy =
            hovered.polyline.reduce((s, pt) => s + pt[1], 0) / hovered.polyline.length;
          lx = cx * sx;
          ly = cy * sy;
        }
        const text = `#${idx + 1}`;
        ctx.font = "bold 22px ui-monospace, monospace";
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.strokeStyle = hiStroke;
        ctx.lineWidth = 4;
        ctx.strokeText(text, lx + 8, ly - 8);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, lx + 8, ly - 8);
      }
    }
  }, [cells, naturalSize, previewPolyline, hoveredCellId, editingCellId]);

  // ── 파일 로드 ─────────────────────────────────────────────────────────────
  function loadImageFile(file: File) {
    if (!isSupportedImage(file)) {
      setError(`지원하지 않는 형식입니다: ${file.name}\nJPG · PNG · BMP · TIFF 만 가능합니다.`);
      return;
    }
    setImageFile(file);
    setCells([]);
    setHoveredCellId(null);
    setEditingCellId(null);
    setReinferPending(false);
    setError(null);
    setLatencies([]);
    setPreviewPolyline(null);
    setIsPreviewPending(false);
    setZoom(1);
    setPan({ x: 0, y: 0 });

    imgRef.current = null;
    sessionIdRef.current = null;

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
      URL.revokeObjectURL(url);
      // 이미지 로드 즉시 인코딩 → session_id 확보 (첫 미리보기 지연 제거)
      const warmCanvas = document.createElement("canvas");
      warmCanvas.width = img.naturalWidth;
      warmCanvas.height = img.naturalHeight;
      warmCanvas.getContext("2d")!.drawImage(img, 0, 0);
      warmCanvas.toBlob((blob) => {
        if (!blob) return;
        const fd = new FormData();
        fd.append("image", blob, "image.png");
        fetch(ENCODE_URL, { method: "POST", body: fd })
          .then((r) => r.json())
          .then(({ session_id }) => { sessionIdRef.current = session_id ?? null; })
          .catch(() => {});
      }, "image/png");
    };
    img.src = url;
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) loadImageFile(f);
    e.target.value = "";
  }

  async function loadSampleImage(src: string, name: string) {
    try {
      const res = await fetch(src);
      const blob = await res.blob();
      const file = new File([blob], name, { type: blob.type });
      loadImageFile(file);
    } catch {
      setError("샘플 이미지를 불러오지 못했습니다.");
    }
  }

  const [isDragOver, setIsDragOver] = useState(false);
  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) loadImageFile(f);
  }

  // ── 휠 이벤트 ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const wrapper = canvasWrapperRef.current;
    if (!wrapper) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey) {
        // ctrl + wheel → 줌
        const rect = wrapper.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        setZoom((prev) => {
          const factor = e.deltaY < 0 ? 1.1 : 0.9;
          const next = clamp(prev * factor, 1, 8);
          const scale = next / prev;
          setPan((p) => ({
            x: clamp(mx * (1 - scale) + p.x * scale, CANVAS_SIZE * (1 - next), 0),
            y: clamp(my * (1 - scale) + p.y * scale, CANVAS_SIZE * (1 - next), 0),
          }));
          return next;
        });
        return;
      }
      // wheel: 도형 전체 크기 (가로·세로 비율 유지, 세부 조정은 우드래그)
      const scale = e.deltaY > 0 ? 1.06 : 0.94;
      setShapeWidth((w) => clamp(w * scale, SHAPE_DIM_MIN, SHAPE_DIM_MAX));
      setShapeHeight((h) => clamp(h * scale, SHAPE_DIM_MIN, SHAPE_DIM_MAX));
    };
    wrapper.addEventListener("wheel", onWheel, { passive: false });
    return () => wrapper.removeEventListener("wheel", onWheel);
  }, [naturalSize.w]);

  // ── hover preview: 클릭 전 세그먼트 미리보기 ────────────────────────────────
  useEffect(() => {
    if (previewTimerRef.current) {
      window.clearTimeout(previewTimerRef.current);
      previewTimerRef.current = null;
    }

    if (
      !shapeCenter ||
      !imgRef.current ||
      !naturalSize.w ||
      isDragging ||
      !!resizeOrigin ||
      (editingCellId !== null && reinferPending)
    ) {
      setPreviewPolyline(null);
      setIsPreviewPending(false);
      return;
    }

    const toImg = naturalSize.w / CANVAS_SIZE;
    const verts = ellipseVertices(
      shapeCenter.x * toImg,
      shapeCenter.y * toImg,
      shapeSides,
      shapeWidth * toImg,
      shapeHeight * toImg,
      shapeRotationDeg
    ).map((v) => ({ x: Math.round(v.x), y: Math.round(v.y), label: 1 as const }));

    const reqId = ++previewReqSeqRef.current;
    setIsPreviewPending(true);
    previewTimerRef.current = window.setTimeout(async () => {
      try {
        if (!imgRef.current) return;
        inferenceExcludeCellIdRef.current = editingCellId;
        const { polyline } = await previewInferenceRef.current(imgRef.current, verts);
        if (reqId === previewReqSeqRef.current) {
          setPreviewPolyline(polyline.length >= 3 ? polyline : null);
        }
      } catch {
        if (reqId === previewReqSeqRef.current) setPreviewPolyline(null);
      } finally {
        if (reqId === previewReqSeqRef.current) setIsPreviewPending(false);
      }
    }, PREVIEW_DEBOUNCE_MS);

    return () => {
      if (previewTimerRef.current) {
        window.clearTimeout(previewTimerRef.current);
        previewTimerRef.current = null;
      }
    };
  }, [
    shapeCenter?.x,
    shapeCenter?.y,
    shapeSides,
    shapeWidth,
    shapeHeight,
    shapeRotationDeg,
    naturalSize.w,
    isDragging,
    resizeOrigin,
    editingCellId,
    reinferPending,
  ]);

  /** canvas(768) 좌표계로 환산 */
  function canvasPointFromEvent(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (CANVAS_SIZE / rect.width),
      y: (e.clientY - rect.top) * (CANVAS_SIZE / rect.height),
    };
  }

  // ── 마우스 다운: 좌(클릭/회전) / 우(크기조정) 분기 ───────────────────────
  function handleCanvasMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    // ctrl + 좌드래그 → 화면 이동(팬), 모드 무관.
    // 캔버스 밖으로 마우스가 나가도 끊김/점프 없이 추적되도록 window 리스너 사용
    if (e.button === 0 && e.ctrlKey) {
      const sx = e.clientX, sy = e.clientY, px = pan.x, py = pan.y, z = zoom;
      const onMove = (ev: MouseEvent) => {
        setPan({
          x: clamp(px + (ev.clientX - sx), CANVAS_SIZE * (1 - z), 0),
          y: clamp(py + (ev.clientY - sy), CANVAS_SIZE * (1 - z), 0),
        });
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
      return;
    }
    if (activeTool === "cursor") {
      if (e.button === 0 && hoveredCellId !== null && editingCellId === null) {
        handleDeleteCell(hoveredCellId);
      } else {
        switchTool("annotate");
      }
      return;
    }
    const p = canvasPointFromEvent(e);
    if (!p || pendingCount > 0) return;
    if (e.button === 2) {
      // 우클릭 → 크기 조정 시작 (시점 폭 저장)
      e.preventDefault();
      const { boxW, boxH } = getScreenBoxSize(shapeWidth, shapeHeight, shapeRotationDeg);
      setResizeOrigin({
        x: p.x,
        y: p.y,
        startW: shapeWidth,
        startH: shapeHeight,
        startBoxW: boxW,
        startBoxH: boxH,
      });
      return;
    }
    if (e.button !== 0) return;
    // 좌클릭 → 클릭/회전 시작
    setDragOrigin(p);
    setIsDragging(false);
  }

  // ── 마우스 무브: 커서 추적 + 좌드래그(회전) / 우드래그(크기조정) ─────────
  function handleCanvasMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const p = canvasPointFromEvent(e);
    if (!p) return;
    setCursorPos(p);

    // 캔버스 hit detection → 세포 목록 하이라이트
    if (!isDragging && !resizeOrigin && editingCellId === null && naturalSize.w) {
      const toImg = naturalSize.w / CANVAS_SIZE;
      let found: number | null = null;
      for (let i = cellsRef.current.length - 1; i >= 0; i--) {
        const cell = cellsRef.current[i];
        if (cell.pending || cell.polyline.length < 3) continue;
        if (isPointInPolyline(p.x * toImg, p.y * toImg, cell.polyline)) {
          found = cell.id;
          break;
        }
      }
      if (found !== null) {
        hoveredFromCanvasRef.current = true;
        setHoveredCellId(found);
      } else if (hoveredFromCanvasRef.current) {
        hoveredFromCanvasRef.current = false;
        setHoveredCellId(null);
      }
    }

    // 우드래그 = 가로/세로 폭 동시 조정 (1px:1px)
    if (resizeOrigin) {
      const dx = p.x - resizeOrigin.x;
      const dy = p.y - resizeOrigin.y;
      const targetBoxW = clamp(resizeOrigin.startBoxW + dx, SHAPE_DIM_MIN, SHAPE_DIM_MAX);
      const targetBoxH = clamp(resizeOrigin.startBoxH + dy, SHAPE_DIM_MIN, SHAPE_DIM_MAX);
      const { w, h } = solveShapeSizeFromScreenBox(
        targetBoxW,
        targetBoxH,
        shapeRotationDeg,
        resizeOrigin.startW,
        resizeOrigin.startH,
        resizeOrigin.startBoxW,
        resizeOrigin.startBoxH
      );
      setShapeWidth(w);
      setShapeHeight(h);
      return;
    }

    // 좌드래그 = 회전
    if (dragOrigin) {
      const dx = p.x - dragOrigin.x;
      const dy = p.y - dragOrigin.y;
      const dist = Math.hypot(dx, dy);
      if (dist > DRAG_THRESHOLD) {
        if (!isDragging) setIsDragging(true);
        // atan2 결과 그대로 회전각으로 사용 — 오른쪽=0°, 아래=90°, 왼쪽=180°, 위=-90°
        setShapeRotationDeg((Math.atan2(dy, dx) * 180) / Math.PI);
      }
    }
  }

  function handleCanvasMouseLeave() {
    setCursorPos(null);
    // 캔버스를 벗어났을 때 mouseup 못 받을 수 있으므로 안전하게 종료
    if (dragOrigin) {
      setDragOrigin(null);
      setIsDragging(false);
    }
    if (resizeOrigin) {
      setResizeOrigin(null);
    }
    if (hoveredFromCanvasRef.current) {
      hoveredFromCanvasRef.current = false;
      setHoveredCellId(null);
    }
  }

  // ── 마우스 업: 짧은 클릭이면 cell 추가 / 드래그/리사이즈는 그냥 종료 ────
  async function handleCanvasMouseUp(e: React.MouseEvent<HTMLCanvasElement>) {
    // 우드래그 종료
    if (e.button === 2 && resizeOrigin) {
      setResizeOrigin(null);
      return;
    }
    if (!dragOrigin) return;
    const wasDragging = isDragging;
    const origin = dragOrigin;
    setDragOrigin(null);
    setIsDragging(false);

    if (wasDragging) return; // 드래그였으면 cell 추가 안 함 (회전만 한 것)

    // 재추론: 좌클릭으로 새 세그먼트 생성(미리보기) → [적용]에서 기존 결과 대체
    if (editingCellId !== null) {
      void handleReinferLeftClick(origin);
      return;
    }

    // 짧은 클릭 → cell 추가
    if (!naturalSize.w) return;
    if (pendingKiLabel === null) {
      setError("양성(+) 또는 음성(-) 라벨을 먼저 선택하세요.");
      return;
    }

    // canvas(768) → image(natural) 좌표 변환
    const toImg = naturalSize.w / CANVAS_SIZE;
    const safeCenter = clampShapeCenter(
      origin,
      shapeWidth,
      shapeHeight,
      shapeRotationDeg,
      shapeSides
    );
    const cx = safeCenter.x * toImg;
    const cy = safeCenter.y * toImg;
    const wImg = shapeWidth * toImg;
    const hImg = shapeHeight * toImg;
    const verts = ellipseVertices(cx, cy, shapeSides, wImg, hImg, shapeRotationDeg).map((v) => ({
      x: Math.round(v.x),
      y: Math.round(v.y),
      label: 1 as 1 | 0,
    }));

    const img = imgRef.current;
    if (!img) {
      setError("이미지가 없습니다.");
      return;
    }

    const id = Date.now() + Math.floor(Math.random() * 1000);
    const label = pendingKiLabel;
    previewReqSeqRef.current += 1;
    setIsPreviewPending(false);
    setPreviewPolyline(null);
    setError(null);

    redoStackRef.current = [];
    setRedoCount(0);
    setCells((prev) => [
      ...prev,
      { id, points: verts, polyline: [], kiLabel: label, inferenceMs: 0, pending: true },
    ]);

    try {
      const { polyline, inferenceMs } = await previewInferenceRef.current(img, verts);
      if (polyline.length < 3) {
        setCells((prev) => prev.filter((c) => c.id !== id));
        setError("세포를 찾지 못했습니다. 프롬프트 위치를 조정해 보세요.");
        return;
      }
      setCells((prev) =>
        prev.map((c) => (c.id === id ? { ...c, polyline, inferenceMs, pending: false } : c))
      );
      setLatencies((prev) => [...prev.slice(-49), { ms: inferenceMs, cellId: id }]);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setCells((prev) => prev.filter((c) => c.id !== id));
        return;
      }
      setError(err instanceof Error ? err.message : String(err));
      setCells((prev) => prev.filter((c) => c.id !== id));
    }
  }

  // ── 액션 ──────────────────────────────────────────────────────────────────
  function handleUndo() {
    const prev = cellsRef.current;
    if (prev.length === 0) return;
    const last = prev[prev.length - 1];
    redoStackRef.current = [...redoStackRef.current, last];
    setRedoCount((n) => n + 1);
    setCells(prev.slice(0, -1));
  }

  function handleRedo() {
    const stack = redoStackRef.current;
    if (stack.length === 0) return;
    const cell = stack[stack.length - 1];
    redoStackRef.current = stack.slice(0, -1);
    setRedoCount((n) => Math.max(0, n - 1));
    setCells((prev) => [...prev, cell]);
  }

  function handleResetAll() {
    if (cells.length === 0) return;
    const ok = window.confirm(`확정된 세포 ${confirmedCount}개를 전부 삭제할까요?`);
    if (!ok) return;
    setCells([]);
    setHoveredCellId(null);
    clearReinferEditState();
    setError(null);
  }

  function handleCellDragStart(e: React.DragEvent, cellId: number) {
    setDragCellId(cellId);
    e.dataTransfer.effectAllowed = "move";
  }

  function handleCellDragOver(e: React.DragEvent, cellId: number) {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    if (cellId !== dragCellId) setDragOverCellId(cellId);
  }

  function handleCellDrop(e: React.DragEvent, targetCellId: number) {
    e.preventDefault();
    if (!dragCellId || dragCellId === targetCellId) return;
    setCells((prev) => {
      const from = prev.findIndex((c) => c.id === dragCellId);
      const to = prev.findIndex((c) => c.id === targetCellId);
      if (from < 0 || to < 0) return prev;
      const next = [...prev];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
    setDragCellId(null);
    setDragOverCellId(null);
  }

  function handleCellDragEnd() {
    setDragCellId(null);
    setDragOverCellId(null);
  }

  function handleDeleteCell(id: number) {
    setCells((prev) => prev.filter((c) => c.id !== id));
    setHoveredCellId((hid) => (hid === id ? null : hid));
    if (editingCellId === id) clearReinferEditState();
  }


  function clearReinferEditState() {
    setEditingCellId(null);
    setReinferPending(false);
    previewReqSeqRef.current += 1;
    setPreviewPolyline(null);
    setIsPreviewPending(false);
  }

  function handleStartReedit(cell: Cell) {
    if (cell.pending || !naturalSize.w) return;
    const { cx, cy, w, h, rotationDeg, sides } = recoverShapeFromPoints(cell.points, naturalSize.w);
    setEditingCellId(cell.id);
    setReinferPending(false);
    setHoveredCellId(cell.id);
    setPendingKiLabel(cell.kiLabel);
    setShapeSides(sides);
    setShapeWidth(w);
    setShapeHeight(h);
    setShapeRotationDeg(rotationDeg);
    setCursorPos(clampShapeCenter({ x: cx, y: cy }, w, h, rotationDeg, sides));
    setError(null);
    previewReqSeqRef.current += 1;
    setPreviewPolyline(null);
  }

  function handleCancelReedit() {
    clearReinferEditState();
  }

  function buildVertsFromCanvasCenter(center: { x: number; y: number }): Point[] {
    const toImg = naturalSize.w / CANVAS_SIZE;
    const safe = clampShapeCenter(center, shapeWidth, shapeHeight, shapeRotationDeg, shapeSides);
    return ellipseVertices(
      safe.x * toImg,
      safe.y * toImg,
      shapeSides,
      shapeWidth * toImg,
      shapeHeight * toImg,
      shapeRotationDeg
    ).map((v) => ({ x: Math.round(v.x), y: Math.round(v.y), label: 1 as const }));
  }

  async function handleReinferLeftClick(origin: { x: number; y: number }) {
    const img = imgRef.current;
    if (!naturalSize.w || !img || editingCellId === null) return;
    if (reinferPending) return;
    const safe = clampShapeCenter(origin, shapeWidth, shapeHeight, shapeRotationDeg, shapeSides);
    setCursorPos(safe);
    const verts = buildVertsFromCanvasCenter(safe);
    const cellId = editingCellId;
    setReinferPending(true);
    setError(null);
    previewReqSeqRef.current += 1;
    setPreviewPolyline(null);

    setCells((prev) =>
      prev.map((c) => (c.id === cellId ? { ...c, points: verts, pending: true } : c))
    );

    try {
      const { polyline, inferenceMs } = await previewInferenceRef.current(img, verts);
      if (polyline.length < 3) {
        setError("세포를 찾지 못했습니다. 프롬프트 위치를 조정해 보세요.");
        setCells((prev) =>
          prev.map((c) => (c.id === cellId ? { ...c, pending: false } : c))
        );
        return;
      }
      setCells((prev) =>
        prev.map((c) =>
          c.id === cellId
            ? { ...c, points: verts, polyline, inferenceMs, pending: false }
            : c
        )
      );
      setLatencies((prev) => [...prev.slice(-49), { ms: inferenceMs, cellId }]);
      clearReinferEditState();
      setError(null);
    } catch (err) {
      if (!(err instanceof Error && err.name === "AbortError")) {
        setError(err instanceof Error ? err.message : String(err));
      }
      setCells((prev) =>
        prev.map((c) => (c.id === cellId ? { ...c, pending: false } : c))
      );
    } finally {
      setReinferPending(false);
    }
  }

  function handleSetKiLabel(id: number, label: "positive" | "negative") {
    setCells((prev) => prev.map((c) => (c.id === id ? { ...c, kiLabel: label } : c)));
  }

  function resetRotation() {
    setShapeRotationDeg(0);
  }

  // ── 캔버스 hover → 세포 목록 자동 스크롤 ───────────────────────────────
  useEffect(() => {
    if (!hoveredFromCanvasRef.current || hoveredCellId === null) return;
    const el = cellRowRefs.current.get(hoveredCellId);
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [hoveredCellId]);

  // ── 모드 전환 (토스트 포함) ──────────────────────────────────────────────
  function switchTool(next: "annotate" | "cursor") {
    setActiveTool((prev) => {
      if (prev !== next) {
        if (modeToastTimerRef.current) clearTimeout(modeToastTimerRef.current);
        setModeToast(next);
        modeToastTimerRef.current = setTimeout(() => setModeToast(null), 1800);
      }
      return next;
    });
  }

  // ── 키보드 ────────────────────────────────────────────────────────────────
  const handlersRef = useRef({ handleUndo, handleRedo, handleResetAll, resetRotation, switchTool });
  useEffect(() => {
    handlersRef.current = { handleUndo, handleRedo, handleResetAll, resetRotation, switchTool };
  });
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if ((e.key === "z" || e.key === "Z") && !e.ctrlKey && !e.metaKey)
        handlersRef.current.handleUndo();
      else if (e.key === "y" || e.key === "Y")
        handlersRef.current.handleRedo();
      else if (e.key === "Escape" && e.ctrlKey)
        handlersRef.current.handleResetAll();
      else if (e.key === "Escape") {
        // Esc 토글: 삭제 모드면 추가 모드로 해제, 아니면 삭제 모드 진입(확정 세포 있을 때만)
        setActiveTool((prev) => {
          const next = prev === "cursor" ? "annotate" : "cursor";
          if (next === "cursor" && !cellsRef.current.some((c) => !c.pending)) return prev;
          if (prev !== next) {
            if (modeToastTimerRef.current) clearTimeout(modeToastTimerRef.current);
            setModeToast(next);
            modeToastTimerRef.current = setTimeout(() => setModeToast(null), 1800);
          }
          return next;
        });
      }
      else if (e.key === "r" || e.key === "R") handlersRef.current.resetRotation();
      else if (e.key === "p" || e.key === "P") {
        setPendingKiLabel("positive");
        handlersRef.current.switchTool("annotate");
      }
      else if (e.key === "n" || e.key === "N") {
        setPendingKiLabel("negative");
        handlersRef.current.switchTool("annotate");
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    function onResize() {
      windowWidthRef.current = window.innerWidth;
    }
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    function onMouseMove(e: MouseEvent) {
      if (!isToolbarDraggingRef.current) return;
      const dx = toolbarDragStartXRef.current - e.clientX;
      // 최솟값: 왼쪽 여백 == 툴바 폭이 되는 시점 → T = (W - CANVAS_SIZE) / 3
      const minBySymmetry = Math.floor((windowWidthRef.current - CANVAS_SIZE) / 3);
      const minToolbar = Math.max(270, minBySymmetry);
      setToolbarWidth(clamp(toolbarDragStartWidthRef.current + dx, minToolbar, 600));
    }
    function onMouseUp() {
      isToolbarDraggingRef.current = false;
    }
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  // ── JSON ──────────────────────────────────────────────────────────────────
  function buildResultJson() {
    const confirmed = cells.filter((c) => !c.pending);
    return {
      image: imageFile?.name ?? "unknown",
      backend: "openvino-server",
      model: MODEL_NAME,
      prompt_shape: {
        sides: shapeSides,
        width_px: shapeWidth,
        height_px: shapeHeight,
        rotation_deg: displayRotation,
      },
      latency_stats_ms: stats,
      cells: confirmed.map((cell, idx) => ({
        cell_index: idx + 1,
        ki67_label: cell.kiLabel,
        inference_ms: Math.round(cell.inferenceMs * 100) / 100,
        prompt_points: cell.points.map((p, i) => ({
          index: i + 1,
          x: p.x,
          y: p.y,
          label: p.label === 1 ? "foreground" : "background",
        })),
        polyline: cell.polyline.map((pt) => ({ x: Math.round(pt[0]), y: Math.round(pt[1]) })),
      })),
    };
  }

  async function saveJson() {
    if (confirmedCount === 0) {
      setError("저장할 확정된 세포가 없습니다.");
      return;
    }
    const json = JSON.stringify(buildResultJson(), null, 2);
    const base = imageFile?.name?.replace(/\.[^.]+$/, "") ?? `ki67_${Date.now()}`;
    const fileName = `ki67_hybrid_${base}.json`;
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ── 파생 ──────────────────────────────────────────────────────────────────
  const labelMissing = pendingKiLabel === null;

  // ── 렌더 ──────────────────────────────────────────────────────────────────
  return (
    <main className="h-screen flex bg-gray-950 text-white overflow-hidden">
      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* ─────────── 좌측: 캔버스 + 결과 ─────────── */}
      <section className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
        <div
          ref={canvasAreaRef}
          className="flex-1 min-h-0 flex items-center justify-center p-6 overflow-hidden"
        >
        {!naturalSize.w ? (
          <div className="max-w-xl w-full flex flex-col gap-4">
            {/* 업로드 zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragOver(true);
              }}
              onDragLeave={() => setIsDragOver(false)}
              className={`w-full border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition ${
                isDragOver
                  ? "border-blue-400 bg-blue-500/10 scale-[1.01]"
                  : "border-gray-600 hover:border-blue-400/60 hover:bg-gray-900/40"
              }`}
            >
              <svg
                className={`w-14 h-14 mx-auto mb-4 ${isDragOver ? "text-blue-400" : "text-gray-500"}`}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <p className="text-gray-200 font-medium mb-1">
                {isDragOver ? "여기에 놓아 업로드" : "클릭하여 이미지 업로드"}
              </p>
              {!isDragOver && (
                <p className="text-gray-500 text-xs mb-1">또는 파일을 이 영역에 드래그</p>
              )}
              <p className="text-gray-600 text-xs">JPG · PNG · BMP · TIFF</p>
            </div>

            {/* 샘플 이미지 */}
            <div>
              <p className="text-[11px] text-gray-500 mb-2 text-center">또는 샘플 이미지로 시작</p>
              <div className="flex gap-2 justify-center flex-wrap">
                {SAMPLE_IMAGES.map((s) => (
                  <button
                    key={s.src}
                    onClick={() => loadSampleImage(s.src, s.name)}
                    className="group relative rounded-lg overflow-hidden border border-gray-700 hover:border-blue-400 transition w-28 h-28 shrink-0"
                    title={s.name}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={s.src} alt={s.name} className="w-full h-full object-cover" />
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
                      <span className="text-white text-[10px] font-medium px-1 text-center">{s.label}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div
            className="relative overflow-hidden rounded-lg border border-gray-700 shadow-2xl shrink-0"
            style={{
              width: CANVAS_SIZE,
              height: CANVAS_SIZE,
              transform: `scale(${viewportScale})`,
              transformOrigin: "center center",
            }}
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragOver(false); }}
          >
            <div
              ref={canvasWrapperRef}
              className="relative"
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: "top left",
                width: CANVAS_SIZE,
                height: CANVAS_SIZE,
              }}
            >
              <canvas
                ref={canvasRef}
                width={CANVAS_SIZE}
                height={CANVAS_SIZE}
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                onMouseLeave={handleCanvasMouseLeave}
                onContextMenu={(e) => e.preventDefault()}
                className="block select-none"
                style={{ cursor: activeTool === "cursor" ? (hoveredCellId !== null && editingCellId === null ? "pointer" : "default") : "none" }}
              />

              {/* 도형 커서 — 가로 폭/세로 폭 분리 + CSS rotate(deg) */}
              {activeTool === "annotate" && shapeCenter && (
                <svg
                  className="pointer-events-none absolute"
                  style={{
                    left: shapeCenter.x - shapeWidth / 2,
                    top: shapeCenter.y - shapeHeight / 2,
                    width: shapeWidth,
                    height: shapeHeight,
                    overflow: "visible",
                    transform: `rotate(${shapeRotationDeg}deg)`,
                    transformOrigin: "center center",
                  }}
                  viewBox={`0 0 ${shapeWidth} ${shapeHeight}`}
                >
                  <polygon
                    points={ellipsePointsSvg(shapeWidth, shapeHeight, shapeSides)}
                    fill={
                      labelMissing
                        ? "rgba(250,204,21,0.08)"
                        : pendingKiLabel === "positive"
                        ? "rgba(239,68,68,0.12)"
                        : "rgba(59,130,246,0.12)"
                    }
                    stroke={
                      labelMissing
                        ? "#facc15"
                        : pendingKiLabel === "positive"
                        ? "#ef4444"
                        : "#3b82f6"
                    }
                    strokeWidth={isDragging ? 2.5 : 1.5}
                  />
                  {/* 꼭짓점 표시 */}
                  {Array.from({ length: shapeSides }, (_, i) => {
                    const vertOffset = shapeSides % 2 === 0 ? Math.PI / shapeSides : 0;
                    const ang = -Math.PI / 2 + vertOffset + (i / shapeSides) * Math.PI * 2;
                    const px = shapeWidth / 2 + Math.cos(ang) * (shapeWidth / 2 - 2);
                    const py = shapeHeight / 2 + Math.sin(ang) * (shapeHeight / 2 - 2);
                    return (
                      <circle
                        key={i}
                        cx={px}
                        cy={py}
                        r={2.2}
                        fill={
                          labelMissing
                            ? "#facc15"
                            : pendingKiLabel === "positive"
                            ? "#ef4444"
                            : "#3b82f6"
                        }
                      />
                    );
                  })}
                </svg>
              )}

              {/* 좌드래그 회전 가이드 */}
              {isDragging && dragOrigin && cursorPos && (
                <svg
                  className="pointer-events-none absolute inset-0"
                  width={CANVAS_SIZE}
                  height={CANVAS_SIZE}
                >
                  <line
                    x1={dragOrigin.x}
                    y1={dragOrigin.y}
                    x2={cursorPos.x}
                    y2={cursorPos.y}
                    stroke="#fbbf24"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    opacity={0.7}
                  />
                  <circle cx={dragOrigin.x} cy={dragOrigin.y} r={3} fill="#fbbf24" />
                </svg>
              )}

              {/* 우드래그 리사이즈 가이드 — X/Y 십자선 + 시작점 */}
              {resizeOrigin && cursorPos && (
                <svg
                  className="pointer-events-none absolute inset-0"
                  width={CANVAS_SIZE}
                  height={CANVAS_SIZE}
                >
                  {/* X축 = 가로폭 (시안) */}
                  <line
                    x1={resizeOrigin.x}
                    y1={resizeOrigin.y}
                    x2={cursorPos.x}
                    y2={resizeOrigin.y}
                    stroke="#22d3ee"
                    strokeWidth={1.5}
                    opacity={0.7}
                  />
                  {/* Y축 = 세로폭 (마젠타) */}
                  <line
                    x1={cursorPos.x}
                    y1={resizeOrigin.y}
                    x2={cursorPos.x}
                    y2={cursorPos.y}
                    stroke="#e879f9"
                    strokeWidth={1.5}
                    opacity={0.7}
                  />
                  <circle cx={resizeOrigin.x} cy={resizeOrigin.y} r={3} fill="#22d3ee" />
                </svg>
              )}
            </div>

            {/* 좌상단 inferring 배지 */}
            {pendingCount > 0 && (
              <div className="absolute top-0 left-0 z-10 pt-2 pl-2 pointer-events-none">
                <div className="pointer-events-auto bg-black/70 backdrop-blur px-2.5 py-1 rounded text-[11px] font-mono text-emerald-300 flex items-center gap-1.5 transition-opacity duration-300 hover:opacity-40">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  확정 추론 {pendingCount}…
                </div>
              </div>
            )}

            {/* 우상단 카운터 + 액션 상태 */}
            <div className="absolute top-0 right-0 z-10 pt-2 pr-2 pointer-events-none">
              <div className={`pointer-events-auto bg-black/70 backdrop-blur px-2.5 py-1 rounded text-[11px] font-mono flex items-center gap-1.5 transition-opacity duration-300 ${cursorPos && cursorPos.x > CANVAS_SIZE - 160 && cursorPos.y < 160 ? "opacity-20" : "opacity-100"}`}>
                <span className="text-emerald-300">{confirmedCount}</span>
                <span className="text-gray-500">cells</span>
                {activeTool === "cursor" && <span className="text-red-400">· 삭제 모드 (Esc)</span>}
                {activeTool === "annotate" && <span className="text-emerald-500">· 추가 모드</span>}
                {isDragging && <span className="text-amber-300">· rotate</span>}
                {resizeOrigin && <span className="text-cyan-300">· resize</span>}
                {isPreviewPending && <span className="text-emerald-300">· preview…</span>}
              </div>
            </div>

            {/* 모드 전환 토스트 */}
            {modeToast && (
              <div className="pointer-events-none absolute top-10 inset-x-0 z-20 flex justify-center">
                <div className={`px-4 py-1.5 rounded-full text-[12px] font-semibold shadow-lg border ${
                  modeToast === "cursor"
                    ? "bg-red-950/90 text-red-300 border-red-700/60"
                    : "bg-emerald-900/95 text-emerald-100 border-emerald-600/70"
                }`}>
                  {modeToast === "cursor"
                    ? "삭제 모드 — 세포를 클릭해 삭제"
                    : "추가 모드 — 클릭으로 세포 추가"}
                </div>
              </div>
            )}

            {editingCellId !== null && (
              <div className="pointer-events-none absolute bottom-2 left-2 right-2 z-10 flex justify-center">
                <div className="bg-amber-500/95 text-black text-xs font-medium px-3 py-1.5 rounded shadow text-center">
                  {reinferPending ? (
                    <>#{cells.findIndex((c) => c.id === editingCellId) + 1} 재추론 중…</>
                  ) : (
                    <>#{cells.findIndex((c) => c.id === editingCellId) + 1} 재추론 모드 — <span className="font-bold">좌클릭</span>으로 즉시 적용</>
                  )}
                </div>
              </div>
            )}

            {/* 라벨 미선택 경고 (하단) */}
            {labelMissing && !editingCellId && activeTool !== "cursor" && (
              <div className="pointer-events-none absolute inset-x-0 bottom-10 z-10 flex justify-center">
                <div className="inline-block bg-yellow-500/90 text-black text-[11px] font-medium px-3 py-1 rounded shadow">
                  양성(+) 또는 음성(−) 라벨을 선택하세요
                </div>
              </div>
            )}

            {/* 파일 드래그 오버레이 */}
            {isDragOver && (
              <div className="pointer-events-none absolute inset-0 z-40 bg-blue-500/20 border-2 border-dashed border-blue-400 rounded-lg flex items-center justify-center">
                <p className="text-blue-300 font-medium text-sm bg-black/60 px-4 py-2 rounded">새 이미지로 교체</p>
              </div>
            )}
          </div>
        )}
        </div>

        {/* 결과 패널 (캔버스 하단) */}
        {naturalSize.w > 0 && (
          <div className="shrink-0 flex flex-col bg-gray-900/60" style={{ height: 240 }}>
            {/* 섹션 타이틀 바 — 삭제 모드일 때 안내로 전환 (높이 유지 → 캔버스 크기 불변) */}
            {activeTool === "cursor" && editingCellId === null ? (
              <div className="shrink-0 border-t border-red-800/60 h-9 px-5 flex items-center gap-3 bg-red-900/70 text-red-200 whitespace-nowrap overflow-hidden">
                <span className="flex items-center gap-1 font-bold text-xs shrink-0">삭제 모드 (Esc)</span>
                <div className="h-4 w-px bg-white/30 shrink-0" />
                <div className="flex items-center gap-x-4 text-[11px] min-w-0">
                  <span className="flex items-center gap-1.5">
                    <kbd className="rounded border border-white/40 bg-white/25 px-1.5 py-0.5 text-[10px] font-semibold leading-none">좌클릭</kbd>
                    세포 삭제
                  </span>
                  <span className="flex items-center gap-1.5">
                    <kbd className="rounded border border-white/40 bg-white/25 px-1.5 py-0.5 text-[10px] font-semibold leading-none">Esc</kbd>
                    모드 해제
                  </span>
                  <span className="flex items-center gap-1.5">
                    <kbd className="rounded border border-white/40 bg-white/25 px-1.5 py-0.5 text-[10px] font-semibold leading-none">Ctrl+Esc</kbd>
                    전체 삭제
                  </span>
                </div>
              </div>
            ) : (
              <div className="shrink-0 border-t border-gray-700 h-9 px-5 flex items-center gap-3 bg-gray-900">
                <span className="text-[9px] uppercase tracking-widest text-gray-500 font-semibold">결과</span>
                <div className="flex-1 h-px bg-gray-800" />
                {confirmedCount > 0 && ki67Rate !== null && (
                  <span className="text-[11px] tabular-nums">
                    <span className="text-gray-500 text-[10px] mr-1.5">Ki-67</span>
                    <span className="font-bold text-emerald-300">{ki67Rate.toFixed(1)}%</span>
                  </span>
                )}
              </div>
            )}
            <div className="flex-1 min-h-0 px-5 py-3 flex gap-6 items-start overflow-x-auto">

              {/* 추론 시간 */}
              <div className="min-w-[160px] shrink-0">
                <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-1.5 font-semibold">추론 시간</div>
                {latencies.length === 0 ? (
                  <p className="text-[11px] text-gray-500">기록 없음</p>
                ) : (
                  <div className="text-[11px] space-y-1">
                    <Row k="방금" v={lastEntry?.ms} cellLabel={cellLabel(lastEntry?.cellId, cells)} highlight />
                    <Row k="평균" v={stats?.avg} />
                  </div>
                )}
                <p className="text-[10px] text-emerald-400 mt-1.5 font-mono truncate" title={MODEL_NAME}>
                  {MODEL_NAME}
                </p>
              </div>

              {/* 세포 목록 */}
              {cells.length > 0 && (
                <div className="flex-1 min-w-[240px] flex flex-col h-full">
                  <div className="flex items-center gap-2 mb-1.5 shrink-0">
                    <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">
                      세포 목록
                    </span>
                    <span className="text-gray-500 text-[10px]">
                      {confirmedCount}개{pendingCount > 0 && ` · 추론중 ${pendingCount}`}
                    </span>
                    <div className="ml-auto flex items-center gap-1">
                      {(["all", "positive", "negative"] as const).map((f) => (
                        <button
                          key={f}
                          onClick={() => setCellFilter(f)}
                          className={`px-1.5 py-0.5 rounded text-[9px] transition border ${
                            cellFilter === f
                              ? f === "positive"
                                ? "bg-red-600 border-red-500 text-white"
                                : f === "negative"
                                ? "bg-blue-600 border-blue-500 text-white"
                                : "bg-gray-600 border-gray-500 text-white"
                              : "bg-gray-800 border-gray-700 text-gray-400 hover:bg-gray-700"
                          }`}
                        >
                          {f === "all" ? "전체" : f === "positive" ? "양성+" : "음성−"}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="rounded border border-gray-800 divide-y divide-gray-800 overflow-y-auto flex-1 min-h-0">
                    {cells
                      .filter((c) => cellFilter === "all" || c.pending || c.kiLabel === cellFilter)
                      .map((cell, _filteredIdx) => {
                        const idx = cells.indexOf(cell);
                        return (
                          <div
                            key={cell.id}
                            ref={(el) => { if (el) cellRowRefs.current.set(cell.id, el); else cellRowRefs.current.delete(cell.id); }}
                            draggable={!cell.pending && editingCellId === null}
                            onMouseEnter={() => { hoveredFromCanvasRef.current = false; setHoveredCellId(cell.id); }}
                            onMouseLeave={() => setHoveredCellId(null)}
                            onDragStart={(e) => handleCellDragStart(e, cell.id)}
                            onDragOver={(e) => handleCellDragOver(e, cell.id)}
                            onDrop={(e) => handleCellDrop(e, cell.id)}
                            onDragEnd={handleCellDragEnd}
                            className={`px-2 py-1.5 flex items-center gap-1.5 text-[11px] transition-colors ${
                              cell.pending ? "opacity-60" : ""
                            } ${
                              dragOverCellId === cell.id && dragCellId !== cell.id
                                ? "border-t-2 border-blue-400"
                                : ""
                            } ${
                              dragCellId === cell.id
                                ? "opacity-40"
                                : editingCellId === cell.id
                                ? "bg-amber-900/30 ring-1 ring-inset ring-amber-500/70"
                                : hoveredCellId === cell.id
                                ? "bg-gray-800/90 ring-1 ring-inset ring-emerald-500/50"
                                : "hover:bg-gray-800/50"
                            }`}
                          >
                            {!cell.pending && editingCellId === null && (
                              <span className="shrink-0 text-gray-600 cursor-grab active:cursor-grabbing select-none text-[10px] leading-none pr-0.5">⠿</span>
                            )}
                            <span
                              className={`shrink-0 tabular-nums transition-all ${
                                hoveredCellId === cell.id
                                  ? "text-lg font-bold text-white min-w-[2.5rem]"
                                  : "text-[11px] text-gray-400 w-10"
                              }`}
                            >
                              #{idx + 1}
                            </span>
                            {cell.pending ? (
                              <span className="text-emerald-300 italic flex-1 font-mono">…확정 추론</span>
                            ) : (
                              <>
                                <button
                                  onClick={() => handleSetKiLabel(cell.id, "positive")}
                                  className={`px-1.5 py-0.5 rounded transition ${
                                    cell.kiLabel === "positive"
                                      ? "bg-red-600 text-white"
                                      : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                                  }`}
                                >
                                  +
                                </button>
                                <button
                                  onClick={() => handleSetKiLabel(cell.id, "negative")}
                                  className={`px-1.5 py-0.5 rounded transition ${
                                    cell.kiLabel === "negative"
                                      ? "bg-blue-600 text-white"
                                      : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                                  }`}
                                >
                                  −
                                </button>
                                <span
                                  className="text-gray-500 text-[10px] ml-0.5 tabular-nums"
                                  title="이 세포를 확정할 때 계산에 걸린 시간"
                                >
                                  {formatLatency(cell.inferenceMs)}
                                </span>
                              </>
                            )}
                            <div className="ml-auto flex items-center gap-1 shrink-0">
                              {!cell.pending && (
                                <>
                                  {editingCellId === cell.id ? (
                                    <>
                                      {reinferPending && (
                                        <span className="text-[10px] text-gray-400 animate-pulse">추론 중…</span>
                                      )}
                                      <button
                                        onClick={handleCancelReedit}
                                        className="px-1.5 py-0.5 rounded text-[10px] bg-gray-800 hover:bg-gray-700 text-gray-400"
                                      >
                                        취소
                                      </button>
                                    </>
                                  ) : (
                                    <button
                                      onClick={() => handleStartReedit(cell)}
                                      disabled={editingCellId !== null}
                                      className="px-1.5 py-0.5 rounded text-[10px] bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 disabled:opacity-40 disabled:cursor-not-allowed"
                                    >
                                      재추론
                                    </button>
                                  )}
                                </>
                              )}
                              <button
                                onClick={() => handleDeleteCell(cell.id)}
                                className="px-1.5 py-0.5 rounded text-[10px] bg-red-900/40 hover:bg-red-800/50 text-red-400"
                              >
                                삭제
                              </button>
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}

              {/* Ki-67 지수 + JSON 저장 */}
              <div className="min-w-[160px] shrink-0 flex flex-col gap-3">
                {confirmedCount > 0 && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-1.5 font-semibold">Ki-67 지수</div>
                    <div className="rounded border border-gray-800 bg-gray-950/60 px-3 py-2 text-[11px] space-y-1.5">
                      <div className="flex justify-between gap-4">
                        <span className="text-red-400">양성</span>
                        <span className="tabular-nums text-gray-200">{positiveCount}개</span>
                      </div>
                      <div className="flex justify-between gap-4">
                        <span className="text-blue-400">음성</span>
                        <span className="tabular-nums text-gray-200">{negativeCount}개</span>
                      </div>
                      <div className="flex justify-between gap-4 border-t border-gray-800 pt-1.5">
                        <span className="text-gray-400 font-medium">지수</span>
                        <span className="tabular-nums font-bold text-emerald-300">
                          {ki67Rate !== null ? `${ki67Rate.toFixed(1)}%` : "—"}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                {error && (
                  <div className="rounded border border-red-800 bg-red-900/30 px-2 py-1.5 text-[11px] text-red-300 whitespace-pre-line">
                    {error}
                  </div>
                )}
                <button
                  onClick={saveJson}
                  disabled={confirmedCount === 0}
                  className="w-full py-2 rounded text-xs font-medium bg-emerald-700 hover:bg-emerald-600 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed transition"
                >
                  JSON 저장 {confirmedCount > 0 && <span className="opacity-80">({confirmedCount})</span>}
                </button>
              </div>

            </div>
          </div>
        )}
      </section>

      {/* 툴바 리사이즈 핸들 */}
      <div
        onMouseDown={(e) => {
          isToolbarDraggingRef.current = true;
          toolbarDragStartXRef.current = e.clientX;
          toolbarDragStartWidthRef.current = toolbarWidth;
          e.preventDefault();
        }}
        className="w-1 shrink-0 cursor-col-resize bg-gray-800 hover:bg-blue-500/50 transition-colors"
      />
      {/* ─────────── 우측: 툴바 ─────────── */}
      <aside className="shrink-0 border-l border-gray-800 bg-gray-900/70 flex flex-col overflow-hidden min-h-0" style={{ width: toolbarWidth }}>
        <div className="px-4 py-3 border-b border-gray-800">
          <div className="flex items-center justify-between">
            <h1 className="text-sm font-semibold tracking-wide">Ki-67 · Realtime</h1>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-900/50 text-emerald-300 border border-emerald-700/60" title="OpenVINO 서버 추론 (FastAPI, 포트 8000)">
              OpenVINO
            </span>
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-hidden px-4 py-2 flex flex-col gap-2">
          {/* 도구 모드 */}
          <div>
            <ToolLabel>도구</ToolLabel>
            <div className="grid grid-cols-2 gap-1.5">
              <button
                onClick={() => switchTool("annotate")}
                className={`py-2 rounded text-xs font-semibold transition border flex items-center justify-center gap-1.5 ${
                  activeTool === "annotate"
                    ? "bg-emerald-700 border-emerald-500 text-white ring-1 ring-emerald-400/30"
                    : "bg-gray-800 border-gray-700 text-gray-400 hover:bg-gray-700 hover:text-gray-200"
                }`}
              >
                추가 모드
              </button>
              <button
                onClick={() => switchTool("cursor")}
                disabled={!cells.some((c) => !c.pending)}
                className={`py-2 rounded text-xs font-semibold transition border flex items-center justify-center gap-1.5 ${
                  activeTool === "cursor"
                    ? "bg-rose-700 border-rose-500 text-white ring-1 ring-rose-400/30"
                    : "bg-gray-800 border-gray-700 text-gray-400 hover:bg-gray-700 hover:text-gray-200 disabled:opacity-40 disabled:cursor-not-allowed"
                }`}
              >
                삭제 모드 (Esc)
              </button>
            </div>
          </div>

          {/* 이미지 */}
          <div>
            <ToolLabel>이미지</ToolLabel>
            <div className="flex items-center gap-1.5">
              <span className="flex-1 text-[11px] text-gray-400 truncate px-2 py-1.5 bg-gray-950 border border-gray-700 rounded">
                {imageFile?.name ?? "—"}
              </span>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="shrink-0 px-2.5 py-1.5 rounded text-[11px] bg-gray-800 hover:bg-gray-700 border border-gray-700 transition"
              >
                불러오기
              </button>
            </div>
          </div>

          {/* 라벨 */}
          <div>
            <ToolLabel>
              Ki-67 라벨
              <span className={`ml-1.5 ${labelMissing ? "text-yellow-400" : "text-gray-500"}`}>
                {labelMissing ? "필수" : "✓ 선택됨"}
              </span>
            </ToolLabel>
            <div className="grid grid-cols-2 gap-1.5">
              <button
                onClick={() => { setPendingKiLabel("positive"); switchTool("annotate"); }}
                className={`py-1.5 rounded text-xs font-medium transition border ${
                  pendingKiLabel === "positive"
                    ? "bg-red-600 border-red-500 text-white"
                    : labelMissing
                    ? "bg-gray-800 border-yellow-500/60 text-gray-200 hover:bg-gray-700 ring-1 ring-yellow-500/40"
                    : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
                }`}
              >
                양성 <span className="opacity-60">(P)</span>
              </button>
              <button
                onClick={() => { setPendingKiLabel("negative"); switchTool("annotate"); }}
                className={`py-1.5 rounded text-xs font-medium transition border ${
                  pendingKiLabel === "negative"
                    ? "bg-blue-600 border-blue-500 text-white"
                    : labelMissing
                    ? "bg-gray-800 border-yellow-500/60 text-gray-200 hover:bg-gray-700 ring-1 ring-yellow-500/40"
                    : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
                }`}
              >
                음성 <span className="opacity-60">(N)</span>
              </button>
            </div>
          </div>

          {/* 프롬프트 도형 */}
          <div>
            <ToolLabel>프롬프트 도형</ToolLabel>
            <div className="grid grid-cols-4 gap-1.5">
              {SHAPE_OPTIONS.map(({ sides, glyph, name }) => (
                <button
                  key={sides}
                  onClick={() => { setShapeSides(sides); switchTool("annotate"); }}
                  title={`${name} (${sides}점)`}
                  className={`py-2 rounded text-base font-medium transition border ${
                    shapeSides === sides
                      ? "bg-blue-600 border-blue-500 text-white"
                      : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
                  }`}
                >
                  {glyph}
                </button>
              ))}
            </div>
          </div>

          {/* 도형 모양 */}
          <div className="flex gap-2 items-stretch">

            {/* 왼쪽: 제목 + 슬라이더 3개 */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">도형 크기 · 회전</span>
                <button
                  onClick={() => { setShapeWidth(DEFAULT_SHAPE_WIDTH); setShapeHeight(DEFAULT_SHAPE_HEIGHT); setShapeRotationDeg(0); }}
                  title="도형 크기·회전 기본값으로 초기화"
                  className="text-[9px] px-1.5 py-0.5 rounded bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 transition"
                >
                  초기화
                </button>
              </div>
              <div className="space-y-1">
                <div>
                  <div className="flex justify-between text-[10px] text-gray-400 mb-0.5">
                    <span>가로폭</span>
                    <span className="font-mono text-gray-300">{Math.round(shapeWidth)}px</span>
                  </div>
                  <input
                    type="range"
                    min={SHAPE_DIM_MIN}
                    max={SHAPE_DIM_MAX}
                    value={shapeWidth}
                    onChange={(e) => setShapeWidth(Number(e.target.value))}
                    className="w-full accent-emerald-500"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] text-gray-400 mb-0.5">
                    <span>세로폭</span>
                    <span className="font-mono text-gray-300">{Math.round(shapeHeight)}px</span>
                  </div>
                  <input
                    type="range"
                    min={SHAPE_DIM_MIN}
                    max={SHAPE_DIM_MAX}
                    value={shapeHeight}
                    onChange={(e) => setShapeHeight(Number(e.target.value))}
                    className="w-full accent-emerald-500"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] text-gray-400 mb-0.5">
                    <span>회전</span>
                    <span className="font-mono text-gray-300">{displayRotation}°</span>
                  </div>
                  <div className="flex gap-1.5">
                    <input
                      type="range"
                      min={-180}
                      max={180}
                      step={1}
                      value={shapeRotationDeg}
                      onChange={(e) => setShapeRotationDeg(Number(e.target.value))}
                      className="flex-1 min-w-0 accent-emerald-500"
                    />
                    <button
                      onClick={resetRotation}
                      title="회전 0° 리셋 (R)"
                      className="shrink-0 px-2 py-0.5 rounded text-[10px] bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 transition"
                    >
                      0°
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* 오른쪽: 미리보기 박스 — 외접원 중심 기준으로 회전 */}
            {(() => {
              const previewPts = ellipsePointsSvg(shapeWidth, shapeHeight, shapeSides)
                .split(" ").map(p => { const [x, y] = p.split(",").map(Number); return [x, y] as [number, number]; });
              // 외접원 중심 = 타원 중심 (삼각형 등 홀수각 도형도 정확히 중심 회전)
              const bcx = shapeWidth / 2;
              const bcy = shapeHeight / 2;
              const R = Math.max(...previewPts.map(([x, y]) => Math.hypot(x - bcx, y - bcy)));
              const halfSize = R * 1.2;
              return (
                <div className="rounded border border-gray-600 bg-gray-800 shrink-0 flex items-center justify-center overflow-hidden w-40">
                  <svg
                    width={120}
                    height={120}
                    viewBox={`${bcx - halfSize} ${bcy - halfSize} ${halfSize * 2} ${halfSize * 2}`}
                    preserveAspectRatio="xMidYMid meet"
                  >
                    <polygon
                      points={ellipsePointsSvg(shapeWidth, shapeHeight, shapeSides)}
                      transform={`rotate(${shapeRotationDeg}, ${bcx}, ${bcy})`}
                      fill={
                        pendingKiLabel === "positive"
                          ? "rgba(239,68,68,0.25)"
                          : pendingKiLabel === "negative"
                          ? "rgba(59,130,246,0.25)"
                          : "rgba(156,163,175,0.2)"
                      }
                      stroke={
                        pendingKiLabel === "positive"
                          ? "#ef4444"
                          : pendingKiLabel === "negative"
                          ? "#3b82f6"
                          : "#9ca3af"
                      }
                      strokeWidth={Math.max(shapeWidth, shapeHeight) / 25}
                    />
                  </svg>
                </div>
              );
            })()}

          </div>

          {/* 단축키 */}
          <div>
            <ToolLabel>단축키</ToolLabel>
            <div className="rounded border border-gray-800 bg-gray-950/40 px-2.5 py-1.5 space-y-1.5">
              <div className="space-y-0.5">
                <p className="text-[9px] text-gray-600 uppercase tracking-widest mb-0.5">마우스</p>
                {([
                  [["좌클릭"], "세포 추가"],
                  [["Esc", "좌클릭"], "세포 삭제", "→"],
                  [["좌드래그"], "도형 회전"],
                  [["우드래그"], "크기 조정"],
                  [["wheel"], "도형 크기"],
                  [["ctrl", "wheel"], "줌"],
                  [["ctrl", "좌드래그"], "화면 이동"],
                ] as [string[], string, string?][]).map(([keys, desc, sep]) => (
                  <div key={desc} className="flex items-center gap-1.5 min-w-0">
                    <div className="flex items-center gap-1 shrink-0 min-w-[5.5rem]">
                      {keys.map((k, i) => (
                        <span key={k} className="flex items-center gap-1">
                          {i > 0 && <span className="text-gray-600 text-[9px]">{sep ?? "+"}</span>}
                          <Kbd>{k}</Kbd>
                        </span>
                      ))}
                    </div>
                    <span className="text-gray-700 text-[9px] shrink-0">→</span>
                    <span className="text-gray-400 text-[10px]">{desc}</span>
                  </div>
                ))}
              </div>
              <div className="border-t border-gray-800 pt-1.5 space-y-0.5">
                <p className="text-[9px] text-gray-600 uppercase tracking-widest mb-0.5">키보드</p>
                {([
                  [["P"], "양성 라벨 선택"],
                  [["N"], "음성 라벨 선택"],
                  [["R"], "회전 0° 리셋"],
                  [["Z"], "마지막 Undo"],
                  [["Y"], "Undo 되돌리기 (Redo)"],
                  [["Esc"], "삭제 모드 토글"],
                  [["ctrl", "Esc"], "전체 세포 삭제"],
                ] as [string[], string][]).map(([keys, desc]) => (
                  <div key={desc} className="flex items-center gap-1.5 min-w-0">
                    <div className="flex items-center gap-1 shrink-0 min-w-[5.5rem]">
                      {keys.map((k, i) => (
                        <span key={k} className="flex items-center gap-1">
                          {i > 0 && <span className="text-gray-600 text-[9px]">+</span>}
                          <Kbd>{k}</Kbd>
                        </span>
                      ))}
                    </div>
                    <span className="text-gray-700 text-[9px] shrink-0">→</span>
                    <span className="text-gray-400 text-[10px]">{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 액션 */}
          <div>
            <ToolLabel>액션</ToolLabel>
            <div className="grid grid-cols-2 gap-1.5 mb-1.5">
              <button
                onClick={handleUndo}
                disabled={cells.length === 0}
                className="py-1.5 rounded text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
              >
                Undo <span className="text-gray-500">(Z)</span>
              </button>
              <button
                onClick={handleRedo}
                disabled={redoCount === 0}
                className="py-1.5 rounded text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
              >
                Redo <span className="text-gray-500">(Y)</span>
              </button>
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <button
                onClick={() => {
                  setZoom(1);
                  setPan({ x: 0, y: 0 });
                }}
                className="py-1.5 rounded text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 transition"
              >
                줌 초기화
              </button>
              <button
                onClick={handleResetAll}
                disabled={cells.length === 0}
                className="py-1.5 rounded text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
              >
                Reset <span className="text-gray-500">(Esc)</span>
              </button>
            </div>
          </div>

        </div>

      </aside>

    </main>
  );
}

// ── 프레젠테이션 ────────────────────────────────────────────────────────────
/** 확정 세그먼트 1회 소요 시간 표시 (초 + ms 병기) */
function formatLatency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  const sec = ms / 1000;
  const secStr = sec < 10 ? sec.toFixed(2) : sec.toFixed(1);
  return `${secStr}초 (${Math.round(ms)}ms)`;
}

function ToolLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-1.5 font-semibold">
      {children}
    </div>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex items-center px-1.5 py-0.5 rounded border border-gray-600 bg-gray-700 text-gray-200 font-mono text-[9px] leading-none whitespace-nowrap">
      {children}
    </kbd>
  );
}

function cellLabel(cellId: number | undefined, cells: Cell[]): string | undefined {
  if (cellId === undefined) return undefined;
  const idx = cells.findIndex((c) => c.id === cellId);
  return idx >= 0 ? `#${idx + 1}` : undefined;
}

function Row({
  k,
  v,
  cellLabel: label,
  kind = "time",
  highlight = false,
}: {
  k: string;
  v: number | null | undefined;
  cellLabel?: string;
  kind?: "time" | "count";
  highlight?: boolean;
}) {
  let display = "—";
  if (v !== null && v !== undefined) {
    display = kind === "count" ? `${Math.round(v)}회` : formatLatency(v);
  }
  return (
    <div className="flex justify-between gap-2">
      <span className="text-gray-500 shrink-0">{k}</span>
      <span
        className={`text-right tabular-nums ${highlight ? "text-emerald-300 font-medium" : "text-gray-200"}`}
      >
        {label && <span className="text-gray-500 font-mono mr-1">{label}</span>}
        {display}
      </span>
    </div>
  );
}
