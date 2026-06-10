export type Point = { x: number; y: number; label: 1 | 0 };

export type Cell = {
  id: number;
  points: Point[];
  polyline: [number, number][];
  kiLabel: "positive" | "negative";
  inferenceMs: number;
  pending: boolean;
};

export type SegmentResult = {
  polyline: [number, number][];
  inferenceMs: number;
};

export type SegmentFn = (
  image: HTMLImageElement,
  points: Point[],
) => Promise<SegmentResult>;

export type ShapeState = {
  sides: 3 | 4 | 5 | 6;
  width: number;
  height: number;
  rotationDeg: number;
};

export type ResizeOrigin = {
  x: number;
  y: number;
  startW: number;
  startH: number;
  startBoxW: number;
  startBoxH: number;
};

export type CellMaskSource = {
  id: number;
  pending: boolean;
  polyline: [number, number][];
};

export const MODEL_NAME = "e11_holdout_float32_webgpu_int8_wasm";
export const CANVAS_SIZE = 768;
export const SHAPE_OPTIONS: {
  sides: 3 | 4 | 5 | 6;
  glyph: string;
  name: string;
}[] = [
  { sides: 3, glyph: "△", name: "삼각형" },
  { sides: 4, glyph: "□", name: "사각형" },
  { sides: 5, glyph: "⬠", name: "오각형" },
  { sides: 6, glyph: "⬡", name: "육각형" },
];
export const SUPPORTED_EXTS = [
  ".jpg",
  ".jpeg",
  ".png",
  ".bmp",
  ".tif",
  ".tiff",
];
export const SAMPLE_IMAGES = [
  { src: "/samples/sample1.png", name: "sample1.png", label: "샘플 1" },
];
export const SHAPE_DIM_MIN = 12;
export const SHAPE_DIM_MAX = 240;
