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

// WebGPU: float32 모델 (양자화 op 미지원, GPU 완전 활용) — HuggingFace CDN
export const ONNX_ENCODER_URL_WEBGPU = "https://huggingface.co/seoyeonlee/ki67-sam-onnx/resolve/main/encoder.onnx";
export const ONNX_DECODER_URL_WEBGPU = "https://huggingface.co/seoyeonlee/ki67-sam-onnx/resolve/main/decoder.onnx";
// WASM: INT8 양자화 모델 (CPU에서 더 빠름)
export const ONNX_ENCODER_URL_WASM = "https://huggingface.co/seoyeonlee/ki67-sam-onnx/resolve/main/encoder.onnx";
export const ONNX_DECODER_URL_WASM = "https://huggingface.co/seoyeonlee/ki67-sam-onnx/resolve/main/decoder.onnx";
// 하위 호환용 (기존 코드가 참조하는 경우)
export const ONNX_ENCODER_URL = ONNX_ENCODER_URL_WASM;
export const ONNX_DECODER_URL = ONNX_DECODER_URL_WASM;
export const MODEL_NAME = "e11_holdout_float32_webgpu_int8_wasm";
export const IMAGE_ENCODER_INPUT_SIZE = 512;
export const PROMPT_ENCODER_INPUT_SIZE = 1024;
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
export const DEFAULT_SHAPE_WIDTH = 64;
export const DEFAULT_SHAPE_HEIGHT = 64;
export const SHAPE_DIM_MIN = 12;
export const SHAPE_DIM_MAX = 240;
export const DRAG_THRESHOLD = 8;
export const PREVIEW_DEBOUNCE_MS = 80;
