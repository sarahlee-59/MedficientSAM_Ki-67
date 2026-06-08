"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const ENCODER_INT8_URL = "/models/encoder.quantized.onnx";
const DECODER_INT8_URL = "/models/decoder.quantized.onnx";
const ENCODER_FP32_URL = "/api/onnx/encoder";
const DECODER_FP32_URL = "/api/onnx/decoder";
const IMAGE_URL        = "/samples/sample1.png";
const PROMPT_SCALE     = 1024 / 256;

const WARMUP_RUNS = 5;
const BENCH_RUNS  = 50;

type Point = { x: number; y: number; label: 1 };

const CELLS: { name: string; ki: "+" | "−"; points: Point[] }[] = [
  {
    name: "Cell 1", ki: "+",
    points: [
      { x: 188, y:  62, label: 1 },
      { x: 201, y:  71, label: 1 },
      { x: 196, y:  85, label: 1 },
      { x: 181, y:  85, label: 1 },
      { x: 176, y:  71, label: 1 },
    ],
  },
  {
    name: "Cell 2", ki: "+",
    points: [
      { x:  81, y: 147, label: 1 },
      { x:  95, y: 158, label: 1 },
      { x:  90, y: 174, label: 1 },
      { x:  72, y: 174, label: 1 },
      { x:  67, y: 158, label: 1 },
    ],
  },
  {
    name: "Cell 3", ki: "+",
    points: [
      { x:  34, y: 191, label: 1 },
      { x:  46, y: 205, label: 1 },
      { x:  42, y: 229, label: 1 },
      { x:  26, y: 229, label: 1 },
      { x:  22, y: 205, label: 1 },
    ],
  },
  {
    name: "Cell 4", ki: "+",
    points: [
      { x: 129, y: 184, label: 1 },
      { x: 143, y: 201, label: 1 },
      { x: 138, y: 227, label: 1 },
      { x: 121, y: 227, label: 1 },
      { x: 116, y: 201, label: 1 },
    ],
  },
  {
    name: "Cell 5", ki: "−",
    points: [
      { x: 135, y:  78, label: 1 },
      { x: 149, y:  92, label: 1 },
      { x: 122, y:  92, label: 1 },
    ],
  },
  {
    name: "Cell 6", ki: "−",
    points: [
      { x:  18, y:  91, label: 1 },
      { x:  29, y: 111, label: 1 },
      { x:   8, y: 111, label: 1 },
    ],
  },
  {
    name: "Cell 7", ki: "−",
    points: [
      { x: 237, y: 130, label: 1 },
      { x: 243, y: 140, label: 1 },
      { x: 237, y: 150, label: 1 },
      { x: 224, y: 150, label: 1 },
      { x: 218, y: 140, label: 1 },
      { x: 224, y: 130, label: 1 },
    ],
  },
  {
    name: "Cell 8", ki: "−",
    points: [
      { x: 174, y:  29, label: 1 },
      { x: 179, y:  38, label: 1 },
      { x: 174, y:  47, label: 1 },
      { x: 163, y:  47, label: 1 },
      { x: 157, y:  38, label: 1 },
      { x: 163, y:  29, label: 1 },
    ],
  },
];

// ─── Statistics ────────────────────────────────────────────

function calcMedian(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
}

function calcP95(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.max(0, Math.ceil(0.95 * s.length) - 1)];
}

function calcStddev(arr: number[]): number {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length);
}

// ─── CPU-fallback Log Analyzer (temporary debug helper) ───

function summarizeCpuFallbacks(lines: string[], label: string): void {
  const cpuLines = lines.filter(l => /\bcpu\b/i.test(l));
  if (cpuLines.length === 0) {
    console.log(
      `[CPU fallback · ${label}] CPU 관련 로그 없음`,
      `(캡처된 전체 로그: ${lines.length}줄)`,
    );
    if (lines.length > 0) {
      console.groupCollapsed("[CPU fallback] 캡처 샘플 (최대 20줄)");
      lines.slice(0, 20).forEach(l => console.log(l));
      console.groupEnd();
    }
    return;
  }

  // Match known ORT op names; multiple matches per line deduped via Set
  const opRe =
    /\b(Shape|Reshape|Gather(?:Elements|ND)?|Slice|Squeeze|Unsqueeze|Concat|Cast|Expand|Tile|NonZero|Where|TopK|Pad(?:ding)?|Flatten|Transpose|Range|ConstantOfShape|ScatterElements|ScatterND|Size|Trilu|Unique|Conv|MatMul|Gemm|BatchNormalization|LayerNormalization|GroupNormalization|Relu|Sigmoid|Softmax|Gelu|Erf|Tanh|Add|Sub|Mul|Div|Sqrt|Pow|Clip|Neg|Abs|Exp|Log)\b/g;

  const opCount: Record<string, number> = {};
  for (const line of cpuLines) {
    opRe.lastIndex = 0;
    const seen = new Set<string>();
    let m: RegExpExecArray | null;
    while ((m = opRe.exec(line)) !== null) {
      if (!seen.has(m[1])) {
        seen.add(m[1]);
        opCount[m[1]] = (opCount[m[1]] ?? 0) + 1;
      }
    }
    if (seen.size === 0) opCount["(op미상)"] = (opCount["(op미상)"] ?? 0) + 1;
  }

  console.group(`[CPU fallback · ${label} encoder] CPU 관련 로그 ${cpuLines.length}줄`);
  console.table(
    Object.entries(opCount)
      .sort(([, a], [, b]) => b - a)
      .map(([op, count]) => ({ op, count })),
  );
  console.groupCollapsed("원본 CPU 로그 전체");
  cpuLines.forEach(l => console.log(l));
  console.groupEnd();
  console.groupEnd();
}

// ─── Image / Tensor Utilities ──────────────────────────────

function imageToRgb(img: HTMLImageElement): { rgb: Uint8Array; w: number; h: number } {
  const { naturalWidth: w, naturalHeight: h } = img;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d")!.drawImage(img, 0, 0);
  const rgba = c.getContext("2d")!.getImageData(0, 0, w, h).data;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4) {
    rgb[j++] = rgba[i]; rgb[j++] = rgba[i + 1]; rgb[j++] = rgba[i + 2];
  }
  return { rgb, w, h };
}

function buildEncFeed(rgb: Uint8Array, w: number, h: number): Record<string, ort.Tensor> {
  return {
    image: new ort.Tensor("uint8", rgb, [h, w, 3]),
    original_size: new ort.Tensor("int16", new Int16Array([h, w]), [2]),
  };
}

function buildDecFeed(emb: ort.Tensor, pts: Point[]): Record<string, ort.Tensor> {
  const k = pts.length;
  const coords = new Float32Array(k * 2);
  const labels = new Float32Array(k);
  for (let i = 0; i < k; i++) {
    coords[i * 2]     = pts[i].x * PROMPT_SCALE;
    coords[i * 2 + 1] = pts[i].y * PROMPT_SCALE;
    labels[i] = pts[i].label;
  }
  return {
    image_embeddings: emb,
    point_coords: new ort.Tensor("float32", coords, [1, k, 2]),
    point_labels: new ort.Tensor("float32", labels, [1, k]),
  };
}

// ─── Result Types ──────────────────────────────────────────

type CellStats = {
  name: string;
  ki: "+" | "−";
  med: number;
  p95: number;
  std: number;
};

type RunResult = {
  label: string;
  backend: "wasm" | "webgpu";
  isApiDelivered: boolean;
  modelLoadMs: number;
  warmupMs: number;
  encMed: number;
  encP95: number;
  encStd: number;
  cells: CellStats[];
  decSumMed: number;
  decSumP95: number;
  decSumStd: number;
  totalMed: number;
  totalP95: number;
  totalStd: number;
  error?: string;
};

// ─── Benchmark Core ────────────────────────────────────────

async function runBrowserBenchmark(
  img: HTMLImageElement,
  ep: "wasm" | "webgpu",
  encoderUrl: string,
  decoderUrl: string,
  label: string,
  isApiDelivered: boolean,
  onStatus: (s: string) => void,
  debugCpuFallback = false,
): Promise<RunResult> {
  const { rgb, w, h } = imageToRgb(img);

  // Sessions created once, reused for all runs
  onStatus(`[${label}] 모델 로드 중…`);
  const t0 = performance.now();

  // Encoder session — verbose only when debugCpuFallback is set (first WebGPU run)
  let enc!: ort.InferenceSession;
  if (debugCpuFallback) {
    const captured: string[] = [];
    const orig = {
      log:   console.log,
      warn:  console.warn,
      debug: console.debug,
      info:  console.info,
    };
    const tap = (...args: unknown[]) =>
      captured.push(args.map(a => (typeof a === "object" ? JSON.stringify(a) : String(a))).join(" "));
    console.log   = (...a) => { tap(...a); orig.log(...a); };
    console.warn  = (...a) => { tap(...a); orig.warn(...a); };
    console.debug = (...a) => { tap(...a); orig.debug(...a); };
    console.info  = (...a) => { tap(...a); orig.info(...a); };
    try {
      enc = await ort.InferenceSession.create(encoderUrl, {
        executionProviders: [ep],
        logSeverityLevel: 0,
        logVerbosityLevel: 10,
      });
    } finally {
      console.log   = orig.log;
      console.warn  = orig.warn;
      console.debug = orig.debug;
      console.info  = orig.info;
    }
    summarizeCpuFallbacks(captured, label);
  } else {
    enc = await ort.InferenceSession.create(encoderUrl, { executionProviders: [ep] });
  }

  const dec = await ort.InferenceSession.create(decoderUrl, { executionProviders: [ep] });
  const modelLoadMs = performance.now() - t0;

  // Encoder inputs created once outside measurement loop
  const encFeed = buildEncFeed(rgb, w, h);

  // For WebGPU: read output data to force GPU→CPU sync before stopping the clock
  async function drainOutputs(out: Record<string, ort.Tensor>): Promise<void> {
    if (ep === "webgpu") {
      await Promise.all(Object.values(out).map(t => t.getData()));
    }
  }

  // Warmup: WARMUP_RUNS encoder + WARMUP_RUNS decoder (excluded from stats)
  onStatus(`[${label}] 워밍업 중… (enc×${WARMUP_RUNS} + dec×${WARMUP_RUNS})`);
  const tw = performance.now();
  let warmEmb!: ort.Tensor;
  for (let i = 0; i < WARMUP_RUNS; i++) {
    const out = await enc.run(encFeed);
    warmEmb = out.image_embeddings as ort.Tensor;
    await drainOutputs(out);
  }
  const warmDecFeed = buildDecFeed(warmEmb, CELLS[0].points);
  for (let i = 0; i < WARMUP_RUNS; i++) {
    const out = await dec.run(warmDecFeed);
    await drainOutputs(out);
  }
  const warmupMs = performance.now() - tw;

  // Encoder benchmark
  onStatus(`[${label}] 인코더 측정 중… (${BENCH_RUNS}회)`);
  const encTimes: number[] = [];
  let lastEmb!: ort.Tensor;
  for (let i = 0; i < BENCH_RUNS; i++) {
    const t = performance.now();
    const out = await enc.run(encFeed);
    lastEmb = out.image_embeddings as ort.Tensor;
    await drainOutputs(out);
    encTimes.push(performance.now() - t);
  }

  // Decoder feeds pre-created per cell outside the measurement loop
  const decFeeds = CELLS.map(cell => buildDecFeed(lastEmb, cell.points));

  // Decoder benchmark
  const cells: CellStats[] = [];
  for (let ci = 0; ci < CELLS.length; ci++) {
    const cell = CELLS[ci];
    onStatus(`[${label}] 디코더 — ${cell.name} (${BENCH_RUNS}회)…`);
    const times: number[] = [];
    for (let i = 0; i < BENCH_RUNS; i++) {
      const t = performance.now();
      const out = await dec.run(decFeeds[ci]);
      await drainOutputs(out);
      times.push(performance.now() - t);
    }
    cells.push({
      name: cell.name,
      ki: cell.ki,
      med: calcMedian(times),
      p95: calcP95(times),
      std: calcStddev(times),
    });
  }

  const encMed  = calcMedian(encTimes);
  const encP95v = calcP95(encTimes);
  const encStdv = calcStddev(encTimes);

  const decSumMed = cells.reduce((s, c) => s + c.med, 0);
  const decSumP95 = cells.reduce((s, c) => s + c.p95, 0);
  // Combine cell std assuming independence: σ_total = √(Σσᵢ²)
  const decSumStd = Math.sqrt(cells.reduce((s, c) => s + c.std ** 2, 0));

  return {
    label, backend: ep, isApiDelivered, modelLoadMs, warmupMs,
    encMed, encP95: encP95v, encStd: encStdv,
    cells, decSumMed, decSumP95, decSumStd,
    totalMed: encMed + decSumMed,
    totalP95: encP95v + decSumP95,
    totalStd: Math.sqrt(encStdv ** 2 + decSumStd ** 2),
  };
}

// ─── Configs ──────────────────────────────────────────────

const CONFIGS: {
  ep: "wasm" | "webgpu";
  encUrl: string;
  decUrl: string;
  label: string;
  isApiDelivered: boolean;
}[] = [
  { ep: "wasm",   encUrl: ENCODER_INT8_URL, decUrl: DECODER_INT8_URL, label: "WASM (int8)",  isApiDelivered: false },
  { ep: "webgpu", encUrl: ENCODER_INT8_URL, decUrl: DECODER_INT8_URL, label: "GPU (int8)",   isApiDelivered: false },
  { ep: "webgpu", encUrl: ENCODER_FP32_URL, decUrl: DECODER_FP32_URL, label: "GPU (fp32) †", isApiDelivered: true  },
];

// ─── Display Helpers ──────────────────────────────────────

function ms(v: number): string {
  return v < 10 ? v.toFixed(1) : Math.round(v).toString();
}

function Ratio({ base, val }: { base: number | undefined; val: number | undefined }) {
  if (!base || !val) return <span className="text-gray-600">—</span>;
  const r = base / val;
  const c = r >= 2 ? "text-emerald-400" : r >= 1.1 ? "text-yellow-300" : "text-gray-400";
  return <span className={`text-[10px] font-bold ${c}`}>{r.toFixed(2)}×</span>;
}

type StatRow = {
  label: string;
  labelColor: string;
  note: string;
  getMed: (r: RunResult) => number;
  getP95: (r: RunResult) => number;
  getStd: (r: RunResult) => number;
};

function StatCell({
  r,
  row,
  className = "",
}: {
  r: RunResult | undefined;
  row: StatRow;
  className?: string;
}) {
  const base = `text-right pr-6 align-top py-3 ${className}`;
  if (!r) return <td className={`${base} text-gray-600`}>—</td>;
  if (r.error) return <td className={`${base} text-red-400 text-xs`}>{r.error.slice(0, 40)}</td>;
  return (
    <td className={base}>
      <div className="font-bold text-green-300 tabular-nums">{ms(row.getMed(r))}</div>
      <div className="text-xs text-yellow-400 tabular-nums mt-0.5">p95 {ms(row.getP95(r))}</div>
      <div className="text-xs text-gray-400 tabular-nums">±{ms(row.getStd(r))}</div>
    </td>
  );
}

function RatioCell({
  base,
  r,
  getMed,
  className = "",
}: {
  base: RunResult | undefined;
  r: RunResult | undefined;
  getMed: (x: RunResult) => number;
  className?: string;
}) {
  const b = base && !base.error ? getMed(base) : undefined;
  const v = r && !r.error ? getMed(r) : undefined;
  return (
    <td className={`text-right pr-6 align-top py-3 ${className}`}>
      <Ratio base={b} val={v} />
    </td>
  );
}

// ─── GPU Adapter Info ─────────────────────────────────────

type GpuInfo = { vendor: string; architecture: string; description: string } | null;

// ─── Page ─────────────────────────────────────────────────

export default function BenchmarkPage() {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [ready, setReady]     = useState(false);
  const [status, setStatus]   = useState("이미지 로드 중…");
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<RunResult[]>([]);
  const [secure, setSecure]   = useState<boolean | null>(null);
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | "unavailable" | null>(null);

  useEffect(() => {
    const isSecure =
      typeof window !== "undefined" &&
      (window.isSecureContext || location.hostname === "localhost");
    setSecure(isSecure);

    if (isSecure && typeof navigator !== "undefined" && "gpu" in navigator) {
      (navigator as unknown as { gpu: { requestAdapter: () => Promise<unknown> } }).gpu
        .requestAdapter()
        .then((adapter) => {
          if (!adapter) { setGpuInfo("unavailable"); return; }
          const info =
            ((adapter as Record<string, unknown>).info as GpuInfo) ??
            { vendor: "", architecture: "", description: "WebGPU 사용 가능" };
          setGpuInfo(info);
        })
        .catch(() => setGpuInfo("unavailable"));
    } else {
      setGpuInfo("unavailable");
    }

    const img = new Image();
    img.onload  = () => { imgRef.current = img; setReady(true); setStatus("준비 완료"); };
    img.onerror = () => setStatus("이미지 로드 실패");
    img.src = IMAGE_URL;
  }, []);

  async function handleRun() {
    if (!ready || !imgRef.current) return;
    setRunning(true);
    setResults([]);
    const acc: RunResult[] = [];
    let gpuDebugDone = false;
    for (const cfg of CONFIGS) {
      // verbose CPU-fallback capture only for the first WebGPU encoder session
      const debugCpuFallback = cfg.ep === "webgpu" && !gpuDebugDone;
      if (debugCpuFallback) gpuDebugDone = true;
      try {
        acc.push(
          await runBrowserBenchmark(
            imgRef.current,
            cfg.ep,
            cfg.encUrl,
            cfg.decUrl,
            cfg.label,
            cfg.isApiDelivered,
            setStatus,
            debugCpuFallback,
          ),
        );
      } catch (e) {
        acc.push({
          label: cfg.label, backend: cfg.ep, isApiDelivered: cfg.isApiDelivered,
          modelLoadMs: 0, warmupMs: 0,
          encMed: 0, encP95: 0, encStd: 0,
          cells: [],
          decSumMed: 0, decSumP95: 0, decSumStd: 0,
          totalMed: 0, totalP95: 0, totalStd: 0,
          error: e instanceof Error ? e.message : String(e),
        });
      }
      setResults([...acc]);
    }
    setStatus(`벤치마크 완료 (워밍업 ${WARMUP_RUNS}회 제외, 본 측정 ${BENCH_RUNS}회)`);

    // Encoder stats — compare to judge if 4% gap (WASM vs GPU int8) is signal or noise
    const encTable = acc
      .filter(r => !r.error)
      .map(r => ({
        label: r.label,
        "enc median (ms)": +r.encMed.toFixed(1),
        "enc p95 (ms)": +r.encP95.toFixed(1),
        "enc ±σ (ms)": +r.encStd.toFixed(1),
        "σ/median (%)": +((r.encStd / r.encMed) * 100).toFixed(1),
      }));
    console.table(encTable);

    setRunning(false);
  }

  const wasmR    = results.find(r => r.label === "WASM (int8)");
  const gpuInt8R = results.find(r => r.label === "GPU (int8)");
  const gpuFp32R = results.find(r => r.isApiDelivered);

  const svgCells = CELLS.map(cell => ({
    ...cell,
    path: cell.points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x} ${p.y}`).join(" ") + " Z",
    cx: cell.points.reduce((s, p) => s + p.x, 0) / cell.points.length,
    cy: cell.points.reduce((s, p) => s + p.y, 0) / cell.points.length,
  }));

  const statRows: StatRow[] = [
    {
      label: "추론 합계", labelColor: "text-white font-bold",
      note: "이미지 1장 처리 총 시간",
      getMed: r => r.totalMed,
      getP95: r => r.totalP95,
      getStd: r => r.totalStd,
    },
    {
      label: "인코더", labelColor: "text-yellow-300 font-semibold",
      note: "이미지 분석 (병목 구간)",
      getMed: r => r.encMed,
      getP95: r => r.encP95,
      getStd: r => r.encStd,
    },
    {
      label: "디코더 (세포 1개)", labelColor: "text-cyan-300 font-semibold",
      note: "세포 추가할수록 누적됨",
      getMed: r => r.cells.length > 0 ? r.decSumMed / r.cells.length : 0,
      getP95: r => r.cells.length > 0 ? r.decSumP95 / r.cells.length : 0,
      getStd: r => r.cells.length > 0 ? r.decSumStd / r.cells.length : 0,
    },
  ];

  const sep = "border-l border-dashed border-gray-700 pl-4";

  return (
    <main className="min-h-screen bg-gray-950 text-white p-8 font-mono text-sm">
      <div className="max-w-4xl mx-auto space-y-6">

        <h1 className="text-2xl font-bold">WebGPU vs WASM 추론 벤치마크</h1>

        {/* WebGPU adapter status */}
        <div className="rounded-lg border px-4 py-3 text-xs border-gray-700 bg-gray-900/50">
          {secure === false ? (
            <p className="text-amber-300">
              ⚠ 비보안 컨텍스트 —{" "}
              <code className="bg-gray-800 px-1 rounded">localhost</code>로 접속하세요.
            </p>
          ) : gpuInfo === null ? (
            <p className="text-gray-500">WebGPU 어댑터 조회 중…</p>
          ) : gpuInfo === "unavailable" ? (
            <p className="text-red-400">✗ WebGPU 어댑터 없음</p>
          ) : (
            <p className="text-emerald-400 font-bold">
              ✓ WebGPU 사용 가능
              {gpuInfo.description && (
                <span className="ml-3 text-gray-300 font-normal">{gpuInfo.description}</span>
              )}
              {gpuInfo.vendor && (
                <span className="ml-3 text-gray-500 font-normal">{gpuInfo.vendor}</span>
              )}
            </p>
          )}
        </div>

        {/* Image preview + cell table */}
        <div className="flex gap-8 items-start">
          <div className="relative shrink-0">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={IMAGE_URL}
              alt="sample1"
              className="w-64 h-64 border border-gray-700 rounded"
              style={{ imageRendering: "pixelated" }}
            />
            <svg
              className="absolute inset-0 w-64 h-64 pointer-events-none"
              viewBox="0 0 256 256"
            >
              {svgCells.map((cell, i) => {
                const color = cell.ki === "+" ? "#ef4444" : "#60a5fa";
                return (
                  <g key={i}>
                    <path d={cell.path} fill="none" stroke={color} strokeWidth="1.5" opacity="0.9" />
                    {cell.points.map((p, j) => (
                      <circle key={j} cx={p.x} cy={p.y} r="2" fill={color} opacity="0.9" />
                    ))}
                    <text x={cell.cx + 4} y={cell.cy - 4} fontSize="7" fill={color} fontWeight="bold">
                      {i + 1}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          <table className="text-xs text-gray-300 border-collapse self-start">
            <thead>
              <tr className="border-b border-gray-700 text-gray-500">
                <th className="text-left pr-4 py-1">세포</th>
                <th className="text-center pr-4">Ki-67</th>
                <th className="text-left pr-4">점 수</th>
                <th className="text-left">중심 (px)</th>
              </tr>
            </thead>
            <tbody>
              {svgCells.map((cell, i) => (
                <tr key={i} className="border-b border-gray-800">
                  <td className="pr-4 py-0.5">{cell.name}</td>
                  <td
                    className={`text-center pr-4 font-bold ${
                      cell.ki === "+" ? "text-red-400" : "text-blue-400"
                    }`}
                  >
                    {cell.ki}
                  </td>
                  <td className="pr-4 text-gray-500">{cell.points.length}pt</td>
                  <td className="text-gray-400">
                    ({Math.round(cell.cx)}, {Math.round(cell.cy)})
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Run button */}
        <div className="flex items-center gap-4">
          <button
            onClick={handleRun}
            disabled={!ready || running}
            className="px-6 py-2 bg-violet-600 hover:bg-violet-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg font-semibold transition text-base"
          >
            {running ? "측정 중…" : "벤치마크 시작"}
          </button>
          <span className="text-gray-400 text-xs">{status}</span>
        </div>

        {/* Results */}
        {results.length > 0 && (
          <div className="space-y-6">

            <div>
              <h2 className="text-base font-semibold text-gray-200 mb-1">
                결과 (단위: ms)
              </h2>
              <p className="text-[10px] text-gray-500 mb-3">
                각 칸: 중앙값 · p95 · ±표준편차 / 워밍업 {WARMUP_RUNS}회 제외 · 본 측정{" "}
                {BENCH_RUNS}회 / 추론 합계 p95·σ는 enc + dec 합산 (독립 가정, 보수적 추정)
              </p>

              <table className="border-collapse text-sm">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400 text-xs">
                    <th className="text-left py-2 pr-10">지표</th>
                    <th className="text-right pr-6">WASM (int8)</th>
                    <th className="text-right pr-2">GPU (int8)</th>
                    <th className="text-right pr-6 text-[10px]">vs WASM</th>
                    <th className={`text-right pr-2 ${sep}`}>
                      GPU (fp32) †
                    </th>
                    <th className="text-right pr-0 text-[10px]">vs WASM</th>
                  </tr>
                </thead>
                <tbody>
                  {statRows.map((row, i) => (
                    <tr key={i} className="border-b border-gray-800">
                      <td className={`pr-10 ${row.labelColor} align-top py-3`}>
                        {row.label}
                        <span className="ml-2 text-gray-500 font-normal text-xs">
                          {row.note}
                        </span>
                      </td>
                      <StatCell r={wasmR}    row={row} />
                      <StatCell r={gpuInt8R} row={row} />
                      <RatioCell base={wasmR} r={gpuInt8R} getMed={row.getMed} />
                      <StatCell r={gpuFp32R} row={row} className={sep} />
                      <RatioCell base={wasmR} r={gpuFp32R} getMed={row.getMed} />
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* GPU fp32 footnote */}
              <div className="mt-3 pt-2 border-t border-gray-800 space-y-1 text-[10px] text-gray-500">
                <p>
                  <span className="text-amber-400">†</span> GPU (fp32) 레이블이지만{" "}
                  <code className="text-gray-400">/api/onnx/encoder</code>·
                  <code className="text-gray-400">decoder</code>는 실제로{" "}
                  <code className="text-gray-400">encoder/decoder.quantized.onnx</code>(int8)를 반환 —
                  별도 fp32 모델 없음.
                </p>
                <p>
                  <span className="font-semibold text-gray-400">측정 경계:</span>{" "}
                  session.run() 타이밍은 WASM/GPU int8과 동일. 네트워크 왕복은 측정에 포함되지 않고
                  모델 로드 시간(InferenceSession.create)에만 HTTP 오버헤드가 더해짐.
                </p>
                <p>
                  <span className="font-semibold text-gray-400">서버 추론 분리 방법 (참고):</span>{" "}
                  API가 브라우저 대신 서버에서 추론을 실행한다면, 응답 헤더{" "}
                  <code className="text-gray-400">X-Inference-Ms</code>로 서버 연산 시간을 분리하거나
                  <code className="text-gray-400"> PerformanceResourceTiming</code>으로
                  네트워크 구간을 측정할 수 있음.
                </p>
              </div>
            </div>

            {results.some(r => !r.error) && (
              <div>
                <h2 className="text-sm font-semibold text-gray-400 mb-2">모델 로드 시간 (ms)</h2>
                <div className="flex gap-6 text-xs">
                  {results
                    .filter(r => !r.error)
                    .map(r => (
                      <div key={r.label} className="flex flex-col">
                        <span className="text-gray-500">{r.label}</span>
                        <span className="text-white font-bold text-base">
                          {ms(r.modelLoadMs)}
                        </span>
                        <span className="text-gray-600">워밍업 {ms(r.warmupMs)} ms</span>
                      </div>
                    ))}
                </div>
              </div>
            )}

          </div>
        )}
      </div>
    </main>
  );
}
