"use client";

import React from "react";
import {
  type ShapeState,
  SHAPE_OPTIONS,
  SHAPE_DIM_MIN,
  SHAPE_DIM_MAX,
  MODEL_NAME,
} from "../types";
import { ellipsePointsSvg } from "../utils/segmentation";
import { ToolLabel, Kbd } from "./ui";

type BackendState = {
  current: "webgpu" | "wasm" | null;
  modelsReady: boolean;
  backendLatencies: { webgpu: number[]; wasm: number[] };
};

type BenchmarkResult = { median: number; p95: number; min: number; max: number; n: number };
type BenchmarkStatus = { backend: "webgpu" | "wasm"; phase: "loading" | "warmup" | "measuring"; step: number; total: number } | null;

type Props = {
  toolbarWidth: number;
  activeTool: "cursor" | "annotate";
  onSwitchTool: (tool: "annotate" | "cursor") => void;
  hasCells: boolean;
  backendState: BackendState;
  onSwitchBackend: (ep: "webgpu" | "wasm") => void;
  benchmarkResults: { webgpu: BenchmarkResult | null; wasm: BenchmarkResult | null };
  benchmarkStatus: BenchmarkStatus;
  onRunBenchmark: (ep: "webgpu" | "wasm") => void;
  onRunFullBenchmark: () => void;
  hasImage: boolean;
  imageFileName: string | undefined;
  onFileInputClick: () => void;
  pendingKiLabel: "positive" | "negative" | null;
  onSetLabel: (label: "positive" | "negative") => void;
  shapeState: ShapeState;
  onShapeChange: (updates: Partial<ShapeState>) => void;
  onResetShapeSize: () => void;
  onResetRotation: () => void;
  onZoomReset: () => void;
  onUndo: () => void;
  onResetAll: () => void;
  cellCount: number;
};

export function Toolbar({
  toolbarWidth,
  activeTool,
  onSwitchTool,
  hasCells,
  backendState,
  onSwitchBackend,
  benchmarkResults,
  benchmarkStatus,
  onRunBenchmark,
  onRunFullBenchmark,
  hasImage,
  imageFileName,
  onFileInputClick,
  pendingKiLabel,
  onSetLabel,
  shapeState,
  onShapeChange,
  onResetShapeSize,
  onResetRotation,
  onZoomReset,
  onUndo,
  onResetAll,
  cellCount,
}: Props) {
  const { current: backend, modelsReady } = backendState;
  const { sides, width, height, rotationDeg } = shapeState;
  const labelMissing = pendingKiLabel === null;
  const displayRotation = ((Math.round(rotationDeg) % 360) + 360) % 360;

  return (
    <aside
      className="shrink-0 border-l border-gray-800 bg-gray-900/70 flex flex-col overflow-hidden min-h-0"
      style={{ width: toolbarWidth }}
    >
      {/* 헤더 */}
      <div className="px-4 py-3 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <h1 className="text-sm font-semibold tracking-wide">
            Ki-67 · Realtime
          </h1>
          <span
            className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-900/50 text-emerald-300 border border-emerald-700/60"
            title="브라우저 ONNX 추론 (onnxruntime-web)"
          >
            ONNX
          </span>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-hidden px-4 py-1 flex flex-col justify-evenly gap-1">
        {/* 도구 모드 */}
        <div>
          <ToolLabel>도구</ToolLabel>
          <div className="grid grid-cols-2 gap-1.5">
            <button
              onClick={() => onSwitchTool("annotate")}
              className={`py-2 rounded text-xs font-semibold transition border flex items-center justify-center gap-1.5 ${
                activeTool === "annotate"
                  ? "bg-emerald-700 border-emerald-500 text-white ring-1 ring-emerald-400/30"
                  : "bg-gray-800 border-gray-700 text-gray-400 hover:bg-gray-700 hover:text-gray-200"
              }`}
            >
              추가 모드
            </button>
            <button
              onClick={() => onSwitchTool("cursor")}
              disabled={!hasCells}
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

        {/* 추론 백엔드 */}
        <div>
          <ToolLabel>
            추론 백엔드
            <span className={`ml-1.5 text-[9px] font-normal ${modelsReady ? "text-emerald-400" : "text-yellow-500"}`}>
              {modelsReady ? "✓ 모델 준비됨" : "버튼 클릭 후 모델 로드"}
            </span>
          </ToolLabel>
          <div className="grid grid-cols-2 gap-1.5">
            {(["webgpu", "wasm"] as const).map((ep) => (
              <button
                key={ep}
                onClick={() => onSwitchBackend(ep)}
                className={`py-1.5 rounded text-xs font-semibold transition border flex items-center justify-center gap-1 ${
                  backend === ep && modelsReady
                    ? "bg-violet-700 border-violet-500 text-white ring-1 ring-violet-400/30"
                    : backend === ep
                      ? "bg-violet-900/40 border-violet-700/60 text-violet-300"
                      : "bg-gray-800 border-gray-700 text-gray-400 hover:bg-gray-700 hover:text-gray-200"
                }`}
              >
                {backend === ep && modelsReady && <span className="w-1.5 h-1.5 rounded-full bg-violet-300 shrink-0" />}
                {ep === "webgpu" ? "WebGPU" : "CPU (WASM)"}
              </button>
            ))}
          </div>
          {!modelsReady && (
            <p className="text-[9px] text-gray-600 mt-1 leading-relaxed">
              ① 백엔드 선택 → ② 이미지 로드 → ③ 세그멘테이션
            </p>
          )}
        </div>

        {/* 벤치마크 */}
        <div>
          <ToolLabel>벤치마크</ToolLabel>
          <p className="text-[9px] text-gray-500 mb-1.5 leading-relaxed">
            이미지 중앙 고정 입력으로 워밍업 5회(결과 제외) 후 20회 측정합니다.
            평균 대신 <span className="text-gray-300">중앙값(median)</span>과{" "}
            <span className="text-gray-300">P95</span>를 사용해 이상치에 강한 비교를 제공합니다.
          </p>

          {/* 전체 비교 실행 버튼 */}
          {(() => {
            const isRunning = benchmarkStatus !== null;
            const hasBothResults = benchmarkResults.webgpu !== null && benchmarkResults.wasm !== null;
            let label = "WebGPU vs CPU 전체 비교 실행";
            if (isRunning) {
              const cur = benchmarkStatus!.backend === "webgpu" ? "WebGPU" : "CPU";
              if (benchmarkStatus!.phase === "loading") label = `${cur} 로드 중…`;
              else if (benchmarkStatus!.phase === "warmup") label = `${cur} 워밍업 ${benchmarkStatus!.step}/${benchmarkStatus!.total}`;
              else label = `${cur} 측정 ${benchmarkStatus!.step}/${benchmarkStatus!.total}`;
            } else if (hasBothResults) {
              label = "전체 비교 재실행";
            }
            return (
              <button
                onClick={onRunFullBenchmark}
                disabled={!hasImage || isRunning}
                className={`w-full py-2 rounded text-xs font-bold transition border flex items-center justify-center gap-1.5 mb-2 ${
                  isRunning
                    ? "bg-yellow-700/60 border-yellow-500/60 text-yellow-200 cursor-wait"
                    : hasBothResults
                      ? "bg-emerald-800/50 border-emerald-600/60 text-emerald-200 hover:bg-emerald-700/50"
                      : "bg-violet-800/60 border-violet-600/60 text-violet-100 hover:bg-violet-700/60 disabled:opacity-40 disabled:cursor-not-allowed"
                }`}
              >
                {isRunning && <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse shrink-0" />}
                {hasBothResults && !isRunning && <span className="text-emerald-400">✓</span>}
                {label}
              </button>
            );
          })()}

          {/* 각 백엔드별 측정 버튼 */}
          <div className="grid grid-cols-2 gap-1.5">
            {(["webgpu", "wasm"] as const).map((ep) => {
              const isRunning = benchmarkStatus?.backend === ep;
              const isDone = benchmarkResults[ep] !== null;
              const isOtherRunning = benchmarkStatus !== null && benchmarkStatus.backend !== ep;
              const label = ep === "webgpu" ? "WebGPU" : "CPU";

              let statusText = `${label} 측정`;
              if (isRunning && benchmarkStatus) {
                if (benchmarkStatus.phase === "loading") statusText = "로드 중…";
                else if (benchmarkStatus.phase === "warmup") statusText = `워밍업 ${benchmarkStatus.step}/${benchmarkStatus.total}`;
                else statusText = `측정 ${benchmarkStatus.step}/${benchmarkStatus.total}`;
              } else if (isDone) {
                statusText = `${label} 재측정`;
              }

              return (
                <button
                  key={ep}
                  onClick={() => onRunBenchmark(ep)}
                  disabled={!hasImage || benchmarkStatus !== null}
                  className={`py-1.5 rounded text-xs font-semibold transition border flex items-center justify-center gap-1 ${
                    isRunning
                      ? "bg-yellow-700/60 border-yellow-500/60 text-yellow-200 cursor-wait"
                      : isDone && !isOtherRunning
                        ? "bg-emerald-900/40 border-emerald-700/50 text-emerald-300 hover:bg-emerald-800/40"
                        : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed"
                  }`}
                >
                  {isRunning && <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse shrink-0" />}
                  {isDone && !isRunning && <span className="text-emerald-400">✓</span>}
                  {statusText}
                </button>
              );
            })}
          </div>

          {/* 진행 상황 안내 */}
          {benchmarkStatus && (
            <div className="mt-2 rounded border border-yellow-800/60 bg-yellow-950/30 px-2.5 py-2">
              {benchmarkStatus.phase === "loading" && (
                <p className="text-[9px] text-yellow-300">
                  {benchmarkStatus.backend === "webgpu" ? "WebGPU" : "CPU"} 모델을 로드하고 있습니다…
                </p>
              )}
              {benchmarkStatus.phase === "warmup" && (
                <>
                  <p className="text-[9px] text-yellow-300 font-semibold mb-0.5">
                    워밍업 중 ({benchmarkStatus.step}/{benchmarkStatus.total})
                  </p>
                  <p className="text-[9px] text-yellow-600 leading-relaxed">
                    첫 추론은 JIT·shader 컴파일로 느립니다. 이 구간을 제외하고 실제 속도를 측정합니다.
                  </p>
                  <div className="mt-1.5 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-yellow-600 rounded-full transition-all duration-300"
                      style={{ width: `${(benchmarkStatus.step / benchmarkStatus.total) * 100}%` }} />
                  </div>
                </>
              )}
              {benchmarkStatus.phase === "measuring" && (
                <>
                  <p className="text-[9px] text-yellow-300 font-semibold mb-0.5">
                    측정 중 ({benchmarkStatus.step}/{benchmarkStatus.total})
                  </p>
                  <p className="text-[9px] text-yellow-600 leading-relaxed">
                    동일한 이미지 중앙 포인트로 반복 추론 중입니다.
                  </p>
                  <div className="mt-1.5 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-emerald-500 rounded-full transition-all duration-300"
                      style={{ width: `${(benchmarkStatus.step / benchmarkStatus.total) * 100}%` }} />
                  </div>
                </>
              )}
            </div>
          )}

          {/* 벤치마크 결과 비교 */}
          {(benchmarkResults.webgpu || benchmarkResults.wasm) && !benchmarkStatus && (() => {
            const r = benchmarkResults;
            const hasBoth = r.webgpu !== null && r.wasm !== null;
            const maxMedian = Math.max(r.webgpu?.median ?? 0, r.wasm?.median ?? 0);
            const faster = hasBoth
              ? (r.webgpu!.median <= r.wasm!.median ? "webgpu" : "wasm")
              : null;
            const ratio = hasBoth
              ? (Math.max(r.webgpu!.median, r.wasm!.median) / Math.min(r.webgpu!.median, r.wasm!.median)).toFixed(1)
              : null;
            return (
              <div className="mt-2 rounded border border-gray-700 bg-gray-950/60 px-2.5 py-2 space-y-2">
                <div className="flex items-center justify-between mb-1">
                  <p className="text-[9px] text-gray-500 uppercase tracking-widest">벤치마크 결과</p>
                  {hasBoth && (
                    <span className="text-[8px] bg-emerald-800/60 text-emerald-200 px-1.5 py-0.5 rounded font-bold">
                      {faster === "webgpu" ? "WebGPU" : "CPU"} {ratio}× 빠름
                    </span>
                  )}
                </div>
                {(["webgpu", "wasm"] as const).map((ep) => {
                  const res = r[ep];
                  const isWinner = faster === ep;
                  const pct = maxMedian > 0 && res ? (res.median / maxMedian) * 100 : 0;
                  return (
                    <div key={ep}>
                      <div className="flex items-center justify-between text-[10px] mb-0.5">
                        <span className={isWinner ? "text-emerald-300 font-semibold" : "text-gray-400"}>
                          {ep === "webgpu" ? "WebGPU" : "CPU (WASM)"}
                          {res && <span className="text-gray-600 font-normal ml-1">{res.n}회</span>}
                        </span>
                        <span className="font-mono text-gray-200">
                          {res ? `median ${res.median.toFixed(0)}ms` : <span className="text-gray-600">미측정</span>}
                        </span>
                      </div>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                        {res && (
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${isWinner || !hasBoth ? "bg-emerald-500" : "bg-gray-500"}`}
                            style={{ width: `${pct}%` }}
                          />
                        )}
                      </div>
                      {res && (
                        <p className="text-[9px] text-gray-600 mt-0.5">
                          P95 {res.p95.toFixed(0)}ms · min {res.min.toFixed(0)}ms · max {res.max.toFixed(0)}ms
                        </p>
                      )}
                    </div>
                  );
                })}
                {!hasBoth && (
                  <p className="text-[9px] text-gray-600 pt-0.5 border-t border-gray-800">
                    반대 백엔드도 측정하면 비교가 표시됩니다
                  </p>
                )}
              </div>
            );
          })()}
        </div>

        {/* 이미지 */}
        <div>
          <ToolLabel>이미지</ToolLabel>
          <div className="flex items-center gap-1.5">
            <span className="flex-1 text-[11px] text-gray-400 truncate px-2 py-1.5 bg-gray-950 border border-gray-700 rounded">
              {imageFileName ?? "—"}
            </span>
            <button
              onClick={onFileInputClick}
              className="shrink-0 px-2.5 py-1.5 rounded text-[11px] bg-gray-800 hover:bg-gray-700 border border-gray-700 transition"
            >
              불러오기
            </button>
          </div>
        </div>

        {/* Ki-67 라벨 */}
        <div>
          <ToolLabel>
            Ki-67 라벨
            <span
              className={`ml-1.5 ${labelMissing ? "text-yellow-400" : "text-gray-500"}`}
            >
              {labelMissing ? "필수" : "✓ 선택됨"}
            </span>
          </ToolLabel>
          <div className="grid grid-cols-2 gap-1.5">
            <button
              onClick={() => onSetLabel("positive")}
              className={`py-1.5 rounded text-xs font-medium transition border flex items-center justify-center gap-1.5 ${
                pendingKiLabel === "positive"
                  ? "bg-red-600 border-red-500 text-white"
                  : labelMissing
                    ? "bg-gray-800 border-yellow-500/60 text-gray-200 hover:bg-gray-700 ring-1 ring-yellow-500/40"
                    : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
              }`}
            >
              양성(+) <Kbd>P</Kbd>
            </button>
            <button
              onClick={() => onSetLabel("negative")}
              className={`py-1.5 rounded text-xs font-medium transition border flex items-center justify-center gap-1.5 ${
                pendingKiLabel === "negative"
                  ? "bg-blue-600 border-blue-500 text-white"
                  : labelMissing
                    ? "bg-gray-800 border-yellow-500/60 text-gray-200 hover:bg-gray-700 ring-1 ring-yellow-500/40"
                    : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
              }`}
            >
              음성(−) <Kbd>N</Kbd>
            </button>
          </div>
        </div>

        {/* 프롬프트 도형 */}
        <div>
          <ToolLabel>프롬프트 도형</ToolLabel>
          <div className="grid grid-cols-4 gap-1.5">
            {SHAPE_OPTIONS.map(({ sides: s, glyph, name }) => (
              <button
                key={s}
                onClick={() => {
                  onShapeChange({ sides: s });
                  onSwitchTool("annotate");
                }}
                title={`${name} (${s}점)`}
                className={`py-2 rounded text-base font-medium transition border ${
                  sides === s
                    ? "bg-blue-600 border-blue-500 text-white"
                    : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
                }`}
              >
                {glyph}
              </button>
            ))}
          </div>
        </div>

        {/* 도형 크기 · 회전 */}
        <div className="flex gap-2 items-stretch">
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">
                도형 크기 · 회전
              </span>
              <button
                onClick={onResetShapeSize}
                title="도형 크기 기본값으로 초기화"
                className="text-[9px] px-1.5 py-0.5 rounded bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 transition"
              >
                초기화
              </button>
            </div>
            <div className="space-y-1.5">
              <div>
                <div className="flex justify-between text-[10px] text-gray-400 mb-0.5">
                  <span>가로폭</span>
                  <span className="font-mono text-gray-300">
                    {Math.round(width)}px
                  </span>
                </div>
                <input
                  type="range"
                  min={SHAPE_DIM_MIN}
                  max={SHAPE_DIM_MAX}
                  value={width}
                  onChange={(e) =>
                    onShapeChange({ width: Number(e.target.value) })
                  }
                  className="w-full accent-emerald-500"
                />
              </div>
              <div>
                <div className="flex justify-between text-[10px] text-gray-400 mb-0.5">
                  <span>세로폭</span>
                  <span className="font-mono text-gray-300">
                    {Math.round(height)}px
                  </span>
                </div>
                <input
                  type="range"
                  min={SHAPE_DIM_MIN}
                  max={SHAPE_DIM_MAX}
                  value={height}
                  onChange={(e) =>
                    onShapeChange({ height: Number(e.target.value) })
                  }
                  className="w-full accent-emerald-500"
                />
              </div>
              <div>
                <div className="flex justify-between text-[10px] text-gray-400 mb-0.5">
                  <span>회전</span>
                  <span className="font-mono text-gray-300">
                    {displayRotation}°
                  </span>
                </div>
                <div className="flex gap-1.5">
                  <input
                    type="range"
                    min={-180}
                    max={180}
                    step={1}
                    value={rotationDeg}
                    onChange={(e) =>
                      onShapeChange({ rotationDeg: Number(e.target.value) })
                    }
                    className="flex-1 accent-emerald-500"
                  />
                  <button
                    onClick={onResetRotation}
                    title="회전 0° 리셋 (R)"
                    className="shrink-0 px-2 py-0.5 rounded text-[10px] bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 transition"
                  >
                    0°
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* 도형 미리보기 */}
          {(() => {
            const previewPts = ellipsePointsSvg(width, height, sides)
              .split(" ")
              .map((p) => {
                const [x, y] = p.split(",").map(Number);
                return [x, y] as [number, number];
              });
            const bcx = width / 2;
            const bcy = height / 2;
            const R = Math.max(
              ...previewPts.map(([x, y]) => Math.hypot(x - bcx, y - bcy)),
            );
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
                    points={ellipsePointsSvg(width, height, sides)}
                    transform={`rotate(${rotationDeg}, ${bcx}, ${bcy})`}
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
                    strokeWidth={Math.max(width, height) / 25}
                  />
                </svg>
              </div>
            );
          })()}
        </div>

        {/* 단축키 */}
        <div>
          <ToolLabel>단축키</ToolLabel>
          <div className="rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5 grid grid-cols-2 gap-x-2">
            <div>
              <p className="text-[9px] text-gray-600 uppercase tracking-widest mb-1">
                마우스
              </p>
              {(
                [
                  [["좌클릭"], "추가"],
                  [["Esc", "클릭"], "삭제", "→"],
                  [["좌드래그"], "회전"],
                  [["우드래그"], "크기"],
                  [["wheel"], "도형 크기"],
                  [["ctrl", "wheel"], "줌"],
                  [["ctrl", "드래그"], "이동"],
                ] as [string[], string, string?][]
              ).map(([keys, desc, sep]) => (
                <div
                  key={desc}
                  className="flex items-center gap-1 min-w-0 mb-0.5"
                >
                  <div className="flex items-center gap-0.5 shrink-0">
                    {keys.map((k, i) => (
                      <span key={k} className="flex items-center gap-0.5">
                        {i > 0 && (
                          <span className="text-gray-600 text-[8px]">
                            {sep ?? "+"}
                          </span>
                        )}
                        <Kbd>{k}</Kbd>
                      </span>
                    ))}
                  </div>
                  <span className="text-gray-600 text-[8px] shrink-0">→</span>
                  <span className="text-gray-400 text-[9px] truncate">
                    {desc}
                  </span>
                </div>
              ))}
            </div>
            <div>
              <p className="text-[9px] text-gray-600 uppercase tracking-widest mb-1">
                키보드
              </p>
              {(
                [
                  [["P"], "양성 라벨"],
                  [["N"], "음성 라벨"],
                  [["R"], "회전 0° 리셋"],
                  [["Z"], "마지막 Undo"],
                  [["Esc"], "삭제 모드"],
                  [["ctrl", "Esc"], "전체 삭제"],
                ] as [string[], string][]
              ).map(([keys, desc]) => (
                <div
                  key={desc}
                  className="flex items-center gap-1 min-w-0 mb-0.5"
                >
                  <div className="flex items-center gap-0.5 shrink-0">
                    {keys.map((k, i) => (
                      <span key={k} className="flex items-center gap-0.5">
                        {i > 0 && (
                          <span className="text-gray-600 text-[8px]">+</span>
                        )}
                        <Kbd>{k}</Kbd>
                      </span>
                    ))}
                  </div>
                  <span className="text-gray-600 text-[8px] shrink-0">→</span>
                  <span className="text-gray-400 text-[9px] truncate">
                    {desc}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 액션 */}
        <div>
          <ToolLabel>액션</ToolLabel>
          <div className="grid grid-cols-3 gap-1.5">
            <button
              onClick={onZoomReset}
              className="py-1.5 rounded text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 transition"
            >
              줌 초기화
            </button>
            <button
              onClick={onUndo}
              disabled={cellCount === 0}
              className="py-1.5 rounded text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
            >
              Undo <span className="text-gray-500">(Z)</span>
            </button>
            <button
              onClick={onResetAll}
              disabled={cellCount === 0}
              className="py-1.5 rounded text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
            >
              Reset <span className="text-gray-500">(Esc)</span>
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
}
