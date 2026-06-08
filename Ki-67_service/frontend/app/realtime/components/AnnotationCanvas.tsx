"use client";

import React from "react";
import {
  type Cell,
  type ShapeState,
  type ResizeOrigin,
  CANVAS_SIZE,
  SAMPLE_IMAGES,
} from "../types";
import { ellipsePointsSvg } from "../utils/segmentation";

type ViewState = {
  zoom: number;
  pan: { x: number; y: number };
  viewportScale: number;
};

type InteractionState = {
  activeTool: "cursor" | "annotate";
  isDragging: boolean;
  dragOrigin: { x: number; y: number } | null;
  resizeOrigin: ResizeOrigin | null;
  cursorPos: { x: number; y: number } | null;
  hoveredCellId: number | null;
  editingCellId: number | null;
};

type OverlayState = {
  loadingMsg: string | null;
  pendingCount: number;
  confirmedCount: number;
  isPreviewPending: boolean;
  modeToast: "annotate" | "cursor" | null;
  reinferPending: boolean;
  isDragOver: boolean;
  modelsReady: boolean;
};

type Props = {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  canvasWrapperRef: React.RefObject<HTMLDivElement | null>;
  canvasAreaRef: React.RefObject<HTMLDivElement | null>;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  naturalSize: { w: number; h: number };
  viewState: ViewState;
  shapeState: ShapeState;
  shapeCenter: { x: number; y: number } | null;
  interactionState: InteractionState;
  overlayState: OverlayState;
  pendingKiLabel: "positive" | "negative" | null;
  cells: Cell[];
  onMouseDown: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  onMouseMove: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  onMouseUp: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  onMouseLeave: () => void;
  onDrop: (e: React.DragEvent) => void;
  setIsDragOver: (v: boolean) => void;
  onLoadSample: (src: string, name: string) => void;
};

export function AnnotationCanvas({
  canvasRef,
  canvasWrapperRef,
  canvasAreaRef,
  fileInputRef,
  naturalSize,
  viewState,
  shapeState,
  shapeCenter,
  interactionState,
  overlayState,
  pendingKiLabel,
  cells,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  onMouseLeave,
  onDrop,
  setIsDragOver,
  onLoadSample,
}: Props) {
  const { zoom, pan, viewportScale } = viewState;
  const { sides, width, height, rotationDeg } = shapeState;
  const {
    activeTool,
    isDragging,
    dragOrigin,
    resizeOrigin,
    cursorPos,
    hoveredCellId,
    editingCellId,
  } = interactionState;
  const {
    loadingMsg,
    pendingCount,
    confirmedCount,
    isPreviewPending,
    modeToast,
    reinferPending,
    isDragOver,
    modelsReady,
  } = overlayState;

  const labelMissing = pendingKiLabel === null;

  return (
    <div
      ref={canvasAreaRef}
      className="flex-1 min-h-0 flex items-center justify-center p-6 overflow-hidden"
    >
      {!naturalSize.w ? (
        /* ── 업로드 존 ── */
        <div className="max-w-xl w-full flex flex-col gap-4">
          {!modelsReady && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-violet-900/40 border border-violet-700/60 text-violet-300 text-xs">
              <span className="text-base">⚠</span>
              우측 패널에서 추론 백엔드(WebGPU 또는 CPU)를 먼저 선택하세요
            </div>
          )}
          <div
            onClick={() => {
              if (modelsReady) fileInputRef.current?.click();
            }}
            onDrop={modelsReady ? onDrop : undefined}
            onDragOver={
              modelsReady
                ? (e) => {
                    e.preventDefault();
                    setIsDragOver(true);
                  }
                : undefined
            }
            onDragLeave={modelsReady ? () => setIsDragOver(false) : undefined}
            className={`w-full border-2 border-dashed rounded-xl p-12 text-center transition ${
              !modelsReady
                ? "border-gray-700 opacity-40 cursor-not-allowed"
                : isDragOver
                  ? "border-blue-400 bg-blue-500/10 scale-[1.01] cursor-pointer"
                  : "border-gray-600 hover:border-blue-400/60 hover:bg-gray-900/40 cursor-pointer"
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
              <p className="text-gray-500 text-xs mb-1">
                또는 파일을 이 영역에 드래그
              </p>
            )}
            <p className="text-gray-600 text-xs">JPG · PNG · BMP · TIFF</p>
          </div>

          <div>
            <p className="text-[11px] text-gray-500 mb-2 text-center">
              또는 샘플 이미지로 시작
            </p>
            <div className="flex gap-2 justify-center flex-wrap">
              {SAMPLE_IMAGES.map((s) => (
                <button
                  key={s.src}
                  onClick={() => onLoadSample(s.src, s.name)}
                  disabled={!modelsReady}
                  className={`group relative rounded-lg overflow-hidden border transition w-28 h-28 shrink-0 ${
                    modelsReady
                      ? "border-gray-700 hover:border-blue-400"
                      : "border-gray-700 opacity-40 cursor-not-allowed"
                  }`}
                  title={s.name}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={s.src}
                    alt={s.name}
                    className="w-full h-full object-cover"
                  />
                  {modelsReady && (
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
                      <span className="text-white text-[10px] font-medium px-1 text-center">
                        {s.label}
                      </span>
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* ── 캔버스 영역 ── */
        <div
          className="relative overflow-hidden rounded-lg border border-gray-700 shadow-2xl shrink-0"
          style={{
            width: CANVAS_SIZE,
            height: CANVAS_SIZE,
            transform: `scale(${viewportScale})`,
            transformOrigin: "center center",
          }}
          onDrop={onDrop}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node))
              setIsDragOver(false);
          }}
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
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onMouseLeave={onMouseLeave}
              onContextMenu={(e) => e.preventDefault()}
              className="block select-none"
              style={{
                cursor:
                  activeTool === "cursor"
                    ? hoveredCellId !== null && editingCellId === null
                      ? "pointer"
                      : "default"
                    : "none",
              }}
            />

            {/* 도형 커서 */}
            {activeTool === "annotate" && shapeCenter && (
              <svg
                className="pointer-events-none absolute"
                style={{
                  left: shapeCenter.x - width / 2,
                  top: shapeCenter.y - height / 2,
                  width,
                  height,
                  overflow: "visible",
                  transform: `rotate(${rotationDeg}deg)`,
                  transformOrigin: "center center",
                }}
                viewBox={`0 0 ${width} ${height}`}
              >
                <polygon
                  points={ellipsePointsSvg(width, height, sides)}
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
                {Array.from({ length: sides }, (_, i) => {
                  const vertOffset = sides % 2 === 0 ? Math.PI / sides : 0;
                  const ang =
                    -Math.PI / 2 + vertOffset + (i / sides) * Math.PI * 2;
                  const px = width / 2 + Math.cos(ang) * (width / 2 - 2);
                  const py = height / 2 + Math.sin(ang) * (height / 2 - 2);
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
                <circle
                  cx={dragOrigin.x}
                  cy={dragOrigin.y}
                  r={3}
                  fill="#fbbf24"
                />
              </svg>
            )}

            {/* 우드래그 리사이즈 가이드 */}
            {resizeOrigin && cursorPos && (
              <svg
                className="pointer-events-none absolute inset-0"
                width={CANVAS_SIZE}
                height={CANVAS_SIZE}
              >
                <line
                  x1={resizeOrigin.x}
                  y1={resizeOrigin.y}
                  x2={cursorPos.x}
                  y2={resizeOrigin.y}
                  stroke="#22d3ee"
                  strokeWidth={1.5}
                  opacity={0.7}
                />
                <line
                  x1={cursorPos.x}
                  y1={resizeOrigin.y}
                  x2={cursorPos.x}
                  y2={cursorPos.y}
                  stroke="#e879f9"
                  strokeWidth={1.5}
                  opacity={0.7}
                />
                <circle
                  cx={resizeOrigin.x}
                  cy={resizeOrigin.y}
                  r={3}
                  fill="#22d3ee"
                />
              </svg>
            )}
          </div>

          {/* 로딩 오버레이 */}
          {loadingMsg && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 backdrop-blur-sm rounded-lg z-20 gap-3">
              <svg
                className="w-10 h-10 text-blue-400 animate-spin"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-20"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="2.5"
                />
                <path
                  className="opacity-80"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  d="M12 2a10 10 0 0 1 10 10"
                />
              </svg>
              <p className="text-sm font-semibold text-gray-100">
                {loadingMsg}
              </p>
            </div>
          )}

          {/* 추론 중 배지 */}
          {pendingCount > 0 && (
            <div className="group absolute top-0 left-0 z-10 pt-2 pl-2 pr-20 pb-20">
              <div className="bg-black/70 backdrop-blur px-2.5 py-1 rounded text-[11px] font-mono text-emerald-300 flex items-center gap-1.5 transition-opacity duration-300 group-hover:opacity-40">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                확정 추론 {pendingCount}…
              </div>
            </div>
          )}

          {/* 우상단 카운터 */}
          <div
            className="absolute top-2 right-2 z-10 bg-black/70 backdrop-blur px-2.5 py-1 rounded text-[11px] font-mono flex items-center gap-1.5 transition-opacity duration-300"
            style={{
              opacity:
                cursorPos &&
                cursorPos.x > CANVAS_SIZE - 220 &&
                cursorPos.y < 160
                  ? 0.15
                  : 1,
            }}
          >
            <span className="text-emerald-300">{confirmedCount}</span>
            <span className="text-gray-500">cells</span>
            {activeTool === "cursor" && (
              <span className="text-red-400">· 삭제 모드 (Esc)</span>
            )}
            {activeTool === "annotate" && (
              <span className="text-emerald-500">· 추가 모드</span>
            )}
            {isDragging && <span className="text-amber-300">· rotate</span>}
            {resizeOrigin && <span className="text-cyan-300">· resize</span>}
            {isPreviewPending && (
              <span className="text-emerald-300">· preview…</span>
            )}
          </div>

          {/* 모드 전환 토스트 */}
          {modeToast && (
            <div className="pointer-events-none absolute top-10 inset-x-0 z-20 flex justify-center">
              <div
                className={`px-4 py-1.5 rounded-full text-[12px] font-semibold shadow-lg border ${
                  modeToast === "cursor"
                    ? "bg-red-950/90 text-red-300 border-red-700/60"
                    : "bg-emerald-900/95 text-emerald-100 border-emerald-600/70"
                }`}
              >
                {modeToast === "cursor"
                  ? "삭제 모드 — 세포를 클릭해 삭제"
                  : "추가 모드 — 클릭으로 세포 추가"}
              </div>
            </div>
          )}

          {/* 재추론 모드 배너 */}
          {editingCellId !== null && (
            <div className="pointer-events-none absolute bottom-2 left-2 right-2 z-10 flex justify-center">
              <div className="bg-amber-500/95 text-black text-xs font-medium px-3 py-1.5 rounded shadow text-center">
                {reinferPending ? (
                  <>
                    #{cells.findIndex((c) => c.id === editingCellId) + 1} 재추론
                    중…
                  </>
                ) : (
                  <>
                    #{cells.findIndex((c) => c.id === editingCellId) + 1} 재추론
                    모드 — <span className="font-bold">좌클릭</span>으로 즉시
                    적용
                  </>
                )}
              </div>
            </div>
          )}

          {/* 라벨 미선택 경고 */}
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
              <p className="text-blue-300 font-medium text-sm bg-black/60 px-4 py-2 rounded">
                새 이미지로 교체
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
