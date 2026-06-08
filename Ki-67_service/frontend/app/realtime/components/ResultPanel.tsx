"use client";

import React from "react";
import { type Cell, MODEL_NAME } from "../types";
import { formatLatency, cellLabel, Row, ToolLabel } from "./ui";

type CellListState = {
  cells: Cell[];
  hoveredCellId: number | null;
  editingCellId: number | null;
  dragCellId: number | null;
  dragOverCellId: number | null;
  reinferPending: boolean;
  cellFilter: "all" | "positive" | "negative";
};

type Ki67Stats = {
  confirmedCount: number;
  pendingCount: number;
  positiveCount: number;
  negativeCount: number;
  ki67Rate: number | null;
};

type Props = {
  activeTool: "cursor" | "annotate";
  latencies: { ms: number; cellId: number }[];
  ki67Stats: Ki67Stats;
  cellListState: CellListState;
  error: string | null;
  cellRowRefs: React.MutableRefObject<Map<number, HTMLDivElement>>;
  hoveredFromCanvasRef: React.MutableRefObject<boolean>;
  onSetCellFilter: (f: "all" | "positive" | "negative") => void;
  onSetKiLabel: (id: number, label: "positive" | "negative") => void;
  onStartReedit: (cell: Cell) => void;
  onCancelReedit: () => void;
  onDeleteCell: (id: number) => void;
  onCellDragStart: (e: React.DragEvent, cellId: number) => void;
  onCellDragOver: (e: React.DragEvent, cellId: number) => void;
  onCellDrop: (e: React.DragEvent, cellId: number) => void;
  onCellDragEnd: () => void;
  onSetHoveredCellId: (id: number | null) => void;
  onSaveJson: () => void;
};

export function ResultPanel({
  activeTool,
  latencies,
  ki67Stats,
  cellListState,
  error,
  cellRowRefs,
  hoveredFromCanvasRef,
  onSetCellFilter,
  onSetKiLabel,
  onStartReedit,
  onCancelReedit,
  onDeleteCell,
  onCellDragStart,
  onCellDragOver,
  onCellDrop,
  onCellDragEnd,
  onSetHoveredCellId,
  onSaveJson,
}: Props) {
  const {
    confirmedCount,
    pendingCount,
    positiveCount,
    negativeCount,
    ki67Rate,
  } = ki67Stats;
  const {
    cells,
    hoveredCellId,
    editingCellId,
    dragCellId,
    dragOverCellId,
    reinferPending,
    cellFilter,
  } = cellListState;

  const lastEntry = latencies.length ? latencies[latencies.length - 1] : null;
  const stats =
    latencies.length > 0
      ? (() => {
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
        })()
      : null;

  return (
    <div
      className="shrink-0 flex flex-col bg-gray-900/60"
      style={{ height: 240 }}
    >
      {/* 섹션 타이틀 바 */}
      {activeTool === "cursor" && editingCellId === null ? (
        <div className="shrink-0 border-t border-red-800/60 h-9 px-5 flex items-center gap-3 bg-red-900/70 text-red-200 whitespace-nowrap overflow-hidden">
          <span className="flex items-center gap-1 font-bold text-xs shrink-0">
            삭제 모드 (Esc)
          </span>
          <div className="h-4 w-px bg-white/30 shrink-0" />
          <div className="flex items-center gap-x-4 text-[11px] min-w-0">
            <span className="flex items-center gap-1.5">
              <kbd className="rounded border border-white/40 bg-white/25 px-1.5 py-0.5 text-[10px] font-semibold leading-none">
                좌클릭
              </kbd>
              세포 삭제
            </span>
            <span className="flex items-center gap-1.5">
              <kbd className="rounded border border-white/40 bg-white/25 px-1.5 py-0.5 text-[10px] font-semibold leading-none">
                Esc
              </kbd>
              모드 해제
            </span>
            <span className="flex items-center gap-1.5">
              <kbd className="rounded border border-white/40 bg-white/25 px-1.5 py-0.5 text-[10px] font-semibold leading-none">
                Ctrl+Esc
              </kbd>
              전체 삭제
            </span>
          </div>
        </div>
      ) : (
        <div className="shrink-0 border-t border-gray-700 h-9 px-5 flex items-center gap-3 bg-gray-900">
          <span className="text-[9px] uppercase tracking-widest text-gray-500 font-semibold">
            결과
          </span>
          <div className="flex-1 h-px bg-gray-800" />
          {confirmedCount > 0 && ki67Rate !== null && (
            <span className="text-[11px] tabular-nums">
              <span className="text-gray-500 text-[10px] mr-1.5">Ki-67</span>
              <span className="font-bold text-emerald-300">
                {ki67Rate.toFixed(1)}%
              </span>
            </span>
          )}
        </div>
      )}

      <div className="flex-1 min-h-0 px-5 py-3 flex gap-6 items-start overflow-x-auto">
        {/* 추론 시간 */}
        <div className="min-w-[160px] shrink-0">
          <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-1.5 font-semibold">
            추론 시간
          </div>
          {latencies.length === 0 ? (
            <p className="text-[11px] text-gray-500">기록 없음</p>
          ) : (
            <div className="text-[11px] space-y-1">
              <Row
                k="방금"
                v={lastEntry?.ms}
                cellLabel={cellLabel(lastEntry?.cellId, cells)}
                highlight
              />
              <Row k="평균" v={stats?.avg} />
            </div>
          )}
          <p
            className="text-[10px] text-emerald-400 mt-1.5 font-mono truncate"
            title={MODEL_NAME}
          >
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
                {confirmedCount}개
                {pendingCount > 0 && ` · 추론중 ${pendingCount}`}
              </span>
              <div className="ml-auto flex items-center gap-1">
                {(["all", "positive", "negative"] as const).map((f) => (
                  <button
                    key={f}
                    onClick={() => onSetCellFilter(f)}
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
                    {f === "all"
                      ? "전체"
                      : f === "positive"
                        ? "양성+"
                        : "음성−"}
                  </button>
                ))}
              </div>
            </div>
            <div className="rounded border border-gray-800 divide-y divide-gray-800 overflow-y-auto flex-1 min-h-0">
              {cells
                .filter(
                  (c) =>
                    cellFilter === "all" ||
                    c.pending ||
                    c.kiLabel === cellFilter,
                )
                .map((cell) => {
                  const idx = cells.indexOf(cell);
                  return (
                    <div
                      key={cell.id}
                      ref={(el) => {
                        if (el) cellRowRefs.current.set(cell.id, el);
                        else cellRowRefs.current.delete(cell.id);
                      }}
                      draggable={!cell.pending && editingCellId === null}
                      onMouseEnter={() => {
                        hoveredFromCanvasRef.current = false;
                        onSetHoveredCellId(cell.id);
                      }}
                      onMouseLeave={() => onSetHoveredCellId(null)}
                      onDragStart={(e) => onCellDragStart(e, cell.id)}
                      onDragOver={(e) => onCellDragOver(e, cell.id)}
                      onDrop={(e) => onCellDrop(e, cell.id)}
                      onDragEnd={onCellDragEnd}
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
                        <span className="shrink-0 text-gray-600 cursor-grab active:cursor-grabbing select-none text-[10px] leading-none pr-0.5">
                          ⠿
                        </span>
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
                        <span className="text-emerald-300 italic flex-1 font-mono">
                          …확정 추론
                        </span>
                      ) : (
                        <>
                          <button
                            onClick={() => onSetKiLabel(cell.id, "positive")}
                            className={`px-1.5 py-0.5 rounded transition ${
                              cell.kiLabel === "positive"
                                ? "bg-red-600 text-white"
                                : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                            }`}
                          >
                            +
                          </button>
                          <button
                            onClick={() => onSetKiLabel(cell.id, "negative")}
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
                                  <span className="text-[10px] text-gray-400 animate-pulse">
                                    추론 중…
                                  </span>
                                )}
                                <button
                                  onClick={onCancelReedit}
                                  className="px-1.5 py-0.5 rounded text-[10px] bg-gray-800 hover:bg-gray-700 text-gray-400"
                                >
                                  취소
                                </button>
                              </>
                            ) : (
                              <button
                                onClick={() => onStartReedit(cell)}
                                disabled={editingCellId !== null}
                                className="px-1.5 py-0.5 rounded text-[10px] bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 disabled:opacity-40 disabled:cursor-not-allowed"
                              >
                                재추론
                              </button>
                            )}
                          </>
                        )}
                        <button
                          onClick={() => onDeleteCell(cell.id)}
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
              <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-1.5 font-semibold">
                Ki-67 지수
              </div>
              <div className="rounded border border-gray-800 bg-gray-950/60 px-3 py-2 text-[11px] space-y-1.5">
                <div className="flex justify-between gap-4">
                  <span className="text-red-400">양성</span>
                  <span className="tabular-nums text-gray-200">
                    {positiveCount}개
                  </span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-blue-400">음성</span>
                  <span className="tabular-nums text-gray-200">
                    {negativeCount}개
                  </span>
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
            onClick={onSaveJson}
            disabled={confirmedCount === 0}
            className="w-full py-2 rounded text-xs font-medium bg-emerald-700 hover:bg-emerald-600 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed transition"
          >
            JSON 저장{" "}
            {confirmedCount > 0 && (
              <span className="opacity-80">({confirmedCount})</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
