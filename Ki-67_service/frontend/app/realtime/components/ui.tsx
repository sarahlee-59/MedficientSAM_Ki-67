"use client";

import { type Cell } from "../types";

export function formatLatency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  const sec = ms / 1000;
  const secStr = sec < 10 ? sec.toFixed(2) : sec.toFixed(1);
  return `${secStr}초 (${Math.round(ms)}ms)`;
}

export function cellLabel(
  cellId: number | undefined,
  cells: Cell[],
): string | undefined {
  if (cellId === undefined) return undefined;
  const idx = cells.findIndex((c) => c.id === cellId);
  return idx >= 0 ? `#${idx + 1}` : undefined;
}

export function ToolLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-1 font-semibold">
      {children}
    </div>
  );
}

export function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex items-center px-1.5 py-0.5 rounded border border-gray-600 bg-gray-700 text-gray-200 font-mono text-[9px] leading-none whitespace-nowrap">
      {children}
    </kbd>
  );
}

export function Row({
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
