"use client";

import { useRef, useState, useEffect, useCallback } from "react";

type Point = { x: number; y: number; label: 1 | 0 };

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8787";

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [points, setPoints] = useState<Point[]>([]);
  const [polyline, setPolyline] = useState<[number, number][] | null>(null);
  const [mode, setMode] = useState<1 | 0>(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !naturalSize.w) return;

    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / naturalSize.w;
    const scaleY = canvas.height / naturalSize.h;

    if (polyline && polyline.length > 1) {
      ctx.beginPath();
      ctx.moveTo(polyline[0][0] * scaleX, polyline[0][1] * scaleY);
      for (let i = 1; i < polyline.length; i++) {
        ctx.lineTo(polyline[i][0] * scaleX, polyline[i][1] * scaleY);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(0,220,70,0.35)";
      ctx.fill();
      ctx.strokeStyle = "#00e646";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x * scaleX, p.y * scaleY, 5, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? "#ef4444" : "#3b82f6";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }, [points, polyline, naturalSize]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setPoints([]);
    setPolyline(null);
    setError(null);

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    };
    img.src = url;
  }

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || !naturalSize.w) return;

    const rect = canvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const cy = (e.clientY - rect.top) * (canvas.height / rect.height);

    const ix = Math.round(cx * (naturalSize.w / canvas.width));
    const iy = Math.round(cy * (naturalSize.h / canvas.height));

    setPoints((prev) => [...prev, { x: ix, y: iy, label: mode }]);
    setPolyline(null);
  }

  async function runSegmentation() {
    if (!imageFile || points.length === 0) return;
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("image", imageFile);
      formData.append("points", JSON.stringify(points.map((p) => [p.x, p.y])));
      formData.append("labels", JSON.stringify(points.map((p) => p.label)));

      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      if (!res.ok) {
        const msg = await res.json();
        throw new Error(msg.detail ?? "API 오류");
      }
      const data = await res.json();
      setPolyline(data.polyline);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center py-10 px-4">
      <h1 className="text-2xl font-bold mb-1">MedficientSAM</h1>
      <p className="text-gray-400 text-sm mb-8">이미지를 업로드하고 세포를 클릭하세요</p>

      {/* 업로드 */}
      <div
        className="w-full max-w-xl border-2 border-dashed border-gray-600 rounded-xl p-8 text-center cursor-pointer hover:border-gray-400 transition mb-4"
        onClick={() => fileInputRef.current?.click()}
      >
        <p className="text-gray-400 text-sm">
          {imageFile ? imageFile.name : "클릭하여 이미지 업로드"}
        </p>
        <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
      </div>

      {naturalSize.w > 0 && (
        <>
          {/* 모드 토글 */}
          <div className="flex gap-3 mb-4">
            {([1, 0] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-4 py-1.5 rounded-full text-sm font-medium transition ${
                  mode === m
                    ? m === 1 ? "bg-red-500 text-white" : "bg-blue-500 text-white"
                    : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                }`}
              >
                {m === 1 ? "● Foreground" : "● Background"}
              </button>
            ))}
          </div>

          {/* 캔버스 */}
          <div className="relative mb-4">
            <canvas
              ref={canvasRef}
              width={512}
              height={512}
              className="rounded-lg cursor-crosshair border border-gray-700"
              onClick={handleCanvasClick}
            />
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg">
                <span className="text-sm animate-pulse">세그멘테이션 중…</span>
              </div>
            )}
          </div>

          {/* 버튼 */}
          <div className="flex gap-3 mb-3">
            <button
              onClick={() => { setPoints((p) => p.slice(0, -1)); setPolyline(null); }}
              disabled={points.length === 0}
              className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-sm transition"
            >
              Undo
            </button>
            <button
              onClick={() => { setPoints([]); setPolyline(null); setError(null); }}
              className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-sm transition"
            >
              Reset
            </button>
            <button
              onClick={runSegmentation}
              disabled={points.length === 0 || loading}
              className="px-5 py-2 rounded-lg bg-green-600 hover:bg-green-500 disabled:opacity-40 font-semibold text-sm transition"
            >
              Segment
            </button>
          </div>

          <p className="text-gray-400 text-xs mb-2">
            포인트: {points.length}개
            {polyline && ` | 폴리라인: ${polyline.length}개 좌표`}
          </p>

          {error && <p className="text-red-400 text-sm">{error}</p>}

          {polyline && (
            <details className="w-full max-w-xl mt-4">
              <summary className="text-gray-400 text-xs cursor-pointer hover:text-gray-200">
                폴리라인 좌표 보기
              </summary>
              <pre className="mt-2 text-xs text-green-400 bg-gray-900 rounded-lg p-4 overflow-auto max-h-48">
                {JSON.stringify(polyline, null, 2)}
              </pre>
            </details>
          )}
        </>
      )}
    </main>
  );
}
