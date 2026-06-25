import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.formData();
  const res = await fetch(`${BACKEND_URL}/encode`, { method: "POST", body });
  const text = await res.text();
  const data = (() => {
    try {
      return JSON.parse(text);
    } catch {
      return { detail: text || "백엔드 서버 응답 오류" };
    }
  })();
  return Response.json(data, { status: res.status });
}
