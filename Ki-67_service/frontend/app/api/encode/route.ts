import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.formData();
  const res = await fetch(`${BACKEND_URL}/encode`, { method: "POST", body });
  return Response.json(await res.json(), { status: res.status });
}
