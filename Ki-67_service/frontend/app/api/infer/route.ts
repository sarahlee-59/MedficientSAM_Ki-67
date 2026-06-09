import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.formData();
  const res = await fetch(`${BACKEND_URL}/infer`, { method: "POST", body });
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
