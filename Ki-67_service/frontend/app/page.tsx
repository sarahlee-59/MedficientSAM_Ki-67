import { redirect } from "next/navigation";

// 메인 페이지는 더 이상 사용하지 않음 — 실시간 ONNX 페이지로 이동
export default function Home() {
  redirect("/realtime");
}
