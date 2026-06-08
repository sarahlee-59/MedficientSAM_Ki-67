import { promises as fs } from "fs";

const ONNX_DIR = process.env.ONNX_DIR ?? "/mnt/Disk1/sylee/deployment";
const ENCODER_FILE = "encoder.quantized.onnx";

export async function GET() {
  try {
    const filePath = `${ONNX_DIR}/${ENCODER_FILE}`;
    const buf = await fs.readFile(filePath);
    return new Response(buf, {
      headers: {
        "content-type": "application/octet-stream",
        "cache-control": "public, max-age=3600",
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return Response.json(
      { detail: `ONNX encoder 로드 실패: ${message}`, path: `${ONNX_DIR}/${ENCODER_FILE}` },
      { status: 500 }
    );
  }
}
