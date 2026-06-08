import { promises as fs } from "fs";

const ONNX_DIR = process.env.ONNX_DIR ?? "/mnt/Disk1/sylee/deployment";
const DECODER_FILE = "decoder.fp16.onnx";

export async function GET() {
  try {
    const filePath = `${ONNX_DIR}/${DECODER_FILE}`;
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
      { detail: `ONNX decoder fp16 로드 실패: ${message}`, path: `${ONNX_DIR}/${DECODER_FILE}` },
      { status: 500 }
    );
  }
}
