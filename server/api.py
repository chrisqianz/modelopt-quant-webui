"""FastAPI 入口占位版（Task 5/6 换成完整 APIRouter）。"""
from fastapi import FastAPI

app = FastAPI(title="ModelOpt Quant WebUI")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
