"""web/dist 静态托管（Plan 3 才产出 dist；本阶段验证缺失分支不白屏）。

SPA fallback：非 /api/* 且无对应文件时回 index.html。用 GET-only catch-all
而非 app.mount，是为了 fallback 可靠（catch-all 注册在 router include 之后，
FastAPI 按注册顺序匹配，/api/* 路由先命中不受影响）。
"""
from pathlib import Path

DIST = Path(__file__).parent.parent / "web" / "dist"


def mount_static(app) -> None:
    """web/dist 存在 → 挂静态 + SPA fallback；不存在 → GET / 返 no_dist JSON。"""
    if not DIST.exists():
        @app.get("/")
        def _no_dist():
            return {"error": {"code": "no_dist",
                              "message": "web/dist 不存在，先 npm run build",
                              "detail": "GET /api/* 仍可用"}}
        return

    from fastapi import Request
    from fastapi.responses import FileResponse, JSONResponse

    @app.get("/{path:path}")   # SPA catch-all：文件存在返文件，否则回 index.html
    async def _spa(request: Request, path: str):
        # 未匹配的 /api/* 不是前端路由：返 404 JSON，不吞成 index.html（docstring 承诺的行为）
        if path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "not_found", "message": f"没有这个接口 /api/{path[4:]}", "detail": ""}})
        f = (DIST / path) if path else (DIST / "index.html")
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
