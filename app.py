"""单入口（端口 7861）：FastAPI API + SSE + 静态 dist 挂同一进程。

设计稿 D2：一个进程同时挂 /api/*、SSE events 和前端 web/dist。legacy Gradio
UI 已改名 legacy_app_gradio.py 并在 Task 6 退役，这里不再是双开对照。

静态资源与 API 路由都在 server.api（include_router + mount_static），本文件只
负责按配置起 uvicorn。端口取 CFG.server_port（config.yaml 默认 7861）。
"""
from webui.config import load_config

import uvicorn

CFG = load_config()


def main() -> None:
    uvicorn.run("server.api:app", host=CFG.server_host, port=CFG.server_port, log_level="info")


if __name__ == "__main__":
    main()
