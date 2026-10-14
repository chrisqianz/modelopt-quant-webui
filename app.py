"""单入口（端口 7861）：FastAPI API + SSE + 静态 dist 挂同一进程。

设计稿 D2：一个进程同时挂 /api/*、SSE events 和前端 web/dist。早期的「React
7862 + Gradio legacy 7861」双开对照已合并为这个单入口，legacy 已删除。

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
