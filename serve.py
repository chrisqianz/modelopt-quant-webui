"""FastAPI 入口（7862，与 legacy Gradio 7861 双开对照）。"""
import uvicorn
from webui.config import load_config

CFG = load_config()


def main() -> None:
    uvicorn.run("server.api:app", host=CFG.server_host, port=CFG.server_port + 1, log_level="info")


if __name__ == "__main__":
    main()
