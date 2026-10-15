#!/usr/bin/env bash
# 部署到 GPU 机。用法：./deploy.sh [user@host:path]
#
# 只同步代码，机器相关的状态一概不过去：
#   .venv/   —— 跑 WebUI 自己的 venv（目标机上单独建）
#   .venvs/  —— 量化用的 ModelOpt venv（D9：装在项目内）。**这行必须留着**，
#               否则 --delete 会把目标机上装好的几 GB venv 直接删掉
#   web/dist —— 前端构建产物，随 rsync 上机（dist 在 .gitignore，由 mac 侧 build 产生）
#   net.json / config.yaml —— 代理与路径是每台机器自己的事实
set -euo pipefail
DEST="${1:-chrisqian@192.168.124.39:~/modelopt-quant-webui/}"
cd "$(dirname "$0")"

rsync -avz --delete \
    --exclude='.git/' --exclude='.venv/' --exclude='.venvs/' \
    --exclude='__pycache__/' --exclude='web/node_modules/' \
    --exclude='.lens/' --exclude='.superpowers/' \
    --exclude='llm_model_quantization_dashboard.tsx' \
    --exclude='hosts.json' --exclude='server.json' --exclude='net.json' \
    --exclude='config.yaml' --exclude='*.log' \
    ./ "$DEST"

echo "部署完成 → $DEST"
echo "在目标机上：ssh \$(echo $DEST | cut -d: -f1) 'cd modelopt-quant-webui && .venv/bin/python app.py'"
