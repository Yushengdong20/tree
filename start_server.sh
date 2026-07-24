#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 关键步骤：先注入 kuavo SDK 的 Python 路径，确保后续构建和 server 都能找到 SDK。
source "../scripts/source_kuavo_sdk_pythonpath.sh"

# 关键步骤：启动前编译抓取搜索 C++ 扩展，避免运行时加载到旧版本。
bash "tools/build_grasp_search_cpp.sh"

# 关键步骤：server 模式预初始化 grasp_object 机器人服务，减少收到任务后的等待时间。
python3 -m tree.server_main --preload-services grasp_object
