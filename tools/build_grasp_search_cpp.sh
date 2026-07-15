#!/usr/bin/env bash
set -euo pipefail

# 关键步骤：始终从 MercuryTree 包根目录构建，避免在不同 cwd 下找不到 setup.py。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PACKAGE_DIR}"

# 关键步骤：机器人 ROS1/catkin 环境里 ccache 可能被误用作 linker，这里显式指定编译和链接命令。
CC=gcc CXX=g++ LDSHARED="g++ -pthread -shared" \
  python3 setup.py build_ext --inplace

python3 -c "from tree.node.grasp_object import _grasp_search_cpp; print('cpp grasp search ok')"
