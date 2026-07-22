"""码垛放箱诊断日志工具。

将关键阶段的结构化数据追加保存到 ``src/MercuryTree/log``，避免实机终端中
导航、YOLO、手臂等高频日志淹没放箱高度/落点信息。文件为 JSON Lines 格式，
每行一个事件，可按 ``event``、``stack_count`` 或时间顺序筛查。
"""

import json
import os
import threading
from datetime import datetime


_WRITE_LOCK = threading.Lock()
# 本文件位于 ``MercuryTree/tree/utils``：回到 MercuryTree 目录只需两层，
# 不能写成三层，否则会误落到 ``src/log`` 而非 ``src/MercuryTree/log``。
_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "log"))
_LOG_FILE = "pallet_place_diagnostics.jsonl"


def write_pallet_place_diagnostic(event, payload):
    """追加一条可供实机问题回放的码垛诊断记录。

    日志失败绝不影响行为树控制，仅让调用方继续执行。
    """
    record = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "event": str(event),
        **dict(payload or {}),
    }
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        path = os.path.join(_LOG_DIR, _LOG_FILE)
        with _WRITE_LOCK, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    except Exception:
        # 诊断文件为可选旁路，不能因磁盘/权限问题破坏放箱主流程。
        pass


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)
