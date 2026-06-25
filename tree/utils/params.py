"""参数解析辅助函数。"""

import ast
import json


def parse_param_value(value):
    """把 JSON 原生值、字符串化 JSON、Python 字面量字符串统一解析成 Python 对象。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict, int, float, bool)):
        return value

    text = str(value).strip()
    if not text:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue

    return value
