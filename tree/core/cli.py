"""MercuryTree 命令行参数与配置路径解析。"""

import argparse
import os


def build_argument_parser():
    """创建项目自身的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="从 JSON 配置启动 MercuryTree 行为树",
    )
    parser.add_argument(
        "--tree",
        "--tree-json-file",
        dest="tree_json_file",
        default=None,
        help=(
            "要运行的行为树 JSON。绝对路径直接使用；相对路径按 "
            "MercuryTree/config 目录解析，例如 tree/box/example.json"
        ),
    )
    return parser


def resolve_tree_json_file(project_root, tree_file_name):
    """将命令行中的树路径统一解析成绝对路径。"""
    tree_file_name = os.path.expanduser(str(tree_file_name).strip())
    if os.path.isabs(tree_file_name):
        return os.path.abspath(tree_file_name)
    return os.path.abspath(os.path.join(project_root, "config", tree_file_name))
