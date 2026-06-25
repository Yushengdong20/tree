"""行为树 JSON 静态可视化工具。

这个脚本用于把 config/tree 下的行为树 JSON 渲染成 Graphviz 图片。
它会按运行时 TreeFactory 的规则递归展开 SubTree，便于在启动前检查整棵树结构。
"""

import argparse
import json
import os
from typing import Any, Dict, Optional

import pydot


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_TREE_JSON = os.path.join(
    PROJECT_ROOT,
    "config",
    "tree",
    "box",
    "move_box_full_direct_grasp_place_turn.json",
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tree", "visualization", "output")


class GraphBuildState:
    """保存构图过程中的递增状态。"""

    def __init__(self):
        self.node_index = 0

    def next_node_id(self) -> str:
        """生成 Graphviz 内部节点 id，避免中文 label 或重复 label 影响连线。"""
        node_id = f"node_{self.node_index}"
        self.node_index += 1
        return node_id


def load_json_file(json_file: str) -> Dict[str, Any]:
    """读取单个 JSON 文件。"""
    with open(json_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_custom_param_value(param_config: Any) -> Any:
    """兼容 {value, source} 包装格式，取出实际参数值。"""
    if isinstance(param_config, dict) and "value" in param_config:
        return param_config.get("value")
    return param_config


def get_param_value(params: Dict[str, Any], key: str, default: Any = None) -> Any:
    """从节点 params 里读取指定参数。"""
    if key not in params:
        return default
    return parse_custom_param_value(params[key])


def resolve_subtree_path(source_dir: str, subtree_file: str) -> str:
    """按当前 JSON 文件目录解析子树路径。"""
    if os.path.isabs(subtree_file):
        return subtree_file
    return os.path.abspath(os.path.join(source_dir, subtree_file))


def get_node_color(node_name: str) -> str:
    """按节点类型给图节点分配背景色。"""
    if node_name == "Sequence":
        return "#D7E8FF"
    if node_name == "Selector":
        return "#DFF3DF"
    if node_name == "Parallel":
        return "#FFE8C7"
    if node_name == "Repeat":
        return "#EFE1FF"
    if node_name == "SubTree":
        return "#F5F5F5"
    return "#FFFFFF"


def get_node_shape(node_name: str) -> str:
    """组合节点和叶子节点使用不同形状，便于快速区分。"""
    if node_name in ("Sequence", "Selector", "Parallel", "Repeat"):
        return "box"
    return "ellipse"


def format_param_value(value: Any) -> str:
    """把参数值压缩成适合放进节点 label 的短文本。"""
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_params_label(params: Dict[str, Any]) -> str:
    """生成可选参数展示文本。"""
    if not params:
        return ""

    lines = []
    for key in sorted(params.keys()):
        value = parse_custom_param_value(params[key])
        lines.append(f"{key}: {format_param_value(value)}")
    return "\\n".join(lines)


def build_node_label(
    node_name: str,
    node_label: str,
    params: Dict[str, Any],
    show_params: bool,
    source_note: Optional[str] = None,
) -> str:
    """生成 Graphviz 节点展示文本。"""
    label_lines = [node_label, f"[{node_name}]"]

    if source_note:
        label_lines.append(source_note)

    if show_params:
        params_label = build_params_label(params)
        if params_label:
            label_lines.append(params_label)

    return "\\n".join(label_lines)


def add_graph_node(
    graph: pydot.Dot,
    state: GraphBuildState,
    node_name: str,
    node_label: str,
    params: Dict[str, Any],
    show_params: bool,
    source_note: Optional[str] = None,
) -> str:
    """把一个行为树节点加入 Graphviz 图，并返回内部节点 id。"""
    node_id = state.next_node_id()
    graph_node = pydot.Node(
        node_id,
        label=build_node_label(
            node_name=node_name,
            node_label=node_label,
            params=params,
            show_params=show_params,
            source_note=source_note,
        ),
        shape=get_node_shape(node_name),
        style='"rounded,filled"',
        fillcolor=get_node_color(node_name),
        fontname="Noto Sans CJK SC",
        fontsize="11",
        margin="0.12,0.08",
    )
    graph.add_node(graph_node)
    return node_id


def add_edge(graph: pydot.Dot, parent_id: Optional[str], child_id: str) -> None:
    """按父子关系加入有向边。"""
    if parent_id is None:
        return
    graph.add_edge(pydot.Edge(parent_id, child_id, color="#555555", arrowsize="0.7"))


def add_node_recursive(
    graph: pydot.Dot,
    state: GraphBuildState,
    node_config: Dict[str, Any],
    source_dir: str,
    parent_id: Optional[str],
    show_params: bool,
    override_label: Optional[str] = None,
    source_note: Optional[str] = None,
) -> str:
    """递归解析一个 JSON 节点，并把它和所有子节点加入图里。"""
    node_name = node_config.get("name", "UnnamedNode")
    node_label = override_label or node_config.get("label", node_name)
    params = node_config.get("params", {})

    # SubTree 和运行时保持一致：不画占位节点，而是直接展开子树根节点。
    if node_name == "SubTree":
        subtree_file = str(get_param_value(params, "file", "")).strip()
        if not subtree_file:
            raise ValueError("SubTree node requires params.file")

        subtree_path = resolve_subtree_path(source_dir, subtree_file)
        subtree_config = load_json_file(subtree_path)
        if "tree" not in subtree_config:
            raise ValueError(f"Invalid subtree JSON structure: missing 'tree' key in {subtree_path}")

        subtree_note = f"SubTree: {subtree_file}"
        return add_node_recursive(
            graph=graph,
            state=state,
            node_config=subtree_config["tree"],
            source_dir=os.path.dirname(subtree_path),
            parent_id=parent_id,
            show_params=show_params,
            override_label=node_label if node_label != node_name else None,
            source_note=subtree_note,
        )

    node_id = add_graph_node(
        graph=graph,
        state=state,
        node_name=node_name,
        node_label=node_label,
        params=params,
        show_params=show_params,
        source_note=source_note,
    )
    add_edge(graph, parent_id, node_id)

    # childs 是现有配置里的字段名，这里保持兼容而不改 JSON 结构。
    for child_config in node_config.get("childs", []):
        add_node_recursive(
            graph=graph,
            state=state,
            node_config=child_config,
            source_dir=source_dir,
            parent_id=node_id,
            show_params=show_params,
        )

    return node_id


def build_tree_graph(
    json_file: str,
    show_params: bool = False,
    rankdir: str = "LR",
) -> pydot.Dot:
    """从入口 JSON 构建完整 Graphviz 图。"""
    tree_config = load_json_file(json_file)
    if "tree" not in tree_config:
        raise ValueError(f"Invalid JSON structure: missing 'tree' key in {json_file}")

    graph = pydot.Dot(
        graph_type="digraph",
        rankdir=rankdir,
        bgcolor="white",
        splines="ortho",
        concentrate="false",
    )
    graph.set_node_defaults(fontname="Noto Sans CJK SC")
    graph.set_edge_defaults(fontname="Noto Sans CJK SC")

    # 从根节点开始递归展开，SubTree 会继续加载并展开它引用的 JSON。
    state = GraphBuildState()
    add_node_recursive(
        graph=graph,
        state=state,
        node_config=tree_config["tree"],
        source_dir=os.path.dirname(os.path.abspath(json_file)),
        parent_id=None,
        show_params=show_params,
    )
    return graph


def default_output_file(json_file: str) -> str:
    """根据入口 JSON 文件名生成默认输出路径。"""
    json_name = os.path.splitext(os.path.basename(json_file))[0]
    return os.path.join(DEFAULT_OUTPUT_DIR, f"{json_name}.png")


def write_graph(graph: pydot.Dot, output_file: str) -> None:
    """写出 DOT 文件和目标图片文件。"""
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)

    dot_file = os.path.splitext(output_file)[0] + ".dot"
    graph.write_raw(dot_file)

    output_ext = os.path.splitext(output_file)[1].lower()
    if output_ext == ".svg":
        graph.write_svg(output_file)
    elif output_ext == ".pdf":
        graph.write_pdf(output_file)
    else:
        graph.write_png(output_file)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="可视化行为树 JSON，并递归展开 SubTree。")
    parser.add_argument(
        "json_file",
        nargs="?",
        default=DEFAULT_TREE_JSON,
        help=f"入口行为树 JSON 文件，默认: {DEFAULT_TREE_JSON}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出文件路径，支持 .png/.svg/.pdf；默认输出到 tree/visualization/output",
    )
    parser.add_argument(
        "--show-params",
        action="store_true",
        help="在节点中显示 params，参数较多时图会明显变大",
    )
    parser.add_argument(
        "--rankdir",
        choices=("TB", "LR"),
        default="LR",
        help="Graphviz 布局方向：TB 为自上而下，LR 为从左到右；默认 LR",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    json_file = os.path.abspath(args.json_file)
    output_file = args.output or default_output_file(json_file)

    graph = build_tree_graph(
        json_file=json_file,
        show_params=args.show_params,
        rankdir=args.rankdir,
    )
    write_graph(graph=graph, output_file=output_file)

    dot_file = os.path.splitext(output_file)[0] + ".dot"
    print(f"行为树图片已生成: {output_file}")
    print(f"DOT 文件已生成: {dot_file}")


if __name__ == "__main__":
    main()
