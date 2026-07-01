"""JSON -> py_trees 运行时对象的转换工厂。

这个文件负责解析配置树，并递归创建：
- Sequence / Selector / Parallel
- SubTree
- 自定义业务叶子节点

它的定位很明确：只做“建树”，不负责“跑树”。
"""

import importlib
import json
import os
from typing import Any, Dict

import py_trees


class ParamsWrapper:
    """Provide a tiny read-only adapter around flattened JSON parameters."""

    def __init__(self, params: Dict[str, Any]):
        # 把嵌套参数拍平成 a.b.c 形式，叶子节点读取参数时更直接。
        self.params = self._flatten(params)

    def _flatten(self, params: Dict[str, Any], parent_key: str = "", sep: str = "."):
        """把嵌套参数拍平成单层字典，方便节点按 key 直接取值。"""
        items = []
        for key, value in params.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def get(self, key: str, default=None):
        return self.params.get(key, default)


class BehaviorTreeFactory:
    """Load a behaviour tree from a single json file.

    它的核心职责只有一个：把配置里的节点描述递归转换成 py_trees 运行时对象。
    """

    def __init__(self, ros_node):
        self.ros_node = ros_node
        # 同一类型的叶子节点类只 import 一次，避免递归建树时重复导入模块。
        self.module_cache = {}

    def load_tree_from_json(self, json_file: str) -> py_trees.trees.BehaviourTree:
        # 整个项目的“树结构来源”都集中在这个 JSON 文件里。
        tree_config = self._load_tree_config(json_file)

        if "tree" not in tree_config:
            raise ValueError("Invalid JSON structure: missing 'tree' key")

        root = self._build_tree_recursive(
            tree_config["tree"],
            source_dir=os.path.dirname(os.path.abspath(json_file)),
        )
        return py_trees.trees.BehaviourTree(root=root)

    def _build_tree_recursive(self, node_config: Dict[str, Any], source_dir: str):
        # name 对应节点类型，label 对应展示名称；JSON 里不写 label 时退回到 name。
        node_name = node_config.get("name", "UnnamedNode")
        node_label = node_config.get("label", node_name)
        node_params = self._parse_params(node_config.get("params", {}))
        children = node_config.get("childs", [])

        # 先处理 py_trees 自带的组合节点。
        if node_name == "SubTree":
            subtree_file = str(node_params.get("file", "")).strip()
            if not subtree_file:
                raise ValueError("SubTree node requires params.file")
            subtree_path = self._resolve_subtree_path(source_dir, subtree_file)
            subtree_config = self._load_tree_config(subtree_path)
            if "tree" not in subtree_config:
                raise ValueError(f"Invalid subtree JSON structure: missing 'tree' key in {subtree_path}")
            node = self._build_tree_recursive(
                subtree_config["tree"],
                source_dir=os.path.dirname(subtree_path),
            )
            # 外层引用可以覆盖展示 label，便于组合测试树里给子树取更清晰的阶段名。
            if node_label != node_name:
                node.name = node_label
                node.json_label = node_label
            return node
        elif node_name == "Sequence":
            node = py_trees.composites.Sequence(
                name=node_label,
                memory=self._to_bool(node_params.get("memory", False)),
            )
        elif node_name == "Selector":
            node = py_trees.composites.Selector(
                name=node_label,
                memory=self._to_bool(node_params.get("memory", False)),
            )
        elif node_name == "Parallel":
            policy_name = str(node_params.get("policy", "SuccessOnAll"))
            synchronise = self._to_bool(node_params.get("synchronise", True))
            if policy_name == "SuccessOnOne":
                policy = py_trees.common.ParallelPolicy.SuccessOnOne()
            else:
                policy = py_trees.common.ParallelPolicy.SuccessOnAll(
                    synchronise=synchronise
                )
            node = py_trees.composites.Parallel(name=node_label, policy=policy)
        elif node_name == "Repeat":
            if len(children) != 1:
                raise ValueError("Repeat 节点必须且只能有一个子节点")
            child = self._build_tree_recursive(children[0], source_dir=source_dir)
            node = py_trees.decorators.Repeat(
                name=node_label,
                child=child,
                num_success=int(node_params.get("num_success", -1)),
            )
        else:
            # 其余节点视为业务叶子节点，按约定去 pytrees_ros2.node 下动态加载。
            node = self._create_leaf(node_name=node_name, node_label=node_label, params=node_params)

        # 额外挂到节点实例上的字段不参与 py_trees 核心逻辑，主要服务于可视化展示。
        node.node_type_raw = node_name
        node.json_label = node_label

        if hasattr(node, "add_child") and node_name != "Repeat":
            # 组合节点继续递归构建子树，叶子节点则没有 children。
            for child_config in children:
                node.add_child(self._build_tree_recursive(child_config, source_dir=source_dir))

        return node

    def _parse_params(self, params: Dict[str, Any]):
        parsed = {}
        for key, value in params.items():
            # 兼容配置里 {source: CUSTOM, value: ...} 这种外层包装格式。
            if isinstance(value, dict) and value.get("source") == "CUSTOM":
                parsed[key] = value.get("value")
            else:
                parsed[key] = value
        return ParamsWrapper(parsed)

    def _create_leaf(self, node_name: str, node_label: str, params: ParamsWrapper):
        # 约定类名 MoveToTarget 对应文件 move_to_target.py。
        module_name = self._camel_to_snake(node_name)
        cache_key = f"{module_name}:{node_name}"
        if cache_key not in self.module_cache:
            self.module_cache[cache_key] = self._load_leaf_class(
                module_name=module_name,
                class_name=node_name,
            )
        node_class = self.module_cache[cache_key]
        return node_class(
            name=node_name,
            config_label=node_label,
            ros_node=self.ros_node,
            params=params,
        )

    @staticmethod
    def _load_leaf_class(module_name: str, class_name: str):
        """Load one leaf node class from known node subpackages.

        当前约定是：
        - 真实 HTTP 节点放在 `tree.node.http`
        - 纯 mock 示例节点放在 `tree.node.mock`
        - `tree.node` 根目录只保留公共基类等共享文件

        这样目录可以分层，但 JSON 里的节点名字和旧配置都不需要改。
        """
        candidate_modules = [
            f"tree.node.http.{module_name}",
            f"tree.node.{module_name}",
            f"tree.node.common.{module_name}",
            f"tree.node.manipulation.{module_name}",
            f"tree.node.grasp_object.{module_name}",
            f"tree.node.move_box.{module_name}",
            f"tree.node.mock.{module_name}",
        ]
        last_error = None
        for module_path in candidate_modules:
            try:
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError) as exc:
                last_error = exc
        raise ImportError(
            f"Unable to load leaf node {class_name} from {candidate_modules}: {last_error}"
        )

    @staticmethod
    def _load_tree_config(json_file: str) -> Dict[str, Any]:
        """读取单个 JSON 文件并反序列化。"""
        with open(json_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _resolve_subtree_path(source_dir: str, subtree_file: str) -> str:
        # 子树优先按“相对当前 JSON 文件目录”解析，便于在 config/ 下直接互相引用。
        if os.path.isabs(subtree_file):
            return subtree_file
        return os.path.abspath(os.path.join(source_dir, subtree_file))

    @staticmethod
    def _camel_to_snake(value: str) -> str:
        chars = []
        for index, char in enumerate(value):
            if char.isupper() and index > 0:
                chars.append("_")
            chars.append(char.lower())
        return "".join(chars)

    @staticmethod
    def _to_bool(value: Any) -> bool:
        # JSON/参数里常见的字符串布尔值也一起兼容掉。
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
