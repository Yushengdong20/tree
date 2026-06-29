"""放箱后从箱子记忆中选择下一个抓取目标。"""

import math

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction
from tree.runtime.http.move_and_grab_flow import (
    DEFAULT_CHASSIS_URL,
    build_chassis_config,
    get_chassis_current_pose,
)
from tree.utils.box_map_polygon import is_map_position_in_polygon, parse_map_polygon


class MoveBoxSelectNextMemoryBox(TimedMockAction):
    """把当前箱子标记为完成，并从记忆列表里选择下一个箱子。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.box_memory_key = str(params.get("box_memory_key", "move_box_box_memory")).strip()
        self.current_box_target_key = str(
            params.get("current_box_target_key", "move_box_current_box_target")
        ).strip()
        self.finished_box_targets_key = str(
            params.get("finished_box_targets_key", "move_box_finished_box_targets")
        ).strip()
        self.select_policy = str(params.get("select_policy", "nearest")).strip()
        self.valid_box_map_polygon = parse_map_polygon(
            params.get("valid_box_map_polygon", [])
        )
        self.valid_box_polygon_required = self._to_bool(
            params.get("valid_box_polygon_required", False)
        )
        if self.valid_box_polygon_required and not self.valid_box_map_polygon:
            raise ValueError("valid_box_polygon_required=True 时必须配置 valid_box_map_polygon")
        self.enable_colored_log = self._to_bool(params.get("enable_colored_log", True))
        self._last_selection_info = {}
        self.chassis_config = build_chassis_config(
            base_url=str(params.get("chassis_url", DEFAULT_CHASSIS_URL)).strip(),
        )

        self.blackboard.register_key(key=self.box_memory_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_memory_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key=self.current_box_target_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.current_box_target_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=self.finished_box_targets_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.finished_box_targets_key,
            access=py_trees.common.Access.WRITE,
        )

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        try:
            raw_memory = self._read_list(self.box_memory_key)
            memory = self._filter_memory_by_polygon(raw_memory)
            current_target = self._read_current_target()
            finished_targets = self._read_list(self.finished_box_targets_key)
            self._log_info(
                "记忆统计",
                "原始数量=%d 有效数量=%d 过滤数量=%d 已完成数量=%d 当前目标=%s"
                % (
                    len(raw_memory),
                    len(memory),
                    len(raw_memory) - len(memory),
                    len(finished_targets),
                    self._format_target(current_target),
                ),
                "cyan",
            )

            if current_target is not None:
                # 关键步骤：放箱流程成功后才进入本节点，因此此处把当前目标记录为已完成。
                finished_targets.append(current_target)
                self._log_info(
                    "记忆完成",
                    "加入已完成列表 序号=%d 目标=%s"
                    % (len(finished_targets), self._format_target(current_target)),
                    "magenta",
                )

            next_target = self._select_next_target(memory)
            if next_target is None:
                self.blackboard.set(self.current_box_target_key, None, overwrite=True)
                self.blackboard.set(self.box_memory_key, memory, overwrite=True)
                self.blackboard.set(
                    self.finished_box_targets_key,
                    finished_targets,
                    overwrite=True,
                )
                self._log_info(
                    "记忆为空",
                    "有效记忆数量=0 已完成数量=%d 当前目标=None，下一轮将等待 YOLO 新检测"
                    % len(finished_targets),
                    "yellow",
                )
                return Status.SUCCESS

            selected_index = self._last_selection_info.get("index")
            selected_count = self._last_selection_info.get("count", len(memory))
            selected_distance = self._last_selection_info.get("distance")
            current_pose = self._last_selection_info.get("current_pose")
            distance_text = "None" if selected_distance is None else "%.3f" % selected_distance
            self._log_info(
                "记忆目标选中",
                "选择策略=%s 选中序号=%s/%d 距离=%s 当前底盘位姿=%s 目标=%s"
                % (
                    self.select_policy,
                    "None" if selected_index is None else selected_index + 1,
                    selected_count,
                    distance_text,
                    self._format_pose2d(current_pose),
                    self._format_target(next_target),
                ),
                "green",
            )

            memory.remove(next_target)
            self.blackboard.set(self.current_box_target_key, next_target, overwrite=True)
            self.blackboard.set(self.box_memory_key, memory, overwrite=True)
            self.blackboard.set(self.finished_box_targets_key, finished_targets, overwrite=True)
            self._log_info(
                "记忆写回",
                "剩余数量=%d 已完成数量=%d 当前目标=%s"
                % (len(memory), len(finished_targets), self._format_target(next_target)),
                "magenta",
            )
            return Status.SUCCESS
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.get_logger().error(f"[{self.config_label}] 选择下一个记忆箱子失败: {exc}")
            return Status.FAILURE

    def _read_list(self, key):
        """读取 blackboard 列表，缺失或类型不符时返回空列表。"""
        if self.blackboard.exists(key):
            value = self.blackboard.get(key)
            if isinstance(value, list):
                return list(value)
        return []

    @staticmethod
    def _to_bool(value):
        """解析字符串/布尔配置。"""
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def _read_current_target(self):
        """读取当前抓取目标。"""
        if self.blackboard.exists(self.current_box_target_key):
            return self.blackboard.get(self.current_box_target_key)
        return None

    def _select_next_target(self, memory):
        """按策略从记忆列表里选择下一个目标。"""
        self._last_selection_info = {"index": None, "count": len(memory), "distance": None}
        if not memory:
            return None
        if self.select_policy != "nearest":
            self._last_selection_info = {
                "index": 0,
                "count": len(memory),
                "distance": None,
                "current_pose": None,
            }
            return memory[0]

        current_pose = get_chassis_current_pose(self.chassis_config)
        nearest_target = None
        nearest_distance = None
        nearest_index = None
        for index, target in enumerate(memory):
            map_position = target.get("map_position", {})
            if "x" not in map_position or "y" not in map_position:
                continue
            if not self._is_target_allowed(target):
                continue
            # 关键步骤：放箱后直接去下一个箱子，优先选择离当前底盘位置最近的记忆目标。
            distance = math.hypot(
                float(map_position["x"]) - current_pose.x,
                float(map_position["y"]) - current_pose.y,
            )
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_target = target
                nearest_index = index

        if nearest_target is None:
            nearest_target, nearest_index = self._get_first_allowed_target(memory)
            nearest_distance = None
        self._last_selection_info = {
            "index": nearest_index,
            "count": len(memory),
            "distance": nearest_distance,
            "current_pose": current_pose,
        }
        return nearest_target

    def _filter_memory_by_polygon(self, memory):
        """过滤掉不在有效 map 区域内的记忆目标。"""
        filtered_memory = []
        for index, target in enumerate(memory):
            if self._is_target_allowed(target):
                filtered_memory.append(target)
            else:
                self._log_info(
                    "记忆目标过滤",
                    "序号=%d/%d 过滤类型=指定区域外 目标=%s"
                    % (index + 1, len(memory), self._format_target(target)),
                    "yellow",
                )
        return filtered_memory

    def _get_first_allowed_target(self, memory):
        """按原始顺序返回第一个有效目标。"""
        for index, target in enumerate(memory):
            if self._is_target_allowed(target):
                return target, index
        return None, None

    def _is_target_allowed(self, target):
        """判断目标是否落在配置的有效 map 区域内。"""
        if target is None:
            return False
        return is_map_position_in_polygon(
            target.get("map_position"),
            self.valid_box_map_polygon,
        )

    def _log_info(self, tag, message, color):
        """输出带固定前缀和可选颜色的调试日志。"""
        text = f"[{self.config_label}] [{tag}] {message}"
        self.ros_node.get_logger().info(self._color_text(text, color))

    def _color_text(self, text, color):
        """按配置给日志添加 ANSI 颜色。"""
        if not self.enable_colored_log:
            return text

        color_codes = {
            "green": "\033[92m",
            "yellow": "\033[93m",
            "cyan": "\033[96m",
            "magenta": "\033[95m",
        }
        color_code = color_codes.get(color, "")
        reset_code = "\033[0m" if color_code else ""
        return f"{color_code}{text}{reset_code}"

    @staticmethod
    def _format_target(target):
        """格式化记忆目标，便于日志排查。"""
        if target is None:
            return "None"
        return "id=%s map坐标=%s base坐标=%s" % (
            target.get("id", ""),
            MoveBoxSelectNextMemoryBox._format_position(target.get("map_position")),
            MoveBoxSelectNextMemoryBox._format_position(target.get("base_position")),
        )

    @staticmethod
    def _format_position(position):
        """格式化三维/二维坐标。"""
        if position is None:
            return "None"
        return "(%.3f, %.3f, %.3f)" % (
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        )

    @staticmethod
    def _format_pose2d(pose):
        """格式化底盘二维位姿。"""
        if pose is None:
            return "None"
        return "(%.3f, %.3f, %.3f)" % (
            float(pose.x),
            float(pose.y),
            float(pose.yaw),
        )

    def describe_start(self):
        return f"[{self.config_label}] MoveBoxSelectNextMemoryBox start"
