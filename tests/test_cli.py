"""不依赖 ROS 的 MercuryTree 命令行测试。"""

import os
import tempfile
import unittest

from tree.core.cli import build_argument_parser, resolve_tree_json_file


class CliTest(unittest.TestCase):
    def test_tree_argument_and_ros_arguments_are_separated(self):
        parser = build_argument_parser()

        cli_args, ros_args = parser.parse_known_args(
            ["--tree", "tree/demo/example.json", "__name:=custom_runner"]
        )

        self.assertEqual(cli_args.tree_json_file, "tree/demo/example.json")
        self.assertEqual(ros_args, ["__name:=custom_runner"])

    def test_long_tree_argument_alias(self):
        parser = build_argument_parser()

        cli_args, ros_args = parser.parse_known_args(
            ["--tree-json-file", "/tmp/custom_tree.json"]
        )

        self.assertEqual(cli_args.tree_json_file, "/tmp/custom_tree.json")
        self.assertEqual(ros_args, [])

    def test_relative_tree_path_is_resolved_from_config_directory(self):
        with tempfile.TemporaryDirectory() as project_root:
            resolved = resolve_tree_json_file(
                project_root,
                "tree/box/example.json",
            )

        self.assertEqual(
            resolved,
            os.path.join(project_root, "config", "tree", "box", "example.json"),
        )

    def test_absolute_tree_path_is_kept_absolute(self):
        with tempfile.TemporaryDirectory() as project_root:
            absolute_tree_path = os.path.join(project_root, "custom_tree.json")
            resolved = resolve_tree_json_file(project_root, absolute_tree_path)

        self.assertEqual(resolved, absolute_tree_path)


if __name__ == "__main__":
    unittest.main()
