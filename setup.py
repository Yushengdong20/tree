"""setuptools 打包入口。

这个文件决定：
- 哪些 Python 包会被安装
- 哪些 JSON / launch / 前端静态资源会被打包到 share/
- `ros2 run mercurytree xxx` 能运行哪些入口命令
"""

from glob import glob
import os

from setuptools import find_packages, setup


package_name = "mercurytree"
python_package_name = "tree"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[python_package_name, f"{python_package_name}.*"]),
    data_files=[
        # 这些资源会被安装到 share/ 下，运行时 get_package_share_directory() 就从这里找。
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            [],
        ),
        (
            f"share/{package_name}/config/blackboard",
            glob("config/blackboard/*.json"),
        ),
        (
            f"share/{package_name}/config/motion/gait_arm",
            glob("config/motion/gait_arm/*.csv"),
        ),
        (
            f"share/{package_name}/config/tree/box",
            glob("config/tree/box/*.json"),
        ),
        (
            f"share/{package_name}/config/tree/demo",
            glob("config/tree/demo/*.json"),
        ),
        (
            f"share/{package_name}/config/tree/mock",
            glob("config/tree/mock/*.json"),
        ),
        (
            f"share/{package_name}/config/tree/http",
            glob("config/tree/http/*.json"),
        ),
        (
            f"share/{package_name}/config/tree/grasp_object",
            glob("config/tree/grasp_object/*.json"),
        ),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (
            f"share/{package_name}/visualization/web",
            [
                f"{python_package_name}/visualization/web/index.html",
                f"{python_package_name}/visualization/web/styles.css",
                f"{python_package_name}/visualization/web/viewer.js",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ysd",
    maintainer_email="ysd@example.com",
    description="ROS 2 Jazzy behaviour tree demo package built with py_trees.",
    license="MIT",
    entry_points={
        "console_scripts": [
            # bt_runner 是 ros2 run mercurytree bt_runner 的入口。
            "bt_runner = tree.main:main",
            # bt_manual_sender 是外部喂手动结果用的小工具。
            "bt_manual_sender = tree.tools.manual_result_sender:main",
            # bt_mock_http_server 用于在本地模拟底盘/抓取 HTTP 服务。
            "bt_mock_http_server = tree.tools.mock_http_server:main",
        ],
    },
)
