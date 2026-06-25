# mercurytree

`mercurytree` 是一个基于 `py_trees` 的行为树调度工程，当前主要用于：

- 从 JSON 配置构建行为树
- 在 ROS1/ROS2 环境中运行行为树
- 通过自定义 Web Viewer 观察树状态
- 用 blackboard 在节点之间共享运行参数
- 在真实节点和 mock 节点之间切换测试

当前这份代码里，`tree/` 是实际的 Python 包，ROS 包名是 `mercurytree`。

## 当前目录结构

```text
src/
├── README.md
├── setup.py
├── package.xml
├── launch/
│   └── run.launch.py
├── config/
│   ├── blackboard/
│   │   └── blackboard.json
│   └── tree/
│       ├── box/
│       │   └── move_box_full_direct_grasp_place_turn.json
│       ├── demo/
│       ├── http/
│       └── mock/
└── tree/
    ├── main.py
    ├── core/
    │   ├── blackboard_bootstrap.py
    │   ├── manual_input.py
    │   ├── runner.py
    │   ├── runner_config.py
    │   └── tree_factory.py
    ├── node/
    │   ├── base.py
    │   ├── common/
    │   ├── http/
    │   ├── manipulation/
    │   └── mock/
    ├── runtime/
    │   ├── http/
    │   ├── manipulation/
    │   ├── mock/
    │   └── move_box/
    ├── ros_interface/
    ├── tools/
    ├── utils/
    └── visualization/
```

## 运行入口

- [tree/main.py](tree/main.py)
  入口文件，只负责：
  - 选择默认树 JSON
  - 读取 `config/blackboard/blackboard.json`
  - 启动 `BehaviorTreeRunner`
- [tree/core/runner.py](tree/core/runner.py)
  行为树运行中枢，负责：
  - 加载树
  - 周期性 tick
  - 维护快照
  - 启动 web viewer
- [tree/core/tree\_factory.py](tree/core/tree_factory.py)
  把 JSON 转成 `py_trees` 运行时对象。

## 节点加载顺序

当前 `tree_factory.py` 的叶子节点加载顺序是：

1. `tree.node.http.*`
2. `tree.node.*`
3. `tree.node.common.*`
4. `tree.node.manipulation.*`
5. `tree.node.move_box.*`
6. `tree.node.mock.*`

这意味着：

- 真实业务节点优先
- mock 节点只在真实实现不存在时才兜底

例如 JSON 里写 `MoveClaw` 时，现在会优先命中：

- [tree/node/manipulation/move\_claw.py](tree/node/manipulation/move_claw.py)

而不是：

- [tree/node/mock/move\_claw.py](tree/node/mock/move_claw.py)

## Blackboard 机制

启动时会自动读取：

- [config/blackboard/blackboard.json](config/blackboard/blackboard.json)

并通过：

- [tree/core/blackboard\_bootstrap.py](tree/core/blackboard_bootstrap.py)

把其中的顶层 key 写入全局 `py_trees` blackboard。

当前常用的 blackboard key 包括：

- `arm_target`
- `arm_target_a`
- `arm_target_b`

`MoveArmBaseTargetPose` 当前的取参顺序是：

1. 先读 blackboard
2. blackboard 没值时回退到节点自身 `params`

## 当前常用测试树

### 1. `ros1_smoke_test.json`

- [config/tree/mock/ros1\_smoke\_test.json](config/tree/mock/ros1_smoke_test.json)

用途：

- 验证 runner/timer/logging/tree tick 主链路
- 主要走 mock 节点

### 2. `move_arm_target_pose_from_params_demo.json`

- [config/tree/demo/move\_arm\_target\_pose\_from\_params\_demo.json](config/tree/demo/move_arm_target_pose_from_params_demo.json)

用途：

- 验证 `MoveArmBaseTargetPose` 从节点自身 `params` 读取目标
- 通过把 `blackboard_*_key` 指到不存在的 key，避免 blackboard 抢占参数来源

### 3. `move_arm_target_pose_from_blackboard_demo.json`

- [config/tree/demo/move\_arm\_target\_pose\_from\_blackboard\_demo.json](config/tree/demo/move_arm_target_pose_from_blackboard_demo.json)

用途：

- 验证 `MoveArmBaseTargetPose` 从 blackboard 读取目标

### 4. `move_arm_repeat_until_enter_demo.json`

- [config/tree/demo/move\_arm\_repeat\_until\_enter\_demo.json](config/tree/demo/move_arm_repeat_until_enter_demo.json)

用途：

- 让手臂在 `arm_target_a` 和 `arm_target_b` 两组目标之间持续来回运动
- 直到终端按下 Enter 后，树根返回 `SUCCESS`

结构大致是：

```text
cd /home/lab/leju_wbc/src/kuavo_humanoid_sdk/mercurytree

PYTHONPATH=/home/lab/leju_wbc/src/kuavo_humanoid_sdk:/home/lab/leju_wbc/src/kuavo_humanoid_sdk/mercurytree:$PYTHONPATH \
python3 -m tree.main
```

### 5. `move_claw_test.json`

- [config/tree/demo/move\_claw\_test.json](config/tree/demo/move_claw_test.json)

用途：

- 单独测试真实 `MoveClaw`
- 先张开夹爪
- 等终端按 Enter
- 再闭合夹爪

说明：

- 这份测试树和真实 `move_claw.py` 已经接入当前项目
- `move_claw_test.json` 已完成实机测试验证

## 当前默认入口配置

截至当前版本，[tree/main.py](tree/main.py) 默认是：

- `tree_file_name = "tree/box/move_box_full_direct_grasp_place_turn.json"`
- `tick_period_ms = 200`
- `enable_web_viewer = True`
- `web_viewer_host = "0.0.0.0"`
- `web_viewer_port = 8765`
- `stop_on_terminal_state = True`
- `manual_result_mode = True`
- `enable_manual_result_input = True`
- `enable_py_trees_ros_viewer = False`

这意味着当前默认行为是：

- 启动后加载 `blackboard.json`
- 执行 `move_box` 真机测试树
- 在关键步骤停下等待人工输入确认

## 运行方式

### 推荐：直接按 Python 包运行

```bash
cd /home/lab/leju_wbc/src/kuavo_humanoid_sdk/mercurytree

PYTHONPATH=/home/lab/leju_wbc/src/kuavo_humanoid_sdk:/home/lab/leju_wbc/src/kuavo_humanoid_sdk/mercurytree:$PYTHONPATH \
python3 -m tree.main
```

### ROS 包方式

```bash
cd MercuryTree
colcon build --packages-select mercurytree
source ../install/setup.bash
ros2 run mercurytree bt_runner
```

如果是 ROS1 环境，也建议优先直接使用：

```bash
cd /home/lab/leju_wbc/src/kuavo_humanoid_sdk/mercurytree

PYTHONPATH=/home/lab/leju_wbc/src/kuavo_humanoid_sdk:/home/lab/leju_wbc/src/kuavo_humanoid_sdk/mercurytree:$PYTHONPATH \
python3 -m tree.main
```

## Web Viewer

当前自定义 Web Viewer 默认地址：

```text
http://127.0.0.1:8765
```

如果要让同一局域网的其他电脑访问，运行机上的 `main.py` 里已经配置为：

```python
web_viewer_host = "0.0.0.0"
```

所以其他机器可以直接访问：

```text
http://运行机IP:8765
```

## 行为树 JSON 静态可视化

项目提供了一个离线 JSON 可视化工具：

- [tree/visualization/tree\_json\_vis.py](tree/visualization/tree_json_vis.py)

用途：

- 不启动 ROS、不运行行为树时，直接查看 JSON 配置里的树结构
- 自动按 `SubTree.params.file` 递归展开子树
- 生成 `.png` 图片和 `.dot` 文件，便于检查完整流程

从项目根目录运行：

```bash
cd MercuryTree
python3 tree/visualization/tree_json_vis.py
```

也可以在工具目录直接运行：

```bash
cd tree/visualization
python3 tree_json_vis.py
```

默认会可视化：

```text
config/tree/box/move_box_full_direct_grasp_place_turn.json
```

并输出到：

```text
tree/visualization/output/
```

指定其他树：

```bash
cd /home/lab/leju_wbc/src/kuavo_humanoid_sdk/mercurytree

PYTHONPATH=/home/lab/leju_wbc/src/kuavo_humanoid_sdk:/home/lab/leju_wbc/src/kuavo_humanoid_sdk/mercurytree:$PYTHONPATH \
python3 -m tree.main
```

常用参数：

```bash
cd MercuryTree
colcon build --packages-select mercurytree
source ../install/setup.bash
ros2 run mercurytree bt_runner
```

说明：

- 这个工具是离线结构图，用来检查 JSON 配置和子树展开结果
- Web Viewer 是运行时状态观察工具，用来查看 tick 过程和节点状态

## 手动 / mock 模式

`TimedMockAction` 基类支持 `manual_result_mode`，用于：

- 软件验证树结构
- 不直接访问真实服务/真实硬件

在 `main.py` 中如果设置：

```python
manual_result_mode = True
enable_manual_result_input = True
```

则终端支持：

```text
cd /home/lab/leju_wbc/src/kuavo_humanoid_sdk/mercurytree

PYTHONPATH=/home/lab/leju_wbc/src/kuavo_humanoid_sdk:/home/lab/leju_wbc/src/kuavo_humanoid_sdk/mercurytree:$PYTHONPATH \
python3 -m tree.main
```

## 这轮新增或重构过的节点

- [tree/node/manipulation/move\_arm\_base\_target\_pose.py](tree/node/manipulation/move_arm_base_target_pose.py)
- [tree/node/manipulation/move\_arm\_base\_target\_pose.py](tree/node/manipulation/move_arm_base_target_pose.py)
  - 支持 blackboard 优先 / params 回退
  - 支持双臂目标结构
- [tree/node/manipulation/move\_claw.py](tree/node/manipulation/move_claw.py)
  - 已适配当前项目
  - 真实夹爪命令通过 `rospy.Publisher` 发布
- [tree/node/common/wait\_for\_enter.py](tree/node/common/wait_for_enter.py)
  - 阻塞式等待 Enter
- [tree/node/common/wait\_for\_enter\_async.py](tree/node/common/wait_for_enter_async.py)
  - 非阻塞等待 Enter，适合并行停止树

## 公共工具

- [tree/utils/params.py](tree/utils/params.py)
  - 公共参数解析
- [tree/utils/arm\_target.py](tree/utils/arm_target.py)
  - 机械臂目标结构判断与 wrench 归一化

## 阅读建议

建议按这个顺序看代码：

1. [tree/main.py](tree/main.py)
2. [tree/core/runner.py](tree/core/runner.py)
3. [tree/core/tree\_factory.py](tree/core/tree_factory.py)
4. [tree/core/blackboard\_bootstrap.py](tree/core/blackboard_bootstrap.py)
5. [tree/node/base.py](tree/node/base.py)
6. 当前正在调的那棵树对应的 JSON
7. 对应的业务节点实现
