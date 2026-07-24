# MercuryTree

`MercuryTree` 是基于 `py_trees` 的机器人行为树调度工程，当前主要用于在 ROS1 环境下运行搬箱、导航、抓取放置等任务。

核心能力：

- 从 JSON 配置构建行为树
- 通过 blackboard 在节点之间共享任务输入和运行状态
- 支持真实机器人节点和 mock/manual 节点
- 提供 Web Viewer 查看行为树 tick 状态
- 提供 HTTP server 模式，按接口触发单次任务

当前 Python 包目录是 `tree/`，ROS 包名是 `mercurytree`。

## 目录结构

```text
src/MercuryTree/
├── config/
│   ├── blackboard/
│   ├── depalletize/
│   ├── palletize/
│   └── tree/
│       ├── grasp_object/
│       ├── move_box/
│       └── service/
├── launch/
├── test/
├── tools/
│   └── build_grasp_search_cpp.sh
├── tree/
│   ├── core/
│   ├── node/
│   ├── runtime/
│   ├── ros_interface/
│   └── visualization/
├── SERVICE.MD
└── start_server.sh
```

## 推荐启动方式

抓取放置专用 server 推荐直接使用脚本：

```bash
cd src/MercuryTree
./start_server.sh
```

脚本会依次执行：

```bash
source ../scripts/source_kuavo_sdk_pythonpath.sh
bash tools/build_grasp_search_cpp.sh
python3 -m tree.server_main --preload-services grasp_object
```

`--preload-services grasp_object` 会在 server 启动时预初始化抓取任务需要的机器人 SDK services，避免收到任务后机器人原地等待 SDK 初始化。

## Server 模式

入口：

```bash
python3 -m tree.server_main
```

常用参数：

```text
--task-host          任务 HTTP 服务监听地址，默认 127.0.0.1
--task-port          任务 HTTP 服务监听端口，默认 8766
--initial-tree       可选启动预加载树；默认不加载树、不启动 tick
--preload-services   可选 none/grasp_object/move_box，默认 none
```

示例：

```bash
python3 -m tree.server_main --preload-services grasp_object
python3 -m tree.server_main --preload-services move_box
python3 -m tree.server_main --task-host 0.0.0.0 --task-port 8766
```

server 默认启动后处于待命态：不加载 idle tree，也不启动行为树 tick。收到任务 POST 后才 reload 对应 service tree，执行到 `SUCCESS` 或 `FAILURE` 后停止 tick，等待下一次任务。

## HTTP 接口

默认地址：

```text
http://127.0.0.1:8766
```

接口：

```text
GET  /health
GET  /api/task_status?taskId=<taskId>
POST /api/start_grasp_and_place
POST /api/start_move_box
POST /api/start_move_box_direct_grasp_place_memory
POST /api/start_navigation
```

完整接口文档见 [SERVICE.MD](SERVICE.MD)。server 内部链路和新增任务开发规范见 [SERVER_DEVELOPMENT.MD](SERVER_DEVELOPMENT.MD)。

查询任务状态：

```bash
curl -s "http://127.0.0.1:8766/api/task_status?taskId=<taskId>"
```

健康检查：

```bash
curl -s "http://127.0.0.1:8766/health"
```

server 一次只允许运行一个任务。当前任务处于 `QUEUED` 或 `RUNNING` 时，新任务会被拒绝，并返回当前正在运行的 `taskId`。

任务历史记录保存在当前 server 进程内存中，默认最多保留 1000 条。server 重启后历史记录不会保留。

## grasp_and_place Server 行为

service 树：

```text
config/tree/service/grasp_object/start_grasp_and_place.json
```

主要行为：

- 接口入参写入 blackboard 后 reload 单次 service tree
- A 点导航和腰部上升并行执行
- 抓取平面和放置平面高度按 `base_link` 参考系处理
- torso 采样 z 范围按 `heightGraspPlane - 0.37614321` 到 `heightGraspPlane - 0.37614321 + 0.4` 动态设置
- 没有检测到物体时任务失败退出，并返回失败原因
- 检测到物体但没有可达抓取目标时，缓存规划失败后只刷新感知重试一次；仍失败则退出
- 达到目标数量后，双臂和腰部会并行回到预备姿态

预备姿态：

```text
torso = [0.1, 0.000, 0.926, 0.0, 0.0, 0.0]
left  = [0.301, 0.3, 0.2, 0.0, -100.0, 0.0]
right = [0.301, -0.3, 0.2, 0.0, -100.0, 0.0]
```

## move_box Server 行为

service 树：

```text
config/tree/service/move_box/start_move_box.json
```

HTTP 入口：

```text
POST /api/start_move_box
```

主要行为：

- 接口入参写入 blackboard 后 reload 单次 service tree
- 导航到 A 点搜索箱子
- 通过 YOLO/FP 流程靠近箱体并完成底盘对齐
- 置位抓取请求，执行双臂直接抓箱
- 上提箱体与预导航到 B 点并行执行
- 根据 C 点箱体放置中心反算最终放置导航站位
- 导航到最终放置站位后，按 C 点和 `heightPlacePlane` 放置箱体

接口入参对应的 blackboard key：

```text
naviPoseFindBox     -> move_box_navi_pose_find_box
validPolygon        -> move_box_valid_polygon
naviPosePlaceBox    -> move_box_navi_pose_place_box
boxPosePlaceCenter  -> move_box_box_pose_place_center
heightPlacePlane    -> move_box_height_place_plane
```

`--preload-services move_box` 会在 server 启动时预初始化搬箱视觉与控制实例，ArmController 的 `target_frame` 为 `base_link`。

## move_box 记忆版直接抓放 Server 行为

service 树：

```text
config/tree/service/move_box/start_move_box_direct_grasp_place_memory.json
```

HTTP 入口：

```text
POST /api/start_move_box_direct_grasp_place_memory
```

该服务以 `tree/staring/move_box_full_direct_grasp_place_memory.json` 为基线，
由 HTTP 指定找箱/返回位姿、有效选箱区域、放置站位、放置平面高度与有限执行数量。
它保留 YOLO 记忆、YOLO/FP 两级靠近、直接抓取、上提、放置和回等待区流程，
但移除了测试用 Enter 暂停和无限循环。协议见 `SERVICE.MD`。

## 码垛任务

主树：

```text
config/palletize/move_box_palletize_strategy_preview.json
```

码垛当前是普通行为树流程，不是独立 HTTP endpoint。当前 server `TaskRegistry` 注册的任务只有 `move_box`、`grasp_and_place` 和 `navigation`。

主要行为：

- 初始化 move_box 视觉与控制实例
- 初始化 `move_box_pallet_stack_count`
- 导航到等待区域后循环执行四格码垛
- 抓箱前通过 YOLO/FP 完成靠近与底盘对齐
- 抓箱并上提箱体
- 根据垛盘 polygon、slot 参考点和当前 `stack_count` 计算本轮槽位、导航站位、放箱策略和动作点
- 导航到垛盘码垛站位，必要时先执行高位安全预落位
- 根据已选策略执行直接放箱或推箱放箱
- 放箱后刷新 FoundationPose 并报告实际箱心与目标箱心偏差
- 码垛成功后推进 `move_box_pallet_stack_count` 并返回等待区

关键子树：

```text
config/palletize/subtree/move_box_pallet_place_execute_strategy.json
```

关键节点：

```text
ComputeMoveBoxPalletPlaceStrategy
ComputeMoveBoxPalletPrePlaceSafeTargets
ComputeMoveBoxPalletPlaceActionPoints
RefreshFpAndReportPalletPlaceError
```

## 拆垛任务

主树：

```text
config/depalletize/move_box_full_dynamic_auto_depalletize.json
```

拆垛当前也是普通行为树流程，不是独立 HTTP endpoint。配置说明见：

```text
config/depalletize/README.md
```

单轮流程拆成五个阶段：

```text
subtree/01_select_and_approach_selected_pose.json  动态选箱与两级靠近
subtree/02_auto_grasp.json                         按黑板策略执行抓取
subtree/03_transport_to_place.json                 搬运到放置点
subtree/04_place_box.json                          放置箱体
subtree/05_return_and_ready.json                   返回等待区并恢复姿态
```

主树默认循环执行六轮动态拆垛。流程会先读取 YOLO 最近箱并粗靠近，再在更完整视野下重新选择最高层最近箱和抓取策略，随后执行锁定目标的 YOLO/FP 靠近、抓取、搬运、放置和回等待区。

## 服务预加载

`--preload-services` 用于指定 server 启动时预初始化哪类机器人服务：

```text
none          不预加载
grasp_object 预加载抓取放置服务，ArmController target_frame 为 waist_yaw_link
move_box     预加载搬箱服务，ArmController target_frame 为 base_link
```

当前 `robot_services` blackboard key 仍是共享 key，因此同一个 server 进程只支持预加载一种服务。实际部署中建议一台机器人负责一类任务，并指定对应 preload 类型。

## 日志

任务启动接口会记录调用信息：

```text
Task API request: endpoint=..., method=POST, client=<ip>:<port>, user_agent='...', payload={...}
```

参数校验失败会记录：

```text
Task API rejected: endpoint=..., method=POST, client=<ip>:<port>, error=..., payload=...
```

`/health` 和 `/api/task_status` 默认不打印访问日志，避免轮询刷屏。

## Web Viewer

默认地址：

```text
http://127.0.0.1:8765
```

server/main 当前配置为：

```text
web_viewer_host = 0.0.0.0
web_viewer_port = 8765
```

局域网访问：

```text
http://<机器人IP>:8765
```

## 普通行为树入口

非 server 模式入口：

```bash
python3 -m tree.main
```

`tree.main` 会读取 `config/blackboard/blackboard.json`，加载默认树并启动 runner。它更适合本地调试、mock/manual 流程或单棵树验证。

## Blackboard

启动时会读取：

```text
config/blackboard/blackboard.json
```

并通过 `tree/core/blackboard_bootstrap.py` 写入全局 `py_trees` blackboard。

常见运行时 key：

```text
robot_services
model_type
grasp_and_place_active_task_id
grasp_and_place_done_count
grasp_and_place_target_count
grasp_object_pick_navigation_target
grasp_object_place_navigation_target
grasp_object_height_grasp_plane
grasp_object_height_place_plane
grasp_object_sorted_grasp_objects
grasp_object_next_grasp_object_index
grasp_object_grasp_mode
grasp_object_torso_sample_z_min_m
grasp_object_torso_sample_z_max_m
move_box_active_task_id
move_box_navi_pose_find_box
move_box_valid_polygon
move_box_navi_pose_place_box
move_box_box_pose_place_center
move_box_height_place_plane
move_box_pallet_stack_count
move_box_pallet_stack_navigation_target
move_box_pallet_stack_place_plane_height
move_box_pallet_stack_slot_pose
move_box_pallet_stack_expected_box_pose
move_box_pallet_place_strategy
move_box_pallet_place_final_box_pose
```

## 行为树配置

常用 service tree：

```text
config/tree/service/grasp_object/start_grasp_and_place.json
config/tree/service/move_box/start_move_box.json
config/tree/service/navigation/start_navigation.json
```

普通抓放树：

```text
config/tree/grasp_object/grasp_and_place.json
config/tree/grasp_object/grasp_and_place_table.json
config/tree/grasp_object/grasp_and_place_stay.json
```

抓放子树：

```text
config/tree/grasp_object/subtree/
```

搬箱 service tree：

```text
config/tree/service/move_box/start_move_box.json
```

搬箱普通树和子树：

```text
config/tree/move_box/
config/tree/service/move_box/subtree/
```

码垛树：

```text
config/palletize/move_box_palletize_strategy_preview.json
config/palletize/subtree/
```

拆垛树：

```text
config/depalletize/move_box_full_dynamic_auto_depalletize.json
config/depalletize/subtree/
config/depalletize/test/
```

## C++ 抓取搜索扩展

抓取搜索 C++ 扩展构建脚本：

```bash
cd src/MercuryTree
bash tools/build_grasp_search_cpp.sh
```

`start_server.sh` 会在启动 server 前自动执行该脚本。

## 离线 JSON 可视化

工具：

```text
tree/visualization/tree_json_vis.py
```

用途：

- 不启动 ROS 时查看 JSON 树结构
- 递归展开 `SubTree.params.file`
- 生成 `.dot` 和图片输出到 `tree/visualization/output/`

运行：

```bash
cd src/MercuryTree
python3 tree/visualization/tree_json_vis.py
```

## 代码阅读入口

建议按这个顺序看：

1. `tree/server_main.py`
2. `tree/runtime/http_service/task_http_server.py`
3. `tree/runtime/http_service/task_manager.py`
4. `tree/runtime/http_service/task_adapters/`
5. `tree/core/runner.py`
6. `tree/core/tree_factory.py`
7. 当前任务对应的 service tree JSON
8. 对应 `tree/node/` 里的业务节点
