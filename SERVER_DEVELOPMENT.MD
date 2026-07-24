# MercuryTree Server 链路与开发规范

本文档面向 service tree 任务开发者，用于说明 `tree.server_main` 启动后的整体链路，以及新增 HTTP 任务时应遵守的开发规范。

## Server 启动链路

server 模式入口：

```bash
python3 -m tree.server_main
```

入口文件：

```text
tree/server_main.py
```

常用启动参数：

```text
--task-host          任务 HTTP 服务监听地址，默认 127.0.0.1
--task-port          任务 HTTP 服务监听端口，默认 8766
--initial-tree       可选启动预加载树；默认不加载树、不启动 tick
--preload-services   可选 none/grasp_object/move_box，默认 none
```

启动后主要流程如下：

1. 解析 server 参数，并把剩余参数交给 ROS runtime。
2. 初始化 ROS1 runtime，当前节点名为 `pytrees_ros1_server`。
3. 从 `config/blackboard/blackboard.json` 加载 blackboard 默认配置。
4. 根据 `--preload-services` 可选预加载机器人 SDK services。
5. 构建 `BehaviorTreeRunner`。
6. 构建 `TaskRegistry`，注册 HTTP endpoint 到 service tree 的映射。
7. 构建 `TaskManager`，统一管理任务启动、运行状态和历史记录。
8. 启动 FastAPI 任务 HTTP 服务 `TaskHttpServer`。
9. 进入 `ros.spin()`，直到进程退出时关闭 HTTP 服务并销毁 runner。

server 默认启动后处于待命态：不加载 idle tree，也不创建 tick timer。只有收到任务 POST 后，`TaskManager` 才会调用 runner reload 对应单次行为树，并恢复 tick。

## 服务预加载

部署端可以按机器人负责的任务类型预初始化 SDK services，减少第一次任务触发后的等待时间：

```bash
python3 -m tree.server_main --preload-services grasp_object
python3 -m tree.server_main --preload-services move_box
```

预加载逻辑在 `preload_robot_services()` 中：

- `grasp_object` 使用 `build_grasp_object_services()`。
- `move_box` 使用 `build_robot_services()`。
- services 写入 `ROBOT_SERVICES_KEY`。
- 如果 services 带 `model_type`，同步写入 `MODEL_TYPE_KEY`。

注意：`start_server.sh` 当前默认使用 `--preload-services grasp_object`。如果联调搬箱任务，应手动使用 `--preload-services move_box`，或按部署目标调整脚本。

## HTTP 任务注册链路

任务注册表由 `build_task_registry()` 构建。当前注册的任务包括：

```text
/api/start_move_box
/api/start_move_box_direct_grasp_place_memory
/api/start_grasp_and_place
/api/start_navigation
```

每个任务由一个 `TaskDefinition` 描述：

```python
TaskDefinition(
    name="move_box",
    endpoint="/api/start_move_box",
    tree_file=move_box_tree,
    adapter=MoveBoxTaskAdapter(),
    root_label="HTTP move_box 单次任务流程",
)
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `name` | 任务名称，用于响应和状态信息 |
| `endpoint` | 外部 HTTP 启动接口 |
| `tree_file` | 本任务对应的 service tree JSON |
| `adapter` | 请求校验和 blackboard 写入适配器 |
| `root_label` | 根节点 label，便于快照和调试识别 |

## 单次任务执行链路

以 `move_box` 为例，一次任务从 HTTP 请求到行为树执行的链路如下：

1. 外部调用 `POST /api/start_move_box`。
2. FastAPI 使用 `MoveBoxRequestModel` 对请求体做首轮校验。
3. route 记录请求日志，包括 endpoint、method、调用者地址、user-agent 和 payload。
4. route 调用 `task_manager.start_task_by_endpoint()`。
5. `TaskManager` 刷新当前任务状态。
6. 如果已有任务处于 `QUEUED` 或 `RUNNING`，拒绝新任务。
7. 通过 `TaskRegistry` 找到 endpoint 对应的 `TaskDefinition`。
8. adapter 调用 `parse_payload()`，对请求做二次归一化。
9. 生成新的 `taskId`。
10. adapter 调用 `write_blackboard()`，把本次任务输入写入 blackboard。
11. runner 调用 `reload_tree()` 加载对应 service tree。
12. runner 重置 `tick_count`、`live_runtime` 和执行计时状态。
13. runner 确保 tick timer 已启动。
14. `TaskManager` 返回 `QUEUED` 状态给调用方。
15. 行为树开始周期性 tick。
16. 调用方通过 `GET /api/task_status?taskId=<taskId>` 查询任务状态。

关键顺序是：先写 blackboard，再 reload tree。这样可以避免新树恢复 tick 后抢先读取到旧输入或空输入。

## 任务状态规则

`TaskManager` 根据当前单次树根节点状态维护任务状态：

| root status | task status | 说明 |
| --- | --- | --- |
| `SUCCESS` | `SUCCEEDED` | 单次任务完成 |
| `FAILURE` | `FAILED` | 单次任务失败 |
| `INVALID` 或未 tick | `QUEUED` | 任务已接受，等待下一次 tick |
| 其他运行态 | `RUNNING` | 任务执行中 |

任务完成或失败只看当前单次树根节点终态，不依赖请求标志位。

任务失败时，`TaskManager` 会从快照中查找最深的失败节点。如果该节点存在 `feedback_message`，接口响应的 `message` 会优先透传这个失败原因。

## move_box 输入链路

`POST /api/start_move_box` 请求字段：

```json
{
  "naviPoseFindBox": {"x": 1.0, "y": 2.0, "yaw": 0.0},
  "validPolygon": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]],
  "naviPosePlaceBox": {"x": 3.0, "y": 2.0, "yaw": 90.0},
  "boxPosePlaceCenter": {"x": 3.5, "y": 2.0, "yaw": 90.0},
  "heightPlacePlane": 0.7
}
```

`MoveBoxTaskAdapter` 会写入以下 blackboard key：

```text
move_box_active_task_id
move_box_navi_pose_find_box
move_box_valid_polygon
move_box_navi_pose_place_box
move_box_box_pose_place_center
move_box_height_place_plane
```

service tree 消费关系：

| blackboard key | 使用位置 | 说明 |
| --- | --- | --- |
| `move_box_navi_pose_find_box` | `start_move_box.json` | 导航到 A 点搜索箱子 |
| `move_box_valid_polygon` | `move_box_yolo_fp_approach.json` | YOLO 有效识别区域 |
| `move_box_navi_pose_place_box` | `start_move_box.json` | 抓箱后预导航到 B 点 |
| `move_box_box_pose_place_center` | `start_move_box.json`、`move_box_place_flow.json` | C 点放置中心，用于反算最终站位和放置目标 |
| `move_box_height_place_plane` | `move_box_place_flow.json` | H 放置平面高度 |

## 新增任务开发规范

新增一个 service tree 任务时，按以下顺序开发：

1. 在 `tree/runtime/http_service/models.py` 中新增 Pydantic 请求模型。
2. 在 `tree/runtime/http_service/task_adapters/` 中新增任务 adapter。
3. 在 adapter 中集中声明 `TASK_INPUT_KEYS`。
4. 在 `config/tree/service/` 下新增对应 service tree JSON。
5. 在 `server_main.py` 的 `build_task_registry()` 中注册 `TaskDefinition`。
6. 在 `TaskHttpServer._make_app()` 中新增 FastAPI route。
7. 在 `SERVICE.MD` 中补充对外接口文档。
8. 增加请求模型、adapter、service tree 加载相关测试。

## 请求模型规范

请求模型负责对 HTTP payload 做结构化管理：

- 字段名使用对外接口约定，不在行为树节点中重新解释 HTTP payload。
- 数值字段统一做类型校验和归一化。
- 数字字符串可以按需要兼容，例如 `"0.7"` 归一化为 `0.7`。
- bool 不应被接受为数值字段。
- 复杂输入可以在 model 中归一化为统一结构，例如 `validPolygon` 同时兼容 `[x, y]` 和 `{ "x": x, "y": y }`。
- 错误提示应能直接指导调用方修正请求。

## Adapter 规范

adapter 只负责 HTTP 请求到 blackboard 的转换，不实现任务业务逻辑。

每个 adapter 应提供：

```python
def parse_payload(self, payload):
    ...

def write_blackboard(self, task_id, request):
    ...
```

规范要求：

- `parse_payload()` 校验并归一化请求。
- `write_blackboard()` 只写当前任务所需 blackboard key。
- blackboard key 集中放在 `TASK_INPUT_KEYS` 中。
- 写入 blackboard 时使用 `overwrite=True`，确保新任务覆盖旧输入。
- 行为树 JSON 和业务节点只读取 blackboard key，不直接依赖 HTTP 请求字段。

## 行为树规范

service tree 是一次性任务树，应满足：

- 单次任务启动时由 runner reload。
- 根节点进入 `SUCCESS` 表示任务完成。
- 根节点进入 `FAILURE` 表示任务失败。
- 不依赖请求标志位判断任务是否完成。
- 长耗时节点应提供清晰的 `feedback_message`，方便 `/api/task_status` 透传失败原因。
- 输入全部来自 adapter 写入的 blackboard key。
- 任务内部产生的中间 key 应有清晰前缀，避免不同任务之间冲突。

## Route 规范

FastAPI route 应保持薄封装：

- 使用 Pydantic model 作为请求体类型。
- 记录任务启动请求日志。
- 调用 `task_manager.start_task_by_endpoint()`。
- 捕获异常后返回 `StartTaskResponseModel`，状态使用 `REJECTED`。
- 不在 route 中写 blackboard。
- 不在 route 中 reload tree。
- 不在 route 中实现任务业务逻辑。

## 测试规范

新增或调整任务时，至少覆盖三类测试：

1. 请求模型测试：字段校验、数字字符串归一化、非法值拒绝。
2. adapter 测试：请求是否正确写入预期 blackboard key。
3. service tree 加载测试：对应 JSON 是否能被 `BehaviorTreeFactory` 加载。

涉及任务状态或失败原因时，还应补充 `TaskManager` 层测试：

- root `SUCCESS` 是否映射为 `SUCCEEDED`。
- root `FAILURE` 是否映射为 `FAILED`。
- 最深失败节点的 `feedback_message` 是否能透传。
- 历史任务记录是否按上限淘汰。

## 文档规范

对外 HTTP 协议更新时，应同步更新：

```text
SERVICE.MD
README.md
SERVER_DEVELOPMENT.MD
```

其中：

- `SERVICE.MD` 只描述对外 HTTP 接口、字段和响应。
- `README.md` 描述工程概览、启动方式和常用入口。
- `SERVER_DEVELOPMENT.MD` 描述 server 内部链路和开发规范。
