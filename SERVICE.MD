# MercuryTree HTTP 服务接口文档

## 基本信息

任务 HTTP 服务由 `tree/server_main.py` 启动，默认监听：

```text
http://127.0.0.1:8766
```

启动参数：

```text
--task-host          任务 HTTP 服务监听地址，默认 127.0.0.1
--task-port          任务 HTTP 服务监听端口，默认 8766
--initial-tree       可选启动预加载树；默认不加载树、不启动 tick
--preload-services   可选 none/grasp_object/move_box，默认 none
```

所有接口响应均为 JSON，响应头：

```text
Content-Type: application/json; charset=utf-8
```

说明：

- 当前任务服务一次只允许运行一个任务。
- server 默认启动后处于待命态，不加载空闲树，也不启动行为树 tick。
- 每次 POST 任务请求都会 reload 对应的单次行为树；该树到达 `SUCCESS` 或 `FAILURE` 后 runner 暂停 tick，下一次 POST 再 reload 新树并恢复 tick。
- 任务完成/失败由当前单次树根节点终态决定，不依赖请求标志位。
- POST 接口的 HTTP 状态码通常为 `200`，业务是否成功通过响应体里的 `success` 判断。
- 未注册的 URL 会返回 HTTP `404`。
- 任务历史记录保存在当前 server 进程内存中，默认最多保留 1000 条；server 重启或记录被淘汰后，查询旧 `taskId` 会返回 `UNKNOWN`。
- 启动任务接口会在服务端记录 endpoint、method、调用者 IP:port、user-agent 和请求 payload，便于联调排查。

部署端可按机器人负责的任务类型预初始化服务，减少第一次任务触发后的等待时间：

```bash
python3 -m tree.server_main --preload-services grasp_object
python3 -m tree.server_main --preload-services move_box
```

FastAPI 自动文档：

| URL | 说明 |
| --- | --- |
| `/docs` | Swagger UI，可在浏览器中查看和调试所有接口 |
| `/redoc` | ReDoc 风格接口文档 |
| `/openapi.json` | OpenAPI JSON schema |

命令行查看 OpenAPI schema：

```bash
curl http://127.0.0.1:8766/openapi.json
```

## 通用数据结构

### Pose2D

二维位姿对象。

```json
{
  "x": 1.0,
  "y": 2.0,
  "yaw": 0.0
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `x` | number | 是 | map 坐标系下 x 坐标 |
| `y` | number | 是 | map 坐标系下 y 坐标 |
| `yaw` | number | 是 | 航向角，单位 degree |

### PolygonPoint

有效区域多边形点。

标准格式：

```json
{
  "x": 0.0,
  "y": 0.0
}
```

兼容输入格式：

```json
[0.0, 0.0]
```

服务内部会统一归一化为对象格式。

## GET /health

健康检查接口。

### 请求

```http
GET /health
```

请求体为空。

### 调用示例

```bash
curl http://127.0.0.1:8766/health
```

### 响应

```json
{
  "success": true,
  "message": "task server running",
  "endpoints": [
    "/api/start_grasp_and_place",
    "/api/start_move_box_direct_grasp_place_memory",
    "/api/start_navigation",
    "/api/start_move_box"
  ]
}
```

Endpoint 列表按服务端注册表排序返回，客户端不应依赖固定顺序。

## POST /api/start_move_box_direct_grasp_place_memory

启动 YOLO 记忆版直接抓箱放箱任务。该任务以
`tree/staring/move_box_full_direct_grasp_place_memory.json` 为业务基线：
导航到找箱等待区、YOLO/FP 对齐并直接抓取、导航至放置点、按高度放置、
返回等待区；重复直至达到 `targetCount`。服务树没有 Enter 暂停节点。

启动服务时应使用 `--preload-services move_box`。

### 请求

```http
POST /api/start_move_box_direct_grasp_place_memory
Content-Type: application/json
```

```json
{
  "naviPoseFindBox": {"x": 0.84, "y": 4.26, "yaw": -105.02},
  "validPolygon": [[-1.5, 3.0], [-1.5, -1.3], [3.0, -1.3], [3.0, 3.0]],
  "naviPosePlaceBox": {"x": 1.84, "y": 4.26, "yaw": 6.3},
  "heightPlacePlane": 0.7,
  "targetCount": 1
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `naviPoseFindBox` | Pose2D | 是 | 初始找箱等待区，也是每轮放置后的返回位姿。 |
| `validPolygon` | PolygonPoint[] | 是 | map 系下允许选择待抓箱的区域，至少 3 点。 |
| `naviPosePlaceBox` | Pose2D | 是 | 抓箱后导航到的放置站位。 |
| `heightPlacePlane` | number | 是 | 放置平面高度，按 `base_link` 系解释。 |
| `targetCount` | integer | 否 | 本次任务完成的箱子数量，默认 `1`，必须不小于 `1`。 |

该接口不使用 `boxPosePlaceCenter`：它保持记忆版树的“到 B 点后原位下降放置”逻辑；若需要 C 点精确反算站位，请使用既有 `/api/start_move_box`。

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `success` | boolean | 服务是否正常 |
| `message` | string | 状态说明 |
| `endpoints` | string[] | 当前注册的任务启动接口列表 |

## POST /api/start_move_box

启动搬箱任务。

### 请求

```http
POST /api/start_move_box
Content-Type: application/json
```

请求体：

```json
{
  "naviPoseFindBox": {
    "x": 1.0,
    "y": 2.0,
    "yaw": 0.0
  },
  "validPolygon": [
    {
      "x": 0.0,
      "y": 0.0
    },
    {
      "x": 2.0,
      "y": 0.0
    },
    {
      "x": 2.0,
      "y": 2.0
    }
  ],
  "naviPosePlaceBox": {
    "x": 3.0,
    "y": 2.0,
    "yaw": 90.0
  },
  "boxPosePlaceCenter": {
    "x": 3.5,
    "y": 2.0,
    "yaw": 90.0
  },
  "heightPlacePlane": 0.7
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `naviPoseFindBox` | Pose2D | 是 | 导航到找箱位置的目标位姿 |
| `validPolygon` | PolygonPoint[] | 是 | 箱体识别有效区域，至少 3 个点 |
| `naviPosePlaceBox` | Pose2D | 是 | 搬箱后预导航到放置区域的目标位姿 |
| `boxPosePlaceCenter` | Pose2D | 是 | 箱体放置中心位姿 |
| `heightPlacePlane` | number | 是 | 放置平面高度 |

校验规则：

- `naviPoseFindBox`、`naviPosePlaceBox`、`boxPosePlaceCenter` 必须包含 `x`、`y`、`yaw`。
- `validPolygon` 必须是数组，且至少包含 3 个点。
- `validPolygon` 的点支持 `{ "x": 0.0, "y": 0.0 }` 或 `[0.0, 0.0]`。
- 数值字段允许传入数字字符串，例如 `"0.7"`，会归一化为 number。
- bool 不接受为数值字段。

### 调用示例

```bash
curl -X POST http://127.0.0.1:8766/api/start_move_box \
  -H 'Content-Type: application/json' \
  -d '{
    "naviPoseFindBox": {"x": 1.0, "y": 2.0, "yaw": 0.0},
    "validPolygon": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]],
    "naviPosePlaceBox": {"x": 3.0, "y": 2.0, "yaw": 90.0},
    "boxPosePlaceCenter": {"x": 3.5, "y": 2.0, "yaw": 90.0},
    "heightPlacePlane": 0.7
  }'
```

### 成功响应

```json
{
  "success": true,
  "taskId": "bcb46d96-7e10-4306-95bc-6b037ddb53d3",
  "status": "QUEUED",
  "message": "move_box task accepted",
  "taskName": "move_box"
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `success` | boolean | 是否成功接受任务 |
| `taskId` | string | 任务 ID，用于查询任务状态 |
| `status` | string | 任务状态 |
| `message` | string | 状态说明 |
| `taskName` | string | 任务名称 |

### 失败响应

已有任务运行中：

```json
{
  "success": false,
  "taskId": "bcb46d96-7e10-4306-95bc-6b037ddb53d3",
  "status": "RUNNING",
  "message": "task is already running",
  "taskName": "move_box"
}
```

请求体校验失败：

```json
{
  "success": false,
  "taskId": "",
  "status": "REJECTED",
  "message": "body.validPolygon: Value error, validPolygon must contain at least 3 points",
  "taskName": ""
}
```

## POST /api/start_grasp_and_place

启动零件抓放任务。抓取点和放置点均为底盘导航目标，抓取候选由抓取视觉服务计算；任务会按“至少搬够 `targetCount` 个”结束，允许最后一轮双抓后完成数超过目标 1 个。

业务说明：

- 未检测到可抓取物体时，任务会进入 `FAILED`，失败原因可从 `message` 或 `failedNode.message` 读取。
- 检测到物体但没有可达抓取目标时，任务会刷新感知重试一次；仍不可达则进入 `FAILED`。
- 达到目标数量后，任务会让腰部和双臂回到预备姿态，再返回 `SUCCEEDED`。

### 请求

```http
POST /api/start_grasp_and_place
Content-Type: application/json
```

请求体：

```json
{
  "naviPoseGrasp": {
    "x": 1.0,
    "y": 2.0,
    "yaw": 0.0
  },
  "naviPosePlace": {
    "x": 3.0,
    "y": 2.0,
    "yaw": 90.0
  },
  "heightGraspPlane": 0.8,
  "heightPlacePlane": 0.9,
  "targetCount": 5
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `naviPoseGrasp` | Pose2D | 是 | 导航到零件抓取区的目标位姿 |
| `naviPosePlace` | Pose2D | 是 | 导航到零件放置区的目标位姿 |
| `heightGraspPlane` | number | 是 | 抓取平面高度；server 会按 `heightGraspPlane - 0.37614321` 到 `heightGraspPlane - 0.37614321 + 0.4` 动态设置 torso sample z 范围 |
| `heightPlacePlane` | number | 是 | 放置平面高度 |
| `targetCount` | integer | 是 | 目标搬运数量，必须大于等于 1 |

校验规则：

- `naviPoseGrasp`、`naviPosePlace` 必须包含 `x`、`y`、`yaw`。
- 数值字段允许传入数字字符串，例如 `"0.8"`，会归一化为 number。
- `targetCount` 允许传入整数字符串，例如 `"5"`，会归一化为 integer。
- bool 不接受为数值字段。

### 调用示例

```bash
curl -X POST http://127.0.0.1:8766/api/start_grasp_and_place \
  -H 'Content-Type: application/json' \
  -d '{
    "naviPoseGrasp": {"x": 1.0, "y": 2.0, "yaw": 0.0},
    "naviPosePlace": {"x": 3.0, "y": 2.0, "yaw": 90.0},
    "heightGraspPlane": 0.8,
    "heightPlacePlane": 0.9,
    "targetCount": 5
  }'
```

### 成功响应

```json
{
  "success": true,
  "taskId": "bcb46d96-7e10-4306-95bc-6b037ddb53d3",
  "status": "QUEUED",
  "message": "grasp_and_place task accepted",
  "taskName": "grasp_and_place"
}
```

## POST /api/start_navigation

启动单点导航任务。

### 请求

```http
POST /api/start_navigation
Content-Type: application/json
```

请求体：

```json
{
  "targetPose": {
    "x": 1.0,
    "y": 2.0,
    "yaw": 0.0
  }
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `targetPose` | Pose2D | 是 | 导航目标位姿 |

校验规则：

- `targetPose` 必须包含 `x`、`y`、`yaw`。
- 数值字段允许传入数字字符串，例如 `"1.0"`，会归一化为 number。
- bool 不接受为数值字段。

### 调用示例

```bash
curl -X POST http://127.0.0.1:8766/api/start_navigation \
  -H 'Content-Type: application/json' \
  -d '{
    "targetPose": {"x": 1.0, "y": 2.0, "yaw": 0.0}
  }'
```

### 成功响应

```json
{
  "success": true,
  "taskId": "bcb46d96-7e10-4306-95bc-6b037ddb53d3",
  "status": "QUEUED",
  "message": "navigation task accepted",
  "taskName": "navigation"
}
```

## GET /api/task_status

查询任务状态。

### 请求

```http
GET /api/task_status?taskId={taskId}
```

Query 参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `taskId` | string | 是 | 任务启动接口返回的任务 ID |

### 调用示例

```bash
curl 'http://127.0.0.1:8766/api/task_status?taskId=bcb46d96-7e10-4306-95bc-6b037ddb53d3'
```

### 响应

```json
{
  "success": true,
  "taskId": "bcb46d96-7e10-4306-95bc-6b037ddb53d3",
  "status": "RUNNING",
  "message": "move_box task running",
  "taskName": "move_box",
  "currentNode": {
    "id": "root/HTTP move_box 单次任务流程[0]/导航到 A 点搜索箱子[1]",
    "name": "HttpNavigateToPose",
    "label": "导航到 A 点搜索箱子",
    "type": "HttpNavigateToPose",
    "status": "RUNNING"
  },
  "failedNode": null,
  "rootStatus": "RUNNING",
  "executionState": "RUNNING",
  "tickCount": 120,
  "liveRuntime": null,
  "createdAt": 1730000000.123,
  "finishedAt": null
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `success` | boolean | 查询是否成功 |
| `taskId` | string | 任务 ID |
| `status` | string | 任务状态 |
| `message` | string | 状态说明 |
| `taskName` | string | 任务名称 |
| `currentNode` | TaskNode/null | 当前运行中的最深行为树节点，可能为空 |
| `failedNode` | FailedTaskNode/null | 失败时最深的失败行为树节点，可能为空 |
| `rootStatus` | string/null | 当前单次行为树根节点状态；`SUCCESS` 表示任务成功，`FAILURE` 表示任务失败 |
| `executionState` | string/null | runner 执行状态；单次树终态后通常为 `STOPPED` |
| `tickCount` | number | 行为树 tick 次数 |
| `liveRuntime` | object/null | 长耗时节点运行时提示信息 |
| `createdAt` | number/null | 任务创建时间，Unix timestamp，单位秒 |
| `finishedAt` | number/null | 任务结束时间，Unix timestamp，单位秒 |

### TaskNode

`currentNode` 的对象结构：

```json
{
  "id": "root/HTTP navigation 单次任务流程[0]/导航到 HTTP 目标点[1]",
  "label": "导航到 HTTP 目标点",
  "name": "HttpNavigateToPose",
  "type": "HttpNavigateToPose",
  "status": "RUNNING"
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 行为树快照中的节点路径 ID，用于调试定位 |
| `label` | string/null | JSON 配置里的节点展示名 |
| `name` | string/null | py_trees 节点实例名 |
| `type` | string/null | JSON 中的原始节点类型，例如 `HttpNavigateToPose` |
| `status` | string/null | 节点状态，通常为 `RUNNING` |

### FailedTaskNode

`failedNode` 的对象结构：

```json
{
  "id": "root/HTTP navigation 单次任务流程[0]/导航到 HTTP 目标点[1]",
  "label": "导航到 HTTP 目标点",
  "name": "HttpNavigateToPose",
  "type": "HttpNavigateToPose",
  "status": "FAILURE",
  "message": "navigation task failed: {\"task_status\": 6}"
}
```

字段说明：

| 字段 | 类型 | 必定存在 | 说明 |
| --- | --- | --- | --- |
| `id` | string | 是 | 行为树快照中的节点路径 ID，用于调试定位 |
| `label` | string/null | 是 | JSON 配置里的节点展示名 |
| `name` | string/null | 是 | py_trees 节点实例名 |
| `type` | string/null | 是 | JSON 中的原始节点类型，例如 `HttpNavigateToPose` |
| `status` | string/null | 是 | 节点状态，失败节点通常为 `FAILURE` |
| `message` | string | 否 | 节点底层失败原因；只有节点提供 `feedback_message` 时才返回 |

客户端展示建议：

- 用户可见错误优先展示顶层 `message`。
- 需要定位失败节点时读取 `failedNode.label`、`failedNode.name`、`failedNode.type`。
- 需要底层原因时读取 `failedNode.message`；该字段存在时通常和顶层 `message` 一致。

### liveRuntime

`liveRuntime` 为长耗时节点的临时运行提示，当前通常是对象或 `null`。字段不是稳定业务协议，客户端可用于调试展示，不建议作为任务状态判断依据。

### 状态枚举

| 状态 | 说明 |
| --- | --- |
| `QUEUED` | 任务已接受，单次树已 reload，等待行为树 tick |
| `RUNNING` | 任务运行中 |
| `SUCCEEDED` | 单次树根节点返回 `SUCCESS`，任务成功完成 |
| `FAILED` | 单次树根节点返回 `FAILURE`，任务失败 |
| `REJECTED` | 任务被拒绝 |
| `UNKNOWN` | 未找到任务或查询参数缺失 |

导航失败时，`message` 会优先透传失败节点的底层原因，`failedNode` 会给出对应节点，例如：

```json
{
  "success": true,
  "taskId": "bcb46d96-7e10-4306-95bc-6b037ddb53d3",
  "status": "FAILED",
  "message": "navigation task failed: {\"task_status\": 6}",
  "taskName": "navigation",
  "currentNode": null,
  "failedNode": {
    "id": "root/HTTP navigation 单次任务流程[0]/导航到 HTTP 目标点[1]",
    "name": "HttpNavigateToPose",
    "label": "导航到 HTTP 目标点",
    "type": "HttpNavigateToPose",
    "status": "FAILURE",
    "message": "navigation task failed: {\"task_status\": 6}"
  },
  "rootStatus": "FAILURE",
  "executionState": "STOPPED",
  "tickCount": 42,
  "liveRuntime": null,
  "createdAt": 1730000000.123,
  "finishedAt": 1730000012.456
}
```

### 失败响应

缺少 `taskId`：

```json
{
  "success": false,
  "taskId": "",
  "status": "UNKNOWN",
  "message": "missing taskId",
  "taskName": "",
  "currentNode": null,
  "failedNode": null,
  "rootStatus": null,
  "executionState": null,
  "tickCount": 0,
  "liveRuntime": null,
  "createdAt": null,
  "finishedAt": null
}
```

未知 `taskId`：

```json
{
  "success": false,
  "taskId": "unknown-task-id",
  "status": "UNKNOWN",
  "message": "unknown taskId",
  "taskName": "",
  "currentNode": null,
  "failedNode": null,
  "rootStatus": null,
  "executionState": null,
  "tickCount": 0,
  "liveRuntime": null,
  "createdAt": null,
  "finishedAt": null
}
```
