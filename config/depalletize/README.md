# 动态拆垛行为树

主树：`move_box_full_dynamic_auto_depalletize.json`

单轮流程拆成五个子树：

1. `subtree/01_select_and_approach.json`：参考 `config/tree/box/subtree/move_box_yolo_fp_approach_no_enter_cn.json` 展开 YOLO/FP 高效靠近流程，并加入动态选箱、预选 map 目标、导航跳过阈值和最终抓取数据刷新；参考子树本身不改动。
2. `subtree/02_auto_grasp.json`：唯一一次 Enter 确认，随后按黑板策略执行左拉、右拉或双爪直接抓取。
3. `subtree/03_transport_to_place.json`：导航到放置点，接近目标后同步调整腰部高度。
4. `subtree/04_place_box.json`：顺序下降、放置并释放箱体。
5. `subtree/05_return_and_ready.json`：返回等待区域，同时恢复双臂和 0.9m 腰部预备位姿。

建议按以下顺序进行累积实机验证：

1. `test/test_01_select_and_approach.json`
2. `test/test_02_through_auto_grasp.json`
3. `test/test_03_through_transport.json`
4. `test/test_04_through_place.json`
5. `test/test_05_full_cycle.json`

第 2～5 棵测试树都只在最新 FoundationPose 数据即将用于抓取动作之前等待一次 Enter；第 1 棵只验证选箱和靠近，因此没有 Enter。
