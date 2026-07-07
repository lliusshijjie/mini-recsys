# mini-recsys 三阶段扩展计划：可解释推荐 MVP + K8s Service

## Summary

目标是把当前 `mini-recsys` 从“召回 + 简单打分 Demo”演进成一个可解释、可部署到 K8s 的推荐服务。路线保持轻量：使用规则排序、可解释特征、多路召回、简单重排和行为反馈，不引入需要大规模训练数据的复杂模型。

## Phase 1：推荐链路 MVP

- 抽出推荐流水线：`Recall -> Rank -> Rerank -> Explain`，避免继续把逻辑堆在 `/recommend` handler 中。
- 实现多路召回：
  - 语义召回：保留现有 HNSW 用户向量召回。
  - 热门召回：把当前 fallback 热门补齐升级成正式召回源。
  - 类目召回：基于用户兴趣类别和商品 `category` 召回。
- 实现规则排序层：
  - 特征使用 `semantic_score`、`popularity`、`category_match`、`price_affinity`、`novelty`。
  - 默认权重：语义 0.50、类目 0.20、热度 0.20、价格/新颖度 0.10。
- 实现轻量重排层：
  - 已看过滤继续使用 Bloom Filter。
  - Top10 做类目多样性控制，避免单一类别刷屏。
  - 保留 1 个探索位，用于非最高分但相关的商品。
- `/recommend` 响应增加解释字段：
  - 每个 item 返回 `reason`，例如 `semantic_match`、`category_match`、`popular_item`、`exploration_slot`。
  - 返回 `source`，标识来自 `semantic`、`popular`、`category` 或混合来源。

## Phase 2：反馈闭环与可评估性

- 新增行为事件接口 `POST /events`：
  - 支持 `impression`、`click`、`like`、`dismiss`。
  - 暂不做 `purchase`，除非后续有真实业务流程。
- 存储用户行为明细：
  - 保留 Bloom Filter 做快速去重。
  - 额外保存最近行为列表，用于解释和画像更新。
- 实现轻量用户画像更新：
  - 点击、喜欢提高对应类别和相似 item 的偏好。
  - dismiss 降低对应类别或 item 的曝光优先级。
  - 不训练模型，只更新可解释的用户偏好权重。
- 增加离线评估/调试能力：
  - 新增调试接口或 CLI 输出某个用户的召回源、排序特征、重排原因。
  - 加基础指标：候选数量、过滤数量、类目分布、最终 TopN 来源分布。
- 前端展示解释信息：
  - 每个推荐卡片显示主要推荐理由。
  - 保留当前分数展示，但把 `Final/Sim/Pop` 扩展成更贴近排序特征的展示。

## Phase 3：K8s Service 化

- 配置外置化：
  - 用环境变量替代硬编码路径和端口。
  - 必须支持 `PORT`、`DATA_DIR`、`MODEL_PATH`、`TOKENIZER_PATH`、`CORS_ORIGIN`。
- 健康检查分层：
  - `/livez`：进程存活。
  - `/readyz`：Sled、HNSW、Tantivy、模型加载完成后才返回成功。
  - 保留 `/health` 作为兼容入口。
- 容器化与部署：
  - 新增 Dockerfile，包含 Rust 后端、C++ 编译环境和模型挂载约定。
  - 新增 K8s Deployment、Service、ConfigMap、PVC 示例。
  - 单副本 MVP 使用 PVC 持久化 `data/`；多副本暂不支持共享写入 Sled/HNSW。
- 运行可观测性：
  - 增加结构化日志：请求 ID、用户 ID、候选数、耗时、召回源分布。
  - 增加 `/metrics`，至少暴露请求数、延迟、错误数、推荐候选数。
- 启动可靠性：
  - 启动时完成模型加载、索引加载、一次 embedding warmup。
  - readiness 在 warmup 结束后才成功，避免 Pod 刚启动就接流量。

## Test Plan

- Phase 1：
  - 单元测试排序权重、召回合并、已看过滤、类目多样性、探索位。
  - API 测试 `/recommend` 返回 TopN、reason、source，且不会推荐已看 item。
- Phase 2：
  - 测试 `POST /events` 写入行为。
  - 测试 click/like/dismiss 对后续推荐结果产生可解释影响。
  - 测试调试输出能说明每个 item 的召回源和排序原因。
- Phase 3：
  - 测试环境变量覆盖默认配置。
  - 测试 `/livez`、`/readyz`、`/metrics`。
  - 本地构建 Docker 镜像并运行容器。
  - 用 K8s manifest 验证 Deployment、Service、PVC、ConfigMap 能启动单副本服务。

## Assumptions

- MVP 只做单服务、单副本可运行版本；多副本一致性和分布式索引不在本计划内。
- 不引入协同过滤、LTR、深度重排、Kafka、A/B 平台等需要大规模数据或复杂基础设施的能力。
- 当前商品数据、类别、图片和本地模型继续作为 MVP 数据源。
- Phase 1 和 Phase 2 优先保证推荐链路可解释；Phase 3 再把服务包装成 K8s 可运行形态。
