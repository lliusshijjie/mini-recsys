# 可解释逻辑回归排序优化方案

## 1. 背景

当前推荐链路已经具备 `Recall -> Rank -> Rerank -> Explain`、四路召回、行为反馈和固定权重排序。现有排序公式能够作为稳定基线，但权重来自人工经验，无法根据真实曝光、点击、喜欢和忽略行为自动调整。

本方案在保留现有可解释能力和固定权重回退的基础上，引入离线逻辑回归训练。训练程序输出纯 JSON 模型，Rust 服务只负责加载模型和执行轻量推理，不引入在线训练服务、Python Sidecar、Kafka 或复杂 LTR 基础设施。

## 2. 目标与非目标

### 目标

- 使用真实行为数据学习排序特征权重。
- 让四路召回的命中情况、分数和排名进入统一排序模型。
- 模型可版本化、可验证、可回退，并可在 K8s 单副本服务中运行。
- 使用线性特征贡献解释每个推荐结果。
- 建立离线指标，能够证明新模型优于固定权重基线。

### 非目标

- 不实现深度模型、LambdaMART、在线学习或实时参数更新。
- 不引入 UserCF、Swing、Kafka、特征平台或独立模型服务。
- 不使用 `uid`、`item_id` 等高基数标识直接训练，避免记忆样本。
- 不以训练集准确率作为上线依据。

## 3. 总体架构

```text
Recommendation request
        |
        v
Four recall strategies
        |
        v
Recall evidence normalization
        |
        v
RankingFeaturesV1
        |--------------------------|
        v                          v
FixedWeightRanker        LogisticRegressionRanker
        |                          |
        |-----------fallback-------|
                    |
                    v
            Diversity reranking
                    |
                    v
       Exposure snapshot and explanation
                    |
                    v
     Offline export -> Python training -> JSON model
```

在线链路不得调用 Python。训练和模型导出属于离线操作；服务启动时加载模型，模型不可用时继续使用 `FixedWeightRanker`。

## 4. 先决条件

机器学习只能优化可信特征。开始训练前应完成以下修正：

1. 用户和商品 ANN 向量必须来自同一语义空间。活跃用户向量由最近 click/like 商品向量按行为强度和时间衰减聚合；冷启动用户使用 MiniLM 编码英文画像文本。
2. 当前扩展后的 `UserProfile.category_weights` 和预算区间作为冷启动类别、价格特征，行为偏好作为增量特征，不覆盖离线画像。
3. `popularity` 不再随机生成，改为带平滑和时间衰减的统计值，例如 `(clicks + 1) / (impressions + 10)`。
4. impression 不应永久屏蔽商品。永久 Bloom Filter 仅保留明确的长期过滤语义；近期曝光使用有上限或 TTL 的集合。
5. 所有训练和推理特征必须有固定范围，并拒绝 `NaN`、无穷值和缺失字段。
6. 现有探索位应从满足约束的候选中执行有界随机采样，并记录选择概率；固定选择“第一个可用候选”无法提供有效探索数据。

## 5. 召回证据

当前候选只保存召回来源集合和 `semantic_score`，会丢失 category、recent-item 和 popular 路的原始分数及路内排名。应为每个候选保存：

```text
RecallEvidence {
  source,
  raw_score,
  normalized_score,
  rank
}
```

每路分数不可直接相加，因为不同召回策略的分数分布不同。第一版同时使用以下两类特征：

- `normalized_score`：按召回策略约定映射到 `[0, 1]`。
- `reciprocal_rank = 1 / (60 + rank)`：提供对分数尺度不敏感的稳定信号。

逻辑回归通过 source 命中标记、标准化分数和倒数排名学习各召回路的实际贡献，无需再维护独立的手工召回加权公式。

## 6. RankingFeaturesV1

模型特征顺序必须固定并写入模型文件：

| 特征组 | 特征 | 范围 |
| --- | --- | --- |
| 基础排序 | semantic、category、popularity、price_affinity、feedback | `[0,1]`，feedback 为 `[-1,1]` |
| 召回命中 | semantic_ann、category_profile、recent_item_similarity、popular_fallback | `0/1` |
| 召回分数 | 四路 normalized score | `[0,1]` |
| 召回排名 | 四路 reciprocal rank | `[0,1/61]`，未命中为 `0` |
| 聚合信息 | source_count | `[1,4]` |

推荐位置不能作为在线排序特征，否则模型会学习旧排序器造成的位置偏差。它只用于训练评估、诊断和后续去偏。第一版也不加入用户、商品 ID 和自由文本，控制过拟合并保持解释稳定。

当前 `novelty = 1 - popularity`，两者完全线性相关，因此 `RankingFeaturesV1` 不训练 novelty。现有 novelty 仍可供固定权重策略兼容使用；只有在以后引入商品发布时间、首次曝光时间等独立信号后，才能在新的特征版本中重新加入。

## 7. 曝光日志与标签

现有事件不足以还原一次推荐时使用的特征。每个已展示商品需要保存不可变曝光快照：

```json
{
  "recommendation_id": "01J...",
  "uid": 1,
  "item_id": 42,
  "position": 3,
  "selection_probability": 0.1,
  "timestamp_ms": 1710000000000,
  "features_version": "ranking_features_v1",
  "features": { "semantic_score": 0.82 },
  "recall_evidence": [],
  "ranking_strategy": "fixed_weights",
  "model_version": "fixed-v1",
  "final_score": 0.71
}
```

`/recommend` 返回 `recommendation_id`；后续 impression、click、like、dismiss 必须携带该 ID，服务通过 `recommendation_id + item_id` 关联曝光与反馈。兼容期内允许旧客户端不传 ID，但此类事件不得进入训练集。

同一次曝光按观察窗口聚合成一个样本，建议规则如下：

| 最终行为 | 标签 | 样本权重 |
| --- | ---: | ---: |
| like | 1 | 3.0 |
| click | 1 | 1.0 |
| dismiss | 0 | 2.0 |
| 24 小时内无后续行为 | 0 | 0.25 |

如果同一曝光存在多个行为，使用 `like > click > dismiss > impression` 的最终优先级。低权重无行为曝光用于控制负样本数量，避免大量 impression 淹没正样本。

## 8. 离线训练

训练程序使用 Python `scikit-learn` 的带 L2 正则逻辑回归：

```text
p(positive | x) = sigmoid(intercept + sum(weight_i * x_i))
```

建议流程：

1. 从仅在调试模式启用的训练数据导出接口生成 JSONL；生产默认关闭该接口。
2. 校验特征版本、字段完整性、数值范围和曝光反馈关联关系。
3. 按时间切分训练集和验证集，禁止随机切分同一用户的连续行为。
4. 使用 `class_weight` 或上述样本权重缓解正负样本不平衡。
5. 在少量候选值中选择正则参数 `C`，以验证集 NDCG@10 为主要依据。
6. 导出模型、特征顺序、归一化参数、训练数据摘要和验证指标。

第一版只记录 `selection_probability` 并用于分群诊断，不立即引入 IPS，避免少量数据下的高方差。积累足够的随机探索样本后，再通过新的训练版本评估位置去偏。

第一版最低训练门槛建议为 500 次有效曝光和 50 个正样本。未达到门槛时训练程序应失败，不生成可部署模型。合成行为只能测试训练链路，不能作为模型效果结论。

## 9. 模型文件

JSON 模型采用显式版本和固定特征顺序。下例仅展示文件结构，正式模型必须列出完整的 `RankingFeaturesV1`，且四个数组长度严格一致：

```json
{
  "schema_version": 1,
  "model_type": "logistic_regression",
  "model_version": "lr-20260710-001",
  "features_version": "ranking_features_v1",
  "feature_order": ["semantic_score", "category_score"],
  "intercept": -0.42,
  "coefficients": [1.37, 0.81],
  "means": [0.51, 0.38],
  "stddevs": [0.19, 0.27],
  "training_summary": {
    "examples": 2400,
    "positives": 310,
    "validation_ndcg_at_10": 0.64
  }
}
```

服务加载时必须验证 schema、模型类型、特征版本、数组长度、重复特征、零标准差和非有限数值。加载失败应记录明确错误并回退固定权重；不得让 Pod 因可选排序模型损坏而停止提供推荐。

建议新增配置：

```text
MINI_RECSYS_RANKING_STRATEGY=logistic_regression
RANKING_MODEL_PATH=/models/ranking-model.json
```

## 10. 在线推理与解释

在线排序使用标准化特征计算 logit。排序只依赖 logit 的大小，响应可额外返回 sigmoid 概率用于调试，但不得描述为未经校准的真实点击率。

线性模型天然支持贡献解释：

```text
contribution_i = coefficient_i * standardized_feature_i
```

对每个商品按绝对值选择贡献最大的正向特征，映射为稳定英文 reason，例如：

- `strong_semantic_match`
- `category_preference`
- `recent_interest_match`
- `positive_feedback`
- `popular_item`

Debug API 应返回完整特征、各特征贡献、intercept、logit、模型版本和重排前后位置。普通 `/recommend` 只返回主要 reason、source、final score 和模型版本，避免暴露过多内部数据。

## 11. 离线评估与上线门槛

模型必须与同一验证集上的固定权重基线比较：

- 主指标：NDCG@10。
- 辅助指标：MRR、Recall@10、ROC-AUC、Log Loss。
- 体验约束：类别覆盖率、列表内多样性、popular fallback 占比、dismiss rate。

建议上线门槛：

- NDCG@10 相对固定权重提升至少 2%。
- Recall@10 不下降超过 1%。
- 类别覆盖率不下降超过 5%。
- 所有分群至少有可用结果，包括冷启动用户和无近期行为用户。
- Rust 对 JSON 模型的预测结果与 Python 测试向量误差小于 `1e-5`。

样本不足或指标未达到门槛时继续使用固定权重，不为了启用机器学习而降低验收标准。

## 12. 发布与回退

模型文件通过 ConfigMap 或只读模型卷挂载，不打进应用镜像。Deployment 先部署到测试环境，通过 readiness、模型加载日志和固定测试向量后再切换策略。

推荐发布顺序：

1. 仅记录曝光快照，在线仍使用固定权重。
2. 离线训练并在 Debug API 中执行 shadow scoring，不影响返回顺序。
3. 对比固定权重和逻辑回归的离线指标及 shadow 分布。
4. 设置 `MINI_RECSYS_RANKING_STRATEGY=logistic_regression` 启用新模型。
5. 出现模型加载失败、非有限分数或指标异常时立即回退 `fixed_weights`。

模型版本必须进入结构化日志和推荐响应，确保问题能够追溯到具体训练产物。

## 13. 实施阶段

### 阶段一：训练数据基础

- 完成先决特征修正。
- 扩展召回证据和 `RankingFeaturesV1`。
- 增加 recommendation ID、曝光快照和反馈关联。
- 增加训练数据质量检查与调试导出。

### 阶段二：离线训练与一致性

- 新增 Python 训练脚本和固定依赖文件。
- 实现时间切分、样本加权、正则参数选择和 JSON 导出。
- 增加 Python/Rust 共用测试向量，验证跨语言推理一致性。
- 输出固定权重与逻辑回归评估报告。

### 阶段三：在线推理与灰度

- 实现 `LogisticRegressionRanker` 和严格模型校验。
- 增加 shadow scoring、特征贡献解释和模型版本日志。
- 增加环境变量、K8s 模型挂载与固定权重回退。
- 达到评估门槛后切换在线排序策略。

## 14. 测试范围

- 单元测试：召回证据合并、特征顺序、归一化、逻辑回归计算、贡献解释、非法模型拒绝和固定权重回退。
- 数据测试：重复曝光、孤立反馈、超出范围特征、过期无行为曝光和标签优先级。
- 一致性测试：Python 与 Rust 对相同模型和输入得到相同 logit。
- Pipeline 测试：无模型时行为不变；有模型时排序变化且 reason 对应最大正贡献。
- API 测试：recommendation ID 能关联事件，旧客户端事件仍可保存但不进入训练集。
- K8s 测试：模型卷缺失或损坏时服务 ready 且使用固定权重；合法模型挂载后暴露正确版本。

## 15. 风险与控制

- **数据稀疏**：设置最低样本门槛，继续保留固定权重。
- **位置偏差**：位置只用于诊断，后续有随机探索数据后再引入 IPS 去偏。
- **流行度反馈循环**：使用平滑、时间衰减和现有探索位，监控热门召回占比。
- **训练与推理漂移**：模型文件携带特征版本和顺序，Rust 严格校验。
- **画像与行为重复计权**：离线画像、增量类别偏好和 item feedback 分开记录，通过模型学习贡献，避免在特征构造阶段重复相加。
- **解释失真**：reason 只能来自实际正向贡献，不使用与模型无关的固定阈值猜测。

本方案的关键不是让机器学习替代整条推荐链路，而是让固定、可验证的特征经过数据驱动的线性组合。这样既能提升排序准确性，也能继续满足 mini-recsys 对轻量、可解释和可部署的要求。
