# 推荐系统排序层与 C++ 服务端工程

## 1. 排序层在推荐系统中的位置

一个典型的工业推荐链路可以抽象为：

```text
全量物品库
    ↓
多路召回
    ↓
数千～数万候选
    ↓
粗排
    ↓
数百～上千候选
    ↓
精排
    ↓
数十～数百候选
    ↓
重排
    ↓
最终曝光列表
```

召回层回答的是：

> 哪些物品可能与当前用户相关？

排序层回答的是：

> 在这些候选物品中，哪些更值得排在前面？

因此，排序层的本质是：

\[
\text{候选物品} + \text{用户特征} + \text{上下文特征}
\longrightarrow
\text{预测分数}
\longrightarrow
\text{有序列表}
\]

排序层并不只是预测点击率。工业系统通常同时考虑点击、转化、时长、互动、负反馈、内容质量和商业价值。

例如：

\[
Score(u,i)=
w_1 pCTR(u,i)
+w_2 pCVR(u,i)
+w_3 E[\text{WatchTime}]
+w_4 P(\text{Like})
-w_5 P(\text{Negative})
\]

这里的最终目标通常是：

> 在延迟、算力和业务约束之下，最大化用户体验与平台长期收益。

---

## 2. 为什么排序层要拆成粗排和精排

排序层面临三个彼此冲突的目标：

1. 候选规模大；
2. 模型预测要准确；
3. 在线延迟必须足够低。

假设召回阶段返回 10,000 个候选，精排模型对单个候选平均消耗 0.1ms，则逐个执行需要：

\[
10000 \times 0.1ms = 1000ms
\]

这无法满足在线推荐服务常见的几十毫秒到几百毫秒预算。

因此工业系统采用漏斗式架构：

```text
大量候选
  ↓ 使用便宜模型快速筛选
粗排
  ↓ 剩余少量候选
精排
  ↓ 使用复杂模型精细预测
```

---

## 3. 粗排阶段

### 3.1 核心目标

粗排的主要目标不是完成最终排序，而是：

> 以较低计算成本淘汰明显不合适的候选，并尽量保留精排真正需要的优质候选。

粗排通常处理数千到数万个候选，将其缩减到数百或一千左右。

粗排最重要的三个关键词是：

```text
低成本
高吞吐
少误杀
```

### 3.2 常见模型

粗排模型通常较轻量：

- 逻辑回归；
- GBDT；
- 小型 DNN；
- 双塔内积模型；
- 蒸馏模型；
- 简化版 Wide & Deep；
- 向量相似度加少量统计特征。

一个典型粗排分数可以写成：

\[
Score_{\text{pre}}(u,i)
=
\alpha \cdot \langle e_u,e_i\rangle
+\beta \cdot popularity(i)
+\gamma \cdot freshness(i)
\]

### 3.3 粗排更关注的指标

粗排不能只看 AUC，还需要关注它是否把精排的优质候选保留下来。

#### Top-K 保留率

假设用完整精排模型在原始候选中得到理想 Top-K：

\[
Retention@K
=
\frac{
|\text{粗排保留集合}\cap\text{精排理想 Top-K}|
}{K}
\]

粗排最重要的风险是：

> 优质候选一旦被粗排过滤，后续精排再准确也无法恢复。

因此粗排通常比精排更重视候选保留能力和单位计算成本。

---

## 4. 精排阶段

### 4.1 核心目标

精排面对的候选更少，因此可以使用更丰富的特征和更复杂的模型。

它的主要目标是：

> 更准确地建模用户、物品和当前场景之间的交互关系，并输出多种用户行为的预测值。

### 4.2 常见输入特征

#### 用户特征

- 用户 ID；
- 年龄、地域等画像；
- 长期兴趣；
- 短期兴趣；
- 最近点击、观看、购买序列；
- 用户活跃度；
- 用户消费能力。

#### 物品特征

- 物品 ID；
- 类别、作者、商家；
- 发布时间；
- 热度；
- 质量分；
- 文本、图片、视频 Embedding；
- 价格或商业价值。

#### 上下文特征

- 当前时间；
- 设备类型；
- 网络环境；
- 当前页面；
- 推荐位置；
- 本次会话行为；
- 当前搜索词或入口。

#### 交叉特征

- 用户是否偏好该类别；
- 用户近期行为与物品的相似程度；
- 用户价格偏好与物品价格的匹配度；
- 用户是否消费过该作者或品牌；
- 用户当前会话意图与物品主题的关系。

### 4.3 常见模型

- Wide & Deep；
- DeepFM；
- DCN；
- DIN；
- DIEN；
- Transformer 序列模型；
- MMoE；
- PLE；
- 多任务学习模型。

精排模型通常输出多个目标：

\[
\hat y =
[
pCTR,\,
pCVR,\,
P(\text{Like}),\,
E(\text{WatchTime}),\,
P(\text{Negative})
]
\]

然后由服务端或模型内部完成多目标融合。

---

## 5. 排序层的主要评价指标

排序层指标应分成四类：

```text
离线模型指标
粗排专项指标
线上业务指标
工程性能指标
```

### 5.1 离线模型指标

#### AUC

\[
AUC=P(score_{positive}>score_{negative})
\]

用于衡量模型区分正负样本的能力。

#### GAUC

按照用户或请求分组计算 AUC，再加权平均：

\[
GAUC=
\frac{\sum_u w_u AUC_u}{\sum_u w_u}
\]

GAUC 更贴近“同一用户候选集合内部排序”的推荐场景。

#### LogLoss

\[
LogLoss=
-\frac{1}{N}
\sum_{i=1}^{N}
[y_i\log p_i+(1-y_i)\log(1-p_i)]
\]

它不仅关心顺序，还关心预测概率是否可靠。

#### NDCG@K

\[
DCG@K=
\sum_{i=1}^{K}
\frac{2^{rel_i}-1}{\log_2(i+1)}
\]

\[
NDCG@K=\frac{DCG@K}{IDCG@K}
\]

NDCG 更重视高相关物品是否出现在列表前部。

#### Precision@K 与 Recall@K

\[
Precision@K=
\frac{\text{Top-K 中相关物品数}}{K}
\]

\[
Recall@K=
\frac{\text{Top-K 中相关物品数}}{\text{所有相关物品数}}
\]

#### 概率校准

当模型输出 0.1 的点击概率时，大量样本的真实点击率应尽量接近 0.1。

校准对于广告出价、多目标融合、阈值过滤和收益估计非常重要。

### 5.2 粗排专项指标

- 精排 Top-K 保留率；
- 粗排与精排 Top-K 重合率；
- Pairwise 排序一致率；
- Spearman 或 Kendall 排序相关性；
- 单候选推理成本；
- 单请求 P95/P99 延迟；
- 候选裁剪率。

### 5.3 线上业务指标

- CTR；
- CVR；
- GMV；
- 观看时长；
- 完播率；
- 点赞率；
- 收藏率；
- 分享率；
- 留存率；
- 负反馈率；
- 作者覆盖率；
- 内容多样性；
- 长尾物品曝光率。

工业实验通常采用：

```text
核心优化指标 + 护栏指标
```

例如：

```text
核心指标：人均观看时长提升
护栏指标：
- 负反馈率不得显著上升
- 次日留存不得下降
- P99 延迟不得超过阈值
- 内容多样性不得明显降低
```

### 5.4 工程性能指标

- 平均延迟；
- P95/P99 延迟；
- QPS；
- 超时率；
- 降级率；
- 特征缺失率；
- 模型推理失败率；
- CPU/GPU 利用率；
- Batch 大小分布；
- 缓存命中率；
- 模型分数分布；
- 输入特征漂移。

---

# 6. 排序层与召回层的 C++ 技术差异

召回层和排序层共享很多基础设施：

- RPC；
- 异步 I/O；
- 线程池；
- 缓存；
- KV 存储；
- 限流；
- 熔断；
- 配置中心；
- 服务发现；
- 监控和日志。

因此从表面上看二者确实很相似。

真正差异在于各自围绕的核心数据对象不同：

| 层级 | 核心对象 | 核心计算 |
|---|---|---|
| 召回层 | 索引、向量、候选 ID | 查询、合并、过滤 |
| 排序层 | 特征、Tensor、模型、预测分数 | 特征组装、推理、Top-K |

可以压缩为：

```text
召回层：高性能检索系统
排序层：高性能特征与在线推理系统
```

---

# 7. 排序服务的典型在线链路

```text
接收召回候选
    ↓
读取用户特征
    ↓
批量读取物品特征
    ↓
读取实时上下文与行为序列
    ↓
特征拼接、离散化、归一化
    ↓
构造连续 Tensor
    ↓
批量模型推理
    ↓
多目标分数融合
    ↓
Top-K 选择
    ↓
返回重排层
```

一个典型排序服务可以拆成以下模块：

```text
RankService
├─ RequestContext
├─ FeatureFetcher
├─ FeatureAssembler
├─ TensorBuilder
├─ ModelManager
├─ InferenceExecutor
├─ ScoreFusion
├─ TopKSelector
├─ DegradeManager
└─ MetricsCollector
```

---

# 8. 在线特征获取

## 8.1 为什么特征获取是排序服务的关键

召回阶段通常只需要：

- 用户向量；
- 物品向量；
- 用户历史集合；
- 少量过滤特征。

精排阶段可能需要数百到上千个特征，且特征来自不同存储：

```text
用户画像 KV
物品静态特征库
实时特征服务
行为序列服务
本地缓存
请求上下文
```

对于 800 个候选，如果逐个访问特征服务，会形成严重的 N+1 查询问题。

错误示例：

```cpp
for (ItemId item_id : candidates) {
    ItemFeature feature = item_feature_service.get(item_id);
}
```

更合理的接口是：

```cpp
std::vector<ItemFeature> features =
    item_feature_service.batch_get(candidates);
```

## 8.2 并行获取不同来源的特征

不同来源之间没有依赖时，应并发执行：

```cpp
auto user_future = get_user_features_async(user_id);
auto item_future = batch_get_item_features_async(item_ids);
auto realtime_future = get_realtime_features_async(user_id);
auto sequence_future = get_behavior_sequence_async(user_id);

auto user_features = user_future.get();
auto item_features = item_future.get();
auto realtime_features = realtime_future.get();
auto sequence_features = sequence_future.get();
```

实际工程中可以使用：

- RPC 框架的异步接口；
- C++20 协程；
- Future/Promise；
- 任务图调度器；
- brpc ParallelChannel 一类的并发机制。

## 8.3 多级缓存

排序特征通常适合多级缓存：

```text
线程本地缓存
    ↓
进程内 LRU
    ↓
分布式 KV / Redis
    ↓
后端特征服务
```

物品静态特征更新频率较低，可以重点使用本地缓存；用户实时特征时效性较高，更依赖近线或实时存储。

---

# 9. 特征拼接与 Tensor 构造

## 9.1 连续内存布局

不建议使用：

```cpp
std::vector<std::vector<float>>
```

因为它会带来多次堆分配、内存碎片和较差的 Cache 局部性。

更合理的布局是：

```cpp
class FeatureMatrix {
public:
    FeatureMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    float* row(std::size_t index) {
        return data_.data() + index * cols_;
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;
};
```

最终模型输入通常是：

\[
[\text{candidate\_count},\text{feature\_dim}]
\]

例如：

```text
800 个候选 × 512 维特征
```

## 9.2 稀疏与稠密特征

排序模型输入通常同时包含：

- 数值型稠密特征；
- 离散 ID 特征；
- 多值稀疏特征；
- 变长序列特征；
- Embedding；
- Mask 和序列长度。

C++ 服务端需要处理：

- 缺失值填充；
- 归一化；
- 离散化；
- 字典映射；
- 哈希桶；
- Padding；
- Mask 构造；
- 数据类型转换；
- Tensor shape 校验。

## 9.3 对象池与 Arena

排序服务每次请求都可能构造大量中间对象。

为了降低堆分配和尾延迟抖动，可以使用：

- 对象池；
- Arena；
- 线程本地缓存；
- Tensor Buffer 复用；
- `std::pmr`；
- 固定容量容器；
- 请求级内存资源。

---

# 10. 批量推理

## 10.1 请求内 Batch

不能为每个候选单独调用模型：

```cpp
for (const auto& candidate : candidates) {
    float score = model.predict(candidate);
}
```

更合理的是：

```text
全部候选特征
    ↓
一个二维 Tensor
    ↓
一次批量推理
    ↓
得到全部候选分数
```

伪代码：

```cpp
FeatureMatrix input(candidates.size(), feature_dim);

for (std::size_t i = 0; i < candidates.size(); ++i) {
    fill_feature_row(
        input.row(i),
        user_features,
        item_features[i],
        context_features,
        sequence_features
    );
}

PredictionBatch predictions = model.run(input);
```

Batch 能减少：

- 推理框架调用次数；
- Tensor 创建次数；
- GPU Kernel Launch；
- 调度开销；
- 内存分配；
- CPU Cache Miss。

## 10.2 动态 Batch

为了提升 GPU 或 CPU 推理吞吐，还可以将多个请求合并：

```text
请求 A：500 条样本
请求 B：300 条样本
请求 C：700 条样本
        ↓
合并推理
```

动态 Batch 一般设置两个触发条件：

```text
累计样本数量达到阈值
或者
最早请求等待时间达到阈值
```

例如：

```text
max_batch_size = 2048
max_wait_time = 1ms
```

简化结构：

```cpp
struct RankingTask {
    FeatureBatch features;
    std::chrono::steady_clock::time_point deadline;
    std::promise<PredictionBatch> promise;
};

class DynamicBatcher {
public:
    std::future<PredictionBatch> submit(FeatureBatch features);

private:
    void worker_loop();

    BlockingQueue<RankingTask> queue_;
    std::shared_ptr<Model> model_;
};
```

动态 Batch 的核心矛盾是：

```text
更大的 Batch → 更高吞吐
等待更久     → 更高延迟
```

因此需要结合 SLA 做权衡。

---

# 11. 模型推理框架

常见在线推理方案包括：

- ONNX Runtime；
- TensorRT；
- TorchScript；
- TensorFlow Serving；
- XGBoost 或 LightGBM 原生推理库；
- 自研算子和推理引擎。

C++ 服务端通常负责：

- 加载模型；
- 创建 Session；
- 管理线程数；
- 构造输入 Tensor；
- 执行推理；
- 解析输出；
- 管理 CPU/GPU 内存；
- 模型预热；
- 模型热更新；
- 推理失败降级。

使用 ONNX Runtime 时，一个常见封装形式如下：

```cpp
class RankModel {
public:
    virtual ~RankModel() = default;

    virtual PredictionBatch predict(
        std::span<const float> input,
        std::size_t batch_size,
        std::size_t feature_dim
    ) const = 0;

    virtual void warmup() = 0;
    virtual std::string_view version() const = 0;
};
```

业务代码不应直接耦合具体推理引擎，方便后续在 ONNX Runtime、TensorRT 或轻量模型之间切换。

---

# 12. CPU 和 GPU 推理的选择

## CPU 更适合

- 模型较小；
- 请求 Batch 不稳定；
- 低延迟优先；
- 单机候选量较小；
- 部署成本要求较低；
- 需要快速扩缩容。

重点优化方向：

- SIMD；
- 线程绑定；
- NUMA；
- 连续内存；
- 减少分配；
- 合理配置推理线程数；
- 避免线程池嵌套。

## GPU 更适合

- 模型较大；
- 候选 Batch 足够大；
- QPS 较稳定；
- Attention 或 Transformer 算子较多；
- 吞吐量比单请求极限延迟更重要。

重点优化方向：

- 动态 Batch；
- Host/Device 内存拷贝；
- Pinned Memory；
- CUDA Stream；
- 模型并发；
- GPU 利用率；
- OOM 和过载保护。

---

# 13. Top-K 选择

模型推理后，每个候选会得到一个或多个分数。

完整排序可以使用：

```cpp
std::sort(
    items.begin(),
    items.end(),
    [](const Item& lhs, const Item& rhs) {
        return lhs.score > rhs.score;
    }
);
```

只需要 Top-K 时可以使用：

```cpp
std::nth_element(
    items.begin(),
    items.begin() + k,
    items.end(),
    [](const Item& lhs, const Item& rhs) {
        return lhs.score > rhs.score;
    }
);

items.resize(k);
```

如果还要求 Top-K 内部有序，再对前 K 个排序即可：

```cpp
std::sort(items.begin(), items.end(), compare);
```

其复杂度通常优于对所有候选完整排序。

不过在精排服务中，真正的主要开销通常是：

```text
特征读取 > 特征拼接 > 模型推理 > 排序
```

---

# 14. 多目标分数融合

模型可能输出：

```cpp
struct Prediction {
    float pctr;
    float pcvr;
    float watch_time;
    float like_prob;
    float negative_prob;
};
```

服务端融合可以写成：

```cpp
struct RankWeights {
    float ctr;
    float cvr;
    float watch_time;
    float like;
    float negative;
};

float calculate_score(
    const Prediction& pred,
    const RankWeights& weights
) {
    return weights.ctr * pred.pctr
         + weights.cvr * pred.pcvr
         + weights.watch_time * pred.watch_time
         + weights.like * pred.like_prob
         - weights.negative * pred.negative_prob;
}
```

工业实现通常还需要：

- 分数归一化；
- 截断异常值；
- 按场景加载不同参数；
- 按实验组使用不同公式；
- 在线动态配置；
- 版本管理；
- 参数回滚。

权重不应直接硬编码在程序中，而应由配置中心或实验平台管理。

---

# 15. DAG 执行引擎

复杂排序链路通常不是纯线性的：

```text
GetUserFeature ──┐
GetItemFeature ──┼─ BuildTensor ─ Predict ─ ScoreFusion ─ TopK
GetSequence ─────┤
GetContext ──────┘
```

某些节点可以并行，某些节点存在依赖关系。

因此工业排序系统经常将链路抽象成 DAG：

- 节点表示特征读取、特征转换、模型推理、分数计算；
- 边表示数据依赖；
- 调度器负责并发执行和失败传播。

C++ DAG 引擎通常需要处理：

- 依赖计数；
- 线程池调度；
- 超时传播；
- 节点跳过；
- 结果缓存；
- 失败降级；
- 埋点监控；
- 请求级资源管理。

---

# 16. 训练与在线服务一致性

这是排序系统中非常重要的问题，通常称为：

```text
Training-Serving Skew
训练与服务偏差
```

例如训练时：

```python
age_bucket = min(age // 10, 8)
```

线上 C++ 却写成：

```cpp
int age_bucket = age / 5;
```

即使模型离线指标很好，线上输入分布也已经发生变化。

需要保证以下逻辑一致：

- 特征定义；
- 缺失值；
- 归一化；
- 离散化；
- 哈希方式；
- 词表版本；
- 时间窗口；
- 数据类型；
- 序列截断；
- Padding 与 Mask。

常见工程手段：

1. 离线和在线共享特征配置；
2. 自动生成 C++ 特征代码；
3. 特征版本化；
4. 样本回放；
5. 在线日志与离线样本逐字段对比；
6. 特征分布监控；
7. 模型输入 Schema 校验。

---

# 17. 模型热更新

排序模型迭代频繁，服务不能每次更新模型都整体重启。

一个典型流程是：

```text
模型平台发布新版本
    ↓
排序服务下载模型
    ↓
校验文件完整性
    ↓
创建新推理 Session
    ↓
模型预热
    ↓
原子切换
    ↓
旧模型等待存量请求完成后释放
```

示例：

```cpp
class ModelManager {
public:
    std::shared_ptr<const RankModel> current() const {
        return std::atomic_load(&model_);
    }

    bool update(std::shared_ptr<RankModel> next) {
        if (!next) {
            return false;
        }

        next->warmup();
        std::atomic_store(
            &model_,
            std::shared_ptr<const RankModel>(std::move(next))
        );
        return true;
    }

private:
    std::shared_ptr<const RankModel> model_;
};
```

每个请求获取一个稳定的模型快照：

```cpp
auto model = model_manager.current();
auto result = model->predict(input, batch_size, feature_dim);
```

模型切换后，正在执行的旧请求仍然持有旧模型，不会产生悬垂引用。

模型更新还需要考虑：

- 模型文件校验；
- 模型版本；
- 灰度流量；
- 预热；
- 内存峰值；
- 回滚；
- 推理框架线程安全；
- 新旧模型输出兼容性。

---

# 18. 超时与降级

召回层某一路失败，可以放弃该路结果；排序层精排失败时仍必须返回可用顺序。

常见降级链路：

```text
复杂精排模型
    ↓ 失败或预算不足
轻量精排模型
    ↓ 失败
使用粗排分数
    ↓ 失败
按召回分数或热度排序
```

还可以进行特征级降级：

- 实时特征超时，使用近线特征；
- 行为序列超时，使用空序列或缓存序列；
- 某类特征缺失，填充默认值；
- 剩余时间不足，裁剪候选数量；
- GPU 过载，切换 CPU 轻量模型。

伪代码：

```cpp
RankResult rank(
    const RankRequest& request,
    const RequestContext& context
) {
    if (context.remaining() > std::chrono::milliseconds(15)) {
        auto result = complex_model_predict(request);
        if (result.ok()) {
            return result.value();
        }
    }

    if (context.remaining() > std::chrono::milliseconds(5)) {
        auto result = lightweight_model_predict(request);
        if (result.ok()) {
            return result.value();
        }
    }

    return sort_by_recall_score(request.candidates);
}
```

---

# 19. 延迟预算管理

总请求预算不能简单平均分给每个模块。

例如总预算为 100ms：

```text
网关：5ms
召回：20ms
粗排：10ms
精排：30ms
重排：10ms
其他：10ms
预留：15ms
```

精排内部继续拆分：

```text
特征获取：10ms
特征拼接：3ms
模型推理：12ms
分数融合：2ms
Top-K：1ms
预留：2ms
```

请求上下文中应传递绝对截止时间：

```cpp
struct RequestContext {
    std::chrono::steady_clock::time_point deadline;

    std::chrono::milliseconds remaining() const {
        const auto now = std::chrono::steady_clock::now();

        if (now >= deadline) {
            return std::chrono::milliseconds{0};
        }

        return std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - now
        );
    }
};
```

下游 RPC 和模型执行应根据剩余预算设置超时，而不是为每个阶段写死相同的超时时间。

---

# 20. 排序服务的可观测性

排序服务不仅要监控普通 RPC 指标，还要监控模型和特征。

## 服务指标

- 请求量；
- 成功率；
- 平均延迟；
- P95/P99；
- 超时率；
- 降级率；
- 线程池队列长度；
- 内存和 CPU；
- GPU 利用率。

## 特征指标

- 各类特征读取耗时；
- 特征缺失率；
- 默认值填充率；
- 缓存命中率；
- 特征分布；
- 序列长度分布；
- Schema 不匹配次数。

## 模型指标

- 当前模型版本；
- 推理耗时；
- Batch 大小；
- 推理失败率；
- 输出分数均值和分位数；
- pCTR、pCVR 分布；
- 新旧模型分数差异；
- 输入数据漂移；
- 模型切换和回滚次数。

例如线上 pCTR 均值突然从 0.08 变成 0.35，可能意味着：

- 特征错位；
- 归一化错误；
- 模型版本异常；
- Tensor shape 错误；
- 流量结构变化。

---

# 21. 一个简化的 C++ 排序服务骨架

```cpp
#include <algorithm>
#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using UserId = std::uint64_t;
using ItemId = std::uint64_t;

struct Candidate {
    ItemId item_id{};
    float recall_score{};
    float rank_score{};
};

struct UserFeatures {};
struct ItemFeatures {};
struct ContextFeatures {};
struct SequenceFeatures {};

struct Prediction {
    float pctr{};
    float pcvr{};
    float negative_prob{};
};

struct RankRequest {
    UserId user_id{};
    std::vector<Candidate> candidates;
    ContextFeatures context;
};

struct RankResponse {
    std::vector<Candidate> items;
    std::string model_version;
    bool degraded{};
};

class RankModel {
public:
    virtual ~RankModel() = default;

    virtual std::vector<Prediction> predict(
        std::span<const float> tensor,
        std::size_t batch_size,
        std::size_t feature_dim
    ) const = 0;

    virtual std::string version() const = 0;
};

class FeatureService {
public:
    UserFeatures get_user_features(UserId user_id) const;
    std::vector<ItemFeatures> batch_get_item_features(
        std::span<const ItemId> item_ids
    ) const;
    SequenceFeatures get_sequence_features(UserId user_id) const;
};

class RankService {
public:
    RankResponse rank(const RankRequest& request) const {
        std::vector<ItemId> item_ids;
        item_ids.reserve(request.candidates.size());

        for (const auto& candidate : request.candidates) {
            item_ids.push_back(candidate.item_id);
        }

        const UserFeatures user_features =
            feature_service_.get_user_features(request.user_id);

        const std::vector<ItemFeatures> item_features =
            feature_service_.batch_get_item_features(item_ids);

        const SequenceFeatures sequence_features =
            feature_service_.get_sequence_features(request.user_id);

        constexpr std::size_t feature_dim = 512;
        std::vector<float> tensor(
            request.candidates.size() * feature_dim
        );

        build_tensor(
            tensor,
            feature_dim,
            user_features,
            item_features,
            request.context,
            sequence_features
        );

        auto model = current_model_;
        const auto predictions = model->predict(
            tensor,
            request.candidates.size(),
            feature_dim
        );

        std::vector<Candidate> ranked = request.candidates;

        for (std::size_t i = 0; i < ranked.size(); ++i) {
            ranked[i].rank_score =
                predictions[i].pctr
                + 0.5F * predictions[i].pcvr
                - 0.3F * predictions[i].negative_prob;
        }

        constexpr std::size_t top_k = 100;

        if (ranked.size() > top_k) {
            std::nth_element(
                ranked.begin(),
                ranked.begin() + top_k,
                ranked.end(),
                [](const Candidate& lhs, const Candidate& rhs) {
                    return lhs.rank_score > rhs.rank_score;
                }
            );

            ranked.resize(top_k);
        }

        std::sort(
            ranked.begin(),
            ranked.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
                return lhs.rank_score > rhs.rank_score;
            }
        );

        return RankResponse{
            .items = std::move(ranked),
            .model_version = model->version(),
            .degraded = false,
        };
    }

private:
    static void build_tensor(
        std::vector<float>& tensor,
        std::size_t feature_dim,
        const UserFeatures& user_features,
        const std::vector<ItemFeatures>& item_features,
        const ContextFeatures& context_features,
        const SequenceFeatures& sequence_features
    );

    FeatureService feature_service_;
    std::shared_ptr<const RankModel> current_model_;
};
```

这个骨架省略了很多工业细节，但已经体现排序服务的主线：

```text
批量特征读取
→ Tensor 构造
→ 批量推理
→ 多目标融合
→ Top-K
```

---

# 22. C++ 工程师需要重点掌握的能力

## 基础服务端能力

- Linux 网络编程；
- RPC；
- 异步 I/O；
- 线程池；
- C++20 协程；
- 缓存和 KV；
- 限流、熔断、降级；
- 配置中心；
- 可观测性；
- 服务治理。

## 排序层专项能力

- Feature Store 基本概念；
- 批量特征读取；
- 特征 Schema；
- 稠密和稀疏特征处理；
- Tensor 内存布局；
- ONNX Runtime；
- TensorRT 基本原理；
- 请求内 Batch；
- 动态 Batch；
- CPU/GPU 推理；
- 模型热更新；
- 训练与服务一致性；
- 模型和特征监控；
- 多目标分数融合。

## 推荐的学习顺序

```text
第一阶段：
理解排序层的数据流、粗排、精排和指标

第二阶段：
使用 ONNX Runtime 完成一个批量推理服务

第三阶段：
增加 Feature Service、Batch 查询和本地缓存

第四阶段：
实现模型热更新、超时降级和监控

第五阶段：
学习动态 Batch、CPU/GPU 优化和 DAG 调度
```

---

# 23. 最终总结

排序层的核心流程可以压缩为：

```text
候选 ID
  ↓
批量获取特征
  ↓
构造 Tensor
  ↓
批量模型推理
  ↓
多目标融合
  ↓
Top-K
```

粗排关注：

```text
低成本、高吞吐、尽量不误杀
```

精排关注：

```text
丰富特征、复杂交互、准确预测、多目标优化
```

从 C++ 服务端角度，排序层最有代表性的能力是：

```text
批量特征获取
连续内存和 Tensor 构造
批量推理
动态 Batch
CPU/GPU 调度
模型热更新
超时降级
训练与服务一致性
特征和模型可观测性
```

最值得记住的一句话是：

> 召回服务的核心是高性能检索，排序服务的核心是高性能在线特征与模型推理。
