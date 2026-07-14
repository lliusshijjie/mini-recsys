# 推荐系统多目标学习与 MMoE 架构

## 1. 为什么推荐系统需要多目标学习

真实推荐系统通常不能只优化点击率。

以短视频推荐为例，平台可能同时关心：

- 是否点击；
- 是否完整播放；
- 观看时长；
- 是否点赞；
- 是否评论；
- 是否关注；
- 是否分享；
- 是否快速划走；
- 是否产生负反馈。

以电商推荐为例，可能同时预测：

- CTR；
- 加购率；
- 收藏率；
- CVR；
- 客单价；
- 退款风险；
- GMV。

因此精排模型经常输出多个目标：

\[
\hat y=
[
pCTR,\,
pCVR,\,
P(\text{Like}),\,
E(\text{WatchTime}),\,
P(\text{Negative})
]
\]

最终排序分数可以由这些目标组合而成：

\[
Score=
w_1pCTR
+w_2pCVR
+w_3E(\text{WatchTime})
+w_4P(\text{Like})
-w_5P(\text{Negative})
\]

---

## 2. 什么是多目标学习

多目标学习，也称为多任务学习，核心思想是：

> 使用同一个模型同时学习多个相关任务，使任务之间能够共享数据和表示。

总损失通常写成：

\[
L=
\sum_{k=1}^{K}\lambda_kL_k
\]

其中：

- \(K\) 是任务数量；
- \(L_k\) 是第 \(k\) 个任务的损失；
- \(\lambda_k\) 是任务权重。

例如：

\[
L=
\lambda_{ctr}L_{ctr}
+\lambda_{cvr}L_{cvr}
+\lambda_{like}L_{like}
\]

多任务学习的优势包括：

1. 共享底层特征，减少重复计算；
2. 数据量较大的任务可以帮助数据稀疏任务；
3. 一个模型同时输出多个业务目标；
4. 在线服务只需要执行一次主干推理。

但它也会带来问题：

> 不同任务的目标、样本分布和梯度方向可能存在冲突。

这通常称为负迁移。

---

## 3. Shared-Bottom 架构

最简单的多任务模型是 Shared-Bottom：

```text
输入特征
   ↓
共享底层网络
   ├─ CTR Tower → pCTR
   ├─ CVR Tower → pCVR
   └─ Like Tower → pLike
```

公式为：

\[
h=SharedBottom(x)
\]

\[
\hat y_k=Tower_k(h)
\]

它的优点是简单、计算开销低。

但所有任务被迫共享同一个底层表示。

例如：

- CTR 更关心标题、图片和兴趣匹配；
- CVR 更关心价格、购买意愿和商品质量；
- 负反馈更关心低质内容和用户厌恶信号。

如果这些任务强制使用完全相同的共享表示，容易发生负迁移。

---

# 4. MMoE 架构

MMoE 全称：

```text
Multi-gate Mixture-of-Experts
多门控混合专家模型
```

MMoE 的核心思想是：

> 多个任务共享一组 Expert，但每个任务通过独立 Gate 选择不同的专家组合。

结构如下：

```text
输入特征 x
   ↓
多个 Expert
   ├─ Expert 1
   ├─ Expert 2
   ├─ Expert 3
   └─ Expert 4
   ↓
每个任务独立 Gate
   ├─ CTR Gate
   ├─ CVR Gate
   └─ Like Gate
   ↓
每个任务独立 Tower
   ├─ CTR Tower
   ├─ CVR Tower
   └─ Like Tower
```

---

## 5. MMoE 的数学形式

假设共有 \(n\) 个专家。

第 \(i\) 个专家：

\[
E_i(x)=MLP_i(x)
\]

任务 \(k\) 的 Gate：

\[
g_k(x)=Softmax(W_kx+b_k)
\]

Gate 输出：

\[
g_k(x)=
[g_{k,1},g_{k,2},...,g_{k,n}]
\]

满足：

\[
\sum_{i=1}^{n}g_{k,i}=1
\]

任务 \(k\) 的专家融合表示：

\[
h_k=
\sum_{i=1}^{n}
g_{k,i}(x)E_i(x)
\]

然后送入该任务的 Tower：

\[
\hat y_k=Tower_k(h_k)
\]

例如：

```text
CTR Gate：
Expert 1 = 0.60
Expert 2 = 0.25
Expert 3 = 0.10
Expert 4 = 0.05

CVR Gate：
Expert 1 = 0.10
Expert 2 = 0.15
Expert 3 = 0.20
Expert 4 = 0.55
```

这表示 CTR 和 CVR 使用相同的专家池，但选择的专家组合不同。

---

## 6. 如何理解 Expert、Gate 和 Tower

### Expert

Expert 是多个并行的小型神经网络，用于学习不同的特征表示。

理想情况下，不同专家可能逐渐形成分工：

```text
Expert 1：长期兴趣
Expert 2：短期意图
Expert 3：物品质量
Expert 4：价格和转化
```

但这种语义不是人工指定的，而是训练过程中自动形成的。

### Gate

Gate 相当于任务级路由器。

它决定：

> 当前任务、当前样本应该以什么比例使用各个 Expert。

因此同一个用户和物品样本，在 CTR 和 CVR 任务中可以获得不同的专家组合。

### Tower

Tower 是每个任务自己的预测网络。

经过 Gate 融合之后，任务专属 Tower 完成最终输出。

---

# 7. MMoE 相比 Shared-Bottom 的优势

Shared-Bottom：

```text
所有任务共享完全相同的底层表示
```

MMoE：

```text
所有任务共享专家池
但每个任务可以选择不同的专家组合
```

因此 MMoE 能够：

1. 保留任务之间的共享；
2. 减少不相关任务之间的相互干扰；
3. 为不同任务提供不同底层表示；
4. 缓解负迁移。

它特别适合：

- 任务之间存在一定相关性；
- 但任务目标又不完全相同；
- 希望共享主干计算；
- 同时保留任务差异。

---

# 8. MMoE 的极化现象

MMoE 的 Gate 通过 Softmax 输出专家权重。

理想情况：

```text
Expert 1 = 0.35
Expert 2 = 0.30
Expert 3 = 0.20
Expert 4 = 0.15
```

极化后可能变成：

```text
Expert 1 = 0.98
Expert 2 = 0.01
Expert 3 = 0.005
Expert 4 = 0.005
```

这表示任务几乎只依赖一个专家。

这种现象通常被称为：

- Gate 极化；
- Expert Collapse；
- 专家坍缩；
- Expert Monopolization；
- 负载不均衡。

---

## 9. 极化为什么会发生

### 9.1 Softmax 正反馈

训练初期，某个 Expert 可能偶然表现更好：

```text
Gate 给它更高权重
    ↓
它获得更多梯度
    ↓
它学习得更快
    ↓
Gate 更倾向于选择它
```

形成“强者愈强”的正反馈。

### 9.2 任务样本量不平衡

例如：

```text
CTR 样本：10 亿
CVR 样本：1000 万
```

CTR 的梯度可能主导共享专家，使专家更偏向 CTR。

### 9.3 任务损失尺度不一致

总损失：

\[
L=
\lambda_{ctr}L_{ctr}
+\lambda_{cvr}L_{cvr}
\]

如果某个任务的损失或梯度明显更大，它会主导 Expert 的训练。

### 9.4 专家初始化和结构过于相似

所有 Expert 使用相同结构、相同输入和相近初始化时，可能学习出相似表示。

此时 Gate 没有必要同时选择多个类似专家，容易固定选择当前略好的一个。

---

# 10. 极化带来的问题

## 10.1 部分专家得不到充分训练

如果大多数任务都依赖 Expert 1：

```text
Expert 1：获得大量梯度
Expert 2：梯度很少
Expert 3：几乎不更新
Expert 4：几乎不更新
```

模型虽然设计了多个专家，实际容量却没有被利用。

## 10.2 专家无法形成差异化分工

多个专家本来应该学习不同模式，但极化会导致：

- 一个专家垄断大部分流量；
- 其他专家退化；
- 专家表示高度相似；
- 模型容量浪费。

## 10.3 任务共享关系失衡

一种极端是所有任务都依赖同一个专家：

```text
MMoE 退化为 Shared-Bottom
```

另一种极端是每个任务完全使用不同专家：

```text
MMoE 退化为多个独立模型
```

理想情况应该是：

> 相关任务适当共享，冲突任务适当分离。

## 10.4 泛化能力下降

Gate 过早固定后，模型容易陷入局部最优。

对于新用户、长尾物品和新场景，Gate 可能无法灵活选择其他专家。

---

# 11. 缓解极化的方法

## 11.1 Gate 熵正则

Gate 熵为：

\[
H(g_k)=
-\sum_i g_{k,i}\log g_{k,i}
\]

Gate 权重过于集中时，熵会变低。

可以在损失中加入熵正则：

\[
L_{total}
=
L_{task}
-\lambda H(g)
\]

鼓励 Gate 在训练早期不要过快变成 one-hot。

但正则不能过强，否则所有权重长期接近平均，Gate 将失去选择能力。

## 11.2 负载均衡损失

统计一个 Batch 内各 Expert 获得的平均权重：

\[
p_i=
\frac{1}{B}
\sum_{b=1}^{B}
g_i(x_b)
\]

希望各专家的负载不要差距过大。

可以增加：

\[
L_{balance}
=
\sum_i
\left(p_i-\frac{1}{n}\right)^2
\]

总损失：

\[
L=
L_{task}
+\lambda L_{balance}
\]

## 11.3 Softmax 温度

\[
g_i=
\frac{\exp(z_i/T)}
{\sum_j\exp(z_j/T)}
\]

- \(T\) 较高：分布更平滑；
- \(T\) 较低：分布更尖锐。

训练初期可以使用较高温度，之后逐渐降低。

## 11.4 Expert Dropout

训练时随机屏蔽部分专家，避免 Gate 长期依赖固定专家。

这类似于：

```text
不允许某个专家一直被当作唯一答案
```

## 11.5 平衡多任务损失

需要避免单一任务主导训练。

常见方法包括：

- 手工调整任务权重；
- 动态任务权重；
- GradNorm；
- 不确定性加权；
- 梯度裁剪；
- 梯度冲突处理。

## 11.6 使用 PLE

PLE 会显式划分：

```text
共享专家
任务专属专家
```

相比 MMoE 完全依赖 Gate 自行学习共享关系，PLE 对共享和任务差异的边界控制更明确。

---

# 12. MMoE 与 PLE 的简单对比

| 架构 | 核心结构 | 特点 |
|---|---|---|
| Shared-Bottom | 一个共享底层 + 多个 Tower | 简单，但容易负迁移 |
| MMoE | 共享 Expert + 每任务独立 Gate | 共享更灵活，但可能极化 |
| PLE | 共享 Expert + 任务专属 Expert | 显式区分共享和专属能力 |

可以简单记忆为：

```text
Shared-Bottom：全部共享
MMoE：通过 Gate 自动选择怎么共享
PLE：显式区分共享专家和任务专家
```

---

# 13. MMoE 的简化伪代码

```python
class MMoE:
    def forward(self, x):
        expert_outputs = [
            expert(x)
            for expert in self.experts
        ]

        task_outputs = []

        for task_id in range(self.task_count):
            gate_weight = softmax(
                self.gates[task_id](x)
            )

            mixed = 0

            for expert_id in range(self.expert_count):
                mixed += (
                    gate_weight[expert_id]
                    * expert_outputs[expert_id]
                )

            output = self.towers[task_id](mixed)
            task_outputs.append(output)

        return task_outputs
```

从 C++ 在线服务角度，MMoE 通常已经被导出为 ONNX 或 TensorRT Engine。

C++ 服务不负责实现训练过程，而主要负责：

```text
构造特征 Tensor
→ 执行一次 MMoE 推理
→ 解析多个任务输出
→ 组合最终排序分数
```

---

# 14. 工业使用时的注意事项

## 14.1 任务并不是越多越好

把大量弱相关或冲突严重的目标放入同一个模型，可能导致负迁移。

需要通过离线实验和线上 A/B 实验判断任务组合。

## 14.2 Loss 权重非常重要

任务损失权重直接影响共享专家的训练方向。

权重设置不当时，某个大样本任务可能压制其他任务。

## 14.3 需要监控 Gate 和 Expert

训练时建议监控：

- 每个任务的 Gate 权重分布；
- Gate 熵；
- 每个 Expert 的平均负载；
- Expert 输出相似度；
- 每个任务的梯度范数；
- 每个任务的离线指标。

## 14.4 离线提升不等于线上提升

多目标模型可能提升 AUC，却未必提升最终业务指标。

需要关注：

- 最终融合公式；
- 预测概率校准；
- 任务之间的线上影响；
- 核心指标和护栏指标；
- 用户长期价值。

---

# 15. 最终总结

多目标学习解决的是：

> 一个推荐模型需要同时预测点击、转化、时长、互动和负反馈等多个目标。

Shared-Bottom 的问题是：

> 所有任务被迫共享完全相同的底层表示，容易产生负迁移。

MMoE 的核心结构是：

```text
多个共享 Expert
+ 每个任务一个独立 Gate
+ 每个任务一个独立 Tower
```

MMoE 的优势是：

> 不同任务可以选择不同的专家组合，在共享与任务差异之间取得平衡。

MMoE 的极化现象是：

> Gate 权重越来越集中，少数专家被过度选择，其他专家得不到充分训练。

最值得记住的一句话是：

> MMoE 通过多 Gate 缓解多任务负迁移，但需要防止 Gate 的“强者愈强”正反馈导致专家坍缩。
