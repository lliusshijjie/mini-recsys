# 推荐系统重排学习笔记：MMR、滑动窗口优化与 DPP

## 1. 重排阶段到底在做什么？

推荐系统的典型在线链路可以简化为：

```text
海量物品库
    ↓
召回：找出“可能感兴趣”的候选
    ↓
数千～数万条
    ↓
粗排：低成本筛掉明显不合适的候选
    ↓
数百～数千条
    ↓
精排：使用复杂模型精确预估用户行为
    ↓
几十～几百条
    ↓
重排：构造最终展示列表
    ↓
Top-K 结果
```

经过召回、粗排和精排之后，重排面对的候选通常已经只有百量级。  
但重排并不只是把这些物品再执行一次 `sort`。

重排通常同时完成三类工作：

1. **最终筛选**：过滤已下架、已看过、重复、超频、不可展示的物品。
2. **顺序调整**：根据多样性、新鲜度、探索性和业务规则调整精排顺序。
3. **列表构造**：从候选集中选出最终的 Top-K，并保证整个列表的整体体验。

因此，重排既是在“确定展示顺序”，也是在“继续筛选最终展示物品”。

---

## 2. 精排与重排的核心区别

精排主要回答：

> 单独看这个物品，用户有多大概率点击、观看、购买或互动？

可以抽象为：

```text
score(u, item)
```

重排主要回答：

> 在已经选出一部分物品的情况下，下一个位置放哪个物品，能让整个列表更好？

可以抽象为：

```text
score(u, item, selected_list)
```

例如精排结果为：

```text
篮球 A    0.98
篮球 B    0.97
篮球 C    0.96
足球 A    0.93
科技 A    0.91
电影 A    0.89
```

如果直接取 Top-5，用户会连续看到三条篮球内容。  
重排后可能变成：

```text
1. 篮球 A
2. 足球 A
3. 篮球 B
4. 科技 A
5. 电影 A
```

篮球内容并没有被完全删除，但被适当打散。

这说明：

- 精排关注的是**单个物品价值**；
- 重排关注的是**整个列表的组合价值**。

---

## 3. 为什么需要多样性重排？

精排模型通常倾向于把同类物品排在一起，因为这些物品拥有相似特征：

- 同一个作者；
- 同一个主题；
- 同一个商品品牌；
- 同一种内容形式；
- 同一种用户兴趣标签。

这会导致推荐列表出现：

- 内容重复；
- 同质化严重；
- 用户疲劳；
- 长期兴趣收窄；
- 新内容没有曝光机会；
- 点击率短期提升，但长期留存下降。

重排算法需要在以下两个目标之间做权衡：

```text
相关性：用户是否喜欢这个物品？
多样性：这个物品是否为当前列表提供了新的信息？
```

MMR 和 DPP 都是用于解决“相关性与多样性平衡”的经典方法。

---

# 4. MMR：最大边际相关性

## 4.1 MMR 的核心思想

MMR 的全称是：

```text
Maximal Marginal Relevance
最大边际相关性
```

“边际”可以理解为：

> 在当前已经选择的列表基础上，再加入一个候选物品，它还能带来多少新增价值？

MMR 每次从剩余候选物品中选择一个物品，其评分通常为：

\[
\operatorname{MMR}(i)
=
\lambda \cdot \operatorname{Rel}(i)
-
(1-\lambda)
\cdot
\max_{j \in S}
\operatorname{Sim}(i,j)
\]

其中：

- \(i\)：当前候选物品；
- \(S\)：已经选入结果列表的物品集合；
- \(\operatorname{Rel}(i)\)：候选物品的相关性，通常来自精排分数；
- \(\operatorname{Sim}(i,j)\)：候选物品与已选物品之间的相似度；
- \(\lambda\)：相关性和多样性的权衡系数。

可以理解为：

```text
MMR 分数
=
精排相关性奖励
-
与已选列表过于相似的惩罚
```

---

## 4.2 λ 参数如何理解？

### λ 接近 1

```text
更加相信精排结果
更加关注相关性
多样性调整较弱
```

例如：

```text
λ = 0.9
```

此时 MMR 基本保留精排顺序，只进行轻微打散。

### λ 接近 0

```text
更加关注多样性
相似内容惩罚较强
可能牺牲相关性
```

例如：

```text
λ = 0.3
```

此时列表会更加丰富，但可能插入一些用户兴趣不强的内容。

工业系统中，通常不会把 λ 设置得太低。  
因为重排的目标不是为了多样性而多样性，而是在保留相关性的基础上降低重复。

---

## 4.3 MMR 的执行流程

假设精排输出 100 个候选，最终需要选择 20 个：

1. 选择精排分数最高的物品作为第一个结果；
2. 遍历所有剩余候选；
3. 计算每个候选与已选列表中最相似物品的相似度；
4. 计算 MMR 分数；
5. 选择 MMR 分数最高的物品；
6. 重复以上过程，直到得到 Top-20。

流程如下：

```text
候选集合 C
已选集合 S = {}

选择精排分最高的物品
    ↓
放入 S
    ↓
对剩余候选计算：
相关性 - 相似度惩罚
    ↓
选择 MMR 分数最高者
    ↓
放入 S
    ↓
直到得到 Top-K
```

---

## 4.4 相似度如何计算？

MMR 效果很大程度上取决于相似度定义。

### 方法一：Embedding 余弦相似度

如果每个物品都有内容向量：

\[
\operatorname{Sim}(i,j)
=
\frac{e_i \cdot e_j}
{\|e_i\|\|e_j\|}
\]

适用于：

- 短视频内容向量；
- 商品语义向量；
- 新闻文本向量；
- 图片多模态向量。

优点是能够发现语义相似内容。

### 方法二：离散标签相似度

例如根据以下特征计算：

- 作者是否相同；
- 类目是否相同；
- 主题标签是否重叠；
- 品牌是否相同；
- 内容形式是否相同。

伪公式：

```text
sim =
    0.4 × 类目相似
  + 0.3 × 作者相似
  + 0.2 × 标签重叠
  + 0.1 × 内容形式相似
```

### 方法三：规则化相似度

某些工业系统并不一定使用复杂向量，而是直接定义规则：

```text
同一作者：sim = 1.0
同一一级类目：sim = 0.7
同一二级主题：sim = 0.5
其他：sim = 0.0
```

这种方式解释性强、计算成本低，适合在线服务。

---

## 4.5 基础 MMR 伪代码

### C++ 风格

```cpp
struct Item {
    int id;
    double rank_score;
    std::vector<float> embedding;
};

double cosine_similarity(const Item& a, const Item& b);

std::vector<Item> mmr_rerank(
    const std::vector<Item>& candidates,
    std::size_t top_k,
    double lambda)
{
    std::vector<Item> selected;
    std::vector<bool> used(candidates.size(), false);

    while (selected.size() < top_k) {
        int best_index = -1;
        double best_mmr_score =
            -std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (used[i]) {
                continue;
            }

            double max_similarity = 0.0;

            for (const auto& selected_item : selected) {
                max_similarity = std::max(
                    max_similarity,
                    cosine_similarity(candidates[i], selected_item)
                );
            }

            double mmr_score =
                lambda * candidates[i].rank_score
                - (1.0 - lambda) * max_similarity;

            if (mmr_score > best_mmr_score) {
                best_mmr_score = mmr_score;
                best_index = static_cast<int>(i);
            }
        }

        if (best_index == -1) {
            break;
        }

        used[best_index] = true;
        selected.push_back(candidates[best_index]);
    }

    return selected;
}
```

---

## 4.6 基础 MMR 的性能问题

设：

- 候选数为 \(N\)；
- 最终选择数为 \(K\)。

基础实现中，每选择一个物品，都需要：

1. 遍历剩余候选；
2. 每个候选再与已选列表比较。

复杂度大致为：

\[
O(NK^2)
\]

当候选数只有几百、Top-K 只有几十时，通常仍然可以接受。  
但在线推荐系统还需要考虑：

- 单机高 QPS；
- 相似度计算可能是高维向量点积；
- 每个请求都需要执行；
- 多路策略叠加；
- 尾延迟要求严格。

因此需要进一步优化。

---

# 5. MMR 的缓存优化

对于每个候选物品，可以维护：

```text
它与当前已选列表的最大相似度
```

每次新增一个物品后，不需要重新遍历整个已选列表，只需要计算：

```text
候选物品与新加入物品的相似度
```

然后更新：

```text
max_sim[i] =
    max(max_sim[i], sim(candidate[i], new_selected))
```

这样复杂度可以降低到近似：

\[
O(NK)
\]

### C++ 风格伪代码

```cpp
std::vector<Item> mmr_rerank_cached(
    const std::vector<Item>& candidates,
    std::size_t top_k,
    double lambda)
{
    const std::size_t n = candidates.size();

    std::vector<Item> selected;
    std::vector<bool> used(n, false);
    std::vector<double> max_similarity(n, 0.0);

    while (selected.size() < top_k) {
        int best_index = -1;
        double best_score =
            -std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < n; ++i) {
            if (used[i]) {
                continue;
            }

            double score =
                lambda * candidates[i].rank_score
                - (1.0 - lambda) * max_similarity[i];

            if (score > best_score) {
                best_score = score;
                best_index = static_cast<int>(i);
            }
        }

        if (best_index == -1) {
            break;
        }

        used[best_index] = true;
        selected.push_back(candidates[best_index]);

        for (std::size_t i = 0; i < n; ++i) {
            if (used[i]) {
                continue;
            }

            double similarity =
                cosine_similarity(
                    candidates[i],
                    candidates[best_index]
                );

            max_similarity[i] =
                std::max(max_similarity[i], similarity);
        }
    }

    return selected;
}
```

---

# 6. MMR 的滑动窗口优化

## 6.1 为什么需要滑动窗口？

标准 MMR 会让候选物品与**所有已经选中的物品**比较。

假设结果列表已经有 30 条：

```text
第 1 条：篮球
第 2 条：科技
...
第 30 条：电影
```

此时选择第 31 条时，候选物品仍然会受到第 1 条篮球内容的影响。

但在信息流场景中，用户通常更加关注局部连续体验：

- 连续两三条同类内容容易疲劳；
- 相隔十几条后再次出现同类内容通常可以接受；
- 用户可能一次只看到当前屏幕附近的若干条内容。

因此，与其要求整个列表完全不重复，不如重点保证：

> 最近几个位置不要出现过于相似的内容。

这就是滑动窗口 MMR。

---

## 6.2 滑动窗口 MMR 的核心思想

只让候选物品与最近 \(W\) 个已选物品比较：

\[
\operatorname{MMR}_{window}(i)
=
\lambda \cdot \operatorname{Rel}(i)
-
(1-\lambda)
\cdot
\max_{j \in S_{recent}}
\operatorname{Sim}(i,j)
\]

其中：

```text
S_recent = 最近 W 个已选物品
```

例如：

```text
窗口大小 W = 5
```

当选择第 20 个物品时，只考虑第 15～19 个物品，不再考虑更早的结果。

---

## 6.3 滑动窗口解决了什么问题？

### 问题一：过度打散

标准 MMR 可能导致：

```text
用户非常喜欢篮球，
但因为列表前面已经出现过篮球，
后续篮球内容持续受到惩罚。
```

滑动窗口允许同类内容在间隔若干位置后重新出现。

### 问题二：相关性损失

全局多样性约束过强时，可能把高相关物品不断后移。

滑动窗口只控制局部重复，可以更好地保留精排相关性。

### 问题三：计算成本

当 Top-K 较大时，滑动窗口将相似度比较范围限制为 \(W\)，复杂度可近似为：

\[
O(NKW)
\]

当 \(W\) 是固定小常数时，可以近似看作：

\[
O(NK)
\]

---

## 6.4 滑动窗口如何选择大小？

窗口大小取决于产品形态。

### 短视频信息流

```text
W = 3～8
```

用户连续刷视频，局部重复感知非常强。

### 新闻信息流

```text
W = 5～15
```

同一热点可以多次出现，但不能连续被不同媒体重复报道。

### 电商列表

```text
W = 4～10
```

用于控制同品牌、同店铺、同类商品的连续曝光。

### 瀑布流或双列列表

窗口不一定只是线性位置，还可以按“当前屏幕可见区域”定义。

---

## 6.5 滑动窗口 MMR 伪代码

### C++ 风格

```cpp
std::vector<Item> sliding_window_mmr(
    const std::vector<Item>& candidates,
    std::size_t top_k,
    std::size_t window_size,
    double lambda)
{
    std::vector<Item> selected;
    std::vector<bool> used(candidates.size(), false);

    while (selected.size() < top_k) {
        int best_index = -1;
        double best_score =
            -std::numeric_limits<double>::infinity();

        std::size_t window_begin = 0;

        if (selected.size() > window_size) {
            window_begin = selected.size() - window_size;
        }

        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (used[i]) {
                continue;
            }

            double max_similarity = 0.0;

            for (std::size_t j = window_begin;
                 j < selected.size();
                 ++j)
            {
                max_similarity = std::max(
                    max_similarity,
                    cosine_similarity(candidates[i], selected[j])
                );
            }

            double mmr_score =
                lambda * candidates[i].rank_score
                - (1.0 - lambda) * max_similarity;

            if (mmr_score > best_score) {
                best_score = mmr_score;
                best_index = static_cast<int>(i);
            }
        }

        if (best_index == -1) {
            break;
        }

        used[best_index] = true;
        selected.push_back(candidates[best_index]);
    }

    return selected;
}
```

---

## 6.6 Rust 风格伪代码

```rust
fn sliding_window_mmr(
    candidates: &[Item],
    top_k: usize,
    window_size: usize,
    lambda: f64,
) -> Vec<Item> {
    let mut selected = Vec::new();
    let mut used = vec![false; candidates.len()];

    while selected.len() < top_k {
        let window_begin =
            selected.len().saturating_sub(window_size);

        let mut best_index: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;

        for (i, candidate) in candidates.iter().enumerate() {
            if used[i] {
                continue;
            }

            let mut max_similarity = 0.0;

            for selected_item in &selected[window_begin..] {
                max_similarity = max_similarity.max(
                    cosine_similarity(candidate, selected_item)
                );
            }

            let mmr_score =
                lambda * candidate.rank_score
                - (1.0 - lambda) * max_similarity;

            if mmr_score > best_score {
                best_score = mmr_score;
                best_index = Some(i);
            }
        }

        let Some(index) = best_index else {
            break;
        };

        used[index] = true;
        selected.push(candidates[index].clone());
    }

    selected
}
```

---

## 6.7 滑动窗口的工程优化：环形队列

实现滑动窗口时，不需要维护整个历史列表用于相似度计算。

可以使用：

```text
deque / ring buffer
```

只保存最近 \(W\) 个物品。

C++ 示例：

```cpp
std::deque<Item> recent_window;

recent_window.push_back(new_item);

if (recent_window.size() > window_size) {
    recent_window.pop_front();
}
```

Rust 示例：

```rust
use std::collections::VecDeque;

let mut recent_window = VecDeque::new();

recent_window.push_back(new_item);

if recent_window.len() > window_size {
    recent_window.pop_front();
}
```

不过实际系统通常仍然会保留完整结果列表，因为最终还需要返回全部 Top-K；  
环形队列只是用于维护“参与局部相似度计算的最近物品”。

---

## 6.8 滑动窗口 MMR 的局限

滑动窗口只保证局部多样性，不保证全局多样性。

例如：

```text
篮球、科技、电影、游戏、篮球、科技、电影、游戏……
```

局部看没有连续重复，但整个列表仍然只覆盖少数类别。

因此工业系统中常见做法是组合多种约束：

```text
滑动窗口：控制局部重复
全局计数器：控制总曝光次数
硬规则：控制作者、广告、店铺频次
```

例如：

```text
最近 5 条内同一类目最多 1 条
整个 Top-20 中同一类目最多 6 条
同一作者最多出现 2 次
```

---

# 7. DPP：行列式点过程

## 7.1 DPP 的核心思想

DPP 的全称是：

```text
Determinantal Point Process
行列式点过程
```

MMR 是一种逐个选择的贪心方法：

```text
每次选择当前边际收益最大的物品
```

DPP 更强调：

> 直接评价一个物品集合整体是否同时具备高质量和高多样性。

DPP 会为一个候选集合 \(Y\) 定义概率：

\[
P(Y) \propto \det(L_Y)
\]

其中：

- \(L\)：候选物品的核矩阵；
- \(L_Y\)：从核矩阵中取出集合 \(Y\) 对应的子矩阵；
- \(\det(L_Y)\)：子矩阵的行列式。

行列式越大，通常表示：

- 物品本身质量较高；
- 物品之间方向差异较大；
- 集合内部重复较少。

---

## 7.2 为什么行列式可以表示多样性？

可以把每个物品理解为向量。

两个向量：

- 方向接近：内容相似；
- 方向不同：内容差异较大。

由这些向量构成的几何体体积可以用行列式表示。

### 如果物品非常相似

向量几乎重合，体积接近 0：

```text
det ≈ 0
```

说明集合缺乏多样性。

### 如果物品差异较大

向量方向不同，张成空间的体积更大：

```text
det 较大
```

说明集合更加多样。

因此，DPP 的直观理解是：

> 选择一组既“长得高”，又“站得开”的物品。

其中：

- “长得高”表示质量高；
- “站得开”表示彼此差异大。

---

## 7.3 DPP 核矩阵的构造

常见形式：

\[
L_{ij}
=
q_i \cdot S_{ij} \cdot q_j
\]

也可以写成：

\[
L
=
\operatorname{diag}(q)
\cdot S
\cdot
\operatorname{diag}(q)
\]

其中：

- \(q_i\)：物品 \(i\) 的质量分数；
- \(S_{ij}\)：物品 \(i\) 和 \(j\) 的相似度；
- \(L\)：同时编码质量和多样性的核矩阵。

质量分数可以来自：

- 精排分数；
- CTR 预估；
- 观看时长预估；
- GMV 预估；
- 多目标融合分数。

相似度矩阵可以来自：

- Embedding 余弦相似度；
- 类目相似度；
- 标签重叠度；
- 作者或品牌相似度；
- 多模态内容向量。

---

## 7.4 一个简单例子

假设有三个候选：

```text
A：篮球视频，质量 0.95
B：篮球视频，质量 0.93
C：科技视频，质量 0.85
```

虽然 A 和 B 的质量都很高，但二者非常相似。

DPP 可能认为：

```text
集合 {A, B}
质量高，但相似度太高
```

而：

```text
集合 {A, C}
质量略低，但组合更加多样
```

最终 DPP 可能更倾向选择 `{A, C}`。

---

# 8. DPP 的贪心求解

精确求解最优 DPP 子集通常成本较高。  
工业系统中通常使用贪心近似：

1. 初始化已选集合为空；
2. 计算每个候选加入当前集合后的行列式增益；
3. 选择增益最大的候选；
4. 更新矩阵分解状态；
5. 重复直到得到 Top-K。

直接反复计算行列式会非常慢，因此常使用：

- Cholesky 分解；
- 增量矩阵更新；
- Fast Greedy MAP Inference。

复杂度通常可以优化到近似：

\[
O(NK^2)
\]

其中：

- \(N\)：候选数；
- \(K\)：最终列表长度。

---

## 8.1 DPP 贪心伪代码

下面是概念化伪代码，重点用于理解流程，并非完整数值稳定实现。

### C++ 风格

```cpp
std::vector<int> dpp_greedy(
    const Matrix& kernel,
    std::size_t top_k)
{
    std::vector<int> selected;
    std::vector<bool> used(kernel.rows(), false);

    while (selected.size() < top_k) {
        int best_index = -1;
        double best_gain =
            -std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < kernel.rows(); ++i) {
            if (used[i]) {
                continue;
            }

            std::vector<int> trial = selected;
            trial.push_back(static_cast<int>(i));

            Matrix sub_matrix =
                extract_sub_matrix(kernel, trial);

            double gain =
                log_determinant(sub_matrix);

            if (gain > best_gain) {
                best_gain = gain;
                best_index = static_cast<int>(i);
            }
        }

        if (best_index == -1) {
            break;
        }

        used[best_index] = true;
        selected.push_back(best_index);
    }

    return selected;
}
```

真实工业实现不会每次完整构造子矩阵和重新计算行列式，  
而是通过 Cholesky 分解增量更新。

---

## 8.2 DPP 核矩阵构造伪代码

```cpp
Matrix build_dpp_kernel(
    const std::vector<Item>& candidates)
{
    const std::size_t n = candidates.size();
    Matrix kernel(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double quality_i =
                transform_quality(candidates[i].rank_score);

            double quality_j =
                transform_quality(candidates[j].rank_score);

            double similarity =
                cosine_similarity(
                    candidates[i],
                    candidates[j]
                );

            kernel(i, j) =
                quality_i * similarity * quality_j;
        }
    }

    return kernel;
}
```

质量分数通常需要做正数变换，例如：

```text
q_i = exp(α × rank_score_i)
```

因为 DPP 核矩阵通常要求满足半正定条件。

---

# 9. MMR 与 DPP 的区别

| 维度 | MMR | DPP |
|---|---|---|
| 核心思想 | 每次选择边际收益最大的物品 | 评价整个集合的质量与多样性 |
| 优化方式 | 贪心逐个选择 | 集合概率或行列式最大化 |
| 多样性范围 | 通常是候选与已选集合的最大相似度 | 同时考虑集合内部整体关系 |
| 工程复杂度 | 低 | 高 |
| 可解释性 | 强 | 中等 |
| 调参难度 | 较低 | 较高 |
| 在线性能 | 较好 | 相对较重 |
| 滑动窗口支持 | 非常自然 | 通常不直接使用滑动窗口 |
| 适用场景 | 大多数在线信息流、搜索、电商 | 对全局集合多样性要求较高的场景 |
| 常见程度 | 非常常见 | 更偏高级或特定场景 |

---

## 9.1 MMR 的优势

- 实现简单；
- 容易解释；
- 容易与规则系统组合；
- 延迟较低；
- 可以直接使用精排分数；
- 适合滑动窗口；
- 可以针对作者、类目、品牌分别设计惩罚。

## 9.2 MMR 的缺点

- 本质是局部贪心；
- 只使用最大相似度时，可能忽略集合整体结构；
- 不一定得到全局最优列表；
- 相似度定义不好时效果有限。

---

## 9.3 DPP 的优势

- 从集合整体角度建模；
- 能同时考虑质量和多样性；
- 数学形式统一；
- 相比简单打散规则，更容易捕捉复杂的相似关系。

## 9.4 DPP 的缺点

- 核矩阵构造复杂；
- 需要保证数值稳定性；
- 计算成本更高；
- 在线服务工程实现难度较大；
- 参数和相似度矩阵不容易解释；
- 与大量业务硬规则结合时不如 MMR 灵活。

---

# 10. 工业系统中如何选择？

## 10.1 优先使用 MMR 的场景

MMR 更适合：

- 短视频信息流；
- 新闻推荐；
- 电商商品列表；
- 搜索结果多样化；
- 在线延迟要求严格；
- 业务规则较多；
- 需要快速迭代和解释。

工程中常见组合：

```text
精排 Top-200
    ↓
硬过滤
    ↓
MMR 多样性重排
    ↓
作者/类目滑动窗口打散
    ↓
广告、运营内容混排
    ↓
Top-20
```

---

## 10.2 适合使用 DPP 的场景

DPP 更适合：

- 候选规模较小；
- 对整个列表的全局多样性要求较高；
- 物品有高质量 Embedding；
- 可以接受更高计算成本；
- 业务规则相对稳定；
- 推荐结果以“一组内容”整体展示。

例如：

- 推荐一组相互差异较大的商品；
- 推荐一页不同主题的新闻；
- 推荐一个多样化歌单；
- 推荐一组代表性搜索结果；
- 从大量候选中选择少量代表样本。

---

# 11. 工业重排通常不是只使用一种算法

真实推荐系统通常不会只部署一个 MMR 或一个 DPP，而是采用分层策略。

一个常见的重排管线：

```text
精排候选
    ↓
硬过滤
    - 下架
    - 已看
    - 风控
    - 地域限制
    ↓
规则打散
    - 同作者
    - 同类目
    - 同品牌
    ↓
算法多样化
    - MMR
    - DPP
    ↓
业务混排
    - 广告
    - 直播
    - 运营位
    ↓
最终频控与兜底
    ↓
Top-K
```

算法负责软优化，规则负责硬约束。

例如：

```text
MMR：让篮球内容适度分散
硬规则：同一作者绝对不能连续出现
```

二者职责不同，不能互相完全替代。

---

# 12. 一个更贴近工业的 MMR 评分

工业系统中的 MMR 往往不仅包含一个相似度项。

可以设计为：

\[
\begin{aligned}
Score(i)
=&\ \lambda_r \cdot RankScore(i) \\
&- \lambda_c \cdot CategorySimilarity(i,S) \\
&- \lambda_a \cdot AuthorSimilarity(i,S) \\
&- \lambda_e \cdot EmbeddingSimilarity(i,S) \\
&+ \lambda_f \cdot Freshness(i) \\
&+ \lambda_x \cdot ExploreBonus(i)
\end{aligned}
\]

对应伪代码：

```cpp
double rerank_score(
    const Item& candidate,
    const std::deque<Item>& recent_window)
{
    double score = candidate.rank_score;

    score -= 0.20 * max_category_similarity(
        candidate,
        recent_window
    );

    score -= 0.15 * max_author_similarity(
        candidate,
        recent_window
    );

    score -= 0.25 * max_embedding_similarity(
        candidate,
        recent_window
    );

    score += 0.10 * freshness_bonus(candidate);
    score += 0.05 * exploration_bonus(candidate);

    return score;
}
```

这样更符合工业系统：

```text
基础相关性
+ 新鲜度
+ 探索价值
- 内容重复
- 作者重复
- 类目重复
```

---

# 13. 重排服务端的工程注意事项

## 13.1 控制候选规模

不要直接对数千条候选运行复杂 DPP。

常见做法：

```text
精排 Top-200
    ↓
重排 Top-20
```

必要时先截断到 Top-N，再进行多样性重排。

---

## 13.2 预计算相似度

对于静态物品特征，可以预计算：

- 归一化 Embedding；
- 类目编码；
- 作者 ID；
- 主题标签；
- 内容聚类 ID。

在线只执行：

- 点积；
- ID 比较；
- bitset 交集；
- 简单计数。

---

## 13.3 限制高维向量计算

如果候选有 200 条，Embedding 为 768 维，完整计算相似度矩阵可能成本较高。

可以使用：

- 向量降维；
- PQ 或量化向量；
- 半精度浮点；
- 内容聚类 ID；
- 只对 Top-N 候选计算；
- SIMD 加速点积；
- 离线缓存相似度。

---

## 13.4 设置超时和降级

重排属于在线链路，必须有降级方案：

```text
DPP 超时
    ↓
降级为 MMR
    ↓
MMR 超时
    ↓
降级为规则打散
    ↓
规则执行异常
    ↓
直接返回精排 Top-K
```

核心原则：

> 重排效果再好，也不能阻塞整个推荐请求。

---

## 13.5 观察列表级指标

重排不能只看单物品 CTR，还需要关注：

- 列表内类目覆盖率；
- 作者覆盖率；
- 平均物品相似度；
- 重复曝光率；
- 用户跳出率；
- 滑动深度；
- 有效播放条数；
- 人均消费时长；
- 次日留存；
- 长期兴趣丰富度。

一个重排策略可能降低短期 CTR，却提高：

- 播放深度；
- 使用时长；
- 长期留存。

因此必须通过 A/B 实验综合评估。

---

# 14. 总结

## 14.1 重排阶段的定位

重排不是简单排序，而是：

```text
从百量级精排候选中，
继续过滤、选择并排列最终 Top-K，
优化整个推荐列表的体验。
```

它关注的是：

```text
整个列表是否相关、丰富、自然、可展示。
```

---

## 14.2 MMR

MMR 的核心是：

```text
选择高相关物品，
同时惩罚与已选物品过于相似的候选。
```

特点：

- 简单；
- 高效；
- 可解释；
- 适合在线服务；
- 容易与业务规则组合。

---

## 14.3 滑动窗口 MMR

滑动窗口只关注最近若干个位置：

```text
重点避免局部连续重复，
允许同类内容间隔一段距离后再次出现。
```

它比全局 MMR 更适合信息流和短视频场景。

---

## 14.4 DPP

DPP 从集合整体角度同时建模：

```text
物品质量 + 集合多样性
```

它使用行列式衡量一个集合是否既高质量又彼此差异明显。

DPP 的理论更加完整，但在线工程成本和实现复杂度通常高于 MMR。

---

## 14.5 最终记忆

可以用下面四句话快速记忆：

```text
精排：每个物品单独有多好？
重排：这些物品放在一起有多好？

MMR：逐个选择，减少与已选结果的相似性。
DPP：整体选择，最大化集合质量与多样性。

滑动窗口 MMR：
只约束最近几个位置，重点解决局部重复。
```
