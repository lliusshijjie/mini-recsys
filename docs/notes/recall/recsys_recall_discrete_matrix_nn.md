# 推荐系统召回层补充：离散特征、矩阵补全、最近邻查找

## 1. 离散特征处理

### 1.1 概念

推荐系统中很多特征不是连续数值，而是离散 ID 或类别：

```text
user_id、item_id、category、city、device、tag、author_id
```

这些特征不能直接输入模型，需要转换成数值向量。工业里最核心的方式是：

```text
离散特征 → ID 映射 → Embedding 查表 → 稠密向量
```

### 1.2 One-Hot

假设城市只有 4 个：

```text
Taipei    -> [1, 0, 0, 0]
Shanghai  -> [0, 1, 0, 0]
Beijing   -> [0, 0, 1, 0]
Tokyo     -> [0, 0, 0, 1]
```

数学表示：

$$
x_i =
\begin{cases}
1, & i = k \\
0, & i \ne k
\end{cases}
$$

问题是：`user_id`、`item_id` 这种特征可能是千万级、亿级，One-Hot 维度极大且稀疏，所以工业里通常不用直接 One-Hot 参与深度模型计算。

### 1.3 Embedding

Embedding 的本质是给每个离散 ID 分配一个低维稠密向量。

假设某个离散特征有 $N$ 个取值，Embedding 维度为 $d$，维护矩阵：

$$
E \in \mathbb{R}^{N \times d}
$$

某个 ID $i$ 的向量为：

$$
e_i = E[i]
$$

例子：

```text
item_id = 0: C++ 教程   -> [0.90, 0.10, 0.20]
item_id = 1: Rust 教程  -> [0.85, 0.20, 0.25]
item_id = 2: 篮球视频   -> [0.10, 0.80, 0.70]
```

如果用户最近看过 C++ 和 Rust，可以简单平均得到用户兴趣向量：

$$
e_u = \frac{e_{C++} + e_{Rust}}{2}
$$

也就是：

$$
e_u =
\frac{
[0.90, 0.10, 0.20] + [0.85, 0.20, 0.25]
}{2}
=
[0.875, 0.15, 0.225]
$$

工业意义：

```text
离散特征处理是双塔召回、向量召回、粗排、精排的基础。
没有 Embedding，就很难把 user_id、item_id、category、tag 等特征喂给模型。
```

---

## 2. 矩阵补全

### 2.1 概念

矩阵补全，也叫 Matrix Completion，解决的问题是：

```text
根据用户已经交互过的物品，预测用户可能喜欢但还没交互过的物品。
```

构造用户-物品交互矩阵：

$$
R \in \mathbb{R}^{m \times n}
$$

其中：

```text
m = 用户数量
n = 物品数量
R[u][i] = 用户 u 对物品 i 的反馈
```

例子：

| 用户 / 物品 | C++教程 | Rust教程 | Tokio教程 | 篮球视频 |
|---|---:|---:|---:|---:|
| 用户A | 5 | 5 | ? | 0 |
| 用户B | 4 | ? | 5 | 0 |
| 用户C | 0 | 0 | ? | 5 |

这里的 `?` 就是希望模型补全的未知兴趣。

### 2.2 矩阵分解

经典做法是把用户和物品都映射到隐向量空间：

$$
P \in \mathbb{R}^{m \times k}
$$

$$
Q \in \mathbb{R}^{n \times k}
$$

用户 $u$ 对物品 $i$ 的预测分数：

$$
\hat{R}_{ui} = p_u^T q_i
$$

优化目标：

$$
\min_{P,Q}
\sum_{(u,i) \in \Omega}
(R_{ui} - p_u^T q_i)^2
+
\lambda(\|p_u\|^2 + \|q_i\|^2)
$$

其中：

```text
Ω = 已观测到的用户-物品交互集合
λ = 正则化参数，防止过拟合
```

### 2.3 计算例子

用户 A 向量：

$$
p_A = [0.9, 0.1]
$$

Rust 教程向量：

$$
q_{Rust} = [0.85, 0.15]
$$

预测兴趣：

$$
\hat{R}_{A,Rust}
=
p_A^T q_{Rust}
=
0.9 \times 0.85 + 0.1 \times 0.15
=
0.78
$$

篮球视频向量：

$$
q_{Basketball} = [0.1, 0.9]
$$

$$
\hat{R}_{A,Basketball}
=
0.9 \times 0.1 + 0.1 \times 0.9
=
0.18
$$

所以用户 A 更可能喜欢 Rust 教程，而不是篮球视频。

工业意义：

```text
矩阵补全可以产生 user embedding 和 item embedding。
这些向量可以进入最近邻查找，用于召回候选物品。
```

---

## 3. 最近邻查找

### 3.1 概念

最近邻查找，Nearest Neighbor Search，指的是：

```text
给定一个查询向量 q，从向量库中找出最相似的 TopK 个向量。
```

推荐系统召回中通常是：

```text
用户向量 user_embedding
        ↓
在 item_embedding 向量库中查 TopK
        ↓
返回候选 item
```

数学定义：

$$
TopK(q) = \operatorname{argTopK}_{x_i \in X} sim(q, x_i)
$$

其中：

```text
q   = 用户向量
x_i = 物品向量
sim = 相似度函数
```

### 3.2 常见相似度

#### 点积

$$
score(u, i) = p_u^T q_i
$$

例子：

$$
p_u = [0.8, 0.2, 0.1]
$$

$$
q_i = [0.9, 0.1, 0.1]
$$

$$
score = 0.8 \times 0.9 + 0.2 \times 0.1 + 0.1 \times 0.1 = 0.75
$$

#### 余弦相似度

$$
sim(q, x) =
\frac{q^T x}{\|q\| \cdot \|x\|}
$$

如果向量已经归一化，那么余弦相似度等价于点积。

#### 欧氏距离

$$
dist(q, x) =
\sqrt{
\sum_{j=1}^{d}(q_j - x_j)^2
}
$$

距离越小越相似。

### 3.3 工业流程

离线阶段：

```text
1. 收集用户行为数据
2. 训练召回模型，例如双塔、矩阵分解、Item2Vec
3. 生成 item_embedding
4. 构建向量索引，例如 HNSW、IVF、PQ、IVF-PQ
5. 将索引加载到线上召回服务
```

在线阶段：

```text
1. 获取用户特征和实时行为
2. 生成 user_embedding
3. 查询 ANN 向量索引
4. 返回 TopK item
5. 过滤已看、已买、下架、低质、黑名单物品
6. 进入粗排 / 精排
```

### 3.4 精确最近邻 vs 近似最近邻

精确最近邻是暴力计算：

$$
O(n \cdot d)
$$

其中：

```text
n = 物品数量
d = 向量维度
```

如果物品有 1 亿个，向量维度 128，那么每次查询需要计算：

$$
100000000 \times 128
$$

线上无法接受。

所以工业里常用 ANN，Approximate Nearest Neighbor，近似最近邻。

ANN 的目标是：

```text
不保证 100% 找到真实 TopK，
但用低延迟找到足够好的 TopK。
```

常见方法：

```text
HNSW:
    图索引，查询快，召回率高，但内存占用大。

IVF:
    先聚类，再只查最近的几个簇，适合大规模向量库。

PQ:
    向量压缩，节省内存，但会损失精度。

IVF-PQ:
    IVF 负责缩小搜索范围，PQ 负责压缩和加速。
```

---

## 4. 最近邻查找 C++ 伪代码

下面是精确 TopK 最近邻查找。工业里千万级以上不会直接这么做，但它可以帮助理解 ANN 的目标：减少全量扫描。

```cpp
#include <vector>
#include <queue>
#include <algorithm>
#include <functional>
using namespace std;

using Vec = vector<float>;

float dot_product(const Vec& a, const Vec& b) {
    float ans = 0.0f;

    for (int i = 0; i < (int)a.size(); i++) {
        ans += a[i] * b[i];
    }

    return ans;
}

vector<pair<int, float>> exact_topk_search(
    const Vec& query,
    const vector<Vec>& item_embeddings,
    int top_k
) {
    // 小根堆：保存当前 TopK。
    // 堆顶是当前 TopK 中分数最低的 item。
    using Node = pair<float, int>; // score, item_id
    priority_queue<Node, vector<Node>, greater<Node>> heap;

    for (int item_id = 0; item_id < (int)item_embeddings.size(); item_id++) {
        float score = dot_product(query, item_embeddings[item_id]);

        if ((int)heap.size() < top_k) {
            heap.push({score, item_id});
        } else if (score > heap.top().first) {
            heap.pop();
            heap.push({score, item_id});
        }
    }

    vector<pair<int, float>> result;

    while (!heap.empty()) {
        auto [score, item_id] = heap.top();
        heap.pop();
        result.push_back({item_id, score});
    }

    sort(result.begin(), result.end(), [](auto& a, auto& b) {
        return a.second > b.second;
    });

    return result;
}
```

复杂度：

$$
O(n \cdot d + n \log K)
$$

工业中使用 HNSW / IVF-PQ 的目的，就是避免对所有 item 做这一步全量扫描。

---

## 5. 三个概念的关系

```text
离散特征处理：
    把 user_id、item_id、category、tag 等离散字段变成 embedding。

矩阵补全：
    根据用户-物品交互矩阵，学习 user embedding 和 item embedding。

最近邻查找：
    用 user embedding 去 item embedding 向量库里找最相似的 TopK item。
```

可以串成一条召回链路：

```text
用户行为 / 离散特征
        ↓
Embedding / 矩阵分解 / 双塔模型
        ↓
user_embedding 和 item_embedding
        ↓
ANN 最近邻查找
        ↓
召回候选物品
        ↓
粗排 / 精排
```

---

## 6. 记忆总结

```text
离散特征处理：
    解决“特征怎么变成向量”。

矩阵补全：
    解决“未知兴趣怎么预测”。

最近邻查找：
    解决“怎么从海量 item 向量中快速找 TopK”。
```

在召回层里，三者经常组合使用：

```text
离散特征 → Embedding → 用户/物品向量 → ANN TopK → 候选召回
```
