# 推荐系统召回层：ItemCF、Swing、UserCF 系统总结

> 适用场景：推荐系统召回层入门与工程复习。  
> 重点内容：三种协同过滤召回方式的直觉、数学公式、具体计算例子、工业应用方式和 C++ 风格伪代码。

---

## 1. 召回层中的协同过滤是什么？

在工业推荐系统中，完整链路通常是：

```text
用户请求
  ↓
多路召回：ItemCF / Swing / UserCF / 热门 / 内容 / 向量召回 / 双塔召回
  ↓
粗排：轻量模型快速筛选
  ↓
精排：复杂模型精准排序
  ↓
重排：多样性、去重、业务规则、探索
  ↓
返回推荐结果
```

召回层的目标不是直接给最终排序，而是：

```text
从千万级、亿级物品池中，快速找出几百到几千个候选物品。
```

ItemCF、Swing、UserCF 都属于**协同过滤召回**。它们不直接理解物品内容，而是利用用户行为日志挖掘关系：

```text
哪些物品经常被同一批用户交互？
哪些用户的兴趣比较相似？
某个用户最近喜欢的东西，能扩展出哪些相似物品？
```

---

## 2. 基础符号约定

用户集合：

\[
U = \{u_1, u_2, ..., u_m\}
\]

物品集合：

\[
I = \{i_1, i_2, ..., i_n\}
\]

用户 \(u\) 交互过的物品集合：

\[
I_u = \{i \mid u \text{ interacted with } i\}
\]

物品 \(i\) 被哪些用户交互过：

\[
U_i = \{u \mid u \text{ interacted with } i\}
\]

用户 \(u\) 对物品 \(i\) 的行为权重：

\[
r_{ui}
\]

在工业系统中，\(r_{ui}\) 不一定只是 0 或 1，通常会结合行为类型：

| 行为 | 示例权重 |
|---|---:|
| 曝光未点击 | 0 |
| 点击 | 1 |
| 长停留 / 长播放 | 2 |
| 收藏 / 点赞 | 3 |
| 加购 | 4 |
| 购买 | 5 |
| 负反馈 / 不感兴趣 | -5 |

还可以加入时间衰减：

\[
r_{ui} = w_{action} \cdot decay(t)
\]

常见时间衰减函数：

\[
decay(t) = e^{-\lambda \Delta t}
\]

其中 \(\Delta t\) 是行为距离当前时间的间隔。越新的行为，权重越大。

---

# 3. UserCF：基于用户相似度的召回

## 3.1 核心直觉

UserCF 的核心思想是：

```text
和你兴趣相似的用户喜欢什么，就把什么推荐给你。
```

例如：

```text
用户 A 看过：C++、Rust、Linux
用户 B 看过：C++、Rust、Linux、推荐系统
```

系统会认为 A 和 B 兴趣相似，因此可以把 B 看过但 A 没看过的“推荐系统”推荐给 A。

---

## 3.2 用户相似度公式

### 3.2.1 0/1 行为下的余弦相似度

如果只考虑“是否交互过”，用户 \(u\) 和用户 \(v\) 的相似度可以写成：

\[
sim(u, v) = \frac{|I_u \cap I_v|}{\sqrt{|I_u| \cdot |I_v|}}
\]

含义：

```text
两个用户共同交互的物品越多，相似度越高；
但如果两个用户本身都非常活跃，需要用分母做归一化，避免活跃用户天然占优。
```

### 3.2.2 带行为权重的用户相似度

如果引入点击、收藏、购买等行为权重，可以使用加权余弦相似度：

\[
sim(u, v) =
\frac{
\sum_{i \in I_u \cap I_v} r_{ui} r_{vi}
}{
\sqrt{\sum_{i \in I_u} r_{ui}^2}
\sqrt{\sum_{i \in I_v} r_{vi}^2}
}
\]

---

## 3.3 UserCF 推荐打分公式

对目标用户 \(u\) 和候选物品 \(j\)，UserCF 的推荐分数可以写成：

\[
score(u, j) = \sum_{v \in N(u), j \in I_v} sim(u, v) \cdot r_{vj}
\]

其中：

\[
N(u)
\]

表示用户 \(u\) 的 TopK 相似用户集合。

通俗理解：

```text
一个物品被越多相似用户喜欢，分数越高；
相似用户和当前用户越像，贡献越大；
相似用户对该物品的行为越强，贡献越大。
```

---

## 3.4 UserCF 具体计算例子

假设有如下用户行为：

| 用户 | 交互物品 |
|---|---|
| u1 | A, B, C |
| u2 | A, B |
| u3 | A, C |
| u4 | B, D |
| u5 | C, D |

现在要给用户 `u3` 推荐物品。

用户 `u3` 的历史行为：

```text
u3 = {A, C}
```

计算 `u3` 和其他用户的相似度。

### u3 和 u1

\[
I_{u3} = \{A, C\}
\]

\[
I_{u1} = \{A, B, C\}
\]

\[
I_{u3} \cap I_{u1} = \{A, C\}
\]

\[
sim(u3, u1) = \frac{2}{\sqrt{2 \cdot 3}} = \frac{2}{\sqrt{6}} \approx 0.816
\]

### u3 和 u2

\[
I_{u2} = \{A, B\}
\]

\[
I_{u3} \cap I_{u2} = \{A\}
\]

\[
sim(u3, u2) = \frac{1}{\sqrt{2 \cdot 2}} = 0.5
\]

### u3 和 u5

\[
I_{u5} = \{C, D\}
\]

\[
I_{u3} \cap I_{u5} = \{C\}
\]

\[
sim(u3, u5) = \frac{1}{\sqrt{2 \cdot 2}} = 0.5
\]

候选物品是 `u3` 没看过的物品：

```text
B, D
```

计算候选物品分数。

### 物品 B

B 被 `u1`、`u2`、`u4` 交互过。

其中和 `u3` 相似度有效的用户主要是 `u1` 和 `u2`：

\[
score(u3, B) = sim(u3, u1) + sim(u3, u2)
\]

\[
score(u3, B) = 0.816 + 0.5 = 1.316
\]

### 物品 D

D 被 `u4`、`u5` 交互过。

其中 `u5` 和 `u3` 有共同物品 C：

\[
score(u3, D) = sim(u3, u5) = 0.5
\]

所以 UserCF 会更倾向于推荐：

```text
B > D
```

---

## 3.5 UserCF 的工业应用

UserCF 直接在超大规模工业推荐中使用得相对少，主要原因是用户规模太大。

如果有 1 亿用户，直接计算用户两两相似度，理论复杂度接近：

\[
O(|U|^2)
\]

这是不可接受的。

但 UserCF 的思想仍然广泛存在于工业系统中，常见变体包括：

```text
相似用户召回
Lookalike 人群扩展
用户聚类召回
社交关系召回
同圈层用户行为扩散
```

适合场景：

```text
社交推荐：朋友喜欢什么
社区推荐：相似用户关注的话题
内容平台：相似兴趣人群最近消费内容
冷启动：根据用户注册信息或早期行为找相似人群
```

工程上很少在线实时计算 UserCF，一般是：

```text
离线计算 user -> similar_users
线上根据 similar_users 扩展候选 item
```

---

## 3.6 UserCF C++ 风格伪代码

```cpp
#include <bits/stdc++.h>
using namespace std;

using UserId = int;
using ItemId = int;

// user -> items
unordered_map<UserId, vector<ItemId>> user_items;

// user -> top similar users
unordered_map<UserId, vector<pair<UserId, double>>> user_sim_index;

static double cosine_user_sim(const vector<ItemId>& a, const vector<ItemId>& b) {
    unordered_set<ItemId> set_a(a.begin(), a.end());
    int common = 0;

    for (ItemId item : b) {
        if (set_a.count(item)) {
            common++;
        }
    }

    if (a.empty() || b.empty()) return 0.0;
    return common / sqrt(1.0 * a.size() * b.size());
}

void build_usercf_index(int top_k) {
    vector<UserId> users;
    for (auto& [u, items] : user_items) {
        users.push_back(u);
    }

    for (int i = 0; i < (int)users.size(); i++) {
        UserId u = users[i];
        vector<pair<UserId, double>> sims;

        for (int j = 0; j < (int)users.size(); j++) {
            if (i == j) continue;

            UserId v = users[j];
            double sim = cosine_user_sim(user_items[u], user_items[v]);
            if (sim > 0) {
                sims.push_back({v, sim});
            }
        }

        sort(sims.begin(), sims.end(), [](auto& x, auto& y) {
            return x.second > y.second;
        });

        if ((int)sims.size() > top_k) {
            sims.resize(top_k);
        }

        user_sim_index[u] = move(sims);
    }
}

vector<pair<ItemId, double>> recall_by_usercf(UserId user, int top_k) {
    unordered_set<ItemId> seen(user_items[user].begin(), user_items[user].end());
    unordered_map<ItemId, double> candidate_score;

    for (auto [sim_user, sim_score] : user_sim_index[user]) {
        for (ItemId item : user_items[sim_user]) {
            if (seen.count(item)) continue;
            candidate_score[item] += sim_score;
        }
    }

    vector<pair<ItemId, double>> result(candidate_score.begin(), candidate_score.end());
    sort(result.begin(), result.end(), [](auto& a, auto& b) {
        return a.second > b.second;
    });

    if ((int)result.size() > top_k) {
        result.resize(top_k);
    }

    return result;
}
```

---

# 4. ItemCF：基于物品相似度的召回

## 4.1 核心直觉

ItemCF 的核心思想是：

```text
你喜欢过什么，就推荐和它相似的东西。
```

例如：

```text
用户看过《Rust 权威指南》
系统发现很多看过《Rust 权威指南》的人也看过《Tokio 教程》
于是推荐《Tokio 教程》
```

ItemCF 是工业推荐系统里非常常见的召回 baseline。

---

## 4.2 物品相似度公式

### 4.2.1 基础 ItemCF 公式

物品 \(i\) 和物品 \(j\) 的相似度：

\[
sim(i, j) = \frac{|U_i \cap U_j|}{\sqrt{|U_i| \cdot |U_j|}}
\]

含义：

```text
同时交互过 i 和 j 的用户越多，两个物品越相似；
但热门物品天然用户多，所以需要归一化。
```

---

### 4.2.2 带用户活跃度惩罚的 ItemCF

工业中经常使用 IUF，即 Inverse User Frequency。

原因是：

```text
有些用户什么都点。
他们同时点击 A 和 B，不一定说明 A 和 B 真相似。
```

改进公式：

\[
sim(i, j) =
\frac{
\sum_{u \in U_i \cap U_j} \frac{1}{\log(1 + |I_u|)}
}{
\sqrt{|U_i| \cdot |U_j|}
}
\]

其中：

\[
|I_u|
\]

表示用户 \(u\) 交互过多少物品。

用户越活跃，贡献越小。

---

### 4.2.3 带行为权重的 ItemCF

更工程化的公式可以写成：

\[
sim(i, j) =
\frac{
\sum_{u \in U_i \cap U_j}
\frac{r_{ui} r_{uj}}{\log(1 + |I_u|)}
}{
\sqrt{|U_i| \cdot |U_j|}
}
\]

这表示：

```text
如果用户只是点击两个物品，贡献较小；
如果用户收藏、购买两个物品，贡献更大；
如果用户过于活跃，贡献会被惩罚。
```

---

## 4.3 ItemCF 推荐打分公式

对用户 \(u\) 推荐候选物品 \(j\)：

\[
score(u, j) = \sum_{i \in I_u} sim(i, j) \cdot r_{ui}
\]

如果加入时间衰减：

\[
score(u, j) = \sum_{i \in I_u} sim(i, j) \cdot r_{ui} \cdot decay(t_{ui})
\]

通俗理解：

```text
用户最近交互过的物品越重要；
用户强行为交互过的物品越重要；
候选物品和用户历史物品越相似，分数越高。
```

---

## 4.4 ItemCF 具体计算例子

仍然使用这份行为数据：

| 用户 | 交互物品 |
|---|---|
| u1 | A, B, C |
| u2 | A, B |
| u3 | A, C |
| u4 | B, D |
| u5 | C, D |

先统计每个物品被哪些用户交互过：

| 物品 | 用户集合 |
|---|---|
| A | u1, u2, u3 |
| B | u1, u2, u4 |
| C | u1, u3, u5 |
| D | u4, u5 |

### 计算 sim(A, B)

\[
U_A = \{u1, u2, u3\}
\]

\[
U_B = \{u1, u2, u4\}
\]

\[
U_A \cap U_B = \{u1, u2\}
\]

\[
sim(A, B) = \frac{2}{\sqrt{3 \cdot 3}} = \frac{2}{3} \approx 0.667
\]

### 计算 sim(A, C)

\[
U_A \cap U_C = \{u1, u3\}
\]

\[
sim(A, C) = \frac{2}{\sqrt{3 \cdot 3}} = 0.667
\]

### 计算 sim(B, C)

\[
U_B \cap U_C = \{u1\}
\]

\[
sim(B, C) = \frac{1}{\sqrt{3 \cdot 3}} = 0.333
\]

### 计算 sim(C, D)

\[
U_C \cap U_D = \{u5\}
\]

\[
sim(C, D) = \frac{1}{\sqrt{3 \cdot 2}} \approx 0.408
\]

现在给用户 `u3` 推荐。

`u3` 已经交互过：

```text
A, C
```

候选物品 `B` 的分数：

\[
score(u3, B) = sim(A, B) + sim(C, B)
\]

\[
score(u3, B) = 0.667 + 0.333 = 1.000
\]

候选物品 `D` 的分数：

\[
score(u3, D) = sim(C, D)
\]

\[
score(u3, D) = 0.408
\]

所以 ItemCF 推荐顺序是：

```text
B > D
```

---

## 4.5 ItemCF 的工业应用

ItemCF 工业落地非常常见，尤其适合电商、内容、视频、文章、音乐等场景。

典型应用：

```text
看了手机 → 推荐手机壳、钢化膜、充电器
买了相机 → 推荐镜头、存储卡、三脚架
看了 Rust 教程 → 推荐 Tokio、Axum、异步编程内容
收藏某篇文章 → 推荐相似文章
播放某首歌 → 推荐相似歌曲
```

ItemCF 在线链路非常简单：

```text
用户最近行为 item
  ↓
查询 item -> similar_items 索引
  ↓
合并候选
  ↓
去重、过滤已读、过滤下架
  ↓
返回候选给排序层
```

工业中的存储形式通常是：

```text
item_id -> [(similar_item_id, score), ...]
```

可以放在：

```text
Redis
RocksDB
HBase
在线 KV
自研高性能索引服务
```

---

## 4.6 ItemCF 的优缺点

### 优点

```text
1. 线上延迟低：直接查 item->items 索引。
2. 可解释性强：因为你看过 A，所以推荐 B。
3. 工程实现简单：离线算相似度，线上查表。
4. 实时兴趣友好：用户刚点击一个 item，就可以触发相似 item 召回。
```

### 缺点

```text
1. 热门偏置：热门物品容易和大量物品相似。
2. 冷启动问题：新物品没有行为，难以被召回。
3. 兴趣窄化：容易围绕用户历史行为不断推荐相似内容。
4. 共现噪声：泛兴趣用户、刷子用户会制造低质量共现。
```

---

## 4.7 ItemCF C++ 风格伪代码：离线构建

```cpp
#include <bits/stdc++.h>
using namespace std;

using UserId = int;
using ItemId = int;

struct BehaviorItem {
    ItemId item_id;
    double weight;    // 行为权重，例如 click=1, buy=5
    int64_t ts;       // 行为时间
};

// user -> items
unordered_map<UserId, vector<BehaviorItem>> user_items;

// item -> user count
unordered_map<ItemId, int> item_user_count;

// pair(item_i, item_j) -> co-occurrence score
unordered_map<long long, double> co_score;

long long make_pair_key(ItemId a, ItemId b) {
    if (a > b) swap(a, b);
    return (static_cast<long long>(a) << 32) | static_cast<unsigned int>(b);
}

void build_itemcf_co_score() {
    for (auto& [user, items] : user_items) {
        // 工业中通常会截断超活跃用户，避免组合爆炸
        if (items.size() > 200) {
            sort(items.begin(), items.end(), [](const auto& x, const auto& y) {
                return x.ts > y.ts;
            });
            items.resize(200);
        }

        // IUF 用户活跃度惩罚
        double user_penalty = 1.0 / log(1.0 + items.size());

        for (const auto& x : items) {
            item_user_count[x.item_id]++;
        }

        for (int i = 0; i < (int)items.size(); i++) {
            for (int j = i + 1; j < (int)items.size(); j++) {
                ItemId item_i = items[i].item_id;
                ItemId item_j = items[j].item_id;

                double weight_i = items[i].weight;
                double weight_j = items[j].weight;

                long long key = make_pair_key(item_i, item_j);

                co_score[key] += user_penalty * weight_i * weight_j;
            }
        }
    }
}

double calc_itemcf_similarity(ItemId i, ItemId j, double co) {
    int cnt_i = item_user_count[i];
    int cnt_j = item_user_count[j];
    if (cnt_i == 0 || cnt_j == 0) return 0.0;

    return co / sqrt(1.0 * cnt_i * cnt_j);
}
```

---

## 4.8 ItemCF C++ 风格伪代码：线上召回

```cpp
#include <bits/stdc++.h>
using namespace std;

using ItemId = int;

struct UserAction {
    ItemId item_id;
    double action_weight;
    double time_decay;
};

// item -> [(similar_item, similarity_score)]
unordered_map<ItemId, vector<pair<ItemId, double>>> item_sim_index;

vector<pair<ItemId, double>> recall_by_itemcf(
    const vector<UserAction>& recent_actions,
    const unordered_set<ItemId>& seen_items,
    int top_k
) {
    unordered_map<ItemId, double> candidate_score;

    for (const auto& action : recent_actions) {
        ItemId trigger_item = action.item_id;

        auto it = item_sim_index.find(trigger_item);
        if (it == item_sim_index.end()) continue;

        for (auto [candidate, sim_score] : it->second) {
            if (seen_items.count(candidate)) continue;

            candidate_score[candidate] +=
                sim_score * action.action_weight * action.time_decay;
        }
    }

    vector<pair<ItemId, double>> result(candidate_score.begin(), candidate_score.end());

    sort(result.begin(), result.end(), [](auto& a, auto& b) {
        return a.second > b.second;
    });

    if ((int)result.size() > top_k) {
        result.resize(top_k);
    }

    return result;
}
```

---

# 5. Swing：工业增强版 ItemCF

## 5.1 核心直觉

Swing 可以理解为 ItemCF 的工业增强版。

普通 ItemCF 只关心：

```text
有多少用户同时交互过物品 i 和物品 j？
```

Swing 更关心：

```text
共同交互 i 和 j 的这些用户，两两之间是不是本来就高度相似？
```

它的核心思想是：

```text
如果两个用户本身只重合少量物品，但刚好都交互了 i 和 j，
那么 i 和 j 的相似关系更可信。

如果两个用户本来什么都一起点，
那么他们共同点了 i 和 j，信息量反而没那么大。
```

---

## 5.2 Swing 相似度公式

一种常见的 Swing 写法是：

\[
sim(i, j) =
\sum_{u, v \in U_i \cap U_j, u < v}
\frac{1}{\alpha + |I_u \cap I_v|}
\]

其中：

\[
U_i \cap U_j
\]

表示同时交互过物品 \(i\) 和物品 \(j\) 的用户集合。

\[
|I_u \cap I_v|
\]

表示用户 \(u\) 和用户 \(v\) 共同交互过多少物品。

\[
\alpha
\]

是平滑参数，用来避免分母过小。

注意：有些资料会用有序用户对 \((u, v), u \neq v\)，有些会用无序用户对 \((u, v), u < v\)。两种写法本质一致，只是差一个常数倍。工程中只要训练和线上使用一致即可。

---

## 5.3 Swing 公式怎么理解？

对于物品 \(i\) 和物品 \(j\)，假设有一批用户同时交互过它们。

Swing 会枚举这些用户对：

```text
(u1, u2), (u1, u3), (u2, u3), ...
```

每一对用户贡献：

\[
\frac{1}{\alpha + |I_u \cap I_v|}
\]

如果两个用户共同交互物品很少，说明这两个用户不是“什么都一起点”的泛兴趣用户，贡献更大。

如果两个用户共同交互物品很多，说明他们本身高度相似，或者可能都是重度用户，贡献更小。

---

## 5.4 Swing 具体计算例子

假设要计算物品 `A` 和 `B` 的 Swing 相似度。

同时交互过 A 和 B 的用户有：

```text
u1, u2, u3
```

这几个用户的完整行为如下：

| 用户 | 交互物品 |
|---|---|
| u1 | A, B, C, D |
| u2 | A, B, E |
| u3 | A, B |

设定：

\[
\alpha = 1
\]

### 用户对 u1 和 u2

\[
I_{u1} = \{A, B, C, D\}
\]

\[
I_{u2} = \{A, B, E\}
\]

\[
I_{u1} \cap I_{u2} = \{A, B\}
\]

\[
|I_{u1} \cap I_{u2}| = 2
\]

贡献为：

\[
\frac{1}{1 + 2} = \frac{1}{3}
\]

### 用户对 u1 和 u3

\[
I_{u1} \cap I_{u3} = \{A, B\}
\]

贡献为：

\[
\frac{1}{1 + 2} = \frac{1}{3}
\]

### 用户对 u2 和 u3

\[
I_{u2} \cap I_{u3} = \{A, B\}
\]

贡献为：

\[
\frac{1}{1 + 2} = \frac{1}{3}
\]

所以：

\[
sim(A, B) = \frac{1}{3} + \frac{1}{3} + \frac{1}{3} = 1
\]

再看一个噪声用户对的例子。

如果两个用户行为如下：

```text
u4 = {A, B, C, D, E, F, G, H, I, J}
u5 = {A, B, C, D, E, F, G, H, I, J, K}
```

它们共同交互了很多物品：

\[
|I_{u4} \cap I_{u5}| = 10
\]

在 \(\alpha = 1\) 时贡献：

\[
\frac{1}{1 + 10} = \frac{1}{11} \approx 0.091
\]

这说明：

```text
两个泛兴趣用户共同点击 A 和 B，并不能强烈说明 A 和 B 相似。
Swing 会降低这类用户对的贡献。
```

---

## 5.5 Swing 和 ItemCF 的区别

| 维度 | ItemCF | Swing |
|---|---|---|
| 计算对象 | item-item | item-item |
| 关注点 | 有多少用户共同交互两个物品 | 共同用户之间的用户对质量 |
| 抗噪能力 | 一般 | 更强 |
| 热门物品偏置 | 需要额外惩罚 | 天然有一定缓解 |
| 计算复杂度 | 较低 | 较高 |
| 工业使用 | 常用 baseline | 常用增强召回 |

一句话理解：

```text
ItemCF：共同用户越多，物品越相似。
Swing：高质量用户对越多，物品越相似。
```

---

## 5.6 Swing 推荐打分公式

Swing 最终算出来的仍然是 item-item 相似度。

所以线上召回公式和 ItemCF 类似：

\[
score(u, j) = \sum_{i \in I_u} sim_{swing}(i, j) \cdot r_{ui} \cdot decay(t_{ui})
\]

工程链路也是：

```text
离线计算 swing item-item 相似度
  ↓
保留每个 item 的 TopK 相似 item
  ↓
写入 KV / 索引服务
  ↓
线上根据用户最近行为查 item->items
```

---

## 5.7 Swing 的工业应用

Swing 常用于对 ItemCF 共现关系进行去噪，尤其适合：

```text
电商商品推荐
内容推荐
短视频推荐
相似商品召回
相关推荐
看了又看
买了又买
```

它主要解决的问题是：

```text
热门商品带来的虚假共现
泛兴趣用户带来的低质量共现
刷子用户或异常用户带来的噪声
高活跃用户让大量物品互相产生弱相关
```

在真实工业系统中，Swing 通常不会单独作为唯一召回源，而是和其他召回源组合：

```text
ItemCF 召回
Swing 召回
Embedding 向量召回
双塔召回
热门召回
新品召回
内容标签召回
运营召回
```

---

## 5.8 Swing C++ 风格伪代码

下面是一个简化版 Swing 伪代码，主要用于理解公式。真实工业中不会直接这么暴力计算。

```cpp
#include <bits/stdc++.h>
using namespace std;

using UserId = int;
using ItemId = int;

// user -> item set
unordered_map<UserId, unordered_set<ItemId>> user_item_set;

// item -> users
unordered_map<ItemId, vector<UserId>> item_users;

// item pair -> swing score
unordered_map<long long, double> swing_score;

long long make_pair_key(ItemId a, ItemId b) {
    if (a > b) swap(a, b);
    return (static_cast<long long>(a) << 32) | static_cast<unsigned int>(b);
}

int common_item_count(UserId u, UserId v) {
    const auto& a = user_item_set[u];
    const auto& b = user_item_set[v];

    int cnt = 0;

    if (a.size() < b.size()) {
        for (ItemId item : a) {
            if (b.count(item)) cnt++;
        }
    } else {
        for (ItemId item : b) {
            if (a.count(item)) cnt++;
        }
    }

    return cnt;
}

vector<UserId> intersect_users(const vector<UserId>& a, const vector<UserId>& b) {
    unordered_set<UserId> set_a(a.begin(), a.end());
    vector<UserId> common;

    for (UserId u : b) {
        if (set_a.count(u)) {
            common.push_back(u);
        }
    }

    return common;
}

void build_swing_index(double alpha) {
    vector<ItemId> items;
    for (auto& [item, users] : item_users) {
        items.push_back(item);
    }

    for (int x = 0; x < (int)items.size(); x++) {
        for (int y = x + 1; y < (int)items.size(); y++) {
            ItemId item_i = items[x];
            ItemId item_j = items[y];

            vector<UserId> common_users = intersect_users(
                item_users[item_i],
                item_users[item_j]
            );

            // 少于两个共同用户，无法形成用户对
            if (common_users.size() < 2) continue;

            double score = 0.0;

            for (int a = 0; a < (int)common_users.size(); a++) {
                for (int b = a + 1; b < (int)common_users.size(); b++) {
                    UserId u = common_users[a];
                    UserId v = common_users[b];

                    int overlap = common_item_count(u, v);
                    score += 1.0 / (alpha + overlap);
                }
            }

            if (score > 0) {
                swing_score[make_pair_key(item_i, item_j)] = score;
            }
        }
    }
}
```

真实工业实现中需要做剪枝：

```text
1. 过滤超活跃用户。
2. 过滤超热门物品。
3. 每个用户只保留最近 N 个高质量行为。
4. 每个物品最多保留 M 个用户。
5. 每个 item 只保留 TopK 相似 item。
6. 使用 Spark / Flink / MapReduce 做离线或近实时计算。
```

---

# 6. 三种方式的综合对比

| 方法 | 相似对象 | 核心思想 | 工业常用程度 | 优点 | 缺点 |
|---|---|---|---|---|---|
| UserCF | user-user | 找相似用户，相似用户喜欢什么就推荐什么 | 中低 | 直观，适合社交和圈层 | 用户规模大，兴趣变化快，在线链路重 |
| ItemCF | item-item | 用户喜欢过 A，就推荐和 A 相似的 B | 高 | 简单、稳定、可解释、低延迟 | 热门偏置、冷启动、容易兴趣窄化 |
| Swing | item-item | 用高质量用户对判断物品相似 | 高 | 去噪能力强，适合工业共现召回 | 计算更重，工程剪枝复杂 |

---

# 7. 工业系统中的完整落地链路

以 ItemCF / Swing 为例，完整链路通常是：

```text
1. 收集行为日志
   user_id, item_id, action_type, timestamp, scene, device, request_id

2. 行为清洗
   去掉爬虫、刷子、异常用户、异常 item、低质量行为

3. 行为加权
   click=1, collect=3, cart=4, buy=5, negative=-5

4. 时间衰减
   新行为权重大，旧行为权重小

5. 构建倒排索引
   user -> items
   item -> users

6. 计算相似度
   ItemCF / Swing / 其他共现算法

7. TopK 截断
   每个 item 只保留最相似的 K 个 item

8. 写入在线存储
   Redis / RocksDB / HBase / KV 服务

9. 线上召回
   根据用户最近行为查询 item->similar_items

10. 业务过滤
   已读、已买、下架、库存不足、黑名单、地域不可见、低质量内容

11. 多路召回融合
   ItemCF + Swing + 向量召回 + 热门召回 + 内容召回

12. 进入粗排 / 精排 / 重排
```

---

# 8. 工业实现中的关键注意事项

## 8.1 不要让超活跃用户污染相似度

如果一个用户一天点击几千个物品，他会让大量物品两两共现。

解决方式：

```text
限制每个用户最大行为数
使用 IUF 惩罚
过滤异常用户
只保留高价值行为
```

---

## 8.2 不要让超热门物品污染相似度

热门物品容易和所有东西都相似。

解决方式：

```text
热门物品降权
过滤过热 item
相似度归一化
按类目分桶计算
```

---

## 8.3 一定要做 TopK 截断

全量 item-item pair 数量非常大。

如果每个 item 都和大量 item 建立相似关系，存储和查询都会爆炸。

工业中一般会保存：

```text
每个 item 的 Top100 / Top200 / Top500 相似 item
```

---

## 8.4 召回结果不能直接展示

召回只是生成候选，不是最终推荐。

召回结果还要经过：

```text
粗排
精排
重排
过滤
打散
多样性控制
业务规则
```

---

## 8.5 协同过滤解决不了所有问题

ItemCF / Swing / UserCF 都依赖用户行为。

如果新物品没有行为，会出现冷启动。

所以工业系统中还需要：

```text
内容召回
标签召回
Embedding 向量召回
双塔召回
热门召回
新品召回
探索流量
```

---

# 9. 面试或学习时的记忆方式

## 9.1 UserCF

公式：

\[
sim(u, v) = \frac{|I_u \cap I_v|}{\sqrt{|I_u| \cdot |I_v|}}
\]

记忆：

```text
找人。
和你相似的人喜欢什么，就推荐什么。
```

工业理解：

```text
直接使用较少，更多变成相似人群、Lookalike、用户聚类、社交扩散。
```

---

## 9.2 ItemCF

公式：

\[
sim(i, j) = \frac{|U_i \cap U_j|}{\sqrt{|U_i| \cdot |U_j|}}
\]

记忆：

```text
找物。
你喜欢过 A，就推荐和 A 相似的 B。
```

工业理解：

```text
低延迟、高可解释、工程简单，是召回层常用 baseline。
```

---

## 9.3 Swing

公式：

\[
sim(i, j) =
\sum_{u, v \in U_i \cap U_j, u < v}
\frac{1}{\alpha + |I_u \cap I_v|}
\]

记忆：

```text
更高质量的 ItemCF。
不是只看共同用户数量，而是看共同用户对的质量。
```

工业理解：

```text
用于减少热门物品、泛兴趣用户、异常用户带来的低质量共现。
```

---

# 10. 最终总结

三种方法的本质区别：

```text
UserCF：用户相似 → 扩散相似用户喜欢的物品。
ItemCF：物品相似 → 根据用户历史物品扩散相似物品。
Swing：物品相似 → 但用用户对质量对共现关系去噪。
```

从工业视角看：

```text
ItemCF 是协同过滤召回的基础款。
Swing 是 ItemCF 的工业增强款。
UserCF 是相似人群和圈层召回的基础思想。
```

在真实系统中，它们通常不是互斥关系，而是多路召回的一部分：

```text
ItemCF 负责稳定可解释召回。
Swing 负责更干净的相似物品召回。
UserCF/Lookalike 负责人群扩散。
向量召回负责语义泛化。
热门/新品/探索召回负责覆盖和冷启动。
```

