# 推荐系统中的行为序列、DIN 与 SIM

> 本文分为两部分：
>
> 1. 用生动、直观的方式理解“行为序列”
> 2. 用更专业的方式理解 DIN 注意力模型与 SIM 长序列建模

---

## 目录

- [一、什么是行为序列](#一什么是行为序列)
- [二、为什么推荐系统需要行为序列](#二为什么推荐系统需要行为序列)
- [三、行为序列到底记录什么](#三行为序列到底记录什么)
- [四、行为序列中的几个关键特征](#四行为序列中的几个关键特征)
- [五、工业系统如何使用行为序列](#五工业系统如何使用行为序列)
- [六、DIN：候选相关的兴趣激活](#六din候选相关的兴趣激活)
- [七、DIN 的结构与计算流程](#七din-的结构与计算流程)
- [八、SIM：面向超长行为序列的搜索式建模](#八sim面向超长行为序列的搜索式建模)
- [九、SIM 的 GSU 与 ESU](#九sim-的-gsu-与-esu)
- [十、DIN 与 SIM 的关系和区别](#十din-与-sim-的关系和区别)
- [十一、工业服务端视角](#十一工业服务端视角)
- [十二、总结](#十二总结)

---

# 一、什么是行为序列

在推荐系统中，**行为序列**是指：

> 用户过去在平台上产生的一系列行为，按照时间顺序排列后形成的记录。

例如，一个用户最近依次进行了这些操作：

```text
10:01 搜索“机械键盘”
10:03 点击“红轴键盘”
10:05 收藏“Keychron K8”
10:08 搜索“静音键盘”
10:10 浏览“茶轴键盘”
10:12 将“茶轴键盘”加入购物车
```

将这些行为按照时间排列，就得到了一条行为序列：

```text
机械键盘
   ↓
红轴键盘
   ↓
Keychron K8
   ↓
静音键盘
   ↓
茶轴键盘
   ↓
加入购物车
```

这里记录的不只是“用户看过哪些物品”，还记录了：

- 用户先做了什么，后做了什么
- 用户最近正在关注什么
- 用户对哪些内容兴趣更强
- 用户的兴趣是否正在发生变化
- 用户当前可能处于什么任务阶段

---

# 二、为什么推荐系统需要行为序列

## 2.1 行为序列像用户留下的脚印

可以把推荐系统想象成一家很大的商场。

一个用户进入商场之后：

```text
先看手机
  ↓
再看手机壳
  ↓
接着看充电器
  ↓
最后看屏幕保护膜
```

即使工作人员不知道这个用户的年龄、职业和收入，也可以根据这串“脚印”猜测：

> 用户可能正在购买手机，或者已经买了手机，现在正在寻找配件。

行为序列就是用户在平台中留下的一串数字脚印。

推荐系统沿着这些脚印，尝试回答：

```text
用户过去做了什么？
用户现在想做什么？
用户下一步可能做什么？
```

---

## 2.2 行为序列像用户和平台的一段无声对话

用户通常不会直接对推荐系统说：

> 我最近正在学习 Rust 异步编程，下一步想看 Tokio Runtime。

但用户可能连续产生这些行为：

```text
点击 Future Trait
  ↓
观看 Waker 教程
  ↓
收藏 Executor 文章
  ↓
搜索 Tokio
```

这些行为连起来，就像用户在“无声地表达”：

> 我当前对 Rust 异步体系有持续兴趣。

所以可以这样理解：

```text
单个行为    = 一个词
行为序列    = 一句话
长期画像    = 一个人的长期性格
```

单独看到一个词，往往无法判断真实含义；看到完整句子，才能理解上下文。

---

## 2.3 用户画像和行为序列解决的问题不同

用户画像可能描述：

```text
用户长期喜欢：
- 编程
- 数码产品
- 游戏
- 机械键盘
```

但用户最近的行为可能是：

```text
日本机票
  ↓
东京酒店
  ↓
东京地铁
  ↓
日本电话卡
```

此时用户当前最需要的，很可能不是编程课程，而是：

- 东京旅游攻略
- 日本交通卡
- 行李箱
- 旅行保险
- 景点门票

因此：

| 信息类型 | 回答的问题 |
|---|---|
| 用户画像 | 这个人长期喜欢什么 |
| 行为序列 | 这个人最近正在做什么 |
| 实时上下文 | 这个人现在处于什么环境 |

工业推荐系统通常会同时使用：

```text
长期兴趣 + 短期兴趣 + 当前上下文
```

---

# 三、行为序列到底记录什么

行为序列中的一个元素，通常不只是一个 `item_id`。

一个更完整的行为记录可能包含：

```cpp
struct Behavior {
    int64_t item_id;
    int32_t category_id;
    ActionType action;
    int64_t timestamp;
    float stay_time;
    bool completed;
    int32_t source_page;
};
```

例如：

```text
item_id: 10086
action: click
category: 机械键盘
timestamp: 10:03
stay_time: 32 秒
source_page: 搜索页
```

因此，行为序列实际表达的是：

```text
用户在什么时间
对什么内容
做了什么行为
行为强度如何
行为发生在什么场景
```

---

# 四、行为序列中的几个关键特征

## 4.1 顺序很重要

假设两个用户都看过：

```text
健身入门
减脂饮食
增肌训练
```

但顺序不同。

用户 A：

```text
健身入门 → 减脂饮食 → 增肌训练
```

用户 B：

```text
增肌训练 → 减脂饮食 → 健身入门
```

虽然两人的行为集合相同，但含义可能不同。

用户 A 可能正在逐步深入学习：

```text
入门 → 饮食 → 系统训练
```

用户 B 最近重新回到“健身入门”，可能说明：

- 之前没有真正掌握
- 兴趣发生了变化
- 想从基础重新开始

所以：

> 行为集合只说明“做过什么”，行为序列还说明“按照什么顺序做”。

---

## 4.2 行为强度不同

不同类型的行为，代表的兴趣强度通常不同。

一个简单的理解方式是：

```text
曝光 < 点击 < 长时间观看 < 收藏 < 加购 < 购买
```

例如：

```text
序列 A：
曝光键盘 → 曝光鼠标 → 曝光耳机

序列 B：
点击键盘 → 收藏键盘 → 加购键盘
```

序列 B 所表达的购买意图明显更强。

工业系统中，通常不会简单认为：

```text
点击 = 喜欢
```

还需要结合：

- 停留时长
- 是否快速退出
- 是否看完
- 是否重复观看
- 是否收藏
- 是否加购
- 是否购买
- 后续是否继续浏览同类内容

---

## 4.3 时间远近不同

一般来说，越近的行为越能反映用户当前意图。

例如：

```text
一年前：看过摄影教程
一个月前：看过健身视频
十分钟前：连续看了四个 Rust 视频
```

当前推荐时，十分钟前的 Rust 行为往往更重要。

可以直观理解为：

```text
行为影响力 ≈ 行为强度 × 时间新鲜度
```

但是，时间衰减并不是绝对规则。

例如：

- 每年春节都会购买年货
- 每年开学季都会购买学习用品
- 每周末固定观看足球比赛

这些周期性行为，即使时间较远，也可能具有很高价值。

---

## 4.4 用户会发生兴趣漂移

用户的兴趣不是永久不变的标签。

例如，用户过去长期关注：

```text
篮球 → NBA → 球鞋 → 篮球训练
```

最近却变成：

```text
跑步 → 马拉松 → 跑鞋 → 心率训练
```

这说明用户可能正在从篮球兴趣迁移到跑步兴趣。

如果系统简单地把所有历史行为平均处理，可能得到：

```text
篮球兴趣 50%
跑步兴趣 50%
```

但这无法体现：

```text
旧兴趣正在减弱
新兴趣正在增强
```

行为序列的价值之一，就是发现这种兴趣漂移。

---

## 4.5 用户通常具有多个兴趣

一个程序员的行为序列可能是：

```text
Rust 教程
  ↓
Tokio 异步
  ↓
台北美食
  ↓
火锅推荐
  ↓
日本旅游
  ↓
Axum Web 开发
```

这里至少包含三个兴趣簇：

```text
兴趣 1：Rust / 服务端开发
兴趣 2：美食 / 火锅
兴趣 3：日本旅游
```

推荐系统不能简单地将这些兴趣平均成一个模糊向量。

更合理的做法是：

> 面对不同候选物品，激活不同的兴趣。

这正是 DIN 模型的核心出发点。

---

# 五、工业系统如何使用行为序列

一个推荐系统通常不会只维护一条行为序列，而会维护多种序列：

```text
最近点击序列
最近观看序列
最近收藏序列
最近购买序列
最近搜索词序列
曝光未点击序列
```

也可能按照时间范围拆分：

```text
最近 10 分钟行为
最近 24 小时行为
最近 7 天行为
长期历史行为
```

不同序列表达不同信息：

| 序列 | 主要含义 |
|---|---|
| 点击序列 | 用户对什么产生过兴趣 |
| 购买序列 | 用户实际消费过什么 |
| 搜索序列 | 用户主动需要什么 |
| 未点击序列 | 用户可能不喜欢什么 |
| 短期序列 | 用户当前意图 |
| 长期序列 | 用户稳定偏好 |

---

## 5.1 行为序列的一般处理流程

```text
用户产生行为
   ↓
日志系统记录
   ↓
按时间整理行为
   ↓
过滤无效和异常行为
   ↓
截断、采样或检索
   ↓
行为转换为 Embedding
   ↓
序列模型提取兴趣
   ↓
与候选物品进行匹配
   ↓
输出 CTR、CVR、时长等预估分数
```

对应的简化伪代码：

```cpp
std::vector<Behavior> build_behavior_sequence(UserId user_id) {
    auto sequence = load_recent_behaviors(user_id);

    std::sort(sequence.begin(), sequence.end(),
              [](const Behavior& lhs, const Behavior& rhs) {
                  return lhs.timestamp < rhs.timestamp;
              });

    sequence = remove_invalid_behaviors(sequence);
    sequence = deduplicate_behaviors(sequence);
    sequence = keep_recent_n(sequence, 200);

    return sequence;
}
```

---

# 六、DIN：候选相关的兴趣激活

DIN 全称：

> **Deep Interest Network，深度兴趣网络**

DIN 要解决的核心问题是：

> 用户具有多个兴趣，面对不同候选物品时，真正有用的历史兴趣不同。

传统做法可能直接把历史行为做平均：

```text
用户兴趣向量 =
    篮球兴趣
  + Rust 兴趣
  + 咖啡兴趣
  + 旅游兴趣
```

这样得到的是一个固定的用户向量。

问题在于：

```text
推荐篮球鞋时使用这个向量
推荐 Tokio 课程时也使用这个向量
推荐咖啡机时仍然使用这个向量
```

这会把大量无关兴趣混在一起。

DIN 的解决方法是：

> 使用当前候选物品作为查询，对历史行为进行候选相关的注意力加权。

---

## 6.1 DIN 的直观例子

用户历史行为：

```text
篮球鞋
Rust 所有权
Tokio 教程
咖啡豆
Axum Web 开发
```

当前候选物品：

```text
《Tokio 异步编程实战》
```

DIN 会学习每个历史行为对当前候选物品的重要程度：

| 历史行为 | 注意力权重 |
|---|---:|
| 篮球鞋 | 0.01 |
| Rust 所有权 | 0.65 |
| Tokio 教程 | 0.98 |
| 咖啡豆 | 0.02 |
| Axum Web 开发 | 0.82 |

最终用户兴趣表示近似为：

```text
0.01 × 篮球鞋
+ 0.65 × Rust 所有权
+ 0.98 × Tokio 教程
+ 0.02 × 咖啡豆
+ 0.82 × Axum Web 开发
```

此时得到的不是“用户的永久兴趣”，而是：

> 用户面对 Tokio 候选物品时，被激活出来的局部兴趣。

---

# 七、DIN 的结构与计算流程

## 7.1 核心结构

DIN 的典型流程：

```text
用户历史行为序列
        ↓
行为 Embedding
        ↓
候选物品 Embedding
        ↓
Local Activation Unit
        ↓
计算每条历史行为的权重
        ↓
加权聚合得到候选相关兴趣
        ↓
与用户画像、候选物品、上下文特征拼接
        ↓
MLP
        ↓
CTR / CVR 等预测结果
```

---

## 7.2 注意力计算

设用户历史行为向量为：

```text
h1, h2, h3, ..., hn
```

当前候选物品向量为：

```text
q
```

DIN 会计算：

```text
a_i = Attention(h_i, q)
```

然后得到候选相关的用户兴趣表示：

\[
u(q)=\sum_{i=1}^{n}a_i h_i
\]

这里的关键是：

```text
q 不同
  ↓
注意力权重不同
  ↓
用户兴趣表示也不同
```

因此 DIN 不是为每个用户生成唯一固定的兴趣向量，而是生成：

```text
用户对于当前候选物品的兴趣向量
```

---

## 7.3 Local Activation Unit

DIN 的注意力通常不是简单使用余弦相似度，而是通过一个小型神经网络学习相关性。

常见输入可以包括：

```text
历史行为向量 h
候选物品向量 q
h - q
h × q
```

其中：

- `h - q` 表示差异关系
- `h × q` 表示逐维交互关系

伪代码：

```cpp
float activation_unit(
    const Vector& behavior,
    const Vector& candidate)
{
    Vector input = concat(
        behavior,
        candidate,
        behavior - candidate,
        behavior * candidate
    );

    return attention_mlp.forward(input);
}
```

这种方式能够学习比简单向量相似度更复杂的关系，例如：

```text
手机 → 手机壳
显示器 → 显示器支架
相机 → 存储卡
```

这些物品未必在内容上高度相似，但在用户行为和购买链路中高度相关。

因此，DIN 学习的不是单纯的“相似性”，而是：

> 某条历史行为对当前候选物品的点击或转化预测是否有帮助。

---

## 7.4 DIN 的伪代码

```cpp
Vector build_din_interest(
    const std::vector<Vector>& behavior_embeddings,
    const Vector& candidate_embedding)
{
    Vector interest(candidate_embedding.size(), 0.0F);

    for (const auto& behavior : behavior_embeddings) {
        float weight =
            activation_unit(behavior, candidate_embedding);

        interest += weight * behavior;
    }

    return interest;
}
```

排序模型：

```cpp
float predict_ctr(
    const UserFeatures& user,
    const ItemFeatures& candidate,
    const ContextFeatures& context)
{
    Vector interest = build_din_interest(
        user.behavior_embeddings,
        candidate.embedding
    );

    Vector model_input = concat(
        user.profile_embedding,
        interest,
        candidate.embedding,
        context.embedding
    );

    return ranking_mlp.forward(model_input);
}
```

---

## 7.5 DIN 的优势

- 能够表达用户的多兴趣
- 面对不同候选物品，动态生成不同兴趣表示
- 比简单平均池化更精准
- 对 CTR 预估等排序任务非常自然
- 容易接入已有的 Embedding + MLP 排序框架

---

## 7.6 DIN 的局限

### 1. 超长序列计算成本高

如果历史序列长度为 `N`，每个候选物品都需要与 `N` 条历史行为计算注意力。

假设：

```text
候选物品数：1000
历史行为数：50000
```

则可能产生大量候选—历史交互计算。

### 2. 长序列噪声多

用户十年前浏览的大量物品，通常与当前候选物品关系很弱。

直接把全部行为送入 DIN，会带来：

- 计算浪费
- 无关噪声
- 内存占用增加
- 在线延迟上升

### 3. 原始 DIN 不重点建模顺序演化

原始 DIN 的重点是：

```text
候选相关兴趣激活
```

而不是重点建模：

```text
行为 A 之后出现行为 B
行为 B 之后又演化成行为 C
```

所以原始 DIN 更接近对历史行为的候选相关加权，而不是完整的时序演化模型。

---

# 八、SIM：面向超长行为序列的搜索式建模

SIM 全称：

> **Search-based Interest Model，基于搜索的兴趣模型**

SIM 要解决的问题是：

> 当用户历史行为非常长时，如何低成本地从数千、数万条行为中找到与当前候选物品最相关的部分。

它的核心思想是：

```text
先检索，再精细建模
```

完整流程：

```text
超长历史行为序列
        ↓
GSU：快速搜索相关行为
        ↓
候选相关子序列
        ↓
ESU：精细兴趣建模
        ↓
长期兴趣向量
        ↓
排序模型
```

这与推荐系统整体的多阶段架构非常相似：

```text
海量物品
  ↓ 召回
少量候选
  ↓ 精排
最终结果
```

SIM 将这种思想应用到了用户行为序列内部：

```text
海量历史行为
  ↓ 行为召回
相关历史子序列
  ↓ 行为精排
精确兴趣表示
```

---

# 九、SIM 的 GSU 与 ESU

## 9.1 GSU：General Search Unit

GSU 的任务是：

> 从超长历史序列中，快速检索出与当前候选物品相关的 Top-K 行为。

例如：

```text
50000 条历史行为
        ↓ GSU
100 条相关行为
```

GSU 主要追求：

- 检索速度快
- 召回覆盖高
- 计算成本低
- 线上延迟稳定

它不要求对每条行为进行特别复杂的判断。

---

## 9.2 Hard Search

Hard Search 使用明确规则进行检索。

例如，候选物品是：

```text
机械键盘
category_id = 电脑外设
```

则从用户历史中检索：

```text
category_id = 电脑外设
```

对应的历史行为可能包括：

```text
显示器
机械键盘
鼠标
键盘轴体
显示器支架
```

常见索引条件：

- 相同类别
- 相同品牌
- 相同店铺
- 相同主题
- 相同标签
- 相同业务场景

可以使用倒排索引：

```text
category_id = 电脑外设
    ↓
[behavior_17, behavior_81, behavior_305, ...]
```

伪代码：

```cpp
std::vector<Behavior> hard_search(
    UserId user_id,
    CategoryId category_id,
    std::size_t top_k)
{
    auto behavior_ids =
        user_category_index.lookup(user_id, category_id);

    return load_latest_behaviors(behavior_ids, top_k);
}
```

优点：

- 速度快
- 规则清晰
- 工程实现简单
- 容易控制延迟

缺点：

- 依赖人工定义的字段
- 跨类别关系容易遗漏
- 表达能力有限

例如：

```text
手机 → 手机壳
显示器 → 屏幕挂灯
相机 → 存储卡
```

这些关系不一定属于同一类别。

---

## 9.3 Soft Search

Soft Search 使用 Embedding 进行向量检索。

流程：

```text
候选物品 Embedding
       ↓
与用户长期行为 Embedding 做近邻检索
       ↓
得到 Top-K 相关行为
```

可以使用：

- 最大内积检索
- 余弦相似度
- HNSW
- Faiss
- ScaNN
- 其他 ANN 索引

伪代码：

```cpp
std::vector<BehaviorId> soft_search(
    UserId user_id,
    const Vector& candidate_embedding,
    std::size_t top_k)
{
    auto& index = user_behavior_index.at(user_id);

    return index.search(
        candidate_embedding,
        top_k
    );
}
```

Soft Search 可以发现更隐式的关系：

```text
Tokio 教程 ↔ Rust 异步开发
显示器 ↔ 屏幕挂灯
跑步 ↔ 马拉松装备
手机 ↔ 手机壳
```

优点：

- 表达能力更强
- 可以发现跨类别和语义关联
- 不完全依赖人工规则

缺点：

- 需要维护行为 Embedding
- 需要维护向量索引
- 存储与更新成本更高
- 在线检索链路更复杂

---

## 9.4 ESU：Exact Search Unit

ESU 的任务是：

> 对 GSU 检索出的少量相关行为进行更复杂、更精确的兴趣建模。

GSU 已经将序列从：

```text
50000 条
```

缩短为：

```text
100 条
```

此时 ESU 可以使用更复杂的结构，例如：

- DIN 式注意力
- Multi-Head Attention
- Transformer
- DIEN 式兴趣演化
- MLP 交互网络

例如 GSU 找到：

```text
Rust 所有权
Rust 生命周期
Tokio 教程
Axum 开发
C++ 协程
Linux epoll
```

当前候选物品：

```text
Tokio Runtime 源码解析
```

ESU 会进一步学习：

| 历史行为 | 精细权重 |
|---|---:|
| Rust 所有权 | 0.35 |
| Rust 生命周期 | 0.52 |
| Tokio 教程 | 0.98 |
| Axum 开发 | 0.72 |
| C++ 协程 | 0.51 |
| Linux epoll | 0.67 |

最后得到长期兴趣向量。

---

## 9.5 SIM 的完整伪代码

```cpp
float predict_ctr_with_sim(
    const User& user,
    const Item& candidate,
    const Context& context)
{
    // 第一阶段：GSU
    std::vector<Behavior> related_behaviors =
        general_search_unit.retrieve(
            user.id,
            candidate,
            100
        );

    // 第二阶段：ESU
    Vector long_term_interest =
        exact_search_unit.encode(
            related_behaviors,
            candidate.embedding
        );

    // 短期行为可以单独建模
    Vector short_term_interest =
        short_sequence_model.encode(
            user.recent_behaviors,
            candidate.embedding
        );

    Vector input = concat(
        user.profile_embedding,
        long_term_interest,
        short_term_interest,
        candidate.embedding,
        context.embedding
    );

    return ranking_mlp.forward(input);
}
```

---

# 十、DIN 与 SIM 的关系和区别

## 10.1 最核心的区别

DIN 解决：

> 已经有一条行为序列时，当前候选物品应该重点关注哪些历史行为？

SIM 解决：

> 当行为序列太长时，如何先从海量历史中快速找到值得进一步分析的行为？

可以简化为：

```text
DIN：怎么精细看历史
SIM：历史太多时，先看哪部分
```

---

## 10.2 SIM 并不是 DIN 的完全替代品

SIM 的 ESU 阶段，可以继续使用 DIN 式注意力。

因此一种常见组合是：

```text
超长历史序列
   ↓
SIM-GSU 检索 Top-K
   ↓
DIN 式注意力建模
   ↓
候选相关兴趣向量
```

也就是说：

> SIM 可以在外层负责缩小序列范围，DIN 可以在内层负责精细计算权重。

---

## 10.3 对比表

| 对比项 | DIN | SIM |
|---|---|---|
| 全称 | Deep Interest Network | Search-based Interest Model |
| 核心目标 | 候选相关兴趣激活 | 超长行为序列高效建模 |
| 主要输入 | 中短行为序列 | 数千到数万级长期行为 |
| 核心结构 | Local Activation Unit | GSU + ESU |
| 是否处理完整序列 | 通常是 | 先检索子序列 |
| 主要计算 | 候选与历史行为逐条交互 | 检索 + 精细建模 |
| 工业类比 | 对历史行为做精排 | 行为召回 + 行为精排 |
| 优势 | 多兴趣表达准确 | 支持更长历史、降低噪声 |
| 局限 | 长序列计算成本高 | 系统结构和索引维护复杂 |
| 二者关系 | 可作为 ESU 的建模方式 | 可在外层包裹 DIN |

---

# 十一、工业服务端视角

从模型视角看，DIN 和 SIM 是兴趣建模方法。

从 C++ / Rust 服务端视角看，还需要解决大量在线工程问题。

---

## 11.1 行为存储

需要考虑：

- 行为写入吞吐量
- 用户维度分片
- 最近行为和长期行为分层存储
- 时间排序
- 去重
- 过期清理
- 热点用户
- 多端行为合并

可能的存储形式：

```text
user_id
   ↓
[behavior_1, behavior_2, ..., behavior_n]
```

短期行为可能放在：

- Redis
- 内存 KV
- 高性能缓存

长期行为可能放在：

- 分布式 KV
- 列式存储
- 特征平台
- 离线数仓

---

## 11.2 行为索引

SIM 的 GSU 需要支持快速检索。

Hard Search 可能维护：

```text
(user_id, category_id) → behavior_list
(user_id, brand_id)    → behavior_list
(user_id, tag_id)      → behavior_list
```

Soft Search 可能维护：

```text
user_id → behavior vector index
```

需要解决：

- 索引更新
- 新行为实时写入
- 旧行为淘汰
- 向量版本一致性
- 模型版本切换
- 索引内存占用
- 查询延迟

---

## 11.3 序列截断与采样

即使使用 SIM，也通常需要限制序列规模。

常见策略：

```text
保留最近 N 条
按行为类型分别采样
高价值行为优先
购买行为长期保留
曝光行为大量下采样
相同物品去重
按时间窗口分桶
```

例如：

```text
最近点击：100 条
最近收藏：50 条
最近购买：100 条
长期行为：通过 GSU 检索 200 条
```

---

## 11.4 在线推理优化

在线排序要求低延迟。

常见优化方式：

- 批量查询 Embedding
- 候选物品批量推理
- 行为 Embedding 缓存
- GSU 结果缓存
- 相同类别候选复用检索结果
- SIMD 优化
- 多线程并行
- 模型量化
- TensorRT / ONNX Runtime
- 控制 Top-K 大小
- 降级策略

例如，同一请求中的多个候选物品可能属于同一类别，可以复用 Hard Search 结果：

```cpp
std::unordered_map<CategoryId, std::vector<Behavior>>
    category_behavior_cache;
```

---

## 11.5 降级策略

当索引、特征服务或模型服务异常时，可以降级：

```text
SIM 长序列兴趣不可用
        ↓
使用最近 50 条短序列
        ↓
仍不可用
        ↓
使用静态用户画像
        ↓
仍不可用
        ↓
热门内容兜底
```

工业系统关注的不只是模型效果，还包括：

```text
效果
延迟
吞吐量
稳定性
资源成本
可降级性
```

---

# 十二、总结

## 12.1 行为序列

行为序列是用户按照时间留下的一串数字脚印。

它帮助推荐系统理解：

- 用户过去做了什么
- 用户当前正在关注什么
- 用户的兴趣如何变化
- 用户具有哪些不同兴趣
- 用户下一步可能做什么

一句话概括：

> 行为序列就是用户与推荐系统之间的一段无声对话。

---

## 12.2 DIN

DIN 的核心是：

> 面对不同候选物品，使用注意力机制激活不同的历史兴趣。

记忆方式：

```text
候选物品不同
   ↓
关注的历史行为不同
   ↓
得到的用户兴趣向量不同
```

---

## 12.3 SIM

SIM 的核心是：

> 面对超长行为序列，先检索相关历史，再对相关子序列进行精细建模。

记忆方式：

```text
数万条长期行为
   ↓ GSU
少量相关行为
   ↓ ESU
候选相关兴趣向量
```

---

## 12.4 DIN 与 SIM 的最终关系

```text
DIN：
解决“怎样精细分析历史行为”

SIM：
解决“历史太长时，先分析哪部分”

典型组合：
SIM-GSU 检索 + DIN 式注意力建模
```

从工业推荐系统的角度看：

> DIN 更偏模型层的兴趣激活，SIM 更体现模型、检索和在线服务的联合设计。
