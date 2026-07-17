# 双塔模型：结构与训练方法

## 1. 模型结构

双塔模型将用户和物品分别编码成向量：

```text
用户特征 → 用户塔 → 用户向量 p_u
物品特征 → 物品塔 → 物品向量 q_i
```

数学表示：

\[
p_u = f_\theta(x_u)
\]

\[
q_i = g_\phi(x_i)
\]

常见匹配分数是点积：

\[
s(u,i)=p_u^Tq_i
\]

双塔适合召回层的关键原因是：**用户和物品可以分开计算**。物品向量可离线计算并构建 ANN 索引；线上只需计算一次用户向量，再执行最近邻检索。

## 2. Pointwise

Pointwise 把每个用户—物品对看成独立样本：

\[
(u,i,y),\quad y\in\{0,1\}
\]

预测概率：

\[
\hat y_{ui}=\sigma(s(u,i))
\]

其中：

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

常见损失是二元交叉熵：

\[
L_{\text{point}}
=
-\left[
y\log \hat y+(1-y)\log(1-\hat y)
\right]
\]

特点：实现简单、训练稳定，适合作为双塔基线；但没有直接优化相对排序。

## 3. Pairwise

Pairwise 使用三元组：

\[
(u,i^+,i^-)
\]

训练目标是：

\[
s(u,i^+)>s(u,i^-)
\]

经典 BPR 损失：

\[
L_{\text{BPR}}
=
-\log \sigma
\left(
s(u,i^+)-s(u,i^-)
\right)
\]

也可使用 Hinge Loss：

\[
L_{\text{hinge}}
=
\max
\left(
0,m-s(u,i^+)+s(u,i^-)
\right)
\]

特点：直接优化正负样本的相对顺序，但对负样本质量非常敏感。

## 4. Listwise

Listwise 一次考虑一个候选列表：

\[
C_u=\{i_1,i_2,\ldots,i_n\}
\]

候选物品的 Softmax 概率：

\[
P(i_j|u)
=
\frac{
\exp(s(u,i_j)/\tau)
}{
\sum_{k\in C_u}\exp(s(u,i_k)/\tau)
}
\]

如果只有一个正样本 \(i^+\)，损失为：

\[
L_{\text{list}}
=
-\log P(i^+|u)
\]

工业中常结合 In-Batch Negative：一个 Batch 中其他用户的正样本，作为当前用户的负样本。

## 5. 三种方式对比

| 方法 | 样本形式 | 关注点 | 常见损失 |
|---|---|---|---|
| Pointwise | \((u,i,y)\) | 单个物品是否合适 | BCE |
| Pairwise | \((u,i^+,i^-)\) | 正样本是否高于负样本 | BPR、Hinge |
| Listwise | \((u,C_u)\) | 正样本在列表中的位置 | Softmax CE、InfoNCE |

记忆方式：

```text
Pointwise：这个物品好不好？
Pairwise：这个物品是否比另一个更好？
Listwise：这一组物品中谁应该排最前？
```

## 6. 工业召回链路

```text
用户行为和特征
        ↓
用户塔生成 user embedding
        ↓
ANN 索引检索 TopK item
        ↓
过滤
        ↓
粗排 / 精排 / 重排
```

双塔的核心价值是：

> 把“对全量物品逐个执行复杂模型”转化为“生成一次用户向量，再做高效向量检索”。
