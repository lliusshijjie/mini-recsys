# docs/

mini-recsys 的学习笔记、思维导图与项目文档。

## 目录说明

| 目录 | 用途 |
|------|------|
| `project/` | mini-recsys 项目计划、架构与设计文档 |
| `notes/` | 推荐系统通用学习笔记（Markdown） |
| `mindmaps/` | XMind 思维导图 |
| `snippets/` | 独立实验与代码草稿（非生产 `src/`） |
| `skills/` | C++ 在线服务与工程实现笔记 |

## 笔记结构

笔记按推荐漏斗阶段组织：

| 目录 | 阶段 | 主题 |
|------|------|------|
| `notes/recall/` | 召回 | 向量检索、协同过滤、双塔模型 |
| `notes/coarse-ranking/` | 粗排 | 预排序模型、Top-K 截断、批量推理 |
| `notes/fine-ranking/` | 精排 | 特征工程、特征交叉、序列模型、多目标学习 |
| `notes/rerank/` | 重排 | 多样性、探索利用、业务规则（预留） |

## 命名规则

```text
[{序号}-]{主题}[-{技术点}].md
```

1. 不写「推荐系统」前缀（目录已表达领域）
2. 不写阶段词（由文件夹表达：`recall/` / `coarse-ranking/` / `fine-ranking/`）
3. 不写文档类型后缀（学习笔记 / 总结 / 方案）
4. 分隔符统一用 `-`
5. 专有名词保留英文原样（`ItemCF`、`DIN`、`MMoE` 等）
6. 同目录用两位序号表示阅读顺序（`01-`、`02-`）

### 索引

**recall/**
- `01-向量召回.md` — 向量召回总览
- `02-协同过滤-ItemCF-Swing-UserCF.md` — ItemCF、Swing、UserCF
- `03-双塔-结构与训练.md` — 双塔架构与训练
- `04-双塔-样本与线上更新.md` — 样本构造与线上更新
- `05-离散特征-矩阵补全与近邻.md` — 离散特征、矩阵补全、近邻检索

**coarse-ranking/**
- `01-粗排模型.md` — 粗排模型与在线服务

**fine-ranking/**
- `01-特征与多模态.md` — 特征与多模态建模
- `02-特征交叉-FM-DCN.md` — FM、DCN、LHUC、SENet
- `03-行为序列-DIN-SIM.md` — DIN 与 SIM 序列建模
- `04-多目标-MMoE.md` — 多目标学习与 MMoE
- `05-可解释逻辑回归.md` — 可解释逻辑回归排序

**skills/**
- `召回-C++服务端.md` — 召回阶段 C++ 服务端技术
- `排序-C++服务端.md` — 排序层 C++ 服务端工程

**project/**
- `mini-recsys三阶段扩展计划.md` — mini-recsys 分阶段扩展计划
