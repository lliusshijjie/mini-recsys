# docs/

Learning notes, mind maps, and project documentation for mini-recsys.

## Layout

| Directory | Purpose |
|-----------|---------|
| `project/` | mini-recsys-specific plans, architecture, and design docs |
| `notes/` | General recommendation-system learning notes (Markdown) |
| `mindmaps/` | XMind mind maps |
| `snippets/` | Standalone experiments and code drafts (not production `src/`) |
| `skills/` | Engineering notes for C++ serving and implementation |

## Notes structure

Notes are organized by recommendation pipeline stage:

| Directory | Stage | Topics |
|-----------|-------|--------|
| `notes/recall/` | Recall | vector search, collaborative filtering, two-tower models |
| `notes/coarse-ranking/` | Coarse ranking | pre-rank models, Top-K filtering, batch inference |
| `notes/fine-ranking/` | Fine ranking | features, feature crossing, sequence models, multi-objective learning |
| `notes/rerank/` | Re-ranking | diversity, exploration, business rules (reserved) |

### Index

**recall/**
- `vector_recall_complete_guide.md` — vector recall overview
- `cf_recall_itemcf_swing_usercf.md` — ItemCF, Swing, UserCF
- `two_tower_structure_and_training.md` — two-tower architecture and training
- `two_tower_samples_online_update.md` — samples and online update
- `recsys_recall_discrete_matrix_nn.md` — discrete features, matrix completion, NN lookup

**coarse-ranking/**
- `推荐系统粗排模型学习笔记.md` — coarse ranking models and serving

**fine-ranking/**
- `推荐系统排序层特征与多模态建模学习笔记.md` — features and multimodal modeling
- `推荐系统特征交叉与常见模型总结.md` — FM, DCN, LHUC, SENet
- `推荐系统_行为序列_DIN_SIM学习笔记.md` — DIN and SIM sequence modeling
- `推荐系统多目标学习与MMoE架构.md` — multi-objective learning and MMoE
- `explainable-logistic-ranking-optimization.md` — explainable logistic regression ranking

**project/**
- `expansion-plan.md` — phased mini-recsys expansion plan
