//! Candidate recall sources for recommendations.

use crate::behavior::{BehaviorEvent, EventType};
use crate::model::Item;
use crate::recommendation::features::normalize_score;
use crate::recommendation::types::{Candidate, RecommendationConfig};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const RECALL_MULTIPLIER: usize = 4;
const MIN_RECALL_POOL: usize = 20;
const RECENT_SIMILARITY_MIN_SCORE: f32 = 0.30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum RecallSource {
    SemanticAnn,
    CategoryProfile,
    RecentItemSimilarity,
    PopularFallback,
}

impl RecallSource {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::SemanticAnn => "semantic_ann",
            Self::CategoryProfile => "category_profile",
            Self::RecentItemSimilarity => "recent_item_similarity",
            Self::PopularFallback => "popular_fallback",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RecallHit {
    item_id: u64,
    score: f32,
    source: RecallSource,
}

struct RecallContext<'a> {
    items: &'a [Item],
    semantic_hits: &'a [(u64, f32)],
    category_scores: &'a HashMap<String, f32>,
    recent_events: &'a [BehaviorEvent],
    pool_size: usize,
}

pub(super) fn recall_candidates(
    items: &[Item],
    semantic_hits: &[(u64, f32)],
    category_scores: &HashMap<String, f32>,
    config: &RecommendationConfig,
) -> HashMap<u64, Candidate> {
    let context = RecallContext {
        items,
        semantic_hits,
        category_scores,
        recent_events: &config.recent_events,
        pool_size: (config.limit * RECALL_MULTIPLIER).max(MIN_RECALL_POOL),
    };
    let mut candidates = HashMap::new();

    merge_hits(&mut candidates, recall_semantic_ann(&context));
    merge_hits(&mut candidates, recall_category_profile(&context));
    merge_hits(&mut candidates, recall_recent_item_similarity(&context));
    merge_popular_fallback(
        &mut candidates,
        recall_popular_fallback(&context),
        context.pool_size,
    );

    candidates
}

fn recall_semantic_ann(context: &RecallContext<'_>) -> Vec<RecallHit> {
    context
        .semantic_hits
        .iter()
        .map(|&(item_id, score)| RecallHit {
            item_id,
            score: normalize_score(score),
            source: RecallSource::SemanticAnn,
        })
        .collect()
}

fn recall_category_profile(context: &RecallContext<'_>) -> Vec<RecallHit> {
    let mut category_items: Vec<&Item> = context
        .items
        .iter()
        .filter(|item| {
            context
                .category_scores
                .get(&item.category)
                .copied()
                .unwrap_or_default()
                > 0.0
        })
        .collect();

    category_items.sort_by(|a, b| {
        let a_score = context
            .category_scores
            .get(&a.category)
            .copied()
            .unwrap_or_default();
        let b_score = context
            .category_scores
            .get(&b.category)
            .copied()
            .unwrap_or_default();
        b_score
            .partial_cmp(&a_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b.popularity
                    .partial_cmp(&a.popularity)
                    .unwrap_or(Ordering::Equal)
            })
    });

    category_items
        .into_iter()
        .take(context.pool_size)
        .map(|item| RecallHit {
            item_id: item.id,
            score: context
                .category_scores
                .get(&item.category)
                .copied()
                .unwrap_or_default(),
            source: RecallSource::CategoryProfile,
        })
        .collect()
}

fn recall_recent_item_similarity(context: &RecallContext<'_>) -> Vec<RecallHit> {
    let item_map: HashMap<u64, &Item> = context.items.iter().map(|item| (item.id, item)).collect();
    let seed_ids = recent_positive_seed_ids(context.recent_events);
    if seed_ids.is_empty() {
        return Vec::new();
    }

    let mut hits = Vec::new();
    for seed_id in &seed_ids {
        let Some(seed) = item_map.get(seed_id) else {
            continue;
        };

        for item in context.items {
            if seed_ids.contains(&item.id) || item.category != seed.category {
                continue;
            }

            let similarity = embedding_similarity(&seed.embedding, &item.embedding);
            if similarity >= RECENT_SIMILARITY_MIN_SCORE {
                hits.push(RecallHit {
                    item_id: item.id,
                    score: similarity,
                    source: RecallSource::RecentItemSimilarity,
                });
            }
        }
    }

    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    hits.truncate(context.pool_size);
    hits
}

fn recent_positive_seed_ids(recent_events: &[BehaviorEvent]) -> HashSet<u64> {
    recent_events
        .iter()
        .rev()
        .filter(|event| matches!(event.event_type, EventType::Click | EventType::Like))
        .map(|event| event.item_id)
        .collect()
}

fn recall_popular_fallback(context: &RecallContext<'_>) -> Vec<RecallHit> {
    let mut popular_items: Vec<&Item> = context.items.iter().collect();
    popular_items.sort_by(|a, b| {
        b.popularity
            .partial_cmp(&a.popularity)
            .unwrap_or(Ordering::Equal)
    });

    popular_items
        .into_iter()
        .map(|item| RecallHit {
            item_id: item.id,
            score: normalize_score(item.popularity),
            source: RecallSource::PopularFallback,
        })
        .collect()
}

fn merge_hits(candidates: &mut HashMap<u64, Candidate>, hits: Vec<RecallHit>) {
    for hit in hits {
        let candidate = candidates
            .entry(hit.item_id)
            .or_insert_with(|| Candidate::new(hit.item_id));
        if hit.source == RecallSource::SemanticAnn {
            candidate.semantic_score = candidate.semantic_score.max(hit.score);
        }
        candidate.add_source(hit.source);
    }
}

fn merge_popular_fallback(
    candidates: &mut HashMap<u64, Candidate>,
    hits: Vec<RecallHit>,
    pool_size: usize,
) {
    for hit in hits {
        if candidates.len() >= pool_size {
            break;
        }
        if candidates.contains_key(&hit.item_id) {
            continue;
        }
        candidates
            .entry(hit.item_id)
            .or_insert_with(|| Candidate::new(hit.item_id))
            .add_source(hit.source);
    }
}

fn embedding_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || left.len() != right.len() {
        return 0.0;
    }

    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }

    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum::<f32>();

    normalize_score(dot / (left_norm * right_norm))
}
