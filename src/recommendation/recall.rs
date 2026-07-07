//! Candidate recall sources for recommendations.

use crate::model::Item;
use crate::recommendation::features::normalize_score;
use crate::recommendation::types::{Candidate, RecommendationConfig};
use std::cmp::Ordering;
use std::collections::HashMap;

const RECALL_MULTIPLIER: usize = 4;
const MIN_RECALL_POOL: usize = 20;

pub(super) fn recall_candidates(
    items: &[Item],
    semantic_hits: &[(u64, f32)],
    category_scores: &HashMap<String, f32>,
    config: &RecommendationConfig,
) -> HashMap<u64, Candidate> {
    let mut candidates = HashMap::new();
    let pool_size = (config.limit * RECALL_MULTIPLIER).max(MIN_RECALL_POOL);

    for &(item_id, score) in semantic_hits {
        let candidate = candidates
            .entry(item_id)
            .or_insert_with(|| Candidate::new(item_id));
        candidate.semantic_score = candidate.semantic_score.max(normalize_score(score));
        candidate.add_source("semantic");
    }

    let mut popular_items: Vec<&Item> = items.iter().collect();
    popular_items.sort_by(|a, b| {
        b.popularity
            .partial_cmp(&a.popularity)
            .unwrap_or(Ordering::Equal)
    });
    for item in popular_items.into_iter().take(pool_size) {
        candidates
            .entry(item.id)
            .or_insert_with(|| Candidate::new(item.id))
            .add_source("popular");
    }

    let mut category_items: Vec<&Item> = items
        .iter()
        .filter(|item| {
            category_scores
                .get(&item.category)
                .copied()
                .unwrap_or_default()
                > 0.0
        })
        .collect();
    category_items.sort_by(|a, b| {
        let a_score = category_scores
            .get(&a.category)
            .copied()
            .unwrap_or_default();
        let b_score = category_scores
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
    for item in category_items.into_iter().take(pool_size) {
        candidates
            .entry(item.id)
            .or_insert_with(|| Candidate::new(item.id))
            .add_source("category");
    }

    candidates
}
