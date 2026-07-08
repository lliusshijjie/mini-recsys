//! Recommendation pipeline orchestration.

use crate::model::{Item, User};
use crate::recommendation::explain::source_label;
use crate::recommendation::features::{user_category_scores, PriceStats};
use crate::recommendation::rank::{rank_candidate, score_desc};
use crate::recommendation::recall::recall_candidates;
use crate::recommendation::rerank::rerank_for_diversity;
use crate::recommendation::types::{
    DebugCandidate, RecommendationConfig, RecommendationDebug, RecommendationOutput,
};
use std::collections::HashMap;

pub fn build_recommendations(
    user: &User,
    items: &[Item],
    semantic_hits: &[(u64, f32)],
    is_seen: &dyn Fn(u64) -> bool,
    config: RecommendationConfig,
) -> RecommendationOutput {
    if items.is_empty() || config.limit == 0 {
        return RecommendationOutput {
            items: Vec::new(),
            filtered_count: 0,
            debug: RecommendationDebug::default(),
        };
    }

    let item_map: HashMap<u64, &Item> = items.iter().map(|item| (item.id, item)).collect();
    let category_scores = user_category_scores(user, items);
    let price_stats = PriceStats::from_items(items);
    let mut candidates = recall_candidates(items, semantic_hits, &category_scores, &config);
    for candidate in candidates.values_mut() {
        candidate.preferences = config.preferences.clone();
    }
    let debug_candidates: Vec<DebugCandidate> = candidates
        .values()
        .map(|candidate| DebugCandidate {
            item_id: candidate.item_id,
            semantic_score: candidate.semantic_score,
            source: source_label(&candidate.sources),
        })
        .collect();
    let candidate_count = candidates.len();
    let ranker = config.ranking_strategy.ranker();

    let mut filtered_count = 0usize;
    let mut ranked = Vec::new();

    for candidate in candidates.into_values() {
        if is_seen(candidate.item_id) {
            filtered_count += 1;
            continue;
        }

        let Some(item) = item_map.get(&candidate.item_id) else {
            continue;
        };

        ranked.push(rank_candidate(
            item,
            &candidate,
            &category_scores,
            &price_stats,
            ranker.as_ref(),
        ));
    }

    ranked.sort_by(score_desc);
    let reranked = rerank_for_diversity(ranked, &config);
    let mut category_distribution = HashMap::new();
    let mut source_distribution = HashMap::new();
    for item in &reranked {
        *category_distribution
            .entry(item.category.clone())
            .or_insert(0) += 1;
        *source_distribution.entry(item.source.clone()).or_insert(0) += 1;
    }

    RecommendationOutput {
        items: reranked,
        filtered_count,
        debug: RecommendationDebug {
            candidate_count,
            candidates: debug_candidates,
            category_distribution,
            source_distribution,
        },
    }
}
