//! Recommendation pipeline orchestration.

use crate::model::{Item, User};
use crate::recommendation::explain::source_label;
use crate::recommendation::exposure::{ExposureDecision, ExposurePolicy};
use crate::recommendation::features::user_category_scores_for_categories;
use crate::recommendation::indexes::RecommendationIndexes;
use crate::recommendation::rank::{rank_candidate, score_desc};
use crate::recommendation::recall::recall_candidates;
use crate::recommendation::rerank::rerank_for_diversity;
use crate::recommendation::types::{
    DebugCandidate, RecommendationConfig, RecommendationDebug, RecommendationOutput,
};
use std::collections::HashMap;
use std::time::Instant;

#[cfg(test)]
pub fn build_recommendations(
    user: &User,
    items: &[Item],
    semantic_hits: &[(u64, f32)],
    config: RecommendationConfig,
) -> RecommendationOutput {
    let indexes = RecommendationIndexes::from_items(items);
    build_recommendations_with_indexes(user, items, &indexes, semantic_hits, config)
}

pub fn build_recommendations_with_indexes(
    user: &User,
    items: &[Item],
    indexes: &RecommendationIndexes,
    semantic_hits: &[(u64, f32)],
    config: RecommendationConfig,
) -> RecommendationOutput {
    if items.is_empty() || config.limit == 0 {
        return RecommendationOutput {
            items: Vec::new(),
            filtered_count: 0,
            debug: RecommendationDebug::default(),
        };
    }

    let pipeline_started = Instant::now();
    let category_scores = user_category_scores_for_categories(user, indexes.categories());
    let recall_output = recall_candidates(items, indexes, semantic_hits, &category_scores, &config);
    let mut stage_durations_micros = recall_output.stage_durations_micros;
    let quality_metrics = recall_output.quality_metrics;
    let candidates = recall_output.candidates;
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
    let exposure_policy = ExposurePolicy::from_events(&config.recent_events);

    let merge_rank_started = Instant::now();
    let mut filtered_count = 0usize;
    let mut exposure_adjusted_count = 0usize;
    let mut exposure_suppressed_count = 0usize;
    let mut ranked = Vec::new();

    for candidate in candidates.into_values() {
        let Some(item) = indexes.item(items, candidate.item_id) else {
            continue;
        };

        let exposure_decision = exposure_policy.decision(candidate.item_id);
        if exposure_decision == ExposureDecision::Suppress {
            filtered_count += 1;
            exposure_suppressed_count += 1;
            continue;
        }

        let mut ranked_item = rank_candidate(
            user,
            item,
            &candidate,
            config.preferences.as_ref(),
            &category_scores,
            indexes.price_stats(),
            ranker.as_ref(),
        );
        if let ExposureDecision::Deboost(amount) = exposure_decision {
            ranked_item.final_score = (ranked_item.final_score - amount).max(0.0);
            exposure_adjusted_count += 1;
        }
        ranked.push(ranked_item);
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
    stage_durations_micros.insert(
        "merge_rank".to_string(),
        merge_rank_started.elapsed().as_micros() as u64,
    );
    stage_durations_micros.insert(
        "total".to_string(),
        pipeline_started.elapsed().as_micros() as u64,
    );

    RecommendationOutput {
        items: reranked,
        filtered_count,
        debug: RecommendationDebug {
            candidate_count,
            candidates: debug_candidates,
            category_distribution,
            source_distribution,
            exposure_adjusted_count,
            exposure_suppressed_count,
            stage_durations_micros,
            quality_metrics,
        },
    }
}
