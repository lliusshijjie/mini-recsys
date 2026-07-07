//! Ranking strategy implementations.

use crate::model::Item;
use crate::recommendation::explain::{reason_for, source_label};
use crate::recommendation::features::{normalize_score, PriceStats};
use crate::recommendation::types::{Candidate, RecommendedItem};
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankingStrategyKind {
    FixedWeights,
    MachineLearningReserved,
}

impl RankingStrategyKind {
    pub fn from_env() -> Self {
        match std::env::var("MINI_RECSYS_RANKING_STRATEGY") {
            Ok(value) if value == "machine_learning_reserved" => Self::MachineLearningReserved,
            _ => Self::FixedWeights,
        }
    }

    pub(super) fn ranker(self) -> Box<dyn Ranker> {
        match self {
            Self::FixedWeights => Box::new(FixedWeightRanker::default()),
            Self::MachineLearningReserved => Box::new(MachineLearningReservedRanker::default()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RankingFeatures {
    semantic_score: f32,
    category_score: f32,
    popularity: f32,
    price_affinity: f32,
    novelty: f32,
}

pub(super) trait Ranker {
    fn name(&self) -> &'static str;
    fn score(&self, features: &RankingFeatures) -> f32;
}

#[derive(Debug, Clone, Copy)]
struct RankWeights {
    semantic: f32,
    category: f32,
    popularity: f32,
    price_and_novelty: f32,
}

impl Default for RankWeights {
    fn default() -> Self {
        Self {
            semantic: 0.50,
            category: 0.20,
            popularity: 0.20,
            price_and_novelty: 0.10,
        }
    }
}

#[derive(Debug, Default)]
struct FixedWeightRanker {
    weights: RankWeights,
}

impl Ranker for FixedWeightRanker {
    fn name(&self) -> &'static str {
        "fixed_weights"
    }

    fn score(&self, features: &RankingFeatures) -> f32 {
        features.semantic_score * self.weights.semantic
            + features.category_score * self.weights.category
            + features.popularity * self.weights.popularity
            + ((features.price_affinity + features.novelty) / 2.0) * self.weights.price_and_novelty
    }
}

#[derive(Debug, Default)]
struct MachineLearningReservedRanker {
    fallback: FixedWeightRanker,
}

impl Ranker for MachineLearningReservedRanker {
    fn name(&self) -> &'static str {
        "machine_learning_reserved"
    }

    fn score(&self, features: &RankingFeatures) -> f32 {
        self.fallback.score(features)
    }
}

pub(super) fn rank_candidate(
    item: &Item,
    candidate: &Candidate,
    category_scores: &HashMap<String, f32>,
    price_stats: &PriceStats,
    ranker: &dyn Ranker,
) -> RecommendedItem {
    let semantic_score = candidate.semantic_score;
    let category_score = category_scores
        .get(&item.category)
        .copied()
        .unwrap_or_default();
    let popularity = normalize_score(item.popularity);
    let price_affinity = price_stats.affinity(item.price);
    let novelty = 1.0 - popularity;
    let features = RankingFeatures {
        semantic_score,
        category_score,
        popularity,
        price_affinity,
        novelty,
    };
    let final_score = ranker.score(&features);

    let source = source_label(&candidate.sources);
    let reason = reason_for(&source, semantic_score, category_score);

    RecommendedItem {
        item_id: item.id,
        name: item.name.clone(),
        category: item.category.clone(),
        image_url: item.image_url.clone(),
        price: item.price,
        semantic_score,
        category_score,
        popularity,
        price_affinity,
        novelty,
        final_score,
        ranking_strategy: ranker.name().to_string(),
        source,
        reason,
    }
}

pub(super) fn score_desc(left: &RecommendedItem, right: &RecommendedItem) -> Ordering {
    right
        .final_score
        .partial_cmp(&left.final_score)
        .unwrap_or(Ordering::Equal)
}
