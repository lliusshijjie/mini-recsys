//! Shared recommendation data types.

use crate::behavior::UserPreferences;
use crate::recommendation::rank::RankingStrategyKind;
use std::collections::{HashMap, HashSet};

const DEFAULT_LIMIT: usize = 10;
const DEFAULT_MAX_PER_CATEGORY: usize = 3;
const DEFAULT_EXPLORATION_SLOTS: usize = 1;

#[derive(Debug, Clone)]
pub struct RecommendationConfig {
    pub limit: usize,
    pub max_per_category: usize,
    pub exploration_slots: usize,
    pub ranking_strategy: RankingStrategyKind,
    pub preferences: Option<UserPreferences>,
}

impl Default for RecommendationConfig {
    fn default() -> Self {
        Self {
            limit: DEFAULT_LIMIT,
            max_per_category: DEFAULT_MAX_PER_CATEGORY,
            exploration_slots: DEFAULT_EXPLORATION_SLOTS,
            ranking_strategy: RankingStrategyKind::FixedWeights,
            preferences: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecommendationOutput {
    pub items: Vec<RecommendedItem>,
    pub filtered_count: usize,
    pub debug: RecommendationDebug,
}

#[derive(Debug, Clone, Default)]
pub struct RecommendationDebug {
    pub candidate_count: usize,
    pub candidates: Vec<DebugCandidate>,
    pub category_distribution: HashMap<String, usize>,
    pub source_distribution: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct DebugCandidate {
    pub item_id: u64,
    pub semantic_score: f32,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct RecommendedItem {
    pub item_id: u64,
    pub name: String,
    pub category: String,
    pub image_url: String,
    pub price: f32,
    pub semantic_score: f32,
    pub category_score: f32,
    pub popularity: f32,
    pub price_affinity: f32,
    pub novelty: f32,
    pub feedback_score: f32,
    pub final_score: f32,
    pub ranking_strategy: String,
    pub source: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub(super) struct Candidate {
    pub item_id: u64,
    pub semantic_score: f32,
    pub sources: HashSet<&'static str>,
    pub preferences: Option<UserPreferences>,
}

impl Candidate {
    pub(super) fn new(item_id: u64) -> Self {
        Self {
            item_id,
            semantic_score: 0.0,
            sources: HashSet::new(),
            preferences: None,
        }
    }

    pub(super) fn add_source(&mut self, source: &'static str) {
        self.sources.insert(source);
    }
}
