//! Shared recommendation data types.

use crate::behavior::{BehaviorEvent, UserPreferences};
use crate::recommendation::rank::RankingStrategyKind;
use crate::recommendation::recall::RecallSource;
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
    pub recent_events: Vec<BehaviorEvent>,
    pub recent_recall_mode: RecentRecallMode,
    pub recent_ann_hits: Vec<(u64, Vec<(u64, f32)>)>,
    pub recall_parallel_min_items: usize,
}

impl Default for RecommendationConfig {
    fn default() -> Self {
        Self {
            limit: DEFAULT_LIMIT,
            max_per_category: DEFAULT_MAX_PER_CATEGORY,
            exploration_slots: DEFAULT_EXPLORATION_SLOTS,
            ranking_strategy: RankingStrategyKind::FixedWeights,
            preferences: None,
            recent_events: Vec::new(),
            recent_recall_mode: RecentRecallMode::from_env(),
            recent_ann_hits: Vec::new(),
            recall_parallel_min_items: usize::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecentRecallMode {
    Exact,
    Shadow,
    Ann,
}

impl RecentRecallMode {
    pub fn from_env() -> Self {
        match std::env::var("MINI_RECSYS_RECENT_RECALL_MODE") {
            Ok(value) if value.eq_ignore_ascii_case("shadow") => Self::Shadow,
            Ok(value) if value.eq_ignore_ascii_case("ann") => Self::Ann,
            _ => Self::Exact,
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
    pub exposure_adjusted_count: usize,
    pub exposure_suppressed_count: usize,
    pub stage_durations_micros: HashMap<String, u64>,
    pub quality_metrics: HashMap<String, f32>,
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
    pub sources: HashSet<RecallSource>,
}

impl Candidate {
    pub(super) fn new(item_id: u64) -> Self {
        Self {
            item_id,
            semantic_score: 0.0,
            sources: HashSet::new(),
        }
    }

    pub(super) fn add_source(&mut self, source: RecallSource) {
        self.sources.insert(source);
    }

    pub(super) fn has_source(&self, source: RecallSource) -> bool {
        self.sources.contains(&source)
    }
}
