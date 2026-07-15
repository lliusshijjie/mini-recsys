//! Request-facing recommendation service and user context cache.

use crate::behavior::{BehaviorEvent, RecentEvents, UserPreferences};
use crate::config::AppConfig;
use crate::ffi::HnswIndex;
use crate::model::{Item, User};
use crate::observability::Metrics;
use crate::recommendation::indexes::RecommendationIndexes;
use crate::recommendation::pipeline::build_recommendations_with_indexes;
use crate::recommendation::rank::RankingStrategyKind;
use crate::recommendation::recall::recent_positive_seed_ids;
use crate::recommendation::types::{RecentRecallMode, RecommendationConfig, RecommendationOutput};
use crate::storage::Storage;
use fastbloom_rs::Membership;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct UserContext {
    pub preferences: UserPreferences,
    pub recent_events: Vec<BehaviorEvent>,
}

#[derive(Debug, Clone, Copy)]
pub struct UserContextCacheConfig {
    pub ttl: Duration,
    pub max_users: usize,
}

#[derive(Debug)]
struct CachedUserContext {
    value: UserContext,
    inserted_at: Instant,
}

#[derive(Debug, Default)]
struct UserContextCacheState {
    entries: HashMap<u64, CachedUserContext>,
    insertion_order: VecDeque<u64>,
}

#[derive(Debug)]
pub struct UserContextCache {
    config: UserContextCacheConfig,
    state: Mutex<UserContextCacheState>,
}

impl UserContextCache {
    pub fn new(config: UserContextCacheConfig) -> Self {
        Self {
            config,
            state: Mutex::new(UserContextCacheState::default()),
        }
    }

    pub fn get(&self, uid: u64) -> Option<UserContext> {
        let mut state = self.state.lock().expect("user context cache poisoned");
        let Some(entry) = state.entries.get(&uid) else {
            return None;
        };
        if entry.inserted_at.elapsed() >= self.config.ttl {
            state.entries.remove(&uid);
            state
                .insertion_order
                .retain(|cached_uid| *cached_uid != uid);
            return None;
        }
        Some(entry.value.clone())
    }

    pub fn insert(&self, uid: u64, value: UserContext) {
        if self.config.max_users == 0 {
            return;
        }

        let mut state = self.state.lock().expect("user context cache poisoned");
        if !state.entries.contains_key(&uid) {
            state.insertion_order.push_back(uid);
        }
        state.entries.insert(
            uid,
            CachedUserContext {
                value,
                inserted_at: Instant::now(),
            },
        );

        while state.entries.len() > self.config.max_users {
            let Some(oldest_uid) = state.insertion_order.pop_front() else {
                break;
            };
            state.entries.remove(&oldest_uid);
        }
    }

    pub fn invalidate(&self, uid: u64) {
        let mut state = self.state.lock().expect("user context cache poisoned");
        state.entries.remove(&uid);
        state
            .insertion_order
            .retain(|cached_uid| *cached_uid != uid);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RecommendationServiceConfig {
    pub recommend_timeout: Duration,
    pub user_context_cache: UserContextCacheConfig,
    pub batch_max_users: usize,
    pub recall_parallel_min_items: usize,
}

impl RecommendationServiceConfig {
    pub fn from_app_config(config: &AppConfig) -> Self {
        Self {
            recommend_timeout: Duration::from_millis(config.recommend_timeout_ms),
            user_context_cache: UserContextCacheConfig {
                ttl: Duration::from_millis(config.user_context_cache_ttl_ms),
                max_users: config.user_context_cache_max_users,
            },
            batch_max_users: config.batch_max_users,
            recall_parallel_min_items: config.recall_parallel_min_items,
        }
    }
}

#[derive(Debug)]
pub enum RecommendationServiceError {
    UserNotFound(u64),
    Storage(anyhow::Error),
}

#[derive(Debug)]
pub struct RecommendationServiceOutput {
    pub user: User,
    pub output: RecommendationOutput,
}

pub struct RecommendationService {
    storage: Arc<Storage>,
    users: Arc<Vec<User>>,
    items: Arc<Vec<Item>>,
    indexes: RecommendationIndexes,
    hnsw_index: Arc<HnswIndex>,
    metrics: Arc<Metrics>,
    cache: UserContextCache,
    config: RecommendationServiceConfig,
}

impl RecommendationService {
    pub fn new(
        storage: Arc<Storage>,
        users: Arc<Vec<User>>,
        items: Arc<Vec<Item>>,
        indexes: RecommendationIndexes,
        hnsw_index: Arc<HnswIndex>,
        metrics: Arc<Metrics>,
        config: RecommendationServiceConfig,
    ) -> Self {
        Self {
            storage,
            users,
            items,
            indexes,
            hnsw_index,
            metrics,
            cache: UserContextCache::new(config.user_context_cache),
            config,
        }
    }

    pub fn recommend_timeout(&self) -> Duration {
        self.config.recommend_timeout
    }

    pub fn batch_max_users(&self) -> usize {
        self.config.batch_max_users
    }

    pub fn recall_parallel_min_items(&self) -> usize {
        self.config.recall_parallel_min_items
    }

    pub fn invalidate_user_context(&self, uid: u64) {
        self.cache.invalidate(uid);
    }

    pub fn record_output_metrics(&self, output: &RecommendationOutput) {
        self.metrics
            .record_recommendation_candidates(output.debug.candidate_count);
        for (stage, latency_micros) in &output.debug.stage_durations_micros {
            self.metrics
                .record_recommendation_stage(stage, Duration::from_micros(*latency_micros));
        }
    }

    pub fn record_timeout(&self) {
        self.metrics.record_recommendation_timeout();
    }

    pub fn recommend_user(
        &self,
        uid: u64,
    ) -> Result<RecommendationServiceOutput, RecommendationServiceError> {
        let total_started = Instant::now();
        let user = self
            .users
            .iter()
            .find(|user| user.id == uid)
            .cloned()
            .ok_or(RecommendationServiceError::UserNotFound(uid))?;

        let semantic_started = Instant::now();
        let semantic_hits = self
            .hnsw_index
            .search(&user.embedding, 100)
            .unwrap_or_default();
        let semantic_duration_micros = semantic_started.elapsed().as_micros() as u64;

        let storage_started = Instant::now();
        let context = self.user_context(uid)?;
        let filter = self
            .storage
            .get_user_filter(uid)
            .map_err(RecommendationServiceError::Storage)?;
        let storage_duration_micros = storage_started.elapsed().as_micros() as u64;

        let recent_recall_mode = RecentRecallMode::from_env();
        let recent_ann_hits = if matches!(
            recent_recall_mode,
            RecentRecallMode::Shadow | RecentRecallMode::Ann
        ) {
            build_recent_ann_hits(&self.hnsw_index, &self.items, &context.recent_events, 100)
        } else {
            Vec::new()
        };

        let mut output = build_recommendations_with_indexes(
            &user,
            &self.items,
            &self.indexes,
            &semantic_hits,
            &|item_id| filter.contains(&item_id.to_le_bytes()),
            RecommendationConfig {
                ranking_strategy: RankingStrategyKind::from_env(),
                preferences: Some(context.preferences),
                recent_events: context.recent_events,
                recent_recall_mode,
                recent_ann_hits,
                ..Default::default()
            },
        );
        output
            .debug
            .stage_durations_micros
            .insert("semantic_ann".to_string(), semantic_duration_micros);
        output
            .debug
            .stage_durations_micros
            .insert("storage".to_string(), storage_duration_micros);
        output.debug.stage_durations_micros.insert(
            "total".to_string(),
            total_started.elapsed().as_micros() as u64,
        );

        Ok(RecommendationServiceOutput { user, output })
    }

    fn user_context(&self, uid: u64) -> Result<UserContext, RecommendationServiceError> {
        if let Some(context) = self.cache.get(uid) {
            self.metrics.record_user_context_cache_hit();
            return Ok(context);
        }

        self.metrics.record_user_context_cache_miss();
        let preferences = self
            .storage
            .get_user_preferences(uid)
            .map_err(RecommendationServiceError::Storage)?;
        let RecentEvents { items } = self
            .storage
            .get_recent_events(uid)
            .map_err(RecommendationServiceError::Storage)?;
        let context = UserContext {
            preferences,
            recent_events: items,
        };
        self.cache.insert(uid, context.clone());
        Ok(context)
    }
}

fn build_recent_ann_hits(
    hnsw_index: &HnswIndex,
    items: &[Item],
    recent_events: &[BehaviorEvent],
    k: usize,
) -> Vec<(u64, Vec<(u64, f32)>)> {
    let seed_ids = recent_positive_seed_ids(recent_events);
    if seed_ids.is_empty() {
        return Vec::new();
    }

    let seed_queries: Vec<(u64, Vec<f32>)> = seed_ids
        .iter()
        .filter_map(|seed_id| {
            items
                .iter()
                .find(|item| item.id == *seed_id)
                .map(|item| (*seed_id, item.embedding.clone()))
        })
        .collect();
    if seed_queries.is_empty() {
        return Vec::new();
    }

    let seed_embeddings: Vec<Vec<f32>> = seed_queries
        .iter()
        .map(|(_, embedding)| embedding.clone())
        .collect();
    let Ok(results) = hnsw_index.search_batch(&seed_embeddings, k) else {
        return Vec::new();
    };

    seed_queries
        .into_iter()
        .map(|(seed_id, _)| seed_id)
        .zip(results)
        .collect()
}
