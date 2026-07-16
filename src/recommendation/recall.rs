//! Candidate recall sources for recommendations.

use crate::algorithms::{
    cosine_similarity_simd, merge_filter_all, partial_topk_by, ScoredCandidate,
};
use crate::behavior::{BehaviorEvent, EventType};
use crate::model::Item;
use crate::recommendation::features::normalize_score;
use crate::recommendation::indexes::RecommendationIndexes;
use crate::recommendation::types::{Candidate, RecentRecallMode, RecommendationConfig};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

const RECALL_MULTIPLIER: usize = 4;
const MIN_RECALL_POOL: usize = 20;
const RECENT_SIMILARITY_MIN_SCORE: f32 = 0.30;
const SEMANTIC_SOURCE_MASK: u8 = 1 << 0;
const CATEGORY_SOURCE_MASK: u8 = 1 << 1;
const RECENT_SOURCE_MASK: u8 = 1 << 2;

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
    indexes: &'a RecommendationIndexes,
    semantic_hits: &'a [(u64, f32)],
    category_scores: &'a HashMap<String, f32>,
    recent_events: &'a [BehaviorEvent],
    recent_ann_hits: &'a [(u64, Vec<(u64, f32)>)],
    pool_size: usize,
}

pub(super) struct RecallOutput {
    pub candidates: HashMap<u64, Candidate>,
    pub stage_durations_micros: HashMap<String, u64>,
    pub quality_metrics: HashMap<String, f32>,
}

pub(super) fn recall_candidates(
    items: &[Item],
    indexes: &RecommendationIndexes,
    semantic_hits: &[(u64, f32)],
    category_scores: &HashMap<String, f32>,
    config: &RecommendationConfig,
) -> RecallOutput {
    let context = RecallContext {
        items,
        indexes,
        semantic_hits,
        category_scores,
        recent_events: &config.recent_events,
        recent_ann_hits: &config.recent_ann_hits,
        pool_size: (config.limit * RECALL_MULTIPLIER).max(MIN_RECALL_POOL),
    };
    let mut stage_durations_micros = HashMap::new();
    let recall_sources = if items.len() >= config.recall_parallel_min_items {
        recall_sources_parallel(&context, config.recent_recall_mode)
    } else {
        recall_sources_serial(&context, config.recent_recall_mode)
    };
    let quality_metrics = recall_sources.recent_quality_metrics;

    stage_durations_micros.insert(
        "semantic_ann".to_string(),
        recall_sources.semantic_elapsed.as_micros() as u64,
    );

    stage_durations_micros.insert(
        "category_recall".to_string(),
        recall_sources.category_elapsed.as_micros() as u64,
    );

    stage_durations_micros.insert(
        "recent_ann".to_string(),
        recall_sources.recent_elapsed.as_micros() as u64,
    );

    stage_durations_micros.insert(
        "popular_fallback".to_string(),
        recall_sources.popular_elapsed.as_micros() as u64,
    );
    let mut candidates = merge_primary_hits(
        recall_sources.semantic_hits,
        recall_sources.category_hits,
        recall_sources.recent_hits,
    );
    merge_popular_fallback(
        &mut candidates,
        recall_sources.popular_hits,
        context.pool_size,
    );

    RecallOutput {
        candidates,
        stage_durations_micros,
        quality_metrics,
    }
}

fn timed<T>(operation: impl FnOnce() -> T) -> (T, Duration) {
    let started = Instant::now();
    let value = operation();
    (value, started.elapsed())
}

struct RecallSourceOutputs {
    semantic_hits: Vec<RecallHit>,
    semantic_elapsed: Duration,
    category_hits: Vec<RecallHit>,
    category_elapsed: Duration,
    recent_hits: Vec<RecallHit>,
    recent_quality_metrics: HashMap<String, f32>,
    recent_elapsed: Duration,
    popular_hits: Vec<RecallHit>,
    popular_elapsed: Duration,
}

fn recall_sources_serial(
    context: &RecallContext<'_>,
    mode: RecentRecallMode,
) -> RecallSourceOutputs {
    let (semantic_hits, semantic_elapsed) = timed(|| recall_semantic_ann(context));
    let (category_hits, category_elapsed) = timed(|| recall_category_profile(context));
    let ((recent_hits, recent_quality_metrics), recent_elapsed) =
        timed(|| recall_recent_item_similarity(context, mode));
    let (popular_hits, popular_elapsed) = timed(|| recall_popular_fallback(context));

    RecallSourceOutputs {
        semantic_hits,
        semantic_elapsed,
        category_hits,
        category_elapsed,
        recent_hits,
        recent_quality_metrics,
        recent_elapsed,
        popular_hits,
        popular_elapsed,
    }
}

fn recall_sources_parallel(
    context: &RecallContext<'_>,
    mode: RecentRecallMode,
) -> RecallSourceOutputs {
    std::thread::scope(|scope| {
        let semantic = scope.spawn(|| timed(|| recall_semantic_ann(context)));
        let category = scope.spawn(|| timed(|| recall_category_profile(context)));
        let recent = scope.spawn(|| timed(|| recall_recent_item_similarity(context, mode)));
        let popular = scope.spawn(|| timed(|| recall_popular_fallback(context)));

        let (semantic_hits, semantic_elapsed) = semantic.join().expect("semantic recall panicked");
        let (category_hits, category_elapsed) = category.join().expect("category recall panicked");
        let ((recent_hits, recent_quality_metrics), recent_elapsed) =
            recent.join().expect("recent recall panicked");
        let (popular_hits, popular_elapsed) = popular.join().expect("popular recall panicked");

        RecallSourceOutputs {
            semantic_hits,
            semantic_elapsed,
            category_hits,
            category_elapsed,
            recent_hits,
            recent_quality_metrics,
            recent_elapsed,
            popular_hits,
            popular_elapsed,
        }
    })
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
    let mut category_items = Vec::new();
    for (category, score) in context.category_scores {
        if *score <= 0.0 {
            continue;
        }
        let Some(item_ids) = context.indexes.category_item_ids(category) else {
            continue;
        };
        category_items.extend(
            item_ids
                .iter()
                .take(context.pool_size)
                .filter_map(|item_id| context.indexes.item(context.items, *item_id)),
        );
    }

    let mut indexed_items: Vec<(usize, &Item)> = category_items.into_iter().enumerate().collect();
    partial_topk_by(
        &mut indexed_items,
        context.pool_size,
        |(left_index, a), (right_index, b)| {
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
                .then_with(|| left_index.cmp(right_index))
        },
    );

    indexed_items
        .into_iter()
        .take(context.pool_size)
        .map(|(_, item)| RecallHit {
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

fn recall_recent_item_similarity(
    context: &RecallContext<'_>,
    mode: RecentRecallMode,
) -> (Vec<RecallHit>, HashMap<String, f32>) {
    let exact_hits = || recall_recent_item_similarity_exact(context);
    match mode {
        RecentRecallMode::Exact => (exact_hits(), HashMap::new()),
        RecentRecallMode::Ann => (recall_recent_item_similarity_ann(context), HashMap::new()),
        RecentRecallMode::Shadow => {
            let exact = exact_hits();
            let ann = recall_recent_item_similarity_ann(context);
            let mut quality_metrics = HashMap::new();
            quality_metrics.insert(
                "recent_ann_overlap".to_string(),
                overlap_ratio(&exact, &ann),
            );
            (exact, quality_metrics)
        }
    }
}

fn recall_recent_item_similarity_exact(context: &RecallContext<'_>) -> Vec<RecallHit> {
    let seed_ids = recent_positive_seed_ids(context.recent_events);
    if seed_ids.is_empty() {
        return Vec::new();
    }

    let mut hits = Vec::new();
    for seed_id in &seed_ids {
        let Some(seed) = context.indexes.item(context.items, *seed_id) else {
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

    truncate_hits_by_score_preserving_ties(&mut hits, context.pool_size);
    hits
}

fn recall_recent_item_similarity_ann(context: &RecallContext<'_>) -> Vec<RecallHit> {
    let seed_ids = recent_positive_seed_ids(context.recent_events);
    if seed_ids.is_empty() {
        return Vec::new();
    }

    let seed_set: HashSet<u64> = seed_ids.iter().copied().collect();
    let seed_categories: HashMap<u64, String> = seed_ids
        .iter()
        .filter_map(|seed_id| {
            context
                .indexes
                .item(context.items, *seed_id)
                .map(|item| (*seed_id, item.category.clone()))
        })
        .collect();
    let mut hits = Vec::new();

    for (seed_id, ann_hits) in context.recent_ann_hits {
        let Some(seed_category) = seed_categories.get(seed_id) else {
            continue;
        };
        for (item_id, score) in ann_hits {
            if seed_set.contains(item_id) || *score < RECENT_SIMILARITY_MIN_SCORE {
                continue;
            }
            let Some(item) = context.indexes.item(context.items, *item_id) else {
                continue;
            };
            if item.category != *seed_category {
                continue;
            }
            hits.push(RecallHit {
                item_id: *item_id,
                score: normalize_score(*score),
                source: RecallSource::RecentItemSimilarity,
            });
        }
    }

    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    hits.dedup_by_key(|hit| hit.item_id);
    hits.truncate(context.pool_size);
    hits
}

pub(crate) fn recent_positive_seed_ids(recent_events: &[BehaviorEvent]) -> Vec<u64> {
    let mut seed_ids = Vec::new();
    for event in recent_events.iter().rev() {
        if !matches!(event.event_type, EventType::Click | EventType::Like) {
            continue;
        }
        if seed_ids.contains(&event.item_id) {
            continue;
        }
        seed_ids.push(event.item_id);
        if seed_ids.len() >= 5 {
            break;
        }
    }
    seed_ids
}

fn overlap_ratio(left: &[RecallHit], right: &[RecallHit]) -> f32 {
    if left.is_empty() {
        return if right.is_empty() { 1.0 } else { 0.0 };
    }
    let right_ids: HashSet<u64> = right.iter().map(|hit| hit.item_id).collect();
    let overlap = left
        .iter()
        .filter(|hit| right_ids.contains(&hit.item_id))
        .count();
    overlap as f32 / left.len() as f32
}

fn recall_popular_fallback(context: &RecallContext<'_>) -> Vec<RecallHit> {
    context
        .indexes
        .popular_item_ids()
        .iter()
        .filter_map(|item_id| context.indexes.item(context.items, *item_id))
        .map(|item| RecallHit {
            item_id: item.id,
            score: normalize_score(item.popularity),
            source: RecallSource::PopularFallback,
        })
        .collect()
}

fn merge_primary_hits(
    semantic_hits: Vec<RecallHit>,
    category_hits: Vec<RecallHit>,
    recent_hits: Vec<RecallHit>,
) -> HashMap<u64, Candidate> {
    let mut scored_hits =
        Vec::with_capacity(semantic_hits.len() + category_hits.len() + recent_hits.len());
    scored_hits.extend(semantic_hits.into_iter().map(scored_hit_for_merge));
    scored_hits.extend(category_hits.into_iter().map(scored_hit_for_merge));
    scored_hits.extend(recent_hits.into_iter().map(scored_hit_for_merge));

    merge_filter_all(&scored_hits, &[])
        .into_iter()
        .map(|scored| (scored.item_id, candidate_from_scored(scored)))
        .collect()
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
    normalize_score(cosine_similarity_simd(left, right))
}

fn truncate_hits_by_score_preserving_ties(hits: &mut Vec<RecallHit>, pool_size: usize) {
    let mut indexed_hits: Vec<(usize, RecallHit)> =
        std::mem::take(hits).into_iter().enumerate().collect();
    partial_topk_by(
        &mut indexed_hits,
        pool_size,
        |(left_index, left), (right_index, right)| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left_index.cmp(right_index))
        },
    );
    *hits = indexed_hits.into_iter().map(|(_, hit)| hit).collect();
}

fn scored_hit_for_merge(hit: RecallHit) -> ScoredCandidate {
    let score = if hit.source == RecallSource::SemanticAnn {
        hit.score
    } else {
        0.0
    };
    ScoredCandidate::new(hit.item_id, score, source_mask(hit.source))
}

fn candidate_from_scored(scored: ScoredCandidate) -> Candidate {
    let mut candidate = Candidate::new(scored.item_id);
    candidate.semantic_score = scored.score;
    for source in sources_from_mask(scored.source_mask) {
        candidate.add_source(source);
    }
    candidate
}

fn source_mask(source: RecallSource) -> u8 {
    match source {
        RecallSource::SemanticAnn => SEMANTIC_SOURCE_MASK,
        RecallSource::CategoryProfile => CATEGORY_SOURCE_MASK,
        RecallSource::RecentItemSimilarity => RECENT_SOURCE_MASK,
        RecallSource::PopularFallback => 0,
    }
}

fn sources_from_mask(mask: u8) -> impl Iterator<Item = RecallSource> {
    [
        (SEMANTIC_SOURCE_MASK, RecallSource::SemanticAnn),
        (CATEGORY_SOURCE_MASK, RecallSource::CategoryProfile),
        (RECENT_SOURCE_MASK, RecallSource::RecentItemSimilarity),
    ]
    .into_iter()
    .filter_map(move |(source_mask, source)| {
        if mask & source_mask != 0 {
            Some(source)
        } else {
            None
        }
    })
}
