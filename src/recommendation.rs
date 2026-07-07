use crate::model::{category_base_vector, Item, User};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const DEFAULT_LIMIT: usize = 10;
const DEFAULT_MAX_PER_CATEGORY: usize = 3;
const DEFAULT_EXPLORATION_SLOTS: usize = 1;
const RECALL_MULTIPLIER: usize = 4;
const MIN_RECALL_POOL: usize = 20;

#[derive(Debug, Clone)]
pub struct RecommendationConfig {
    pub limit: usize,
    pub max_per_category: usize,
    pub exploration_slots: usize,
}

impl Default for RecommendationConfig {
    fn default() -> Self {
        Self {
            limit: DEFAULT_LIMIT,
            max_per_category: DEFAULT_MAX_PER_CATEGORY,
            exploration_slots: DEFAULT_EXPLORATION_SLOTS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecommendationOutput {
    pub items: Vec<RecommendedItem>,
    pub filtered_count: usize,
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
    pub final_score: f32,
    pub source: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
struct Candidate {
    item_id: u64,
    semantic_score: f32,
    sources: HashSet<&'static str>,
}

impl Candidate {
    fn new(item_id: u64) -> Self {
        Self {
            item_id,
            semantic_score: 0.0,
            sources: HashSet::new(),
        }
    }

    fn add_source(&mut self, source: &'static str) {
        self.sources.insert(source);
    }
}

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
        };
    }

    let item_map: HashMap<u64, &Item> = items.iter().map(|item| (item.id, item)).collect();
    let category_scores = user_category_scores(user, items);
    let price_stats = PriceStats::from_items(items);
    let candidates = recall_candidates(items, semantic_hits, &category_scores, &config);

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
        ));
    }

    ranked.sort_by(score_desc);
    let reranked = rerank_for_diversity(ranked, &config);

    RecommendationOutput {
        items: reranked,
        filtered_count,
    }
}

fn recall_candidates(
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

fn rank_candidate(
    item: &Item,
    candidate: &Candidate,
    category_scores: &HashMap<String, f32>,
    price_stats: &PriceStats,
) -> RecommendedItem {
    let semantic_score = candidate.semantic_score;
    let category_score = category_scores
        .get(&item.category)
        .copied()
        .unwrap_or_default();
    let popularity = normalize_score(item.popularity);
    let price_affinity = price_stats.affinity(item.price);
    let novelty = 1.0 - popularity;
    let final_score = semantic_score * 0.50
        + category_score * 0.20
        + popularity * 0.20
        + ((price_affinity + novelty) / 2.0) * 0.10;

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
        source,
        reason,
    }
}

fn rerank_for_diversity(
    mut ranked: Vec<RecommendedItem>,
    config: &RecommendationConfig,
) -> Vec<RecommendedItem> {
    let limit = config.limit;
    let exploration_slots = config.exploration_slots.min(limit);
    let primary_limit = limit.saturating_sub(exploration_slots);
    let mut selected = Vec::new();
    let mut selected_ids = HashSet::new();
    let mut category_counts: HashMap<String, usize> = HashMap::new();

    select_with_category_cap(
        &mut ranked,
        &mut selected,
        &mut selected_ids,
        &mut category_counts,
        primary_limit,
        config.max_per_category,
    );

    for _ in 0..exploration_slots {
        if selected.len() >= limit {
            break;
        }

        if let Some(pos) = find_exploration_candidate(
            &ranked,
            &selected_ids,
            &category_counts,
            config.max_per_category,
        ) {
            let mut exploration = ranked.remove(pos);
            exploration.reason = "exploration_slot".to_string();
            *category_counts
                .entry(exploration.category.clone())
                .or_insert(0) += 1;
            selected_ids.insert(exploration.item_id);
            selected.push(exploration);
        }
    }

    select_with_category_cap(
        &mut ranked,
        &mut selected,
        &mut selected_ids,
        &mut category_counts,
        limit,
        config.max_per_category,
    );

    selected
}

fn find_exploration_candidate(
    ranked: &[RecommendedItem],
    selected_ids: &HashSet<u64>,
    category_counts: &HashMap<String, usize>,
    max_per_category: usize,
) -> Option<usize> {
    ranked
        .iter()
        .position(|item| {
            can_select(item, selected_ids, category_counts, max_per_category)
                && (item.category_score > 0.15 || item.popularity > 0.5)
        })
        .or_else(|| {
            ranked
                .iter()
                .position(|item| can_select(item, selected_ids, category_counts, max_per_category))
        })
}

fn can_select(
    item: &RecommendedItem,
    selected_ids: &HashSet<u64>,
    category_counts: &HashMap<String, usize>,
    max_per_category: usize,
) -> bool {
    !selected_ids.contains(&item.item_id)
        && category_counts
            .get(&item.category)
            .copied()
            .unwrap_or_default()
            < max_per_category
}

fn select_with_category_cap(
    ranked: &mut Vec<RecommendedItem>,
    selected: &mut Vec<RecommendedItem>,
    selected_ids: &mut HashSet<u64>,
    category_counts: &mut HashMap<String, usize>,
    target_len: usize,
    max_per_category: usize,
) {
    let mut idx = 0;
    while selected.len() < target_len && idx < ranked.len() {
        let item = &ranked[idx];
        if can_select(item, selected_ids, category_counts, max_per_category) {
            let item = ranked.remove(idx);
            selected_ids.insert(item.item_id);
            *category_counts.entry(item.category.clone()).or_insert(0) += 1;
            selected.push(item);
        } else {
            idx += 1;
        }
    }
}

fn user_category_scores(user: &User, items: &[Item]) -> HashMap<String, f32> {
    let mut scores = HashMap::new();
    for item in items {
        scores.entry(item.category.clone()).or_insert_with(|| {
            let base = category_base_vector(&item.category);
            cosine_like_score(&user.embedding, &base)
        });
    }
    scores
}

fn cosine_like_score(user_embedding: &[f32], base: &[f32]) -> f32 {
    if user_embedding.is_empty() || user_embedding.len() != base.len() {
        return 0.0;
    }

    let base_norm = base.iter().map(|value| value * value).sum::<f32>().sqrt();
    if base_norm == 0.0 {
        return 0.0;
    }

    let dot = user_embedding
        .iter()
        .zip(base.iter())
        .map(|(left, right)| left * (right / base_norm))
        .sum::<f32>();
    normalize_score(dot)
}

fn normalize_score(score: f32) -> f32 {
    score.clamp(0.0, 1.0)
}

fn source_label(sources: &HashSet<&'static str>) -> String {
    if sources.len() > 1 {
        return "mixed".to_string();
    }

    sources
        .iter()
        .next()
        .copied()
        .unwrap_or("unknown")
        .to_string()
}

fn reason_for(source: &str, semantic_score: f32, category_score: f32) -> String {
    if semantic_score >= 0.50 {
        "semantic_match".to_string()
    } else if category_score >= 0.50 {
        "category_match".to_string()
    } else if source == "popular" || source == "mixed" {
        "popular_item".to_string()
    } else {
        "category_match".to_string()
    }
}

fn score_desc(left: &RecommendedItem, right: &RecommendedItem) -> Ordering {
    right
        .final_score
        .partial_cmp(&left.final_score)
        .unwrap_or(Ordering::Equal)
}

#[derive(Debug, Clone)]
struct PriceStats {
    average: f32,
    max_delta: f32,
}

impl PriceStats {
    fn from_items(items: &[Item]) -> Self {
        let average = items.iter().map(|item| item.price).sum::<f32>() / items.len() as f32;
        let max_delta = items
            .iter()
            .map(|item| (item.price - average).abs())
            .fold(0.0, f32::max);
        Self { average, max_delta }
    }

    fn affinity(&self, price: f32) -> f32 {
        if self.max_delta == 0.0 {
            return 1.0;
        }

        (1.0 - ((price - self.average).abs() / self.max_delta)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{category_base_vector, Item, User, DIM};
    use std::collections::HashSet;

    fn normalized_category(category: &str) -> Vec<f32> {
        let base = category_base_vector(category);
        let norm = base.iter().map(|v| v * v).sum::<f32>().sqrt();
        base.into_iter().map(|v| v / norm).collect()
    }

    fn item(id: u64, category: &str, popularity: f32, price: f32) -> Item {
        Item {
            id,
            name: format!("Item {}", id),
            category: category.to_string(),
            image_url: String::new(),
            price,
            embedding: normalized_category(category),
            popularity,
        }
    }

    #[test]
    fn pipeline_merges_sources_filters_seen_and_explains_results() {
        let user = User {
            id: 1,
            name: "Test user".to_string(),
            embedding: normalized_category("Books"),
        };
        let items = vec![
            item(1, "Books", 0.40, 20.0),
            item(2, "Books", 0.90, 22.0),
            item(3, "Electronics", 0.95, 200.0),
            item(4, "Home", 0.80, 35.0),
            item(5, "Clothing", 0.30, 45.0),
        ];
        let semantic_hits = vec![(1, 0.95), (2, 0.92), (3, 0.10)];
        let seen: HashSet<u64> = [1].into_iter().collect();

        let output = build_recommendations(
            &user,
            &items,
            &semantic_hits,
            &|item_id| seen.contains(&item_id),
            RecommendationConfig {
                limit: 4,
                ..Default::default()
            },
        );

        assert_eq!(output.filtered_count, 1);
        assert_eq!(output.items.len(), 4);
        assert!(!output.items.iter().any(|item| item.item_id == 1));
        assert!(output.items.iter().any(|item| item.source == "mixed"));
        assert!(output.items.iter().all(|item| !item.reason.is_empty()));
        assert!(output
            .items
            .iter()
            .any(|item| item.reason == "exploration_slot"));
    }

    #[test]
    fn pipeline_limits_category_dominance_in_top_results() {
        let user = User {
            id: 1,
            name: "Book heavy user".to_string(),
            embedding: normalized_category("Books"),
        };
        let mut items = vec![
            item(1, "Books", 0.99, 20.0),
            item(2, "Books", 0.98, 21.0),
            item(3, "Books", 0.97, 22.0),
            item(4, "Books", 0.96, 23.0),
            item(5, "Electronics", 0.80, 200.0),
            item(6, "Home", 0.70, 35.0),
        ];
        for item in &mut items {
            item.embedding = vec![0.0; DIM];
        }
        let semantic_hits = vec![
            (1, 0.99),
            (2, 0.98),
            (3, 0.97),
            (4, 0.96),
            (5, 0.30),
            (6, 0.20),
        ];

        let output = build_recommendations(
            &user,
            &items,
            &semantic_hits,
            &|_| false,
            RecommendationConfig {
                limit: 5,
                max_per_category: 2,
                ..Default::default()
            },
        );

        let books_in_top = output
            .items
            .iter()
            .filter(|item| item.category == "Books")
            .count();
        assert!(books_in_top <= 2);
        assert!(output.items.iter().any(|item| item.category != "Books"));
    }
}
