//! Lightweight reranking rules for diversity and exploration.

use crate::recommendation::types::{RecommendationConfig, RecommendedItem};
use std::collections::{HashMap, HashSet};

pub(super) fn rerank_for_diversity(
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
