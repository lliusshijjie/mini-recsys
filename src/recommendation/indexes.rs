//! Precomputed recommendation indexes shared across requests.

use crate::model::Item;
use crate::recommendation::features::PriceStats;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RecommendationIndexes {
    item_positions: HashMap<u64, usize>,
    popular_item_ids: Vec<u64>,
    category_item_ids: HashMap<String, Vec<u64>>,
    categories: Vec<String>,
    price_stats: PriceStats,
}

impl RecommendationIndexes {
    pub fn from_items(items: &[Item]) -> Self {
        let item_positions: HashMap<u64, usize> = items
            .iter()
            .enumerate()
            .map(|(index, item)| (item.id, index))
            .collect();

        let mut popular_item_ids: Vec<u64> = items.iter().map(|item| item.id).collect();
        popular_item_ids
            .sort_by(|left, right| compare_popularity(items, &item_positions, *left, *right));

        let mut category_item_ids: HashMap<String, Vec<u64>> = HashMap::new();
        let mut categories = Vec::new();
        for item in items {
            if !category_item_ids.contains_key(&item.category) {
                categories.push(item.category.clone());
            }
            category_item_ids
                .entry(item.category.clone())
                .or_default()
                .push(item.id);
        }
        for item_ids in category_item_ids.values_mut() {
            item_ids
                .sort_by(|left, right| compare_popularity(items, &item_positions, *left, *right));
        }

        Self {
            item_positions,
            popular_item_ids,
            category_item_ids,
            categories,
            price_stats: PriceStats::from_items(items),
        }
    }

    pub(super) fn item<'a>(&self, items: &'a [Item], item_id: u64) -> Option<&'a Item> {
        items.get(*self.item_positions.get(&item_id)?)
    }

    pub(super) fn popular_item_ids(&self) -> &[u64] {
        &self.popular_item_ids
    }

    pub(super) fn categories(&self) -> &[String] {
        &self.categories
    }

    pub(super) fn price_stats(&self) -> &PriceStats {
        &self.price_stats
    }

    pub(super) fn category_item_ids(&self, category: &str) -> Option<&[u64]> {
        self.category_item_ids
            .get(category)
            .map(|item_ids| item_ids.as_slice())
    }
}

fn compare_popularity(
    items: &[Item],
    item_positions: &HashMap<u64, usize>,
    left_id: u64,
    right_id: u64,
) -> Ordering {
    let left = item_positions
        .get(&left_id)
        .and_then(|index| items.get(*index));
    let right = item_positions
        .get(&right_id)
        .and_then(|index| items.get(*index));
    match (left, right) {
        (Some(left), Some(right)) => right
            .popularity
            .partial_cmp(&left.popularity)
            .unwrap_or(Ordering::Equal),
        _ => Ordering::Equal,
    }
}
