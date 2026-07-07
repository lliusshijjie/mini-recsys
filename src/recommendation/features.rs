//! Feature extraction helpers for recommendation ranking.

use crate::model::{category_base_vector, Item, User};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(super) struct PriceStats {
    average: f32,
    max_delta: f32,
}

impl PriceStats {
    pub(super) fn from_items(items: &[Item]) -> Self {
        let average = items.iter().map(|item| item.price).sum::<f32>() / items.len() as f32;
        let max_delta = items
            .iter()
            .map(|item| (item.price - average).abs())
            .fold(0.0, f32::max);
        Self { average, max_delta }
    }

    pub(super) fn affinity(&self, price: f32) -> f32 {
        if self.max_delta == 0.0 {
            return 1.0;
        }

        (1.0 - ((price - self.average).abs() / self.max_delta)).clamp(0.0, 1.0)
    }
}

pub(super) fn user_category_scores(user: &User, items: &[Item]) -> HashMap<String, f32> {
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

pub(super) fn normalize_score(score: f32) -> f32 {
    score.clamp(0.0, 1.0)
}
