//! Data model definitions.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIM: usize = 384;

/// Product categories aligned with `assets/products.json`.
pub const PRODUCT_CATEGORIES: &[&str] = &[
    "Electronics",
    "Books",
    "Home",
    "Clothing",
    "Sports",
    "Beauty",
    "Toys",
    "Food",
    "Automotive",
    "Misc",
];

const CATEGORY_SLICE: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub profile: UserProfile,
}

/// Offline user features used for recall and ranking cold-start users.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub category_weights: HashMap<String, f32>,
    pub budget_min: f32,
    pub budget_max: f32,
}

impl Default for UserProfile {
    fn default() -> Self {
        Self {
            category_weights: HashMap::new(),
            budget_min: 0.0,
            budget_max: 0.0,
        }
    }
}

impl UserProfile {
    pub fn build(category_weights: &[(&str, f32)], budget_min: f32, budget_max: f32) -> Self {
        let mut weights = HashMap::new();
        for (category, weight) in category_weights {
            weights.insert((*category).to_string(), weight.clamp(0.0, 1.0));
        }
        Self {
            category_weights: weights,
            budget_min,
            budget_max,
        }
    }

    pub fn top_categories(&self, limit: usize) -> Vec<String> {
        let mut ranked: Vec<(String, f32)> = self
            .category_weights
            .iter()
            .map(|(category, weight)| (category.clone(), *weight))
            .collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
            .into_iter()
            .take(limit)
            .map(|(category, _)| category)
            .collect()
    }
}

/// Temporary struct for JSON loading (no embedding or popularity).
#[derive(Debug, Deserialize)]
pub struct ItemJson {
    pub id: u64,
    #[serde(rename = "title")]
    pub name: String,
    pub category: String,
    pub image_url: String,
    pub price: f32,
}

/// Full item struct for storage and runtime use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub id: u64,
    pub name: String,
    pub category: String,
    pub image_url: String,
    pub price: f32,
    pub embedding: Vec<f32>,
    pub popularity: f32,
}

impl Item {
    pub fn from_json(json: ItemJson, embedding: Vec<f32>, popularity: f32) -> Self {
        Self {
            id: json.id,
            name: json.name,
            category: json.category,
            image_url: json.image_url,
            price: json.price,
            embedding,
            popularity,
        }
    }

    #[cfg(test)]
    pub fn new(id: u64, name: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id,
            name: name.into(),
            category: "Test".to_string(),
            image_url: String::new(),
            price: 0.0,
            embedding,
            popularity: 0.5,
        }
    }
}

/// Category anchor vector.
pub fn category_base_vector(category: &str) -> Vec<f32> {
    let mut vec = vec![0.0f32; DIM];
    let range = category_dimension_range(category);
    for i in range {
        vec[i] = 1.0;
    }
    vec
}

pub fn category_dimension_range(category: &str) -> std::ops::Range<usize> {
    let index = PRODUCT_CATEGORIES
        .iter()
        .position(|value| *value == category)
        .unwrap_or(PRODUCT_CATEGORIES.len() - 1);
    let start = index * CATEGORY_SLICE;
    start..start + CATEGORY_SLICE
}

pub fn generate_category_embedding(category: &str) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let base = category_base_vector(category);
    let vec: Vec<f32> = base
        .iter()
        .map(|&v| v + rng.gen::<f32>() * 0.2 - 0.1)
        .collect();
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec.into_iter().map(|x| x / norm).collect()
}

pub fn generate_user_embedding(categories: &[&str]) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut combined = vec![0.0f32; DIM];
    for cat in categories {
        let base = category_base_vector(cat);
        for (i, &v) in base.iter().enumerate() {
            combined[i] += v;
        }
    }
    let vec: Vec<f32> = combined
        .iter()
        .map(|&v| v + rng.gen::<f32>() * 0.1 - 0.05)
        .collect();
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec.into_iter().map(|x| x / norm).collect()
}

/// Generate a fully random vector (for cold-start/noise users).
pub fn generate_random_embedding() -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let vec: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec.into_iter().map(|x| x / norm).collect()
}

pub fn build_user(
    id: u64,
    name: impl Into<String>,
    category_weights: &[(&str, f32)],
    budget_min: f32,
    budget_max: f32,
) -> User {
    let categories: Vec<&str> = category_weights
        .iter()
        .map(|(category, _)| *category)
        .collect();
    User {
        id,
        name: name.into(),
        embedding: generate_user_embedding(&categories),
        profile: UserProfile::build(category_weights, budget_min, budget_max),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_categories_have_distinct_dimension_ranges() {
        let electronics = category_dimension_range("Electronics");
        let books = category_dimension_range("Books");
        let misc = category_dimension_range("Misc");
        let unknown = category_dimension_range("Unknown");

        assert_eq!(electronics, 0..16);
        assert_eq!(books, 16..32);
        assert_eq!(misc, 144..160);
        assert_eq!(unknown, misc);
    }
}
