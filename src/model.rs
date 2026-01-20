//! 数据模型定义

use rand::Rng;
use serde::{Deserialize, Serialize};

pub const DIM: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub embedding: Vec<f32>,
}

/// 用于从 JSON 加载的临时结构（不含 embedding 和 popularity）
#[derive(Debug, Deserialize)]
pub struct ItemJson {
    pub id: u64,
    #[serde(rename = "title")]
    pub name: String,
    pub category: String,
    pub image_url: String,
    pub price: f32,
}

/// 完整的 Item 结构（用于存储和运行时）
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

/// 类别锚点向量
pub fn category_base_vector(category: &str) -> Vec<f32> {
    let mut vec = vec![0.0f32; DIM];
    let range = match category {
        "Electronics" => 0..16,
        "Books" => 16..32,
        "Home" => 32..48,
        "Clothing" => 48..64,
        _ => 0..16,
    };
    for i in range {
        vec[i] = 1.0;
    }
    vec
}

pub fn generate_category_embedding(category: &str) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let base = category_base_vector(category);
    let vec: Vec<f32> = base.iter()
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
    let vec: Vec<f32> = combined.iter()
        .map(|&v| v + rng.gen::<f32>() * 0.1 - 0.05)
        .collect();
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec.into_iter().map(|x| x / norm).collect()
}
