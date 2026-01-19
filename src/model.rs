//! 数据模型定义

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub id: u64,
    pub name: String,
    pub embedding: Vec<f32>,
    pub popularity: f32,  // 0.0 ~ 1.0
}

impl User {
    pub fn new(id: u64, embedding: Vec<f32>) -> Self {
        Self { id, embedding }
    }
}

impl Item {
    pub fn new(id: u64, name: impl Into<String>, embedding: Vec<f32>, popularity: f32) -> Self {
        Self {
            id,
            name: name.into(),
            embedding,
            popularity,
        }
    }
}

/// 应用状态
pub struct AppState {
    pub users: Vec<User>,
    pub items: Vec<Item>,
    pub item_map: HashMap<u64, usize>,  // id -> index 快速查找
}

fn random_normalized_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec.into_iter().map(|x| x / norm).collect()
}

pub fn init_data() -> Arc<AppState> {
    const DIM: usize = 64;
    const NUM_USERS: u64 = 100;
    const NUM_ITEMS: u64 = 1000;

    let mut rng = rand::thread_rng();

    let users: Vec<User> = (1..=NUM_USERS)
        .map(|id| User::new(id, random_normalized_vector(DIM)))
        .collect();

    let items: Vec<Item> = (1..=NUM_ITEMS)
        .map(|id| {
            Item::new(
                id,
                format!("Item_{}", id),
                random_normalized_vector(DIM),
                rng.gen::<f32>(),  // 随机 popularity
            )
        })
        .collect();

    let item_map: HashMap<u64, usize> = items.iter()
        .enumerate()
        .map(|(idx, item)| (item.id, idx))
        .collect();

    Arc::new(AppState { users, items, item_map })
}
