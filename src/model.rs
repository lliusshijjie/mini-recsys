//! 数据模型定义

use rand::Rng;
use serde::{Deserialize, Serialize};
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
}

impl User {
    pub fn new(id: u64, embedding: Vec<f32>) -> Self {
        Self { id, embedding }
    }
}

impl Item {
    pub fn new(id: u64, name: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id,
            name: name.into(),
            embedding,
        }
    }
}

/// 应用状态：存储用户和物品数据
/// 
/// 使用 Arc<AppState> 实现多线程共享：
/// - 数据在 init_data() 后不再修改，是只读的
/// - Arc 提供引用计数的共享所有权，允许多个线程同时持有引用
/// - 因为只读，不需要 Mutex（Mutex 只在需要写时用于互斥）
pub struct AppState {
    pub users: Vec<User>,
    pub items: Vec<Item>,
}

/// 生成归一化的随机向量
fn random_normalized_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    
    // L2 归一化
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec.into_iter().map(|x| x / norm).collect()
}

/// 初始化模拟数据
/// 生成 100 个用户和 1000 个物品，每个有 64 维归一化向量
pub fn init_data() -> Arc<AppState> {
    const DIM: usize = 64;
    const NUM_USERS: u64 = 100;
    const NUM_ITEMS: u64 = 1000;

    let users: Vec<User> = (1..=NUM_USERS)
        .map(|id| User::new(id, random_normalized_vector(DIM)))
        .collect();

    let items: Vec<Item> = (1..=NUM_ITEMS)
        .map(|id| Item::new(id, format!("Item_{}", id), random_normalized_vector(DIM)))
        .collect();

    Arc::new(AppState { users, items })
}
