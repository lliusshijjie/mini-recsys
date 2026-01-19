//! 数据模型定义

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

const DIM: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub id: u64,
    #[serde(rename = "title")]
    pub name: String,
    pub category: String,
    pub image_url: String,
    pub price: f32,
    #[serde(skip)]
    pub embedding: Vec<f32>,
    #[serde(skip)]
    pub popularity: f32,
}

pub struct AppState {
    pub users: Vec<User>,
    pub items: Vec<Item>,
    pub item_map: HashMap<u64, usize>,
}

/// 类别锚点向量：每个类别占据向量空间的不同区域（正交性）
/// Electronics: 前16维为1, Books: 16-32维为1, Home: 32-48维为1, Clothing: 48-64维为1
fn category_base_vector(category: &str) -> Vec<f32> {
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

/// 生成带噪声的类别向量并归一化
fn generate_category_embedding(category: &str) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let base = category_base_vector(category);
    
    // 添加少量噪声
    let vec: Vec<f32> = base.iter()
        .map(|&v| v + rng.gen::<f32>() * 0.2 - 0.1)
        .collect();
    
    // L2 归一化
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec.into_iter().map(|x| x / norm).collect()
}

/// 生成混合类别的用户向量
fn generate_user_embedding(categories: &[&str]) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut combined = vec![0.0f32; DIM];
    
    for cat in categories {
        let base = category_base_vector(cat);
        for (i, &v) in base.iter().enumerate() {
            combined[i] += v;
        }
    }
    
    // 添加噪声
    let vec: Vec<f32> = combined.iter()
        .map(|&v| v + rng.gen::<f32>() * 0.1 - 0.05)
        .collect();
    
    // 归一化
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec.into_iter().map(|x| x / norm).collect()
}

pub fn init_data() -> Arc<AppState> {
    let mut rng = rand::thread_rng();

    // 加载商品数据
    let json_str = fs::read_to_string("assets/products.json")
        .expect("Failed to read products.json");
    let mut items: Vec<Item> = serde_json::from_str(&json_str)
        .expect("Failed to parse products.json");

    // 为每个商品生成类别向量和随机热度
    for item in &mut items {
        item.embedding = generate_category_embedding(&item.category);
        item.popularity = rng.gen::<f32>();
    }

    // 创建固定用户
    let users = vec![
        User {
            id: 1,
            name: "Coder (Electronics + Books)".into(),
            embedding: generate_user_embedding(&["Electronics", "Books"]),
        },
        User {
            id: 2,
            name: "Home Maker (Home)".into(),
            embedding: generate_user_embedding(&["Home"]),
        },
        User {
            id: 3,
            name: "Fashionista (Clothing)".into(),
            embedding: generate_user_embedding(&["Clothing"]),
        },
    ];

    let item_map: HashMap<u64, usize> = items.iter()
        .enumerate()
        .map(|(idx, item)| (item.id, idx))
        .collect();

    Arc::new(AppState { users, items, item_map })
}
