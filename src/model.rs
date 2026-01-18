//! 数据模型定义
//! 
//! 定义推荐系统中的核心数据结构：User 和 Item

use serde::{Deserialize, Serialize};

/// 用户模型
/// 
/// # Fields
/// * `id` - 用户唯一标识
/// * `embedding` - 用户的向量表示 (用户画像)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub embedding: Vec<f32>,
}

/// 物品模型
/// 
/// # Fields
/// * `id` - 物品唯一标识
/// * `name` - 物品名称
/// * `embedding` - 物品的向量表示
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub id: u64,
    pub name: String,
    pub embedding: Vec<f32>,
}

impl User {
    /// 创建新用户
    pub fn new(id: u64, embedding: Vec<f32>) -> Self {
        Self { id, embedding }
    }
}

impl Item {
    /// 创建新物品
    pub fn new(id: u64, name: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id,
            name: name.into(),
            embedding,
        }
    }
}
