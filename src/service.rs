//! 业务逻辑层
//! 
//! 包含推荐系统的核心业务逻辑：召回 (Recall) 和排序 (Rank)

use crate::ffi;
use crate::model::{Item, User};

/// 推荐服务
pub struct RecommendationService {
    items: Vec<Item>,
}

impl RecommendationService {
    /// 创建推荐服务实例
    pub fn new(items: Vec<Item>) -> Self {
        Self { items }
    }

    /// 为用户推荐物品
    /// 
    /// # Arguments
    /// * `user` - 目标用户
    /// * `top_k` - 返回的推荐数量
    /// 
    /// # Returns
    /// 按相似度排序的物品列表（相似度从高到低）
    pub fn recommend(&self, user: &User, top_k: usize) -> Vec<(&Item, f32)> {
        // 计算用户与所有物品的相似度
        let mut scored_items: Vec<(&Item, f32)> = self
            .items
            .iter()
            .filter_map(|item| {
                // 使用 FFI 调用 C++ 点积函数计算相似度
                ffi::compute_dot_product(&user.embedding, &item.embedding)
                    .map(|score| (item, score))
            })
            .collect();

        // 按相似度降序排序
        scored_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 返回 Top-K
        scored_items.into_iter().take(top_k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommendation_service() {
        // 创建测试物品
        let items = vec![
            Item::new(1, "Item A", vec![1.0, 0.0, 0.0]),
            Item::new(2, "Item B", vec![0.0, 1.0, 0.0]),
            Item::new(3, "Item C", vec![0.5, 0.5, 0.0]),
        ];

        let service = RecommendationService::new(items);

        // 创建测试用户 - 偏好第一维度
        let user = User::new(1, vec![1.0, 0.0, 0.0]);

        let recommendations = service.recommend(&user, 2);

        // 应该返回 Item A (分数 1.0) 和 Item C (分数 0.5)
        assert_eq!(recommendations.len(), 2);
        assert_eq!(recommendations[0].0.name, "Item A");
        assert_eq!(recommendations[1].0.name, "Item C");
    }
}
