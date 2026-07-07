use super::*;
use crate::model::{category_base_vector, Item, User, DIM};
use std::collections::HashSet;

fn normalized_category(category: &str) -> Vec<f32> {
    let base = category_base_vector(category);
    let norm = base.iter().map(|v| v * v).sum::<f32>().sqrt();
    base.into_iter().map(|v| v / norm).collect()
}

fn item(id: u64, category: &str, popularity: f32, price: f32) -> Item {
    Item {
        id,
        name: format!("Item {}", id),
        category: category.to_string(),
        image_url: String::new(),
        price,
        embedding: normalized_category(category),
        popularity,
    }
}

#[test]
fn pipeline_merges_sources_filters_seen_and_explains_results() {
    let user = User {
        id: 1,
        name: "Test user".to_string(),
        embedding: normalized_category("Books"),
    };
    let items = vec![
        item(1, "Books", 0.40, 20.0),
        item(2, "Books", 0.90, 22.0),
        item(3, "Electronics", 0.95, 200.0),
        item(4, "Home", 0.80, 35.0),
        item(5, "Clothing", 0.30, 45.0),
    ];
    let semantic_hits = vec![(1, 0.95), (2, 0.92), (3, 0.10)];
    let seen: HashSet<u64> = [1].into_iter().collect();

    let output = build_recommendations(
        &user,
        &items,
        &semantic_hits,
        &|item_id| seen.contains(&item_id),
        RecommendationConfig {
            limit: 4,
            ..Default::default()
        },
    );

    assert_eq!(output.filtered_count, 1);
    assert_eq!(output.items.len(), 4);
    assert!(!output.items.iter().any(|item| item.item_id == 1));
    assert!(output.items.iter().any(|item| item.source == "mixed"));
    assert!(output.items.iter().all(|item| !item.reason.is_empty()));
    assert!(output
        .items
        .iter()
        .any(|item| item.reason == "exploration_slot"));
}

#[test]
fn pipeline_limits_category_dominance_in_top_results() {
    let user = User {
        id: 1,
        name: "Book heavy user".to_string(),
        embedding: normalized_category("Books"),
    };
    let mut items = vec![
        item(1, "Books", 0.99, 20.0),
        item(2, "Books", 0.98, 21.0),
        item(3, "Books", 0.97, 22.0),
        item(4, "Books", 0.96, 23.0),
        item(5, "Electronics", 0.80, 200.0),
        item(6, "Home", 0.70, 35.0),
    ];
    for item in &mut items {
        item.embedding = vec![0.0; DIM];
    }
    let semantic_hits = vec![
        (1, 0.99),
        (2, 0.98),
        (3, 0.97),
        (4, 0.96),
        (5, 0.30),
        (6, 0.20),
    ];

    let output = build_recommendations(
        &user,
        &items,
        &semantic_hits,
        &|_| false,
        RecommendationConfig {
            limit: 5,
            max_per_category: 2,
            ..Default::default()
        },
    );

    let books_in_top = output
        .items
        .iter()
        .filter(|item| item.category == "Books")
        .count();
    assert!(books_in_top <= 2);
    assert!(output.items.iter().any(|item| item.category != "Books"));
}

#[test]
fn default_ranking_strategy_uses_fixed_weights() {
    let user = User {
        id: 1,
        name: "Test user".to_string(),
        embedding: normalized_category("Books"),
    };
    let items = vec![item(1, "Books", 0.80, 20.0)];
    let output = build_recommendations(
        &user,
        &items,
        &[(1, 0.70)],
        &|_| false,
        RecommendationConfig {
            limit: 1,
            exploration_slots: 0,
            ..Default::default()
        },
    );

    assert_eq!(output.items[0].ranking_strategy, "fixed_weights");
}

#[test]
fn reserved_machine_learning_strategy_keeps_current_scores_until_model_exists() {
    let user = User {
        id: 1,
        name: "Test user".to_string(),
        embedding: normalized_category("Books"),
    };
    let items = vec![
        item(1, "Books", 0.80, 20.0),
        item(2, "Electronics", 0.95, 200.0),
    ];
    let fixed = build_recommendations(
        &user,
        &items,
        &[(1, 0.70), (2, 0.20)],
        &|_| false,
        RecommendationConfig {
            limit: 2,
            exploration_slots: 0,
            ranking_strategy: RankingStrategyKind::FixedWeights,
            ..Default::default()
        },
    );
    let reserved = build_recommendations(
        &user,
        &items,
        &[(1, 0.70), (2, 0.20)],
        &|_| false,
        RecommendationConfig {
            limit: 2,
            exploration_slots: 0,
            ranking_strategy: RankingStrategyKind::MachineLearningReserved,
            ..Default::default()
        },
    );

    assert_eq!(
        reserved.items[0].ranking_strategy,
        "machine_learning_reserved"
    );
    assert_eq!(fixed.items[0].item_id, reserved.items[0].item_id);
    assert!((fixed.items[0].final_score - reserved.items[0].final_score).abs() < 1e-6);
}
