use super::*;
use crate::behavior::{BehaviorEvent, EventType, UserPreferences};
use crate::model::{category_base_vector, Item, User, UserProfile, DIM};
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
        profile: UserProfile::default(),
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
        profile: UserProfile::default(),
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
        profile: UserProfile::default(),
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
        profile: UserProfile::default(),
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

#[test]
fn ranking_changes_when_feedback_preferences_are_present() {
    let user = User {
        id: 1,
        name: "Test user".to_string(),
        embedding: normalized_category("Books"),
        profile: UserProfile::default(),
    };
    let items = vec![
        item(1, "Books", 0.80, 20.0),
        item(2, "Electronics", 0.80, 20.0),
    ];
    let semantic_hits = vec![(1, 0.80), (2, 0.80)];

    let baseline = build_recommendations(
        &user,
        &items,
        &semantic_hits,
        &|_| false,
        RecommendationConfig {
            limit: 2,
            exploration_slots: 0,
            ..Default::default()
        },
    );

    let mut preferences = UserPreferences::default();
    preferences.set_item_weight(2, 1.0);
    preferences.set_category_weight("Electronics", 1.0);

    let with_feedback = build_recommendations(
        &user,
        &items,
        &semantic_hits,
        &|_| false,
        RecommendationConfig {
            limit: 2,
            exploration_slots: 0,
            preferences: Some(preferences),
            ..Default::default()
        },
    );

    assert_eq!(baseline.items[0].item_id, 1);
    assert_eq!(with_feedback.items[0].item_id, 2);
    assert!(with_feedback.items[0].feedback_score > 0.0);
    assert_eq!(with_feedback.items[0].reason, "feedback_match");
}

#[test]
fn output_includes_debug_metadata_for_evaluation() {
    let user = User {
        id: 1,
        name: "Test user".to_string(),
        embedding: normalized_category("Books"),
        profile: UserProfile::default(),
    };
    let items = vec![
        item(1, "Books", 0.80, 20.0),
        item(2, "Electronics", 0.90, 30.0),
    ];

    let output = build_recommendations(
        &user,
        &items,
        &[(1, 0.90), (2, 0.30)],
        &|_| false,
        RecommendationConfig {
            limit: 2,
            exploration_slots: 0,
            ..Default::default()
        },
    );

    assert!(output.debug.candidate_count >= output.items.len());
    assert!(!output.debug.candidates.is_empty());
    assert_eq!(
        output.debug.category_distribution.values().sum::<usize>(),
        output.items.len()
    );
    assert_eq!(
        output.debug.source_distribution.values().sum::<usize>(),
        output.items.len()
    );
}

#[test]
fn recent_click_recall_adds_similar_items() {
    let user = User {
        id: 1,
        name: "Recent interest user".to_string(),
        embedding: vec![0.0; DIM],
        profile: UserProfile::default(),
    };
    let items = vec![
        item(10, "Books", 0.20, 20.0),
        item(11, "Books", 0.70, 22.0),
        item(12, "Electronics", 0.95, 200.0),
    ];
    let recent_events = vec![BehaviorEvent::new(1, 10, EventType::Click, "Books")];

    let output = build_recommendations(
        &user,
        &items,
        &[],
        &|_| false,
        RecommendationConfig {
            limit: 3,
            exploration_slots: 0,
            recent_events,
            ..Default::default()
        },
    );

    let similar_item = output
        .items
        .iter()
        .find(|item| item.item_id == 11)
        .expect("similar item should be recalled from recent click");

    assert_eq!(similar_item.source, "recent_item_similarity");
    assert_eq!(similar_item.reason, "similar_to_recent_interest");
    assert!(output
        .debug
        .source_distribution
        .contains_key("recent_item_similarity"));
}

#[test]
fn recent_item_recall_ignores_impression_and_dismiss_events() {
    let user = User {
        id: 1,
        name: "Passive event user".to_string(),
        embedding: vec![0.0; DIM],
        profile: UserProfile::default(),
    };
    let items = vec![
        item(10, "Books", 0.20, 20.0),
        item(11, "Books", 0.70, 22.0),
        item(12, "Electronics", 0.95, 200.0),
    ];
    let recent_events = vec![
        BehaviorEvent::new(1, 10, EventType::Impression, "Books"),
        BehaviorEvent::new(1, 10, EventType::Dismiss, "Books"),
    ];

    let output = build_recommendations(
        &user,
        &items,
        &[],
        &|_| false,
        RecommendationConfig {
            limit: 3,
            exploration_slots: 0,
            recent_events,
            ..Default::default()
        },
    );

    assert!(!output
        .debug
        .candidates
        .iter()
        .any(|candidate| candidate.source == "recent_item_similarity"));
    assert!(!output
        .debug
        .source_distribution
        .contains_key("recent_item_similarity"));
}
