use super::*;
use crate::behavior::{BehaviorEvent, EventType, UserPreferences};
use crate::ffi::{HnswConfig, HnswIndex};
use crate::model::{category_base_vector, Item, User, UserProfile, DIM};
use std::collections::HashSet;
use std::time::Instant;

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
fn indexed_pipeline_matches_default_pipeline_output() {
    let user = User {
        id: 1,
        name: "Indexed user".to_string(),
        embedding: normalized_category("Books"),
        profile: UserProfile::build(&[("Books", 0.8), ("Electronics", 0.4)], 10.0, 250.0),
    };
    let items = vec![
        item(1, "Books", 0.40, 20.0),
        item(2, "Books", 0.90, 22.0),
        item(3, "Electronics", 0.95, 200.0),
        item(4, "Home", 0.80, 35.0),
        item(5, "Clothing", 0.30, 45.0),
    ];
    let indexes = RecommendationIndexes::from_items(&items);
    let semantic_hits = vec![(1, 0.95), (2, 0.92), (3, 0.10)];
    let recent_events = vec![BehaviorEvent::new(1, 1, EventType::Click, "Books")];
    let mut preferences = UserPreferences::default();
    preferences.set_item_weight(3, 1.0);
    let seen: HashSet<u64> = [1].into_iter().collect();
    let config = RecommendationConfig {
        limit: 4,
        preferences: Some(preferences),
        recent_events,
        ..Default::default()
    };

    let baseline = build_recommendations(
        &user,
        &items,
        &semantic_hits,
        &|item_id| seen.contains(&item_id),
        config.clone(),
    );
    let indexed = build_recommendations_with_indexes(
        &user,
        &items,
        &indexes,
        &semantic_hits,
        &|item_id| seen.contains(&item_id),
        config,
    );

    assert_eq!(
        baseline
            .items
            .iter()
            .map(|item| item.item_id)
            .collect::<Vec<_>>(),
        indexed
            .items
            .iter()
            .map(|item| item.item_id)
            .collect::<Vec<_>>()
    );
    assert_eq!(baseline.filtered_count, indexed.filtered_count);
    assert_eq!(
        baseline.debug.candidate_count,
        indexed.debug.candidate_count
    );
    assert_eq!(
        baseline.debug.source_distribution,
        indexed.debug.source_distribution
    );
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
    assert!(output
        .debug
        .stage_durations_micros
        .contains_key("category_recall"));
    assert!(output
        .debug
        .stage_durations_micros
        .contains_key("recent_ann"));
    assert!(output
        .debug
        .stage_durations_micros
        .contains_key("popular_fallback"));
    assert!(output
        .debug
        .stage_durations_micros
        .contains_key("merge_rank"));
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
fn recent_ann_mode_uses_injected_ann_hits() {
    let user = User {
        id: 1,
        name: "Recent ANN user".to_string(),
        embedding: vec![0.0; DIM],
        profile: UserProfile::default(),
    };
    let items = vec![
        item(10, "Books", 0.20, 20.0),
        item(11, "Books", 0.70, 22.0),
        item(12, "Books", 0.60, 24.0),
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
            recent_recall_mode: RecentRecallMode::Ann,
            recent_ann_hits: vec![(10, vec![(12, 0.91)])],
            ..Default::default()
        },
    );

    let ann_item = output
        .items
        .iter()
        .find(|item| item.item_id == 12)
        .expect("ANN hit should be recalled from recent click");

    assert_eq!(ann_item.source, "recent_item_similarity");
    assert_eq!(ann_item.reason, "similar_to_recent_interest");
    assert!(!output
        .debug
        .candidates
        .iter()
        .any(|candidate| candidate.item_id == 11 && candidate.source == "recent_item_similarity"));
}

#[test]
fn recent_shadow_mode_records_quality_but_keeps_exact_results() {
    let user = User {
        id: 1,
        name: "Recent shadow user".to_string(),
        embedding: vec![0.0; DIM],
        profile: UserProfile::default(),
    };
    let items = vec![
        item(10, "Books", 0.20, 20.0),
        item(11, "Books", 0.70, 22.0),
        item(12, "Books", 0.60, 24.0),
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
            recent_recall_mode: RecentRecallMode::Shadow,
            recent_ann_hits: vec![(10, vec![(12, 0.91)])],
            ..Default::default()
        },
    );

    assert!(output.items.iter().any(|item| item.item_id == 11));
    assert!(output
        .debug
        .quality_metrics
        .contains_key("recent_ann_overlap"));
}

#[test]
#[ignore]
fn recent_ann_quality_against_exact() {
    run_recent_ann_quality_benchmark();
}

fn run_recent_ann_quality_benchmark() {
    let dataset_sizes = parse_usize_list_env("MINI_RECSYS_PERF_DATASETS", &[10_000]);
    let ann_k = env_usize("MINI_RECSYS_RECENT_ANN_K", 200);
    let required_recall = env_f32("MINI_RECSYS_RECENT_RECALL_THRESHOLD", 0.95);
    let required_top10 = env_f32("MINI_RECSYS_TOP10_OVERLAP_THRESHOLD", 0.90);

    for dataset_size in dataset_sizes {
        let items = benchmark_items(dataset_size);
        let indexes = RecommendationIndexes::from_items(&items);
        let hnsw = HnswIndex::new(&HnswConfig {
            dim: DIM,
            max_elements: dataset_size + 16,
            m: 16,
            ef_construction: 200,
            ef_search: ann_k.max(100),
        })
        .unwrap();
        for item in &items {
            hnsw.add_item(item.id, &item.embedding).unwrap();
        }

        let user = User {
            id: 1,
            name: "Recent ANN quality user".to_string(),
            embedding: vec![0.0; DIM],
            profile: UserProfile::default(),
        };
        let recent_events = vec![BehaviorEvent::new(1, 1, EventType::Click, "Books")];
        let seed_embedding = items
            .iter()
            .find(|item| item.id == 1)
            .map(|item| item.embedding.clone())
            .unwrap();

        let exact_started = Instant::now();
        let exact = build_recommendations_with_indexes(
            &user,
            &items,
            &indexes,
            &[],
            &|_| false,
            RecommendationConfig {
                limit: 10,
                exploration_slots: 0,
                recent_events: recent_events.clone(),
                recent_recall_mode: RecentRecallMode::Exact,
                ..Default::default()
            },
        );
        let exact_ms = exact_started.elapsed().as_secs_f64() * 1000.0;

        let ann_started = Instant::now();
        let ann_hits = hnsw.search_batch(&[seed_embedding], ann_k).unwrap();
        let ann = build_recommendations_with_indexes(
            &user,
            &items,
            &indexes,
            &[],
            &|_| false,
            RecommendationConfig {
                limit: 10,
                exploration_slots: 0,
                recent_events: recent_events.clone(),
                recent_recall_mode: RecentRecallMode::Ann,
                recent_ann_hits: vec![(1, ann_hits[0].clone())],
                ..Default::default()
            },
        );
        let ann_ms = ann_started.elapsed().as_secs_f64() * 1000.0;

        let shadow = build_recommendations_with_indexes(
            &user,
            &items,
            &indexes,
            &[],
            &|_| false,
            RecommendationConfig {
                limit: 10,
                exploration_slots: 0,
                recent_events,
                recent_recall_mode: RecentRecallMode::Shadow,
                recent_ann_hits: vec![(1, ann_hits[0].clone())],
                ..Default::default()
            },
        );

        let recall_overlap =
            overlap_ratio(&recent_candidate_ids(&exact), &recent_candidate_ids(&ann));
        let top10_overlap = overlap_ratio(&top_item_ids(&exact), &top_item_ids(&ann));
        let shadow_overlap = shadow
            .debug
            .quality_metrics
            .get("recent_ann_overlap")
            .copied()
            .unwrap_or_default();

        println!(
            "recent_ann_quality dataset={} ann_k={} exact_ms={:.2} ann_ms={:.2} recall_overlap={:.3} top10_overlap={:.3} shadow_overlap={:.3}",
            dataset_size, ann_k, exact_ms, ann_ms, recall_overlap, top10_overlap, shadow_overlap
        );

        assert!(
            recall_overlap >= required_recall,
            "recent ANN recall overlap {:.3} below threshold {:.3}",
            recall_overlap,
            required_recall
        );
        assert!(
            top10_overlap >= required_top10,
            "recent ANN top10 overlap {:.3} below threshold {:.3}",
            top10_overlap,
            required_top10
        );
    }
}

fn benchmark_items(count: usize) -> Vec<Item> {
    let categories = ["Books", "Electronics", "Home", "Clothing"];
    (1..=count)
        .map(|id| {
            let category = categories[(id - 1) % categories.len()];
            Item {
                id: id as u64,
                name: format!("Benchmark Item {}", id),
                category: category.to_string(),
                image_url: String::new(),
                price: 10.0 + (id % 100) as f32,
                embedding: benchmark_embedding(id as u64, category),
                popularity: ((id * 37) % 1000) as f32 / 1000.0,
            }
        })
        .collect()
}

fn benchmark_embedding(id: u64, category: &str) -> Vec<f32> {
    let mut vector = category_base_vector(category);
    let mut state = id ^ 0xA076_1D64_78BD_642F;
    for value in &mut vector {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = (((state >> 32) as u32) as f32 / u32::MAX as f32) * 0.08 - 0.04;
        *value += noise;
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    vector.into_iter().map(|value| value / norm).collect()
}

fn recent_candidate_ids(output: &RecommendationOutput) -> Vec<u64> {
    output
        .debug
        .candidates
        .iter()
        .filter(|candidate| candidate.source == "recent_item_similarity")
        .map(|candidate| candidate.item_id)
        .collect()
}

fn top_item_ids(output: &RecommendationOutput) -> Vec<u64> {
    output.items.iter().map(|item| item.item_id).collect()
}

fn overlap_ratio(left: &[u64], right: &[u64]) -> f32 {
    if left.is_empty() {
        return if right.is_empty() { 1.0 } else { 0.0 };
    }
    let right_ids: HashSet<u64> = right.iter().copied().collect();
    let overlap = left
        .iter()
        .filter(|item_id| right_ids.contains(item_id))
        .count();
    overlap as f32 / left.len() as f32
}

fn parse_usize_list_env(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
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
