//! Application initialization, routing, and lifecycle support.

use crate::behavior::{BehaviorEvent, EventType, UserPreferences};
use crate::config::AppConfig;
use crate::embedding;
use crate::ffi::{add_item_to_hnsw, get_hnsw_count, hnsw_search, load_hnsw_index, save_hnsw_index};
use crate::hybrid;
use crate::model::{
    generate_category_embedding, generate_random_embedding, generate_user_embedding, Item,
    ItemJson, User, DIM,
};
use crate::observability::Metrics;
use crate::recommendation::{
    build_recommendations, RankingStrategyKind, RecommendationConfig, RecommendationOutput,
    RecommendedItem as PipelineRecommendedItem,
};
use crate::storage::Storage;
use crate::text_search::TextSearch;
use anyhow::Result;
use axum::body::Body;
use axum::http::{HeaderValue, Method};
use axum::middleware::{self, Next};
use axum::{
    extract::{Query, State},
    http::{Request, StatusCode},
    response::{Json, Response},
    routing::{get, post},
    Router,
};
use fastbloom_rs::Membership;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};

// ============================================================================
// AppState
// ============================================================================

pub struct AppState {
    pub storage: Arc<Storage>,
    pub users: Vec<User>,
    pub items: Vec<Item>,
    pub item_map: HashMap<u64, usize>,
    pub embedding_model: Option<Arc<embedding::EmbeddingModel>>,
    pub text_search: Arc<TextSearch>,
    pub metrics: Arc<Metrics>,
    pub readiness: Arc<ReadinessState>,
}

#[derive(Debug, Default)]
pub struct ReadinessState {
    ready: AtomicBool,
}

impl ReadinessState {
    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

// ============================================================================
// Request/Response
// ============================================================================

#[derive(Deserialize)]
struct RecommendQuery {
    uid: u64,
}

#[derive(Serialize)]
struct RecommendItem {
    item_id: u64,
    name: String,
    category: String,
    image_url: String,
    price: f32,
    sim_score: f32,
    category_score: f32,
    popularity: f32,
    price_affinity: f32,
    novelty: f32,
    feedback_score: f32,
    final_score: f32,
    ranking_strategy: String,
    source: String,
    reason: String,
}

#[derive(Serialize)]
struct UserInfo {
    id: u64,
    name: String,
}

#[derive(Serialize)]
struct RecommendResponse {
    user: UserInfo,
    recommendations: Vec<RecommendItem>,
    filtered_count: usize,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct UsersResponse {
    users: Vec<UserInfo>,
}

#[derive(Deserialize)]
struct MarkSeenRequest {
    uid: u64,
    item_ids: Vec<u64>,
}

#[derive(Serialize)]
struct MarkSeenResponse {
    marked: usize,
}

#[derive(Deserialize)]
struct EventRequest {
    uid: u64,
    item_id: u64,
    event_type: String,
}

#[derive(Debug, Serialize)]
struct EventResponse {
    recorded: bool,
    event_type: String,
    recent_event_count: usize,
    category_weight: f32,
    item_weight: f32,
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
}

#[derive(Serialize)]
struct SearchResponse {
    query: String,
    results: Vec<RecommendItem>,
}

#[derive(Serialize)]
struct DebugCandidateResponse {
    item_id: u64,
    semantic_score: f32,
    source: String,
}

#[derive(Serialize)]
struct DebugRecommendationResponse {
    user: UserInfo,
    recommendations: Vec<RecommendItem>,
    filtered_count: usize,
    candidate_count: usize,
    candidates: Vec<DebugCandidateResponse>,
    category_distribution: HashMap<String, usize>,
    source_distribution: HashMap<String, usize>,
    recent_events: Vec<BehaviorEvent>,
    preferences: UserPreferences,
}

// ============================================================================
// Handlers
// ============================================================================

async fn recommend_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RecommendQuery>,
) -> Result<Json<RecommendResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (user, output) = build_user_recommendation_output(&state, params.uid)?;
    let filtered_count = output.filtered_count;
    state
        .metrics
        .record_recommendation_candidates(output.debug.candidate_count);
    println!(
        "{}",
        json!({
            "event": "recommendation",
            "user_id": params.uid,
            "candidate_count": output.debug.candidate_count,
            "filtered_count": filtered_count,
            "source_distribution": output.debug.source_distribution,
        })
    );
    let recommendations = output
        .items
        .into_iter()
        .map(recommend_item_from_output)
        .collect();

    Ok(Json(RecommendResponse {
        user: UserInfo {
            id: user.id,
            name: user.name.clone(),
        },
        recommendations,
        filtered_count,
    }))
}

fn build_user_recommendation_output(
    state: &AppState,
    uid: u64,
) -> Result<(&User, RecommendationOutput), (StatusCode, Json<ErrorResponse>)> {
    let user = state.users.iter().find(|u| u.id == uid).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("User {} not found", uid),
            }),
        )
    })?;

    let semantic_hits = hnsw_search(&user.embedding, 100);
    let preferences = state.storage.get_user_preferences(uid).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to get preferences: {}", e),
            }),
        )
    })?;
    let filter = state.storage.get_user_filter(uid).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to get filter: {}", e),
            }),
        )
    })?;

    let output = build_recommendations(
        user,
        &state.items,
        &semantic_hits,
        &|item_id| filter.contains(&item_id.to_le_bytes()),
        RecommendationConfig {
            ranking_strategy: RankingStrategyKind::from_env(),
            preferences: Some(preferences),
            ..Default::default()
        },
    );

    Ok((user, output))
}

fn recommend_item_from_output(item: PipelineRecommendedItem) -> RecommendItem {
    RecommendItem {
        item_id: item.item_id,
        name: item.name,
        category: item.category,
        image_url: item.image_url,
        price: item.price,
        sim_score: item.semantic_score,
        category_score: item.category_score,
        popularity: item.popularity,
        price_affinity: item.price_affinity,
        novelty: item.novelty,
        feedback_score: item.feedback_score,
        final_score: item.final_score,
        ranking_strategy: item.ranking_strategy,
        source: item.source,
        reason: item.reason,
    }
}

async fn mark_seen_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<MarkSeenRequest>,
) -> Result<Json<MarkSeenResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Load the user's bloom filter.
    let mut filter = state.storage.get_user_filter(payload.uid).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to get filter: {}", e),
            }),
        )
    })?;

    // Insert all item IDs.
    for item_id in &payload.item_ids {
        filter.add(&item_id.to_le_bytes());
    }

    // Persist back to Sled.
    state
        .storage
        .save_user_filter(payload.uid, &filter)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to save filter: {}", e),
                }),
            )
        })?;

    Ok(Json(MarkSeenResponse {
        marked: payload.item_ids.len(),
    }))
}

fn record_behavior_event(
    state: &AppState,
    uid: u64,
    item_id: u64,
    event_type: EventType,
) -> Result<EventResponse, (StatusCode, Json<ErrorResponse>)> {
    if state
        .storage
        .get_user(uid)
        .map_err(internal_error("Failed to get user"))?
        .is_none()
    {
        return Err(not_found_error(format!("User {} not found", uid)));
    }

    let item = state
        .storage
        .get_item(item_id)
        .map_err(internal_error("Failed to get item"))?
        .ok_or_else(|| not_found_error(format!("Item {} not found", item_id)))?;
    let event = BehaviorEvent::new(uid, item_id, event_type, &item.category);

    state.storage.append_user_event(&event).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to save event: {}", e),
            }),
        )
    })?;

    let mut preferences = state.storage.get_user_preferences(uid).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to get preferences: {}", e),
            }),
        )
    })?;
    preferences.apply_event(event_type, &item);
    state
        .storage
        .save_user_preferences(uid, &preferences)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to save preferences: {}", e),
                }),
            )
        })?;

    if event_type == EventType::Impression {
        let mut filter = state.storage.get_user_filter(uid).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to get filter: {}", e),
                }),
            )
        })?;
        filter.add(&item_id.to_le_bytes());
        state.storage.save_user_filter(uid, &filter).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to save filter: {}", e),
                }),
            )
        })?;
    }

    let recent_events = state.storage.get_recent_events(uid).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to get recent events: {}", e),
            }),
        )
    })?;

    Ok(EventResponse {
        recorded: true,
        event_type: event_type.to_string(),
        recent_event_count: recent_events.items.len(),
        category_weight: preferences.category_weight(&item.category),
        item_weight: preferences.item_weight(item_id),
    })
}

fn not_found_error(message: String) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse { error: message }),
    )
}

fn internal_error(
    context: &'static str,
) -> impl FnOnce(anyhow::Error) -> (StatusCode, Json<ErrorResponse>) {
    move |error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("{}: {}", context, error),
            }),
        )
    }
}

async fn events_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<EventRequest>,
) -> Result<Json<EventResponse>, (StatusCode, Json<ErrorResponse>)> {
    let event_type = EventType::from_str(&payload.event_type)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?;

    record_behavior_event(&state, payload.uid, payload.item_id, event_type).map(Json)
}

async fn debug_recommendation_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RecommendQuery>,
) -> Result<Json<DebugRecommendationResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (user, output) = build_user_recommendation_output(&state, params.uid)?;
    let recent_events = state.storage.get_recent_events(params.uid).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to get recent events: {}", e),
            }),
        )
    })?;
    let preferences = state
        .storage
        .get_user_preferences(params.uid)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to get preferences: {}", e),
                }),
            )
        })?;

    let debug = output.debug;
    let recommendations = output
        .items
        .into_iter()
        .map(recommend_item_from_output)
        .collect();
    let candidates = debug
        .candidates
        .into_iter()
        .map(|candidate| DebugCandidateResponse {
            item_id: candidate.item_id,
            semantic_score: candidate.semantic_score,
            source: candidate.source,
        })
        .collect();

    Ok(Json(DebugRecommendationResponse {
        user: UserInfo {
            id: user.id,
            name: user.name.clone(),
        },
        recommendations,
        filtered_count: output.filtered_count,
        candidate_count: debug.candidate_count,
        candidates,
        category_distribution: debug.category_distribution,
        source_distribution: debug.source_distribution,
        recent_events: recent_events.items,
        preferences,
    }))
}

async fn users_handler(State(state): State<Arc<AppState>>) -> Json<UsersResponse> {
    let users = state
        .users
        .iter()
        .map(|u| UserInfo {
            id: u.id,
            name: u.name.clone(),
        })
        .collect();
    Json(UsersResponse { users })
}

async fn search_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    if contains_cjk_text(&params.q) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Semantic search only supports English text with the current MiniLM model"
                    .to_string(),
            }),
        ));
    }

    let model = state.embedding_model.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Embedding model not loaded".to_string(),
            }),
        )
    })?;

    // 1. Semantic Search (Vector)
    let query_vec = model.encode(&params.q).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Encoding failed: {}", e),
            }),
        )
    })?;

    let vec_candidates = hnsw_search(&query_vec, 50); // Top 50 vector results
    let vec_results: Vec<(u32, f32)> = vec_candidates
        .into_iter()
        .map(|(id, score)| (id as u32, score))
        .collect();

    // 2. Keyword Search (Tantivy)
    let kw_results = state.text_search.search(&params.q, 50).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Text search failed: {}", e),
            }),
        )
    })?;

    // 3. RRF Merge
    let merged_results = hybrid::rrf_merge(vec_results, kw_results);

    // 4. Transform to Response
    let results: Vec<RecommendItem> = merged_results
        .into_iter()
        .take(20)
        .filter_map(|res| {
            let idx = *state.item_map.get(&(res.id as u64))?;
            let item = &state.items[idx];
            Some(RecommendItem {
                item_id: res.id as u64,
                name: item.name.clone(),
                category: item.category.clone(),
                image_url: item.image_url.clone(),
                price: item.price,
                sim_score: res.score, // RRF Score
                category_score: 0.0,
                popularity: item.popularity,
                price_affinity: 0.0,
                novelty: 0.0,
                feedback_score: 0.0,
                final_score: res.score,
                ranking_strategy: "hybrid_search".to_string(),
                source: "search".to_string(),
                reason: "hybrid_search_match".to_string(),
            })
        })
        .collect();

    Ok(Json(SearchResponse {
        query: params.q,
        results,
    }))
}

async fn health_handler() -> &'static str {
    "OK"
}

async fn livez_handler() -> &'static str {
    "OK"
}

async fn readyz_handler(State(state): State<Arc<AppState>>) -> (StatusCode, &'static str) {
    match readiness_status(&state.readiness) {
        StatusCode::OK => (StatusCode::OK, "READY"),
        status => (status, "NOT_READY"),
    }
}

fn readiness_status(readiness: &ReadinessState) -> StatusCode {
    if readiness.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> String {
    state.metrics.render_prometheus()
}

async fn metrics_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let request_id = state.metrics.next_request_id();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let query = request.uri().query().unwrap_or_default().to_string();
    let started = Instant::now();
    let response = next.run(request).await;
    let status = response.status().as_u16();
    let elapsed = started.elapsed();
    state
        .metrics
        .record_http_request(&method, &path, status, elapsed);

    println!(
        "{}",
        json!({
            "event": "http_request",
            "request_id": request_id,
            "method": method,
            "path": path,
            "query": query,
            "status": status,
            "duration_ms": elapsed.as_secs_f64() * 1000.0,
        })
    );

    response
}

fn contains_cjk_text(text: &str) -> bool {
    text.chars().any(|ch| {
        matches!(
            ch,
            '\u{3400}'..='\u{4DBF}'
                | '\u{4E00}'..='\u{9FFF}'
                | '\u{F900}'..='\u{FAFF}'
                | '\u{20000}'..='\u{2A6DF}'
                | '\u{2A700}'..='\u{2B73F}'
                | '\u{2B740}'..='\u{2B81F}'
                | '\u{2B820}'..='\u{2CEAF}'
        )
    })
}

// ============================================================================
// Data initialization
// ============================================================================

fn init_users() -> Vec<User> {
    vec![
        // Single-interest users.
        User {
            id: 1,
            name: "Dev Xiaoming (Electronics + Books)".into(),
            embedding: generate_user_embedding(&["Electronics", "Books"]),
        },
        User {
            id: 2,
            name: "Home Enthusiast Xiaohong (Home)".into(),
            embedding: generate_user_embedding(&["Home"]),
        },
        User {
            id: 3,
            name: "Fashion Fan Xiaomei (Clothing)".into(),
            embedding: generate_user_embedding(&["Clothing"]),
        },
        // Dual-interest users.
        User {
            id: 4,
            name: "Gadget Geek (Electronics)".into(),
            embedding: generate_user_embedding(&["Electronics"]),
        },
        User {
            id: 5,
            name: "Bookworm (Books)".into(),
            embedding: generate_user_embedding(&["Books"]),
        },
        User {
            id: 6,
            name: "Lifestyle Maven (Home + Clothing)".into(),
            embedding: generate_user_embedding(&["Home", "Clothing"]),
        },
        // Multi-category users.
        User {
            id: 7,
            name: "All-Rounder (All Categories)".into(),
            embedding: generate_user_embedding(&["Electronics", "Books", "Home", "Clothing"]),
        },
        User {
            id: 8,
            name: "Tech Homebody (Electronics + Home)".into(),
            embedding: generate_user_embedding(&["Electronics", "Home"]),
        },
        // Cold-start users with random embeddings.
        User {
            id: 9,
            name: "New User A (Random)".into(),
            embedding: generate_random_embedding(),
        },
        User {
            id: 10,
            name: "New User B (Random)".into(),
            embedding: generate_random_embedding(),
        },
    ]
}

fn load_items_from_json(embedding_model: &embedding::EmbeddingModel) -> Result<Vec<Item>> {
    use rand::Rng;
    let json_str = std::fs::read_to_string("assets/products.json")?;
    let items_json: Vec<ItemJson> = serde_json::from_str(&json_str)?;
    let mut rng = rand::thread_rng();
    let total = items_json.len();
    println!("🧠 Encoding {} items with ONNX model...", total);

    let items: Vec<Item> = items_json
        .into_iter()
        .enumerate()
        .map(|(i, json)| {
            // Encode real semantic vectors with the ONNX model.
            let embedding = embedding_model
                .encode(&json.name)
                .unwrap_or_else(|_| generate_category_embedding(&json.category));
            let popularity = rng.gen::<f32>();
            if (i + 1) % 50 == 0 {
                println!("   Encoded {}/{} items", i + 1, total);
            }
            Item::from_json(json, embedding, popularity)
        })
        .collect();

    println!("✅ All {} items encoded with semantic vectors", total);
    Ok(items)
}

fn load_items_from_json_fallback() -> Result<Vec<Item>> {
    use rand::Rng;
    let json_str = std::fs::read_to_string("assets/products.json")?;
    let items_json: Vec<ItemJson> = serde_json::from_str(&json_str)?;
    let mut rng = rand::thread_rng();
    Ok(items_json
        .into_iter()
        .map(|json| {
            let embedding = generate_category_embedding(&json.category);
            let popularity = rng.gen::<f32>();
            Item::from_json(json, embedding, popularity)
        })
        .collect())
}

pub fn init_data_with_storage(
    storage: Arc<Storage>,
    embedding_model: Option<Arc<embedding::EmbeddingModel>>,
    text_search: Arc<TextSearch>,
    metrics: Arc<Metrics>,
    readiness: Arc<ReadinessState>,
) -> Result<Arc<AppState>> {
    let items = if storage.items_count() == 0 {
        println!("📂 Database empty, loading from products.json...");
        let items = match &embedding_model {
            Some(model) => load_items_from_json(model)?,
            None => {
                println!("⚠️  No embedding model, using category-based vectors");
                load_items_from_json_fallback()?
            }
        };
        for item in &items {
            storage.save_item(item)?;
        }
        println!("💾 Saved {} items to database", items.len());

        // Hydrate Tantivy
        println!("🔍 Building text search index...");
        for item in &items {
            text_search.index_item(item)?;
        }
        text_search.commit()?;
        println!("✅ Text index built");

        items
    } else {
        println!("📂 Loading items from database...");
        let items: Vec<Item> = storage.iter_items().filter_map(|r| r.ok()).collect();
        println!("📦 Loaded {} items from database", items.len());
        items
    };

    let users = if storage.users_count() == 0 {
        let users = init_users();
        for user in &users {
            storage.save_user(user)?;
        }
        println!("💾 Saved {} users to database", users.len());
        users
    } else {
        storage.get_all_users()?
    };

    let item_map: HashMap<u64, usize> = items
        .iter()
        .enumerate()
        .map(|(i, item)| (item.id, i))
        .collect();

    Ok(Arc::new(AppState {
        storage,
        users,
        items,
        item_map,
        embedding_model,
        text_search,
        metrics,
        readiness,
    }))
}

// ============================================================================
// Index initialization (hydration)
// ============================================================================

pub fn init_hnsw_with_hydration(items: &[Item], config: &AppConfig) -> Result<()> {
    let max_elements = items.len() + 1000;
    let index_path = config.index_path_str();

    println!("🔧 Loading HNSW index from {}...", index_path);
    let loaded =
        load_hnsw_index(&index_path, DIM, max_elements, 100).map_err(|e| anyhow::anyhow!(e))?;

    let index_count = get_hnsw_count();
    let db_count = items.len();

    if loaded && index_count == db_count {
        println!(
            "✅ HNSW index loaded: {} items (consistent with DB)",
            index_count
        );
        return Ok(());
    }

    if !loaded {
        println!("📝 Index file not found, created new empty index");
    } else {
        println!(
            "⚠️  Index count ({}) != DB count ({}), rebuilding...",
            index_count, db_count
        );
    }

    println!("🔄 Hydrating index from database...");
    let mut success = 0;
    for item in items {
        if add_item_to_hnsw(item.id, &item.embedding).is_ok() {
            success += 1;
        }
    }
    println!("✅ HNSW index rebuilt with {} items", success);

    Ok(())
}

// ============================================================================
// Graceful shutdown
// ============================================================================

async fn graceful_shutdown(storage: Arc<Storage>, config: AppConfig) {
    println!("\n🛑 Shutting down...");
    let index_path = config.index_path_str();

    match save_hnsw_index(&index_path) {
        Ok(()) => println!("💾 HNSW index saved to {}", index_path),
        Err(e) => eprintln!("❌ Failed to save index: {}", e),
    }

    match storage.flush() {
        Ok(()) => println!("💾 Sled database flushed"),
        Err(e) => eprintln!("❌ Failed to flush database: {}", e),
    }

    println!("👋 Goodbye!");
}

pub fn load_embedding_model(config: &AppConfig) -> Option<Arc<embedding::EmbeddingModel>> {
    match embedding::EmbeddingModel::new_with_paths(&config.model_path, &config.tokenizer_path) {
        Ok(model) => {
            println!(
                "🧠 Embedding model loaded (dimension: {})\n",
                model.dimension()
            );
            Some(Arc::new(model))
        }
        Err(e) => {
            eprintln!(
                "⚠️  Failed to load embedding model: {}\n   /search will be unavailable\n",
                e
            );
            None
        }
    }
}

pub fn open_storage(config: &AppConfig) -> Result<Arc<Storage>> {
    let db_path = config.db_path_str();
    let storage = Arc::new(Storage::new(&db_path)?);
    println!("💾 Sled database opened at {}", db_path);
    Ok(storage)
}

pub fn open_text_search(config: &AppConfig) -> Result<Arc<TextSearch>> {
    let tantivy_path = config.tantivy_path_str();
    let text_search = Arc::new(TextSearch::new(&tantivy_path)?);
    println!("🔍 Text search index initialized at {}\n", tantivy_path);
    Ok(text_search)
}

pub fn log_loaded_state(state: &AppState) {
    println!(
        "📊 Loaded {} users, {} items",
        state.users.len(),
        state.items.len()
    );
}

pub fn warmup_embedding_model(model: Option<&embedding::EmbeddingModel>) -> Result<bool> {
    match model {
        Some(model) => {
            let _ = model.encode("warmup recommendation query")?;
            println!("🔥 Embedding warmup completed");
            Ok(true)
        }
        None => {
            eprintln!("⚠️  Embedding warmup skipped because the model is unavailable");
            Ok(false)
        }
    }
}

pub fn build_app(state: Arc<AppState>, config: &AppConfig) -> Router {
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);
    let cors = if config.cors_origin == "*" {
        cors.allow_origin(Any)
    } else {
        cors.allow_origin(config.cors_origin.parse::<HeaderValue>().unwrap())
    };

    Router::new()
        .route("/health", get(health_handler))
        .route("/livez", get(livez_handler))
        .route("/readyz", get(readyz_handler))
        .route("/metrics", get(metrics_handler))
        .route("/users", get(users_handler))
        .route("/recommend", get(recommend_handler))
        .route("/debug/recommendation", get(debug_recommendation_handler))
        .route("/search", get(search_handler))
        .route("/events", post(events_handler))
        .route("/mark_seen", post(mark_seen_handler))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            metrics_middleware,
        ))
        .layer(cors)
        .with_state(state)
}

pub async fn serve_app(app: Router, storage: Arc<Storage>, config: AppConfig) -> Result<()> {
    let addr = format!("0.0.0.0:{}", config.port);
    println!("🌐 Server running at http://{}", addr);
    println!("   GET  /search?q=<query> - semantic search");
    println!("   POST /events - record behavior feedback");
    println!("   Press Ctrl+C to shutdown gracefully\n");

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    let storage_for_shutdown = Arc::clone(&storage);
    let config_for_shutdown = config.clone();
    tokio::select! {
        result = axum::serve(listener, app) => {
            result?;
        }
        _ = tokio::signal::ctrl_c() => {
            graceful_shutdown(storage_for_shutdown, config_for_shutdown).await;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DIM;
    use fastbloom_rs::Membership;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn detects_cjk_text_as_unsupported_search_input() {
        assert!(contains_cjk_text("wireless mouse \u{9F20}\u{6807}"));
        assert!(!contains_cjk_text("wireless mouse"));
    }

    #[test]
    fn readiness_status_changes_after_warmup_completes() {
        let readiness = ReadinessState::default();

        assert_eq!(
            readiness_status(&readiness),
            StatusCode::SERVICE_UNAVAILABLE
        );

        readiness.mark_ready();

        assert_eq!(readiness_status(&readiness), StatusCode::OK);
    }

    fn temp_path(name: &str) -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("mini-recsys-{}-{}", name, now))
            .to_string_lossy()
            .into_owned()
    }

    fn test_item(id: u64, category: &str) -> Item {
        Item {
            id,
            name: format!("Item {}", id),
            category: category.to_string(),
            image_url: String::new(),
            price: 10.0,
            embedding: vec![0.0; DIM],
            popularity: 0.5,
        }
    }

    fn test_state(path: &str) -> AppState {
        let storage = Arc::new(Storage::new(path).unwrap());
        let users = vec![User {
            id: 1,
            name: "Test user".to_string(),
            embedding: vec![0.0; DIM],
        }];
        let items = vec![test_item(10, "Books")];
        storage.save_user(&users[0]).unwrap();
        storage.save_item(&items[0]).unwrap();
        let item_map = items
            .iter()
            .enumerate()
            .map(|(index, item)| (item.id, index))
            .collect();
        let text_search = Arc::new(TextSearch::new(&format!("{}-tantivy", path)).unwrap());

        AppState {
            storage,
            users,
            items,
            item_map,
            embedding_model: None,
            text_search,
            metrics: Arc::new(Metrics::default()),
            readiness: Arc::new(ReadinessState::default()),
        }
    }

    #[test]
    fn record_event_rejects_unknown_user() {
        let path = temp_path("unknown-user");
        let state = test_state(&path);

        let err = record_behavior_event(&state, 99, 10, EventType::Click).unwrap_err();

        assert_eq!(err.0, StatusCode::NOT_FOUND);
        drop(state);
        let _ = fs::remove_dir_all(&path);
        let _ = fs::remove_dir_all(format!("{}-tantivy", path));
    }

    #[test]
    fn record_event_rejects_unknown_item() {
        let path = temp_path("unknown-item");
        let state = test_state(&path);

        let err = record_behavior_event(&state, 1, 99, EventType::Click).unwrap_err();

        assert_eq!(err.0, StatusCode::NOT_FOUND);
        drop(state);
        let _ = fs::remove_dir_all(&path);
        let _ = fs::remove_dir_all(format!("{}-tantivy", path));
    }

    #[test]
    fn impression_event_updates_seen_filter() {
        let path = temp_path("impression");
        let state = test_state(&path);

        let response = record_behavior_event(&state, 1, 10, EventType::Impression).unwrap();
        let filter = state.storage.get_user_filter(1).unwrap();

        assert!(response.recorded);
        assert_eq!(response.recent_event_count, 1);
        assert!(filter.contains(&10u64.to_le_bytes()));
        drop(state);
        let _ = fs::remove_dir_all(&path);
        let _ = fs::remove_dir_all(format!("{}-tantivy", path));
    }

    #[test]
    fn like_event_updates_persisted_preferences() {
        let path = temp_path("like");
        let state = test_state(&path);

        let response = record_behavior_event(&state, 1, 10, EventType::Like).unwrap();
        let preferences = state.storage.get_user_preferences(1).unwrap();

        assert_eq!(response.event_type, "like");
        assert!(preferences.category_weight("Books") > 0.0);
        assert!(preferences.item_weight(10) > 0.0);
        drop(state);
        let _ = fs::remove_dir_all(&path);
        let _ = fs::remove_dir_all(format!("{}-tantivy", path));
    }
}
