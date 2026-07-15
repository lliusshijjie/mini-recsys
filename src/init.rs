//! Application initialization, routing, and lifecycle support.

use crate::behavior::{BehaviorEvent, EventType, UserPreferences};
use crate::config::AppConfig;
use crate::embedding;
use crate::ffi::{HnswConfig, HnswIndex};
use crate::hybrid;
use crate::model::{
    build_user, generate_category_embedding, generate_random_embedding, Item, ItemJson, User,
    UserProfile, DIM, PRODUCT_CATEGORIES,
};
use crate::observability::Metrics;
use crate::recommendation::{
    RecommendationIndexes, RecommendationOutput, RecommendationService,
    RecommendationServiceConfig, RecommendationServiceError, RecommendationServiceOutput,
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
    pub users: Arc<Vec<User>>,
    pub items: Arc<Vec<Item>>,
    pub item_map: HashMap<u64, usize>,
    pub recommendation_service: Arc<RecommendationService>,
    pub hnsw_index: Arc<HnswIndex>,
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

#[derive(Debug, Serialize)]
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

#[derive(Debug, Serialize)]
struct UserInfo {
    id: u64,
    name: String,
    top_categories: Vec<String>,
    budget_min: f32,
    budget_max: f32,
}

#[derive(Debug, Serialize)]
struct RecommendResponse {
    user: UserInfo,
    recommendations: Vec<RecommendItem>,
    filtered_count: usize,
}

#[derive(Debug, Deserialize)]
struct BatchRecommendRequest {
    uids: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct BatchRecommendUserResult {
    uid: u64,
    status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<RecommendResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BatchRecommendResponse {
    results: Vec<BatchRecommendUserResult>,
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
    quality_metrics: HashMap<String, f32>,
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
    let RecommendationServiceOutput { user, output } =
        recommend_with_timeout(Arc::clone(&state), params.uid).await?;
    let filtered_count = output.filtered_count;
    state.recommendation_service.record_output_metrics(&output);
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

    Ok(Json(recommend_response_from_output(&user, output)))
}

fn recommend_response_from_output(user: &User, output: RecommendationOutput) -> RecommendResponse {
    let filtered_count = output.filtered_count;
    let recommendations = output
        .items
        .into_iter()
        .map(recommend_item_from_output)
        .collect();

    RecommendResponse {
        user: user_info_from(user),
        recommendations,
        filtered_count,
    }
}

async fn batch_recommend_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<BatchRecommendRequest>,
) -> Result<Json<BatchRecommendResponse>, (StatusCode, Json<ErrorResponse>)> {
    if payload.uids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "uids must not be empty".to_string(),
            }),
        ));
    }

    let max_users = state.recommendation_service.batch_max_users();
    if payload.uids.len() > max_users {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Batch recommendation supports at most {} users", max_users),
            }),
        ));
    }

    state.metrics.record_batch_users(payload.uids.len());
    let mut handles = Vec::with_capacity(payload.uids.len());
    for (index, uid) in payload.uids.into_iter().enumerate() {
        let state_for_task = Arc::clone(&state);
        handles.push((
            index,
            uid,
            tokio::spawn(async move { recommend_with_timeout(state_for_task, uid).await }),
        ));
    }

    let mut results = Vec::with_capacity(handles.len());
    results.resize_with(handles.len(), || None);
    for (index, uid, handle) in handles {
        let item = match handle.await {
            Ok(Ok(RecommendationServiceOutput { user, output })) => {
                state.recommendation_service.record_output_metrics(&output);
                BatchRecommendUserResult {
                    uid,
                    status: StatusCode::OK.as_u16(),
                    response: Some(recommend_response_from_output(&user, output)),
                    error: None,
                }
            }
            Ok(Err((status, Json(error)))) => BatchRecommendUserResult {
                uid,
                status: status.as_u16(),
                response: None,
                error: Some(error.error),
            },
            Err(error) => BatchRecommendUserResult {
                uid,
                status: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                response: None,
                error: Some(format!("Recommendation task failed: {}", error)),
            },
        };
        results[index] = Some(item);
    }

    Ok(Json(BatchRecommendResponse {
        results: results
            .into_iter()
            .map(|item| item.expect("batch result slot should be filled"))
            .collect(),
    }))
}

async fn recommend_with_timeout(
    state: Arc<AppState>,
    uid: u64,
) -> Result<RecommendationServiceOutput, (StatusCode, Json<ErrorResponse>)> {
    let timeout = state.recommendation_service.recommend_timeout();
    if timeout.is_zero() {
        state.recommendation_service.record_timeout();
        return Err((
            StatusCode::GATEWAY_TIMEOUT,
            Json(ErrorResponse {
                error: "Recommendation request timed out".to_string(),
            }),
        ));
    }
    let service = Arc::clone(&state.recommendation_service);
    let task = tokio::task::spawn_blocking(move || service.recommend_user(uid));

    match tokio::time::timeout(timeout, task).await {
        Ok(Ok(result)) => result.map_err(recommendation_service_error),
        Ok(Err(error)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Recommendation task failed: {}", error),
            }),
        )),
        Err(_) => {
            state.recommendation_service.record_timeout();
            Err((
                StatusCode::GATEWAY_TIMEOUT,
                Json(ErrorResponse {
                    error: "Recommendation request timed out".to_string(),
                }),
            ))
        }
    }
}

fn recommendation_service_error(
    error: RecommendationServiceError,
) -> (StatusCode, Json<ErrorResponse>) {
    match error {
        RecommendationServiceError::UserNotFound(uid) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("User {} not found", uid),
            }),
        ),
        RecommendationServiceError::Storage(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Recommendation storage failed: {}", error),
            }),
        ),
    }
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
    state
        .recommendation_service
        .invalidate_user_context(payload.uid);

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
    state.recommendation_service.invalidate_user_context(uid);

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
    let RecommendationServiceOutput { user, output } = state
        .recommendation_service
        .recommend_user(params.uid)
        .map_err(recommendation_service_error)?;
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
        user: user_info_from(&user),
        recommendations,
        filtered_count: output.filtered_count,
        candidate_count: debug.candidate_count,
        candidates,
        category_distribution: debug.category_distribution,
        source_distribution: debug.source_distribution,
        quality_metrics: debug.quality_metrics,
        recent_events: recent_events.items,
        preferences,
    }))
}

fn user_info_from(user: &User) -> UserInfo {
    UserInfo {
        id: user.id,
        name: user.name.clone(),
        top_categories: user.profile.top_categories(3),
        budget_min: user.profile.budget_min,
        budget_max: user.profile.budget_max,
    }
}

async fn users_handler(State(state): State<Arc<AppState>>) -> Json<UsersResponse> {
    let users = state.users.iter().map(user_info_from).collect();
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

    let vec_candidates = state.hnsw_index.search(&query_vec, 50).unwrap_or_default();
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

const USER_SEED_COUNT: usize = 24;

fn init_users() -> Vec<User> {
    let all_category_weights: Vec<(&str, f32)> = PRODUCT_CATEGORIES
        .iter()
        .map(|category| (*category, 0.4))
        .collect();
    let all_category_refs: Vec<(&str, f32)> = all_category_weights
        .iter()
        .map(|(category, weight)| (*category, *weight))
        .collect();

    vec![
        build_user(
            1,
            "Gadget Geek (Electronics)",
            &[("Electronics", 1.0)],
            50.0,
            500.0,
        ),
        build_user(2, "Bookworm (Books)", &[("Books", 1.0)], 10.0, 80.0),
        build_user(3, "Home Enthusiast (Home)", &[("Home", 1.0)], 20.0, 200.0),
        build_user(
            4,
            "Fashion Fan (Clothing)",
            &[("Clothing", 1.0)],
            30.0,
            300.0,
        ),
        build_user(5, "Fitness Pro (Sports)", &[("Sports", 1.0)], 25.0, 250.0),
        build_user(6, "Beauty Lover (Beauty)", &[("Beauty", 1.0)], 15.0, 150.0),
        build_user(7, "Toy Collector (Toys)", &[("Toys", 1.0)], 10.0, 120.0),
        build_user(8, "Foodie (Food)", &[("Food", 1.0)], 5.0, 80.0),
        build_user(
            9,
            "Car Enthusiast (Automotive)",
            &[("Automotive", 1.0)],
            40.0,
            600.0,
        ),
        build_user(10, "Bargain Hunter (Misc)", &[("Misc", 1.0)], 5.0, 60.0),
        build_user(
            11,
            "Fitness Tech (Electronics + Sports)",
            &[("Electronics", 0.85), ("Sports", 0.75)],
            80.0,
            400.0,
        ),
        build_user(
            12,
            "Self-Care Reader (Books + Beauty)",
            &[("Books", 0.8), ("Beauty", 0.55)],
            15.0,
            100.0,
        ),
        build_user(
            13,
            "Home Cook (Home + Food)",
            &[("Home", 0.8), ("Food", 0.75)],
            15.0,
            150.0,
        ),
        build_user(
            14,
            "Style & Beauty (Clothing + Beauty)",
            &[("Clothing", 0.85), ("Beauty", 0.8)],
            40.0,
            350.0,
        ),
        build_user(
            15,
            "Outdoor Driver (Sports + Automotive)",
            &[("Sports", 0.8), ("Automotive", 0.65)],
            50.0,
            500.0,
        ),
        build_user(
            16,
            "Tech Parent (Toys + Electronics)",
            &[("Toys", 0.8), ("Electronics", 0.7)],
            30.0,
            250.0,
        ),
        build_user(
            17,
            "Pantry Restock (Food + Home)",
            &[("Food", 0.8), ("Home", 0.7)],
            10.0,
            120.0,
        ),
        build_user(
            18,
            "Tech Professional (Electronics + Books + Home)",
            &[("Electronics", 0.75), ("Books", 0.65), ("Home", 0.55)],
            40.0,
            300.0,
        ),
        build_user(
            19,
            "Active Lifestyle (Sports + Clothing + Beauty)",
            &[("Sports", 0.8), ("Clothing", 0.7), ("Beauty", 0.55)],
            35.0,
            280.0,
        ),
        build_user(20, "All-Category Explorer", &all_category_refs, 10.0, 400.0),
        build_user(
            21,
            "Premium Shopper",
            &[
                ("Electronics", 0.75),
                ("Automotive", 0.7),
                ("Clothing", 0.6),
            ],
            200.0,
            800.0,
        ),
        build_user(
            22,
            "Budget Shopper",
            &[("Food", 0.85), ("Misc", 0.8), ("Home", 0.65)],
            5.0,
            40.0,
        ),
        cold_start_user(23, "New User A (Random)"),
        cold_start_user(24, "New User B (Random)"),
    ]
}

fn cold_start_user(id: u64, name: &str) -> User {
    User {
        id,
        name: name.to_string(),
        embedding: generate_random_embedding(),
        profile: UserProfile::default(),
    }
}

fn seed_user_offline_data(storage: &Storage, user: &User) -> Result<()> {
    storage.save_user(user)?;
    if !user.profile.category_weights.is_empty() {
        storage.save_user_preferences(user.id, &UserPreferences::from_profile(&user.profile))?;
    }
    Ok(())
}

fn should_reseed_users(storage: &Storage) -> Result<bool> {
    if storage.users_count() == 0 {
        return Ok(true);
    }

    match storage.get_all_users() {
        Ok(users) => Ok(users.len() < USER_SEED_COUNT
            || users
                .iter()
                .all(|user| user.profile.category_weights.is_empty())),
        Err(_) => Ok(true),
    }
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
    config: &AppConfig,
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

    let users = if should_reseed_users(&storage)? {
        if storage.users_count() > 0 {
            println!("👤 Reseeding users with expanded offline profiles...");
            storage.clear_users()?;
            storage.clear_user_preferences()?;
        } else {
            println!("👤 Seeding users with offline profiles...");
        }

        let users = init_users();
        for user in &users {
            seed_user_offline_data(&storage, user)?;
        }
        println!("💾 Saved {} users to database", users.len());
        users
    } else {
        println!("📂 Loading users from database...");
        storage.get_all_users()?
    };

    let item_map: HashMap<u64, usize> = items
        .iter()
        .enumerate()
        .map(|(i, item)| (item.id, i))
        .collect();
    let recommendation_indexes = RecommendationIndexes::from_items(&items);
    let hnsw_index = init_hnsw_with_hydration(&items, config)?;
    let users = Arc::new(users);
    let items = Arc::new(items);
    let recommendation_service = Arc::new(RecommendationService::new(
        Arc::clone(&storage),
        Arc::clone(&users),
        Arc::clone(&items),
        recommendation_indexes.clone(),
        Arc::clone(&hnsw_index),
        Arc::clone(&metrics),
        RecommendationServiceConfig::from_app_config(config),
    ));

    Ok(Arc::new(AppState {
        storage,
        users,
        items,
        item_map,
        recommendation_service,
        hnsw_index,
        embedding_model,
        text_search,
        metrics,
        readiness,
    }))
}

// ============================================================================
// Index initialization (hydration)
// ============================================================================

pub fn init_hnsw_with_hydration(items: &[Item], config: &AppConfig) -> Result<Arc<HnswIndex>> {
    let max_elements = items.len() + 1000;
    let index_path = config.index_path_str();

    println!("🔧 Loading HNSW index from {}...", index_path);
    let hnsw_config = HnswConfig {
        dim: DIM,
        max_elements,
        m: 16,
        ef_construction: 200,
        ef_search: 100,
    };
    let (mut index, loaded) = HnswIndex::load_or_create(&index_path, DIM, max_elements, 100)
        .map_err(|e| anyhow::anyhow!(e))?;

    let index_count = index.count();
    let db_count = items.len();

    if loaded && index_count == db_count {
        println!(
            "✅ HNSW index loaded: {} items (consistent with DB)",
            index_count
        );
        return Ok(Arc::new(index));
    }

    if !loaded {
        println!("📝 Index file not found, created new empty index");
    } else {
        println!(
            "⚠️  Index count ({}) != DB count ({}), rebuilding...",
            index_count, db_count
        );
        index = HnswIndex::new(&hnsw_config).map_err(|e| anyhow::anyhow!(e))?;
    }

    println!("🔄 Hydrating index from database...");
    let mut success = 0;
    for item in items {
        if index.add_item(item.id, &item.embedding).is_ok() {
            success += 1;
        }
    }
    println!("✅ HNSW index rebuilt with {} items", success);

    Ok(Arc::new(index))
}

// ============================================================================
// Graceful shutdown
// ============================================================================

async fn graceful_shutdown(storage: Arc<Storage>, hnsw_index: Arc<HnswIndex>, config: AppConfig) {
    println!("\n🛑 Shutting down...");
    let index_path = config.index_path_str();

    match hnsw_index.save(&index_path) {
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
    configure_onnx_runtime_dylib();

    if !config.model_path.exists() || !config.tokenizer_path.exists() {
        eprintln!(
            "⚠️  Embedding assets not found at {} and {}; /search will be unavailable\n",
            config.model_path.display(),
            config.tokenizer_path.display()
        );
        return None;
    }

    match std::panic::catch_unwind(|| {
        embedding::EmbeddingModel::new_with_paths(&config.model_path, &config.tokenizer_path)
    }) {
        Ok(Ok(model)) => {
            println!(
                "🧠 Embedding model loaded (dimension: {})\n",
                model.dimension()
            );
            Some(Arc::new(model))
        }
        Ok(Err(e)) => {
            eprintln!(
                "⚠️  Failed to load embedding model: {}\n   /search will be unavailable\n",
                e
            );
            None
        }
        Err(_) => {
            eprintln!(
                "⚠️  Failed to initialize ONNX Runtime; set ORT_DYLIB_PATH to a valid libonnxruntime shared library\n   /search will be unavailable\n"
            );
            None
        }
    }
}

fn configure_onnx_runtime_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }

    let candidates = [
        "libs/onnxruntime/onnxruntime.dll",
        "libs/onnxruntime/libonnxruntime.so",
        "libs/onnxruntime/libonnxruntime.dylib",
    ];

    for candidate in candidates {
        let path = std::path::Path::new(candidate);
        if path.exists() {
            std::env::set_var("ORT_DYLIB_PATH", path);
            println!("🔗 Using ONNX Runtime from {}", path.display());
            return;
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
        .route("/recommend/batch", post(batch_recommend_handler))
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

pub async fn serve_app(
    app: Router,
    storage: Arc<Storage>,
    hnsw_index: Arc<HnswIndex>,
    config: AppConfig,
) -> Result<()> {
    let addr = format!("0.0.0.0:{}", config.port);
    println!("🌐 Server running at http://{}", addr);
    println!("   GET  /search?q=<query> - semantic search");
    println!("   POST /events - record behavior feedback");
    println!("   Press Ctrl+C to shutdown gracefully\n");

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    let storage_for_shutdown = Arc::clone(&storage);
    let hnsw_for_shutdown = Arc::clone(&hnsw_index);
    let config_for_shutdown = config.clone();
    tokio::select! {
        result = axum::serve(listener, app) => {
            result?;
        }
        _ = tokio::signal::ctrl_c() => {
            graceful_shutdown(storage_for_shutdown, hnsw_for_shutdown, config_for_shutdown).await;
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
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
        test_state_with_timeout(path, Duration::from_millis(150))
    }

    fn test_state_with_timeout(path: &str, recommend_timeout: Duration) -> AppState {
        test_state_with_limits(path, recommend_timeout, 32, 10_000)
    }

    fn test_state_with_limits(
        path: &str,
        recommend_timeout: Duration,
        batch_max_users: usize,
        recall_parallel_min_items: usize,
    ) -> AppState {
        let storage = Arc::new(Storage::new(path).unwrap());
        let users = Arc::new(vec![User {
            id: 1,
            name: "Test user".to_string(),
            embedding: vec![0.0; DIM],
            profile: UserProfile::default(),
        }]);
        let items = Arc::new(vec![test_item(10, "Books")]);
        storage.save_user(&users[0]).unwrap();
        storage.save_item(&items[0]).unwrap();
        let item_map = items
            .iter()
            .enumerate()
            .map(|(index, item)| (item.id, index))
            .collect();
        let recommendation_indexes = RecommendationIndexes::from_items(&items);
        let hnsw_index = Arc::new(
            HnswIndex::new(&HnswConfig {
                dim: DIM,
                max_elements: 10,
                m: 16,
                ef_construction: 100,
                ef_search: 50,
            })
            .unwrap(),
        );
        let text_search = Arc::new(TextSearch::new(&format!("{}-tantivy", path)).unwrap());
        let metrics = Arc::new(Metrics::default());
        let recommendation_service = Arc::new(RecommendationService::new(
            Arc::clone(&storage),
            Arc::clone(&users),
            Arc::clone(&items),
            recommendation_indexes.clone(),
            Arc::clone(&hnsw_index),
            Arc::clone(&metrics),
            RecommendationServiceConfig {
                recommend_timeout,
                user_context_cache: crate::recommendation::service::UserContextCacheConfig {
                    ttl: Duration::from_millis(1000),
                    max_users: 1024,
                },
                batch_max_users,
                recall_parallel_min_items,
            },
        ));

        AppState {
            storage,
            users,
            items,
            item_map,
            recommendation_service,
            hnsw_index,
            embedding_model: None,
            text_search,
            metrics,
            readiness: Arc::new(ReadinessState::default()),
        }
    }

    #[tokio::test]
    async fn recommend_with_timeout_returns_recommendations() {
        let path = temp_path("recommend-ok");
        let state = Arc::new(test_state(&path));

        let result = recommend_with_timeout(Arc::clone(&state), 1).await.unwrap();

        assert_eq!(result.user.id, 1);
        assert!(!result.output.items.is_empty());
        drop(state);
        let _ = fs::remove_dir_all(&path);
        let _ = fs::remove_dir_all(format!("{}-tantivy", path));
    }

    #[tokio::test]
    async fn recommend_with_timeout_returns_gateway_timeout() {
        let path = temp_path("recommend-timeout");
        let state = Arc::new(test_state_with_timeout(&path, Duration::from_millis(0)));

        let err = recommend_with_timeout(Arc::clone(&state), 1)
            .await
            .unwrap_err();

        assert_eq!(err.0, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(err.1 .0.error, "Recommendation request timed out");
        assert!(state
            .metrics
            .render_prometheus()
            .contains("mini_recsys_recommendation_timeouts_total 1"));
        drop(state);
        let _ = fs::remove_dir_all(&path);
        let _ = fs::remove_dir_all(format!("{}-tantivy", path));
    }

    #[tokio::test]
    async fn batch_recommend_rejects_empty_and_oversized_requests() {
        let path = temp_path("batch-invalid");
        let state = Arc::new(test_state_with_limits(
            &path,
            Duration::from_millis(150),
            1,
            10_000,
        ));

        let empty = batch_recommend_handler(
            State(Arc::clone(&state)),
            Json(BatchRecommendRequest { uids: Vec::new() }),
        )
        .await
        .unwrap_err();
        assert_eq!(empty.0, StatusCode::BAD_REQUEST);

        let oversized = batch_recommend_handler(
            State(Arc::clone(&state)),
            Json(BatchRecommendRequest { uids: vec![1, 2] }),
        )
        .await
        .unwrap_err();
        assert_eq!(oversized.0, StatusCode::BAD_REQUEST);
        drop(state);
        let _ = fs::remove_dir_all(&path);
        let _ = fs::remove_dir_all(format!("{}-tantivy", path));
    }

    #[tokio::test]
    async fn batch_recommend_returns_independent_success_and_failure_items() {
        let path = temp_path("batch-partial");
        let state = Arc::new(test_state(&path));

        let Json(response) = batch_recommend_handler(
            State(Arc::clone(&state)),
            Json(BatchRecommendRequest { uids: vec![1, 99] }),
        )
        .await
        .unwrap();

        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].uid, 1);
        assert_eq!(response.results[0].status, StatusCode::OK.as_u16());
        assert!(response.results[0].response.is_some());
        assert!(response.results[0].error.is_none());
        assert_eq!(response.results[1].uid, 99);
        assert_eq!(response.results[1].status, StatusCode::NOT_FOUND.as_u16());
        assert!(response.results[1].response.is_none());
        assert_eq!(
            response.results[1].error.as_deref(),
            Some("User 99 not found")
        );
        assert!(state
            .metrics
            .render_prometheus()
            .contains("mini_recsys_batch_users_total 2"));
        drop(state);
        let _ = fs::remove_dir_all(&path);
        let _ = fs::remove_dir_all(format!("{}-tantivy", path));
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
