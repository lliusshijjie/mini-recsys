//! Mini-RecSys - hybrid Rust/C++ recommendation system demo.

mod embedding;
mod ffi;
mod hybrid;
mod model;
mod recommendation;
mod storage;
mod text_search;

use anyhow::Result;
use axum::http::{HeaderValue, Method};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use fastbloom_rs::Membership;
use ffi::{add_item_to_hnsw, get_hnsw_count, hnsw_search, load_hnsw_index, save_hnsw_index};
use model::{
    generate_category_embedding, generate_random_embedding, generate_user_embedding, Item,
    ItemJson, User, DIM,
};
use recommendation::{build_recommendations, RecommendationConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use storage::Storage;
use text_search::TextSearch;
use tower_http::cors::CorsLayer;

const INDEX_PATH: &str = "data/index.bin";
const DB_PATH: &str = "data/db";

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
    final_score: f32,
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

#[derive(Serialize)]
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
struct SearchQuery {
    q: String,
}

#[derive(Serialize)]
struct SearchResponse {
    query: String,
    results: Vec<RecommendItem>,
}

// ============================================================================
// Handlers
// ============================================================================

async fn recommend_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RecommendQuery>,
) -> Result<Json<RecommendResponse>, (StatusCode, Json<ErrorResponse>)> {
    let user = state
        .users
        .iter()
        .find(|u| u.id == params.uid)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("User {} not found", params.uid),
                }),
            )
        })?;

    let semantic_hits = hnsw_search(&user.embedding, 100);

    let filter = state.storage.get_user_filter(params.uid).map_err(|e| {
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
        RecommendationConfig::default(),
    );

    let recommendations = output
        .items
        .into_iter()
        .map(|item| RecommendItem {
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
            final_score: item.final_score,
            source: item.source,
            reason: item.reason,
        })
        .collect();

    Ok(Json(RecommendResponse {
        user: UserInfo {
            id: user.id,
            name: user.name.clone(),
        },
        recommendations,
        filtered_count: output.filtered_count,
    }))
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
                final_score: res.score,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_cjk_text_as_unsupported_search_input() {
        assert!(contains_cjk_text("wireless mouse \u{9F20}\u{6807}"));
        assert!(!contains_cjk_text("wireless mouse"));
    }
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

fn init_data_with_storage(
    storage: Arc<Storage>,
    embedding_model: Option<Arc<embedding::EmbeddingModel>>,
    text_search: Arc<TextSearch>,
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
    }))
}

// ============================================================================
// Index initialization (hydration)
// ============================================================================

fn init_hnsw_with_hydration(items: &[Item]) -> Result<()> {
    let max_elements = items.len() + 1000;

    println!("🔧 Loading HNSW index from {}...", INDEX_PATH);
    let loaded =
        load_hnsw_index(INDEX_PATH, DIM, max_elements, 100).map_err(|e| anyhow::anyhow!(e))?;

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

async fn graceful_shutdown(storage: Arc<Storage>) {
    println!("\n🛑 Shutting down...");

    match save_hnsw_index(INDEX_PATH) {
        Ok(()) => println!("💾 HNSW index saved to {}", INDEX_PATH),
        Err(e) => eprintln!("❌ Failed to save index: {}", e),
    }

    match storage.flush() {
        Ok(()) => println!("💾 Sled database flushed"),
        Err(e) => eprintln!("❌ Failed to flush database: {}", e),
    }

    println!("👋 Goodbye!");
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Initializing Mini-RecSys...\n");

    // Initialize ONNX embedding model.
    let embedding_model = match embedding::EmbeddingModel::new() {
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
    };

    let storage = Arc::new(Storage::new(DB_PATH)?);
    println!("💾 Sled database opened at {}", DB_PATH);

    let text_search = Arc::new(TextSearch::new("data/tantivy_index")?);
    println!("🔍 Text search index initialized at data/tantivy_index\n");

    let state = init_data_with_storage(Arc::clone(&storage), embedding_model, text_search)?;
    println!(
        "📊 Loaded {} users, {} items",
        state.users.len(),
        state.items.len()
    );

    init_hnsw_with_hydration(&state.items)?;
    println!();

    let cors = CorsLayer::new()
        .allow_origin("http://localhost:5173".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/users", get(users_handler))
        .route("/recommend", get(recommend_handler))
        .route("/search", get(search_handler))
        .route("/mark_seen", post(mark_seen_handler))
        .layer(cors)
        .with_state(Arc::clone(&state));

    let addr = "0.0.0.0:3000";
    println!("🌐 Server running at http://{}", addr);
    println!("   GET  /search?q=<query> - semantic search");
    println!("   Press Ctrl+C to shutdown gracefully\n");

    let listener = tokio::net::TcpListener::bind(addr).await?;

    let storage_for_shutdown = Arc::clone(&storage);
    tokio::select! {
        result = axum::serve(listener, app) => {
            result?;
        }
        _ = tokio::signal::ctrl_c() => {
            graceful_shutdown(storage_for_shutdown).await;
        }
    }

    Ok(())
}
