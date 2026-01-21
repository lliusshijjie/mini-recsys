//! Mini-RecSys - 混合 Rust/C++ 推荐系统 Demo

mod ffi;
mod model;
mod storage;
mod embedding;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use fastbloom_rs::Membership;
use ffi::{add_item_to_hnsw, get_hnsw_count, hnsw_search, load_hnsw_index, save_hnsw_index};
use model::{generate_category_embedding, generate_user_embedding, generate_random_embedding, Item, ItemJson, User, DIM};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use storage::Storage;
use tower_http::cors::CorsLayer;
use axum::http::{Method, HeaderValue};

const INDEX_PATH: &str = "data/index.bin";
const DB_PATH: &str = "data/db";
const MIN_RECOMMENDATIONS: usize = 5;

// ============================================================================
// AppState
// ============================================================================

pub struct AppState {
    pub storage: Arc<Storage>,
    pub users: Vec<User>,
    pub items: Vec<Item>,
    pub item_map: HashMap<u64, usize>,
    pub embedding_model: Option<Arc<embedding::EmbeddingModel>>,
}

// ============================================================================
// Request/Response
// ============================================================================

#[derive(Deserialize)]
struct RecommendQuery { uid: u64 }

#[derive(Serialize)]
struct RecommendItem {
    item_id: u64,
    name: String,
    category: String,
    image_url: String,
    price: f32,
    sim_score: f32,
    popularity: f32,
    final_score: f32,
}

#[derive(Serialize)]
struct UserInfo { id: u64, name: String }

#[derive(Serialize)]
struct RecommendResponse { user: UserInfo, recommendations: Vec<RecommendItem>, filtered_count: usize }

#[derive(Serialize)]
struct ErrorResponse { error: String }

#[derive(Serialize)]
struct UsersResponse { users: Vec<UserInfo> }

#[derive(Deserialize)]
struct MarkSeenRequest { uid: u64, item_ids: Vec<u64> }

#[derive(Serialize)]
struct MarkSeenResponse { marked: usize }

#[derive(Deserialize)]
struct SearchQuery { q: String }

#[derive(Serialize)]
struct SearchResponse { query: String, results: Vec<RecommendItem> }

// ============================================================================
// Handlers
// ============================================================================

async fn recommend_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RecommendQuery>,
) -> Result<Json<RecommendResponse>, (StatusCode, Json<ErrorResponse>)> {
    let user = state.users.iter()
        .find(|u| u.id == params.uid)
        .ok_or_else(|| (StatusCode::NOT_FOUND, Json(ErrorResponse {
            error: format!("User {} not found", params.uid),
        })))?;

    // Step A: 召回 Top-100
    let candidates = hnsw_search(&user.embedding, 100);

    // Step B: 获取用户的 Bloom Filter
    let filter = state.storage.get_user_filter(params.uid)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: format!("Failed to get filter: {}", e),
        })))?;

    // Step C: 过滤已看过的商品
    let mut filtered_count = 0;
    let mut recommendations: Vec<RecommendItem> = candidates.into_iter()
        .filter_map(|(item_id, sim_score)| {
            // 检查是否已看过
            if filter.contains(&item_id.to_le_bytes()) {
                filtered_count += 1;
                return None;
            }
            
            let idx = *state.item_map.get(&item_id)?;
            let item = &state.items[idx];
            let final_score = sim_score * 0.7 + item.popularity * 0.3;
            Some(RecommendItem {
                item_id,
                name: item.name.clone(),
                category: item.category.clone(),
                image_url: item.image_url.clone(),
                price: item.price,
                sim_score,
                popularity: item.popularity,
                final_score,
            })
        })
        .collect();

    recommendations.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());
    
    // Step D: 降级填充 (Fallback)
    if recommendations.len() < MIN_RECOMMENDATIONS {
        // 从热门商品中随机补充
        let mut popular_items: Vec<_> = state.items.iter()
            .filter(|item| !filter.contains(&item.id.to_le_bytes()))
            .collect();
        popular_items.sort_by(|a, b| b.popularity.partial_cmp(&a.popularity).unwrap());
        
        for item in popular_items.into_iter().take(MIN_RECOMMENDATIONS - recommendations.len()) {
            if !recommendations.iter().any(|r| r.item_id == item.id) {
                recommendations.push(RecommendItem {
                    item_id: item.id,
                    name: item.name.clone(),
                    category: item.category.clone(),
                    image_url: item.image_url.clone(),
                    price: item.price,
                    sim_score: 0.0,
                    popularity: item.popularity,
                    final_score: item.popularity * 0.3,
                });
            }
        }
    }
    
    recommendations.truncate(10);

    Ok(Json(RecommendResponse {
        user: UserInfo { id: user.id, name: user.name.clone() },
        recommendations,
        filtered_count,
    }))
}

async fn mark_seen_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<MarkSeenRequest>,
) -> Result<Json<MarkSeenResponse>, (StatusCode, Json<ErrorResponse>)> {
    // 加载用户的 Filter
    let mut filter = state.storage.get_user_filter(payload.uid)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: format!("Failed to get filter: {}", e),
        })))?;
    
    // 插入所有 item_id
    for item_id in &payload.item_ids {
        filter.add(&item_id.to_le_bytes());
    }
    
    // 保存回 Sled
    state.storage.save_user_filter(payload.uid, &filter)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: format!("Failed to save filter: {}", e),
        })))?;
    
    Ok(Json(MarkSeenResponse { marked: payload.item_ids.len() }))
}

async fn users_handler(State(state): State<Arc<AppState>>) -> Json<UsersResponse> {
    let users = state.users.iter()
        .map(|u| UserInfo { id: u.id, name: u.name.clone() })
        .collect();
    Json(UsersResponse { users })
}

async fn search_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let model = state.embedding_model.as_ref()
        .ok_or_else(|| (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse {
            error: "Embedding model not loaded".to_string(),
        })))?;

    // 编码查询为向量
    let query_vec = model.encode(&params.q)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: format!("Encoding failed: {}", e),
        })))?;

    // HNSW 搜索
    let candidates = hnsw_search(&query_vec, 20);

    let results: Vec<RecommendItem> = candidates.into_iter()
        .filter_map(|(item_id, sim_score)| {
            let idx = *state.item_map.get(&item_id)?;
            let item = &state.items[idx];
            Some(RecommendItem {
                item_id,
                name: item.name.clone(),
                category: item.category.clone(),
                image_url: item.image_url.clone(),
                price: item.price,
                sim_score,
                popularity: item.popularity,
                final_score: sim_score,
            })
        })
        .collect();

    Ok(Json(SearchResponse { query: params.q, results }))
}

async fn health_handler() -> &'static str { "OK" }

// ============================================================================
// 数据初始化
// ============================================================================

fn init_users() -> Vec<User> {
    vec![
        // 明确单一兴趣的用户
        User { id: 1, name: "程序员小明 (Electronics + Books)".into(), embedding: generate_user_embedding(&["Electronics", "Books"]) },
        User { id: 2, name: "居家达人小红 (Home)".into(), embedding: generate_user_embedding(&["Home"]) },
        User { id: 3, name: "时尚达人小美 (Clothing)".into(), embedding: generate_user_embedding(&["Clothing"]) },
        
        // 双兴趣用户
        User { id: 4, name: "极客玩家 (Electronics)".into(), embedding: generate_user_embedding(&["Electronics"]) },
        User { id: 5, name: "书虫 (Books)".into(), embedding: generate_user_embedding(&["Books"]) },
        User { id: 6, name: "生活家 (Home + Clothing)".into(), embedding: generate_user_embedding(&["Home", "Clothing"]) },
        
        // 混合兴趣用户
        User { id: 7, name: "全能选手 (All Categories)".into(), embedding: generate_user_embedding(&["Electronics", "Books", "Home", "Clothing"]) },
        User { id: 8, name: "科技宅 (Electronics + Home)".into(), embedding: generate_user_embedding(&["Electronics", "Home"]) },
        
        // 噪声用户 - 使用随机embedding
        User { id: 9, name: "新用户A (Random)".into(), embedding: generate_random_embedding() },
        User { id: 10, name: "新用户B (Random)".into(), embedding: generate_random_embedding() },
    ]
}

fn load_items_from_json(embedding_model: &embedding::EmbeddingModel) -> Result<Vec<Item>> {
    use rand::Rng;
    let json_str = std::fs::read_to_string("assets/products.json")?;
    let items_json: Vec<ItemJson> = serde_json::from_str(&json_str)?;
    let mut rng = rand::thread_rng();
    let total = items_json.len();
    println!("🧠 Encoding {} items with ONNX model...", total);
    
    let items: Vec<Item> = items_json.into_iter()
        .enumerate()
        .map(|(i, json)| {
            // 使用 ONNX 模型生成真实语义向量
            let embedding = embedding_model.encode(&json.name)
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
    Ok(items_json.into_iter()
        .map(|json| {
            let embedding = generate_category_embedding(&json.category);
            let popularity = rng.gen::<f32>();
            Item::from_json(json, embedding, popularity)
        })
        .collect())
}

fn init_data_with_storage(storage: Arc<Storage>, embedding_model: Option<Arc<embedding::EmbeddingModel>>) -> Result<Arc<AppState>> {
    let items = if storage.items_count() == 0 {
        println!("📂 Database empty, loading from products.json...");
        let items = match &embedding_model {
            Some(model) => load_items_from_json(model)?,
            None => {
                println!("⚠️  No embedding model, using category-based vectors");
                load_items_from_json_fallback()?
            }
        };
        for item in &items { storage.save_item(item)?; }
        println!("💾 Saved {} items to database", items.len());
        items
    } else {
        println!("📂 Loading items from database...");
        let items: Vec<Item> = storage.iter_items().filter_map(|r| r.ok()).collect();
        println!("📦 Loaded {} items from database", items.len());
        items
    };

    let users = if storage.users_count() == 0 {
        let users = init_users();
        for user in &users { storage.save_user(user)?; }
        println!("💾 Saved {} users to database", users.len());
        users
    } else {
        storage.get_all_users()?
    };

    let item_map: HashMap<u64, usize> = items.iter().enumerate().map(|(i, item)| (item.id, i)).collect();

    Ok(Arc::new(AppState { storage, users, items, item_map, embedding_model }))
}

// ============================================================================
// 索引初始化 (Hydration)
// ============================================================================

fn init_hnsw_with_hydration(items: &[Item]) -> Result<()> {
    let max_elements = items.len() + 1000;
    
    println!("🔧 Loading HNSW index from {}...", INDEX_PATH);
    let loaded = load_hnsw_index(INDEX_PATH, DIM, max_elements, 100)
        .map_err(|e| anyhow::anyhow!(e))?;
    
    let index_count = get_hnsw_count();
    let db_count = items.len();
    
    if loaded && index_count == db_count {
        println!("✅ HNSW index loaded: {} items (consistent with DB)", index_count);
        return Ok(());
    }
    
    if !loaded {
        println!("📝 Index file not found, created new empty index");
    } else {
        println!("⚠️  Index count ({}) != DB count ({}), rebuilding...", index_count, db_count);
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
// 优雅退出
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

    // 1. 初始化 ONNX 模型
    let embedding_model = match embedding::EmbeddingModel::new() {
        Ok(model) => {
            println!("🧠 Embedding model loaded (dimension: {})\n", model.dimension());
            Some(Arc::new(model))
        }
        Err(e) => {
            eprintln!("⚠️  Failed to load embedding model: {}\n   /search will be unavailable\n", e);
            None
        }
    };

    let storage = Arc::new(Storage::new(DB_PATH)?);
    println!("💾 Sled database opened at {}\n", DB_PATH);

    let state = init_data_with_storage(Arc::clone(&storage), embedding_model)?;
    println!("📊 Loaded {} users, {} items\n", state.users.len(), state.items.len());

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
    println!("   GET  /search?q=<query> - 语义搜索");
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
