//! Mini-RecSys - 混合 Rust/C++ 推荐系统 Demo

mod ffi;
mod model;
mod storage;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use ffi::{add_item_to_hnsw, get_hnsw_count, hnsw_search, load_hnsw_index, save_hnsw_index};
use model::{generate_category_embedding, generate_user_embedding, Item, ItemJson, User, DIM};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use storage::Storage;
use tower_http::cors::CorsLayer;
use axum::http::{Method, HeaderValue};

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
struct RecommendResponse { user: UserInfo, recommendations: Vec<RecommendItem> }

#[derive(Serialize)]
struct ErrorResponse { error: String }

#[derive(Serialize)]
struct UsersResponse { users: Vec<UserInfo> }

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

    let candidates = hnsw_search(&user.embedding, 50);

    let mut recommendations: Vec<RecommendItem> = candidates.into_iter()
        .filter_map(|(item_id, sim_score)| {
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
    recommendations.truncate(10);

    Ok(Json(RecommendResponse {
        user: UserInfo { id: user.id, name: user.name.clone() },
        recommendations,
    }))
}

async fn users_handler(State(state): State<Arc<AppState>>) -> Json<UsersResponse> {
    let users = state.users.iter()
        .map(|u| UserInfo { id: u.id, name: u.name.clone() })
        .collect();
    Json(UsersResponse { users })
}

async fn health_handler() -> &'static str { "OK" }

// ============================================================================
// 数据初始化
// ============================================================================

fn init_users() -> Vec<User> {
    vec![
        User { id: 1, name: "Coder (Electronics + Books)".into(), embedding: generate_user_embedding(&["Electronics", "Books"]) },
        User { id: 2, name: "Home Maker (Home)".into(), embedding: generate_user_embedding(&["Home"]) },
        User { id: 3, name: "Fashionista (Clothing)".into(), embedding: generate_user_embedding(&["Clothing"]) },
    ]
}

fn load_items_from_json() -> Result<Vec<Item>> {
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

fn init_data_with_storage(storage: Arc<Storage>) -> Result<Arc<AppState>> {
    let items = if storage.items_count() == 0 {
        println!("📂 Database empty, loading from products.json...");
        let items = load_items_from_json()?;
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

    Ok(Arc::new(AppState { storage, users, items, item_map }))
}

// ============================================================================
// 索引初始化 (Hydration)
// ============================================================================

fn init_hnsw_with_hydration(items: &[Item]) -> Result<()> {
    let max_elements = items.len() + 1000;
    
    // Step A: 尝试加载索引
    println!("🔧 Loading HNSW index from {}...", INDEX_PATH);
    let loaded = load_hnsw_index(INDEX_PATH, DIM, max_elements, 100)
        .map_err(|e| anyhow::anyhow!(e))?;
    
    let index_count = get_hnsw_count();
    let db_count = items.len();
    
    if loaded && index_count == db_count {
        // 索引与数据库一致
        println!("✅ HNSW index loaded: {} items (consistent with DB)", index_count);
        return Ok(());
    }
    
    // Step B & C: 索引为空或不一致，需要重建
    if !loaded {
        println!("📝 Index file not found, created new empty index");
    } else {
        println!("⚠️  Index count ({}) != DB count ({}), rebuilding...", index_count, db_count);
    }
    
    // 重建索引：遍历 Sled 中的所有 Items
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
    
    // 保存 HNSW 索引
    match save_hnsw_index(INDEX_PATH) {
        Ok(()) => println!("💾 HNSW index saved to {}", INDEX_PATH),
        Err(e) => eprintln!("❌ Failed to save index: {}", e),
    }
    
    // Flush Sled 数据库
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

    // 1. Sled 存储
    let storage = Arc::new(Storage::new(DB_PATH)?);
    println!("💾 Sled database opened at {}\n", DB_PATH);

    // 2. 加载数据
    let state = init_data_with_storage(Arc::clone(&storage))?;
    println!("📊 Loaded {} users, {} items\n", state.users.len(), state.items.len());

    // 3. HNSW 索引 (带一致性检查)
    init_hnsw_with_hydration(&state.items)?;
    println!();

    // 4. CORS
    let cors = CorsLayer::new()
        .allow_origin("http://localhost:5173".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    // 5. 路由
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/users", get(users_handler))
        .route("/recommend", get(recommend_handler))
        .layer(cors)
        .with_state(Arc::clone(&state));

    // 6. 启动服务器
    let addr = "0.0.0.0:3000";
    println!("🌐 Server running at http://{}", addr);
    println!("   Press Ctrl+C to shutdown gracefully\n");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    // 使用 tokio::select! 监听信号
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
