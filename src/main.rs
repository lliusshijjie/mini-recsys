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
use ffi::{add_item_to_hnsw, hnsw_search, init_hnsw_index, HnswConfig};
use model::{generate_category_embedding, generate_user_embedding, Item, ItemJson, User, DIM};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use storage::Storage;
use tower_http::cors::CorsLayer;
use axum::http::{Method, HeaderValue};

// ============================================================================
// AppState - 使用 Storage 进行持久化
// ============================================================================

pub struct AppState {
    pub storage: Arc<Storage>,
    pub users: Vec<User>,           // 用户数据缓存在内存
    pub items: Vec<Item>,           // 物品数据缓存在内存（用于推荐计算）
    pub item_map: HashMap<u64, usize>,
}

// ============================================================================
// Request/Response 数据结构
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
    popularity: f32,
    final_score: f32,
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
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct UsersResponse {
    users: Vec<UserInfo>,
}

// ============================================================================
// API Handlers
// ============================================================================

async fn recommend_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RecommendQuery>,
) -> Result<Json<RecommendResponse>, (StatusCode, Json<ErrorResponse>)> {
    let user = state.users.iter()
        .find(|u| u.id == params.uid)
        .ok_or_else(|| {
            (StatusCode::NOT_FOUND, Json(ErrorResponse {
                error: format!("User {} not found", params.uid),
            }))
        })?;

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

async fn health_handler() -> &'static str {
    "OK"
}

// ============================================================================
// 数据初始化
// ============================================================================

fn init_users() -> Vec<User> {
    vec![
        User {
            id: 1,
            name: "Coder (Electronics + Books)".into(),
            embedding: generate_user_embedding(&["Electronics", "Books"]),
        },
        User {
            id: 2,
            name: "Home Maker (Home)".into(),
            embedding: generate_user_embedding(&["Home"]),
        },
        User {
            id: 3,
            name: "Fashionista (Clothing)".into(),
            embedding: generate_user_embedding(&["Clothing"]),
        },
    ]
}

fn load_items_from_json() -> Result<Vec<Item>> {
    let json_str = std::fs::read_to_string("assets/products.json")?;
    let items_json: Vec<ItemJson> = serde_json::from_str(&json_str)?;
    
    let mut rng = rand::thread_rng();
    use rand::Rng;
    
    let items: Vec<Item> = items_json.into_iter()
        .map(|json| {
            let embedding = generate_category_embedding(&json.category);
            let popularity = rng.gen::<f32>();
            Item::from_json(json, embedding, popularity)
        })
        .collect();
    
    Ok(items)
}

fn init_data_with_storage(storage: Arc<Storage>) -> Result<Arc<AppState>> {
    // 检查数据库是否有数据
    let items = if storage.items_count() == 0 {
        println!("📂 Database is empty, loading from products.json...");
        let items = load_items_from_json()?;
        
        // 写入数据库
        for item in &items {
            storage.save_item(item)?;
        }
        println!("💾 Saved {} items to database", items.len());
        items
    } else {
        println!("📂 Loading items from database...");
        let items: Vec<Item> = storage.iter_items()
            .filter_map(|r| r.ok())
            .collect();
        println!("📦 Loaded {} items from database", items.len());
        items
    };

    // 用户数据（也可以持久化，这里简化处理）
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

    let item_map: HashMap<u64, usize> = items.iter()
        .enumerate()
        .map(|(idx, item)| (item.id, idx))
        .collect();

    Ok(Arc::new(AppState {
        storage,
        users,
        items,
        item_map,
    }))
}

// ============================================================================
// HNSW 索引初始化
// ============================================================================

fn init_hnsw_with_items(items: &[Item]) {
    let config = HnswConfig {
        dim: DIM,
        max_elements: items.len() + 1000,
        m: 16,
        ef_construction: 200,
        ef_search: 100,
    };

    println!("🔧 Initializing HNSW index...");

    if let Err(e) = init_hnsw_index(&config) {
        eprintln!("❌ Failed to initialize HNSW index: {}", e);
        return;
    }

    let mut success_count = 0;
    for item in items {
        if add_item_to_hnsw(item.id, &item.embedding).is_ok() {
            success_count += 1;
        }
    }

    println!("✅ HNSW index initialized with {} items", success_count);
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Initializing Mini-RecSys...\n");

    // 1. 初始化 Sled 存储
    let storage = Arc::new(Storage::new("data/db")?);
    println!("💾 Sled database opened at data/db\n");

    // 2. 加载或初始化数据
    let state = init_data_with_storage(storage)?;
    println!("📊 Loaded {} users, {} items\n", state.users.len(), state.items.len());

    // 3. 初始化 HNSW 索引
    init_hnsw_with_items(&state.items);
    println!();

    // 4. 配置 CORS
    let cors = CorsLayer::new()
        .allow_origin("http://localhost:5173".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    // 5. 配置路由
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/users", get(users_handler))
        .route("/recommend", get(recommend_handler))
        .layer(cors)
        .with_state(state);

    // 6. 启动服务器
    let addr = "0.0.0.0:3000";
    println!("🌐 Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}
