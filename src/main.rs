//! Mini-RecSys - 混合 Rust/C++ 推荐系统 Demo
//!
//! 本项目演示了 Rust 与 C++ 的 FFI 集成，使用 HNSW 算法进行高效的向量近似最近邻搜索。

mod ffi;
mod model;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use ffi::{add_item_to_hnsw, hnsw_search, init_hnsw_index, HnswConfig};
use model::{init_data, AppState};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use axum::http::{Method, HeaderValue};

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

    // 使用 HNSW 索引进行高效近似最近邻搜索
    // 召回 50 个候选物品，比暴力搜索快得多
    let candidates = hnsw_search(&user.embedding, 50);

    let mut recommendations: Vec<RecommendItem> = candidates.into_iter()
        .filter_map(|(item_id, sim_score)| {
            let idx = *state.item_map.get(&item_id)?;
            let item = &state.items[idx];
            // 融合相似度分数 (70%) 和热度分数 (30%)
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

    // 按最终分数排序并取 Top 10
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
// HNSW 索引初始化
// ============================================================================

/// 初始化 HNSW 索引并灌入所有物品数据
fn init_hnsw_with_items(state: &AppState) {
    // HNSW 参数解释:
    // - dim: 向量维度 (我们使用 64 维)
    // - max_elements: 最大物品数量
    // - M: 每个节点的最大连接数
    //   - 更高的 M = 更好的召回率，但更多内存和更慢的构建速度
    //   - 推荐值: 16 (平衡), 32-64 (高精度)
    // - ef_construction: 构建时的搜索深度
    //   - 更高 = 更好的索引质量，但更慢的构建
    //   - 推荐值: 200
    // - ef_search: 查询时的搜索深度
    //   - 更高 = 更好的召回率，但更慢的查询
    //   - 必须 >= k (返回的结果数)
    //   - 推荐值: 50-100
    let config = HnswConfig {
        dim: 64,                    // 向量维度
        max_elements: state.items.len() + 1000,  // 预留一些空间
        m: 16,                      // 节点连接数 (平衡模式)
        ef_construction: 200,       // 构建深度
        ef_search: 100,             // 查询深度
    };

    println!("🔧 Initializing HNSW index...");
    println!("   - Dimension: {}", config.dim);
    println!("   - M (connections): {}", config.m);
    println!("   - ef_construction: {}", config.ef_construction);
    println!("   - ef_search: {}", config.ef_search);

    // 初始化索引
    if let Err(e) = init_hnsw_index(&config) {
        eprintln!("❌ Failed to initialize HNSW index: {}", e);
        return;
    }

    // 将所有物品添加到索引中
    let mut success_count = 0;
    for item in &state.items {
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
async fn main() {
    println!("🚀 Initializing Mini-RecSys...\n");

    // 1. 加载商品和用户数据
    let state = init_data();
    println!("📊 Loaded {} users, {} items\n", state.users.len(), state.items.len());

    // 2. 初始化 HNSW 索引并灌入商品数据
    init_hnsw_with_items(&state);
    println!();

    // 3. 配置 CORS
    let cors = CorsLayer::new()
        .allow_origin("http://localhost:5173".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    // 4. 配置路由
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/users", get(users_handler))
        .route("/recommend", get(recommend_handler))
        .layer(cors)
        .with_state(state);

    // 5. 启动服务器
    let addr = "0.0.0.0:3000";
    println!("🌐 Server running at http://{}", addr);
    println!("   - GET /health     - 健康检查");
    println!("   - GET /users      - 获取用户列表");
    println!("   - GET /recommend?uid=<id> - 获取推荐\n");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
