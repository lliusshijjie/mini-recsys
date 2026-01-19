//! Mini-RecSys - 混合 Rust/C++ 推荐系统 Demo

mod ffi;
mod model;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use model::{init_data, AppState};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
struct RecommendQuery {
    uid: u64,
}

#[derive(Serialize)]
struct RecommendItem {
    item_id: u64,
    name: String,
    sim_score: f32,
    popularity: f32,
    final_score: f32,
}

#[derive(Serialize)]
struct RecommendResponse {
    user_id: u64,
    recommendations: Vec<RecommendItem>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

async fn recommend_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RecommendQuery>,
) -> Result<Json<RecommendResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Step 1: 查找用户
    let user = state.users.iter()
        .find(|u| u.id == params.uid)
        .ok_or_else(|| {
            (StatusCode::NOT_FOUND, Json(ErrorResponse {
                error: format!("User {} not found", params.uid),
            }))
        })?;

    // Step 2: 调用 FFI 获取 Top-50 候选项
    let candidates = ffi::recommend_recall(&user.embedding, &state.items, 50);

    // Step 3: 重排序 FinalScore = SimScore * 0.7 + Popularity * 0.3
    let mut recommendations: Vec<RecommendItem> = candidates.into_iter()
        .filter_map(|(item_id, sim_score)| {
            let idx = *state.item_map.get(&item_id)?;
            let item = &state.items[idx];
            let final_score = sim_score * 0.7 + item.popularity * 0.3;
            Some(RecommendItem {
                item_id,
                name: item.name.clone(),
                sim_score,
                popularity: item.popularity,
                final_score,
            })
        })
        .collect();

    // 按 final_score 降序排序
    recommendations.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());

    // Step 4: 返回 Top 10
    recommendations.truncate(10);

    Ok(Json(RecommendResponse {
        user_id: user.id,
        recommendations,
    }))
}

async fn health_handler() -> &'static str {
    "OK"
}

#[tokio::main]
async fn main() {
    println!("🚀 Initializing Mini-RecSys...");

    let state = init_data();
    println!("� Loaded {} users, {} items", state.users.len(), state.items.len());

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/recommend", get(recommend_handler))
        .with_state(state);

    let addr = "0.0.0.0:3000";
    println!("🌐 Server running at http://{}", addr);
    println!("📝 Try: http://localhost:3000/recommend?uid=1");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
