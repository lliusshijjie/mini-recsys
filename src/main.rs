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
use tower_http::cors::CorsLayer;
use axum::http::{Method, HeaderValue};

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

    let candidates = ffi::recommend_recall(&user.embedding, &state.items, 50);

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

#[tokio::main]
async fn main() {
    println!("🚀 Initializing Mini-RecSys...");

    let state = init_data();
    println!("📊 Loaded {} users, {} items", state.users.len(), state.items.len());

    let cors = CorsLayer::new()
        .allow_origin("http://localhost:5173".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/users", get(users_handler))
        .route("/recommend", get(recommend_handler))
        .layer(cors)
        .with_state(state);

    let addr = "0.0.0.0:3000";
    println!("🌐 Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
