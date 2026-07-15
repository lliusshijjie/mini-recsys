//! Mini-RecSys service entrypoint.

mod behavior;
mod config;
mod embedding;
mod ffi;
mod hybrid;
mod init;
mod model;
mod observability;
mod recommendation;
mod storage;
mod text_search;

use anyhow::Result;
use observability::Metrics;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Initializing Mini-RecSys...\n");

    let config = config::AppConfig::from_env()?;
    let metrics = Arc::new(Metrics::default());
    let readiness = Arc::new(init::ReadinessState::default());

    let embedding_model = init::load_embedding_model(&config);
    let storage = init::open_storage(&config)?;
    let text_search = init::open_text_search(&config)?;
    let state = init::init_data_with_storage(
        &config,
        Arc::clone(&storage),
        embedding_model,
        text_search,
        Arc::clone(&metrics),
        Arc::clone(&readiness),
    )?;

    init::log_loaded_state(&state);
    if init::warmup_embedding_model(state.embedding_model.as_deref())? {
        state.readiness.mark_ready();
    }
    println!();

    let app = init::build_app(Arc::clone(&state), &config);
    init::serve_app(app, storage, Arc::clone(&state.hnsw_index), config).await
}
