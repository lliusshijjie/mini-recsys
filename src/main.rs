//! Mini-RecSys service entrypoint.

mod behavior;
mod embedding;
mod ffi;
mod hybrid;
mod init;
mod model;
mod recommendation;
mod storage;
mod text_search;

use anyhow::Result;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Initializing Mini-RecSys...\n");

    let embedding_model = init::load_embedding_model();
    let storage = init::open_storage()?;
    let text_search = init::open_text_search()?;
    let state = init::init_data_with_storage(Arc::clone(&storage), embedding_model, text_search)?;

    init::log_loaded_state(&state);
    init::init_hnsw_with_hydration(&state.items)?;
    println!();

    let app = init::build_app(Arc::clone(&state));
    init::serve_app(app, storage).await
}
