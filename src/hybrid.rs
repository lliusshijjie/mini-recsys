use std::collections::HashMap;

/// Reciprocal Rank Fusion
/// Score = 1 / (k + rank)
const RRF_K: f32 = 60.0;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u32,
    pub score: f32, // RRF score
}

pub fn rrf_merge(
    vector_results: Vec<(u32, f32)>, // (id, similarity)
    keyword_results: Vec<u32>,       // id only, rank implies score
) -> Vec<SearchResult> {
    let mut scores: HashMap<u32, f32> = HashMap::new();

    // 1. Process Vector Results
    for (rank, (id, _sim)) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (RRF_K + rank as f32 + 1.0);
        *scores.entry(*id).or_insert(0.0) += rrf_score;
    }

    // 2. Process Keyword Results
    for (rank, id) in keyword_results.iter().enumerate() {
        let rrf_score = 1.0 / (RRF_K + rank as f32 + 1.0);
        *scores.entry(*id).or_insert(0.0) += rrf_score;
    }

    // 3. Convert to Vec and Sort
    let mut results: Vec<SearchResult> = scores
        .into_iter()
        .map(|(id, score)| SearchResult { id, score })
        .collect();

    // Sort descending by score
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    results
}
