//! Explanation helpers for recommendation outputs.

use crate::recommendation::recall::RecallSource;
use std::collections::HashSet;

pub(super) fn source_label(sources: &HashSet<RecallSource>) -> String {
    if sources.contains(&RecallSource::RecentItemSimilarity) {
        return RecallSource::RecentItemSimilarity.as_str().to_string();
    }

    if sources.len() > 1 {
        return "mixed".to_string();
    }

    sources
        .iter()
        .next()
        .copied()
        .map(RecallSource::as_str)
        .unwrap_or("unknown")
        .to_string()
}

pub(super) fn reason_for(source: &str, semantic_score: f32, category_score: f32) -> String {
    if semantic_score >= 0.50 {
        "semantic_match".to_string()
    } else if category_score >= 0.50 {
        "category_match".to_string()
    } else if source == "popular_fallback" || source == "mixed" {
        "popular_item".to_string()
    } else {
        "category_match".to_string()
    }
}
