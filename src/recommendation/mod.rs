//! Recommendation pipeline modules.

mod explain;
mod features;
mod pipeline;
mod rank;
mod recall;
mod rerank;
mod types;

#[cfg(test)]
mod tests;

pub use pipeline::build_recommendations;
pub use rank::RankingStrategyKind;
pub use types::{RecommendationConfig, RecommendationOutput, RecommendedItem};
