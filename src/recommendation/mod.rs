//! Recommendation pipeline modules.

mod explain;
mod features;
mod indexes;
mod pipeline;
mod rank;
mod recall;
mod rerank;
mod types;

#[cfg(test)]
mod tests;

pub use indexes::RecommendationIndexes;
#[cfg(test)]
pub use pipeline::build_recommendations;
pub use pipeline::build_recommendations_with_indexes;
pub use rank::RankingStrategyKind;
pub(crate) use recall::recent_positive_seed_ids;
pub use types::{RecentRecallMode, RecommendationConfig, RecommendationOutput, RecommendedItem};
