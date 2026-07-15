//! Recommendation pipeline modules.

mod explain;
mod features;
mod indexes;
mod pipeline;
mod rank;
mod recall;
mod rerank;
pub(crate) mod service;
mod types;

#[cfg(test)]
mod tests;

pub use indexes::RecommendationIndexes;
#[cfg(test)]
pub use pipeline::build_recommendations;
#[cfg(test)]
pub use pipeline::build_recommendations_with_indexes;
#[cfg(test)]
pub use rank::RankingStrategyKind;
pub use service::{
    RecommendationService, RecommendationServiceConfig, RecommendationServiceError,
    RecommendationServiceOutput,
};
#[cfg(test)]
pub use types::{RecentRecallMode, RecommendationConfig};
pub use types::{RecommendationOutput, RecommendedItem};
