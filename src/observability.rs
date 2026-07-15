//! Lightweight observability metrics.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const RECOMMENDATION_STAGES: [&str; 7] = [
    "storage",
    "semantic_ann",
    "category_recall",
    "recent_ann",
    "popular_fallback",
    "merge_rank",
    "total",
];

#[derive(Debug, Default)]
pub struct Metrics {
    http_requests_total: AtomicU64,
    http_errors_total: AtomicU64,
    http_latency_micros_total: AtomicU64,
    recommendation_candidates_total: AtomicU64,
    recommendation_timeouts_total: AtomicU64,
    user_context_cache_hits_total: AtomicU64,
    user_context_cache_misses_total: AtomicU64,
    batch_users_total: AtomicU64,
    recommendation_stage_latency_micros_total: [AtomicU64; RECOMMENDATION_STAGES.len()],
    request_sequence: AtomicU64,
}

impl Metrics {
    pub fn next_request_id(&self) -> u64 {
        self.request_sequence.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn record_http_request(&self, _method: &str, _path: &str, status: u16, latency: Duration) {
        self.http_requests_total.fetch_add(1, Ordering::Relaxed);
        self.http_latency_micros_total
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
        if status >= 500 {
            self.http_errors_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_recommendation_candidates(&self, count: usize) {
        self.recommendation_candidates_total
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    pub fn record_recommendation_timeout(&self) {
        self.recommendation_timeouts_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_user_context_cache_hit(&self) {
        self.user_context_cache_hits_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_user_context_cache_miss(&self) {
        self.user_context_cache_misses_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_batch_users(&self, count: usize) {
        self.batch_users_total
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    pub fn record_recommendation_stage(&self, stage: &str, latency: Duration) {
        let Some(index) = RECOMMENDATION_STAGES
            .iter()
            .position(|known_stage| *known_stage == stage)
        else {
            return;
        };
        self.recommendation_stage_latency_micros_total[index]
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn render_prometheus(&self) -> String {
        let requests = self.http_requests_total.load(Ordering::Relaxed);
        let errors = self.http_errors_total.load(Ordering::Relaxed);
        let latency_seconds =
            self.http_latency_micros_total.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let candidates = self.recommendation_candidates_total.load(Ordering::Relaxed);
        let timeouts = self.recommendation_timeouts_total.load(Ordering::Relaxed);
        let cache_hits = self.user_context_cache_hits_total.load(Ordering::Relaxed);
        let cache_misses = self.user_context_cache_misses_total.load(Ordering::Relaxed);
        let batch_users = self.batch_users_total.load(Ordering::Relaxed);

        let mut output = format!(
            concat!(
                "# HELP mini_recsys_http_requests_total Total HTTP requests.\n",
                "# TYPE mini_recsys_http_requests_total counter\n",
                "mini_recsys_http_requests_total {}\n",
                "# HELP mini_recsys_http_errors_total Total HTTP responses with status >= 500.\n",
                "# TYPE mini_recsys_http_errors_total counter\n",
                "mini_recsys_http_errors_total {}\n",
                "# HELP mini_recsys_http_latency_seconds_sum Total HTTP request latency in seconds.\n",
                "# TYPE mini_recsys_http_latency_seconds_sum counter\n",
                "mini_recsys_http_latency_seconds_sum {:.6}\n",
                "# HELP mini_recsys_recommendation_candidates_total Total recommendation candidates observed.\n",
                "# TYPE mini_recsys_recommendation_candidates_total counter\n",
                "mini_recsys_recommendation_candidates_total {}\n",
                "# HELP mini_recsys_recommendation_timeouts_total Total recommendation requests that timed out.\n",
                "# TYPE mini_recsys_recommendation_timeouts_total counter\n",
                "mini_recsys_recommendation_timeouts_total {}\n",
                "# HELP mini_recsys_user_context_cache_hits_total Total user context cache hits.\n",
                "# TYPE mini_recsys_user_context_cache_hits_total counter\n",
                "mini_recsys_user_context_cache_hits_total {}\n",
                "# HELP mini_recsys_user_context_cache_misses_total Total user context cache misses.\n",
                "# TYPE mini_recsys_user_context_cache_misses_total counter\n",
                "mini_recsys_user_context_cache_misses_total {}\n",
                "# HELP mini_recsys_batch_users_total Total users submitted through batch recommendation requests.\n",
                "# TYPE mini_recsys_batch_users_total counter\n",
                "mini_recsys_batch_users_total {}\n",
            ),
            requests, errors, latency_seconds, candidates, timeouts, cache_hits, cache_misses, batch_users
        );
        output.push_str(
            "# HELP mini_recsys_recommendation_stage_latency_seconds_sum Total recommendation stage latency in seconds.\n",
        );
        output.push_str("# TYPE mini_recsys_recommendation_stage_latency_seconds_sum counter\n");
        for (index, stage) in RECOMMENDATION_STAGES.iter().enumerate() {
            let latency_seconds = self.recommendation_stage_latency_micros_total[index]
                .load(Ordering::Relaxed) as f64
                / 1_000_000.0;
            output.push_str(&format!(
                "mini_recsys_recommendation_stage_latency_seconds_sum{{stage=\"{}\"}} {:.6}\n",
                stage, latency_seconds
            ));
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn metrics_render_request_latency_error_and_candidate_counts() {
        let metrics = Metrics::default();

        metrics.record_http_request("GET", "/recommend", 200, Duration::from_millis(10));
        metrics.record_http_request("GET", "/recommend", 500, Duration::from_millis(20));
        metrics.record_recommendation_candidates(42);
        metrics.record_recommendation_stage("category_recall", Duration::from_millis(3));
        metrics.record_recommendation_timeout();
        metrics.record_user_context_cache_hit();
        metrics.record_user_context_cache_miss();
        metrics.record_batch_users(3);

        let output = metrics.render_prometheus();

        assert!(output.contains("mini_recsys_http_requests_total 2"));
        assert!(output.contains("mini_recsys_http_errors_total 1"));
        assert!(output.contains("mini_recsys_http_latency_seconds_sum 0.030000"));
        assert!(output.contains("mini_recsys_recommendation_candidates_total 42"));
        assert!(output.contains("mini_recsys_recommendation_timeouts_total 1"));
        assert!(output.contains("mini_recsys_user_context_cache_hits_total 1"));
        assert!(output.contains("mini_recsys_user_context_cache_misses_total 1"));
        assert!(output.contains("mini_recsys_batch_users_total 3"));
        assert!(output.contains(
            "mini_recsys_recommendation_stage_latency_seconds_sum{stage=\"category_recall\"} 0.003000"
        ));
    }
}
