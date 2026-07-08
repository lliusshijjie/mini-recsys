//! Lightweight observability metrics.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Debug, Default)]
pub struct Metrics {
    http_requests_total: AtomicU64,
    http_errors_total: AtomicU64,
    http_latency_micros_total: AtomicU64,
    recommendation_candidates_total: AtomicU64,
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

    pub fn render_prometheus(&self) -> String {
        let requests = self.http_requests_total.load(Ordering::Relaxed);
        let errors = self.http_errors_total.load(Ordering::Relaxed);
        let latency_seconds =
            self.http_latency_micros_total.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let candidates = self.recommendation_candidates_total.load(Ordering::Relaxed);

        format!(
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
            ),
            requests, errors, latency_seconds, candidates
        )
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

        let output = metrics.render_prometheus();

        assert!(output.contains("mini_recsys_http_requests_total 2"));
        assert!(output.contains("mini_recsys_http_errors_total 1"));
        assert!(output.contains("mini_recsys_http_latency_seconds_sum 0.030000"));
        assert!(output.contains("mini_recsys_recommendation_candidates_total 42"));
    }
}
