mod rust_kernels;

use rust_kernels::{dot_scalar, dot_simd, naive_merge_filter_topk, Candidate, Workspace};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    candidate_count: usize,
    max_id: usize,
    seen_count: usize,
    k: usize,
    iterations: usize,
}

fn main() {
    let cases = [
        Case {
            name: "small",
            candidate_count: 1_000,
            max_id: 2_000,
            seen_count: 200,
            k: 100,
            iterations: 2_000,
        },
        Case {
            name: "medium",
            candidate_count: 10_000,
            max_id: 20_000,
            seen_count: 2_000,
            k: 100,
            iterations: 300,
        },
        Case {
            name: "large",
            candidate_count: 100_000,
            max_id: 200_000,
            seen_count: 20_000,
            k: 100,
            iterations: 40,
        },
    ];

    println!("language,algorithm,case,candidates,max_id,seen,k,iterations,total_ms,avg_us,checksum");
    for case in cases {
        run_case(case);
    }
    run_dot_case("dot384", 384, 20_000, 300);
}

fn run_case(case: Case) {
    let candidates = generate_candidates(case.candidate_count, case.max_id as u32, 0xC0FFEE);
    let seen_ids = generate_seen_ids(case.seen_count, case.max_id as u32, 0xBAD5EED);

    let (naive_duration, naive_checksum) = measure(case.iterations, || {
        checksum(&naive_merge_filter_topk(
            &candidates,
            &seen_ids,
            case.max_id,
            case.k,
        ))
    });

    let mut workspace = Workspace::new(case.max_id);
    let (optimized_duration, optimized_checksum) = measure(case.iterations, || {
        checksum(&workspace.merge_filter_topk(&candidates, &seen_ids, case.k))
    });

    print_result("rust", "naive_hash", case, naive_duration, naive_checksum);
    print_result(
        "rust",
        "generation_topk",
        case,
        optimized_duration,
        optimized_checksum,
    );
}

fn run_dot_case(name: &'static str, dim: usize, vector_count: usize, iterations: usize) {
    let query = generate_f32_values(dim, 0x5151);
    let vectors = generate_f32_values(dim * vector_count, 0x9191);

    let (scalar_duration, scalar_checksum) = measure(iterations, || {
        let mut bits = 0u64;
        for vector in vectors.chunks_exact(dim) {
            bits = bits.wrapping_add(dot_scalar(&query, vector).to_bits() as u64);
        }
        bits
    });
    let (simd_duration, simd_checksum) = measure(iterations, || {
        let mut bits = 0u64;
        for vector in vectors.chunks_exact(dim) {
            bits = bits.wrapping_add(dot_simd(&query, vector).to_bits() as u64);
        }
        bits
    });

    print_dot_result(
        "rust",
        "dot_scalar",
        name,
        dim,
        vector_count,
        iterations,
        scalar_duration,
        scalar_checksum,
    );
    print_dot_result(
        "rust",
        "dot_simd",
        name,
        dim,
        vector_count,
        iterations,
        simd_duration,
        simd_checksum,
    );
}

fn measure(iterations: usize, mut operation: impl FnMut() -> u64) -> (Duration, u64) {
    let mut checksum_acc = 0u64;
    let started = Instant::now();
    for _ in 0..iterations {
        checksum_acc = checksum_acc.wrapping_mul(1_099_511_628_211).wrapping_add(operation());
    }
    (started.elapsed(), checksum_acc)
}

fn print_result(language: &str, algorithm: &str, case: Case, duration: Duration, checksum: u64) {
    let total_ms = duration.as_secs_f64() * 1000.0;
    let avg_us = duration.as_secs_f64() * 1_000_000.0 / case.iterations as f64;
    println!(
        "{},{},{},{},{},{},{},{},{:.3},{:.3},{}",
        language,
        algorithm,
        case.name,
        case.candidate_count,
        case.max_id,
        case.seen_count,
        case.k,
        case.iterations,
        total_ms,
        avg_us,
        checksum
    );
}

fn print_dot_result(
    language: &str,
    algorithm: &str,
    name: &str,
    dim: usize,
    vector_count: usize,
    iterations: usize,
    duration: Duration,
    checksum: u64,
) {
    let total_ms = duration.as_secs_f64() * 1000.0;
    let avg_us = duration.as_secs_f64() * 1_000_000.0 / iterations as f64;
    println!(
        "{},{},{},{},{},{},{},{},{:.3},{:.3},{}",
        language,
        algorithm,
        name,
        dim * vector_count,
        dim,
        vector_count,
        dim,
        iterations,
        total_ms,
        avg_us,
        checksum
    );
}

fn checksum(items: &[Candidate]) -> u64 {
    items.iter().fold(0u64, |acc, item| {
        acc.wrapping_mul(16_777_619)
            ^ item.id as u64
            ^ ((item.source_mask as u64) << 32)
            ^ ((item.score.to_bits() as u64) << 1)
    })
}

fn generate_candidates(count: usize, max_id: u32, seed: u64) -> Vec<Candidate> {
    let mut rng = Lcg::new(seed);
    let mut candidates = Vec::with_capacity(count);
    for _ in 0..count {
        let id = rng.next_u32() % (max_id + 1);
        let score = rng.next_f32();
        let source_mask = 1u8 << (rng.next_u32() % 4);
        candidates.push(Candidate {
            id,
            score,
            source_mask,
        });
    }
    candidates
}

fn generate_seen_ids(count: usize, max_id: u32, seed: u64) -> Vec<u32> {
    let mut rng = Lcg::new(seed);
    let mut seen = Vec::with_capacity(count);
    for _ in 0..count {
        seen.push(rng.next_u32() % (max_id + 1));
    }
    seen
}

fn generate_f32_values(count: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..count).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}
