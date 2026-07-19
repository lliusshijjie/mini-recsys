#![allow(dead_code)]

#[path = "rust_rerank.rs"]
mod rust_rerank;

use rust_rerank::{
    build_similarity_matrix, dpp_rerank, load_candidates_csv, mmr_rerank, Candidate, EMBEDDING_DIM,
};
use std::time::Instant;

fn synth_candidates(n: usize) -> Vec<Candidate> {
    let categories = ["Books", "Electronics", "Home", "Clothing", "Sports", "Beauty"];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let category = categories[i % categories.len()];
        let mut embedding = vec![0.0f32; EMBEDDING_DIM];
        let base = (i % categories.len()) * 5;
        for d in 0..5 {
            embedding[base + d] = 1.0;
        }
        // Mild item-specific noise keeps similarities < 1 for same category.
        embedding[i % EMBEDDING_DIM] += 0.05;
        rust_rerank::l2_normalize(&mut embedding);
        out.push(Candidate {
            id: (i + 1) as u64,
            score: 1.0 - (i as f32) * 0.001,
            category: category.into(),
            embedding,
        });
    }
    out
}

fn bench(name: &str, rounds: u32, mut f: impl FnMut()) {
    // Warmup
    for _ in 0..3 {
        f();
    }
    let started = Instant::now();
    for _ in 0..rounds {
        f();
    }
    let avg_us = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    println!("rust,{name},{avg_us:.3}");
}

fn main() {
    let fixture = load_candidates_csv("experiments/rerank_diversity/data/candidates.csv").ok();

    for &n in &[64usize, 128, 256] {
        let candidates = synth_candidates(n);
        let sim = build_similarity_matrix(&candidates);
        let rounds = if n <= 128 { 200 } else { 80 };

        bench(&format!("mmr_n{n}_k10"), rounds, || {
            let _ = mmr_rerank(&candidates, &sim, 10, 0.7);
        });
        bench(&format!("dpp_n{n}_k10"), rounds, || {
            let _ = dpp_rerank(&candidates, &sim, 10, 1.0);
        });
    }

    if let Some(candidates) = fixture {
        let sim = build_similarity_matrix(&candidates);
        let n = candidates.len();
        bench(&format!("fixture_mmr_n{n}_k10"), 100, || {
            let _ = mmr_rerank(&candidates, &sim, 10, 0.7);
        });
        bench(&format!("fixture_dpp_n{n}_k10"), 100, || {
            let _ = dpp_rerank(&candidates, &sim, 10, 1.0);
        });
    }
}
