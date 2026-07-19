//! Standalone MMR / DPP rerank kernels for experiments (not wired into production).
//!
//! # MMR (Maximal Marginal Relevance)
//! Idea: greedily pick the next item that balances relevance and novelty:
//!   MMR(i) = λ * Rel(i) - (1-λ) * max_{j in S} Sim(i, j)
//! Iterative process:
//!   1. Start with S empty and max_sim[i] = 0.
//!   2. Pick argmax MMR(i); append to S.
//!   3. Update max_sim[r] = max(max_sim[r], Sim(r, picked)) for remaining r.
//!   4. Repeat until |S| = K.
//! Complexity with a precomputed similarity matrix: O(N*K) after O(N^2*D) build.
//!
//! # Greedy MAP DPP
//! Idea: maximize set volume det(L_Y) where L_ij = q_i * S_ij * q_j, so high-quality
//! and mutually diverse items are preferred (near-duplicates shrink volume).
//! Iterative process (incremental Cholesky):
//!   1. Build L once; keep residual marginal gains d2[i].
//!   2. Each round pick argmax d2[i] (largest log-det gain).
//!   3. Append one Cholesky column and update d2[j] -= e_j^2.
//!   4. Repeat until K items are chosen.
//! Complexity: O(N^2) kernel build + O(N*K^2) selection.

use std::fs::File;
use std::io::{BufRead, BufReader};

pub const EMBEDDING_DIM: usize = 32;

#[derive(Clone, Debug)]
pub struct Candidate {
    pub id: u64,
    pub score: f32,
    pub category: String,
    pub embedding: Vec<f32>,
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq <= 1e-12 {
        return;
    }
    let inv = norm_sq.sqrt().recip();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Precompute cosine similarity for L2-normalized embeddings. O(N^2 * D).
pub fn build_similarity_matrix(candidates: &[Candidate]) -> Vec<f32> {
    let n = candidates.len();
    let mut sim = vec![0.0f32; n * n];
    for i in 0..n {
        sim[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let s = dot(&candidates[i].embedding, &candidates[j].embedding).max(0.0);
            sim[i * n + j] = s;
            sim[j * n + i] = s;
        }
    }
    sim
}

pub fn mmr_rerank(
    candidates: &[Candidate],
    similarity: &[f32],
    top_k: usize,
    lambda: f32,
) -> Vec<usize> {
    let n = candidates.len();
    if n == 0 || top_k == 0 {
        return Vec::new();
    }
    let top_k = top_k.min(n);
    let lambda = lambda.clamp(0.0, 1.0);
    let diversity = 1.0 - lambda;

    let mut selected = Vec::with_capacity(top_k);
    let mut used = vec![false; n];
    let mut max_sim = vec![0.0f32; n];

    for _ in 0..top_k {
        let mut best: Option<usize> = None;
        let mut best_mmr = f32::NEG_INFINITY;

        for i in 0..n {
            if used[i] {
                continue;
            }
            let mmr = lambda * candidates[i].score - diversity * max_sim[i];
            let take = match best {
                None => true,
                Some(b) => {
                    mmr > best_mmr
                        || (mmr == best_mmr && candidates[i].id < candidates[b].id)
                }
            };
            if take {
                best_mmr = mmr;
                best = Some(i);
            }
        }

        let Some(best) = best else {
            break;
        };
        used[best] = true;
        selected.push(best);

        let row = &similarity[best * n..(best + 1) * n];
        for i in 0..n {
            if used[i] {
                continue;
            }
            if row[i] > max_sim[i] {
                max_sim[i] = row[i];
            }
        }
    }

    selected
}

fn quality_from_score(score: f32, theta: f32) -> f32 {
    (0.5 * theta * score.clamp(0.0, 1.0)).exp()
}

pub fn build_dpp_kernel(candidates: &[Candidate], similarity: &[f32], theta: f32) -> Vec<f32> {
    let n = candidates.len();
    let quality: Vec<f32> = candidates
        .iter()
        .map(|c| quality_from_score(c.score, theta))
        .collect();
    let mut kernel = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let s = if i == j {
                similarity[i * n + j] + 1e-5
            } else {
                similarity[i * n + j]
            };
            kernel[i * n + j] = quality[i] * s * quality[j];
        }
    }
    kernel
}

pub fn dpp_greedy(candidates: &[Candidate], kernel: &[f32], top_k: usize) -> Vec<usize> {
    let n = candidates.len();
    if n == 0 || top_k == 0 {
        return Vec::new();
    }
    let top_k = top_k.min(n);

    let mut selected = Vec::with_capacity(top_k);
    let mut cis = vec![0.0f32; n * top_k];
    let mut d2: Vec<f32> = (0..n)
        .map(|i| kernel[i * n + i].max(0.0))
        .collect();

    for t in 0..top_k {
        let mut best: Option<usize> = None;
        let mut best_gain = f32::NEG_INFINITY;
        for i in 0..n {
            if d2[i] <= 0.0 {
                continue;
            }
            let take = match best {
                None => true,
                Some(b) => {
                    d2[i] > best_gain
                        || (d2[i] == best_gain && candidates[i].id < candidates[b].id)
                }
            };
            if take {
                best_gain = d2[i];
                best = Some(i);
            }
        }
        let Some(best) = best else {
            break;
        };

        selected.push(best);
        let sqrt_d = d2[best].max(1e-12).sqrt();

        for j in 0..n {
            let mut dot = 0.0f32;
            for p in 0..t {
                dot += cis[best * top_k + p] * cis[j * top_k + p];
            }
            let e = (kernel[best * n + j] - dot) / sqrt_d;
            cis[j * top_k + t] = e;
            if j == best {
                continue;
            }
            d2[j] -= e * e;
            if d2[j] < 1e-12 {
                d2[j] = 0.0;
            }
        }
        d2[best] = 0.0;
    }

    selected
}

pub fn dpp_rerank(
    candidates: &[Candidate],
    similarity: &[f32],
    top_k: usize,
    theta: f32,
) -> Vec<usize> {
    let kernel = build_dpp_kernel(candidates, similarity, theta);
    dpp_greedy(candidates, &kernel, top_k)
}

pub fn load_candidates_csv(path: &str) -> Result<Vec<Candidate>, String> {
    let file = File::open(path).map_err(|e| format!("open fixture: {e}"))?;
    let mut lines = BufReader::new(file).lines();
    let _header = lines
        .next()
        .ok_or_else(|| "empty fixture".to_string())?
        .map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    for line in lines {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let id = parts
            .next()
            .ok_or("missing id")?
            .parse::<u64>()
            .map_err(|e| e.to_string())?;
        let score = parts
            .next()
            .ok_or("missing score")?
            .parse::<f32>()
            .map_err(|e| e.to_string())?;
        let category = parts.next().ok_or("missing category")?.to_string();
        let mut embedding = Vec::with_capacity(EMBEDDING_DIM);
        for _ in 0..EMBEDDING_DIM {
            let v = parts
                .next()
                .ok_or("missing embedding")?
                .parse::<f32>()
                .map_err(|e| e.to_string())?;
            embedding.push(v);
        }
        l2_normalize(&mut embedding);
        out.push(Candidate {
            id,
            score,
            category,
            embedding,
        });
    }
    Ok(out)
}

pub fn unique_category_count(candidates: &[Candidate], indices: &[usize]) -> usize {
    let mut cats: Vec<&str> = indices
        .iter()
        .map(|&i| candidates[i].category.as_str())
        .collect();
    cats.sort_unstable();
    cats.dedup();
    cats.len()
}

pub fn score_topk(candidates: &[Candidate], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| {
        candidates[b]
            .score
            .partial_cmp(&candidates[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| candidates[a].id.cmp(&candidates[b].id))
    });
    order.truncate(k.min(candidates.len()));
    order
}
