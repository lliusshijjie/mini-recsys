#[path = "rust_rerank.rs"]
mod rust_rerank;

use rust_rerank::{
    build_similarity_matrix, dpp_rerank, load_candidates_csv, mmr_rerank, score_topk,
    unique_category_count, Candidate, EMBEDDING_DIM,
};

fn toy_candidates() -> Vec<Candidate> {
    // Two near-duplicate "Books" vectors and one orthogonal "Sports" vector.
    let mut books_a = vec![0.0f32; EMBEDDING_DIM];
    books_a[0] = 1.0;
    let mut books_b = vec![0.0f32; EMBEDDING_DIM];
    books_b[0] = 0.98;
    books_b[1] = 0.2;
    let mut sports = vec![0.0f32; EMBEDDING_DIM];
    sports[8] = 1.0;

    let mut items = vec![
        Candidate {
            id: 1,
            score: 0.99,
            category: "Books".into(),
            embedding: books_a,
        },
        Candidate {
            id: 2,
            score: 0.98,
            category: "Books".into(),
            embedding: books_b,
        },
        Candidate {
            id: 3,
            score: 0.90,
            category: "Sports".into(),
            embedding: sports,
        },
    ];
    for item in &mut items {
        rust_rerank::l2_normalize(&mut item.embedding);
    }
    items
}

#[test]
fn mmr_prefers_diverse_second_item() {
    let candidates = toy_candidates();
    let sim = build_similarity_matrix(&candidates);
    let selected = mmr_rerank(&candidates, &sim, 2, 0.5);
    assert_eq!(selected.len(), 2);
    assert_eq!(candidates[selected[0]].id, 1);
    assert_eq!(
        candidates[selected[1]].id, 3,
        "second pick should diversify away from near-duplicate Books"
    );
}

#[test]
fn mmr_lambda_one_keeps_score_order() {
    let candidates = toy_candidates();
    let sim = build_similarity_matrix(&candidates);
    let selected = mmr_rerank(&candidates, &sim, 3, 1.0);
    let ids: Vec<u64> = selected.iter().map(|&i| candidates[i].id).collect();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[test]
fn dpp_also_breaks_near_duplicates() {
    let candidates = toy_candidates();
    let sim = build_similarity_matrix(&candidates);
    let selected = dpp_rerank(&candidates, &sim, 2, 1.0);
    assert_eq!(selected.len(), 2);
    assert_eq!(candidates[selected[0]].id, 1);
    assert_eq!(candidates[selected[1]].id, 3);
}

#[test]
fn fixture_from_products_improves_category_coverage() {
    let path = "experiments/rerank_diversity/data/candidates.csv";
    let candidates = match load_candidates_csv(path) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip fixture test: {err}");
            return;
        }
    };
    assert!(candidates.len() >= 32);

    let sim = build_similarity_matrix(&candidates);
    let k = 10;
    let baseline = score_topk(&candidates, k);
    let mmr = mmr_rerank(&candidates, &sim, k, 0.7);
    let dpp = dpp_rerank(&candidates, &sim, k, 1.0);

    let base_cats = unique_category_count(&candidates, &baseline);
    let mmr_cats = unique_category_count(&candidates, &mmr);
    let dpp_cats = unique_category_count(&candidates, &dpp);

    assert_eq!(mmr.len(), k);
    assert_eq!(dpp.len(), k);
    assert!(
        mmr_cats >= base_cats,
        "MMR category coverage {} should be >= score-sort {}",
        mmr_cats,
        base_cats
    );
    assert!(
        dpp_cats >= base_cats,
        "DPP category coverage {} should be >= score-sort {}",
        dpp_cats,
        base_cats
    );

    // First item should remain the top relevance item for moderate diversity.
    assert_eq!(candidates[mmr[0]].id, candidates[baseline[0]].id);
    assert_eq!(candidates[dpp[0]].id, candidates[baseline[0]].id);
}
