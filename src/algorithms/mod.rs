//! Low-level reusable recommendation algorithms.

mod aligned;
mod candidate_merge;
mod topk;
mod vector;

#[cfg(test)]
pub(crate) use aligned::AlignedBuffer;
#[cfg(test)]
pub(crate) use candidate_merge::CandidateMergeWorkspace;
pub(crate) use candidate_merge::{merge_filter_topk, ScoredCandidate};
pub(crate) use topk::partial_topk_by;
pub(crate) use vector::cosine_similarity_simd;
#[cfg(test)]
pub(crate) use vector::{cosine_similarity_scalar, dot_scalar, dot_simd};

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    fn score_desc(left: &ScoredCandidate, right: &ScoredCandidate) -> Ordering {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.item_id.cmp(&right.item_id))
    }

    #[test]
    fn aligned_buffer_allocates_zeroed_cache_line_aligned_storage() {
        let mut buffer = AlignedBuffer::<u32>::new_zeroed(16);

        assert_eq!(buffer.as_slice(), &[0; 16]);
        assert_eq!(buffer.ptr_alignment() % 64, 0);

        buffer.as_mut_slice()[3] = 7;
        assert_eq!(buffer.as_slice()[3], 7);
    }

    #[test]
    fn partial_topk_preserves_deterministic_top_order() {
        let mut items = vec![
            ScoredCandidate::new(30, 0.7, 1),
            ScoredCandidate::new(10, 0.9, 1),
            ScoredCandidate::new(20, 0.9, 1),
            ScoredCandidate::new(40, 0.1, 1),
        ];

        partial_topk_by(&mut items, 3, score_desc);

        assert_eq!(
            items,
            vec![
                ScoredCandidate::new(10, 0.9, 1),
                ScoredCandidate::new(20, 0.9, 1),
                ScoredCandidate::new(30, 0.7, 1),
            ]
        );
    }

    #[test]
    fn candidate_merge_workspace_filters_dedups_and_merges_sources() {
        let mut workspace = CandidateMergeWorkspace::new(100);
        let hits = vec![
            ScoredCandidate::new(10, 0.4, 0b0001),
            ScoredCandidate::new(20, 0.8, 0b0010),
            ScoredCandidate::new(10, 0.9, 0b0100),
            ScoredCandidate::new(30, 0.6, 0b1000),
        ];

        let merged = workspace.merge_filter_topk(&hits, &[30], 10);

        assert_eq!(
            merged,
            vec![
                ScoredCandidate::new(10, 0.9, 0b0101),
                ScoredCandidate::new(20, 0.8, 0b0010),
            ]
        );
    }

    #[test]
    fn candidate_merge_workspace_reuses_generations_between_calls() {
        let mut workspace = CandidateMergeWorkspace::new(100);

        let first = workspace.merge_filter_topk(&[ScoredCandidate::new(10, 0.4, 0b0001)], &[], 10);
        let second = workspace.merge_filter_topk(&[ScoredCandidate::new(20, 0.5, 0b0010)], &[], 10);

        assert_eq!(first, vec![ScoredCandidate::new(10, 0.4, 0b0001)]);
        assert_eq!(second, vec![ScoredCandidate::new(20, 0.5, 0b0010)]);
    }

    #[test]
    fn candidate_merge_falls_back_for_sparse_item_ids() {
        let hits = vec![
            ScoredCandidate::new(u64::MAX - 1, 0.4, 0b0001),
            ScoredCandidate::new(u64::MAX - 1, 0.7, 0b0010),
            ScoredCandidate::new(u64::MAX, 0.9, 0b0100),
        ];

        let merged = merge_filter_topk(&hits, &[u64::MAX], 10);

        assert_eq!(
            merged,
            vec![ScoredCandidate::new(u64::MAX - 1, 0.7, 0b0011)]
        );
    }

    #[test]
    fn simd_dot_and_cosine_match_scalar_results() {
        let left = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let right = vec![0.5, 1.5, -1.0, 2.0, 0.25, -0.5, 3.0, 1.0, 0.75];

        let scalar_dot = dot_scalar(&left, &right);
        let simd_dot = dot_simd(&left, &right);
        assert!((scalar_dot - simd_dot).abs() < 0.0001);

        let scalar_cosine = cosine_similarity_scalar(&left, &right);
        let simd_cosine = cosine_similarity_simd(&left, &right);
        assert!((scalar_cosine - simd_cosine).abs() < 0.0001);
    }
}
