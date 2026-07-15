mod rust_kernels;

use rust_kernels::{dot_scalar, dot_simd, merge_filter_topk, AlignedBuffer, Candidate, Workspace};

#[test]
fn merge_filter_topk_dedups_filters_and_orders() {
    let candidates = vec![
        Candidate {
            id: 4,
            score: 0.40,
            source_mask: 0b0001,
        },
        Candidate {
            id: 2,
            score: 0.90,
            source_mask: 0b0010,
        },
        Candidate {
            id: 4,
            score: 0.80,
            source_mask: 0b0100,
        },
        Candidate {
            id: 7,
            score: 0.70,
            source_mask: 0b1000,
        },
    ];
    let seen_ids = vec![2u32];

    let result = merge_filter_topk(&candidates, &seen_ids, 8, 2);

    assert_eq!(result.len(), 2);
    assert_eq!(result[0].id, 4);
    assert_eq!(result[0].source_mask, 0b0101);
    assert_eq!(result[1].id, 7);
}

#[test]
fn workspace_uses_64_byte_aligned_hot_arrays() {
    let workspace = Workspace::new(128);

    assert_eq!(workspace.seen_alignment() % 64, 0);
    assert_eq!(workspace.candidate_alignment() % 64, 0);
    assert_eq!(workspace.score_alignment() % 64, 0);
    assert_eq!(workspace.source_alignment() % 64, 0);
}

#[test]
fn aligned_buffer_exposes_aligned_mutable_slice() {
    let mut buffer = AlignedBuffer::<f32>::new_zeroed(16);
    assert_eq!(buffer.ptr_alignment() % 64, 0);

    buffer.as_mut_slice()[3] = 2.5;

    assert_eq!(buffer.as_slice()[3], 2.5);
}

#[test]
fn simd_dot_matches_scalar_dot() {
    let left: Vec<f32> = (0..384).map(|idx| idx as f32 * 0.001).collect();
    let right: Vec<f32> = (0..384).map(|idx| 1.0 - idx as f32 * 0.0005).collect();

    let scalar = dot_scalar(&left, &right);
    let simd = dot_simd(&left, &right);

    assert!((scalar - simd).abs() < 0.001);
}
