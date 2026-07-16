//! Candidate filtering, deduplication, source merging, and top-k selection.

use crate::algorithms::aligned::AlignedBuffer;
use crate::algorithms::topk::partial_topk_by;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const MAX_DENSE_ITEM_ID: u64 = 5_000_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScoredCandidate {
    pub(crate) item_id: u64,
    pub(crate) score: f32,
    pub(crate) source_mask: u8,
}

impl ScoredCandidate {
    pub(crate) fn new(item_id: u64, score: f32, source_mask: u8) -> Self {
        Self {
            item_id,
            score,
            source_mask,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CandidateMergeWorkspace {
    max_item_id: usize,
    seen_generation: AlignedBuffer<u32>,
    candidate_generation: AlignedBuffer<u32>,
    scores: AlignedBuffer<f32>,
    source_masks: AlignedBuffer<u8>,
    touched: Vec<u64>,
    generation: u32,
}

impl CandidateMergeWorkspace {
    pub(crate) fn new(max_item_id: u64) -> Self {
        let max_item_id = usize::try_from(max_item_id).expect("max item id fits usize");
        let len = max_item_id
            .checked_add(1)
            .expect("candidate workspace length overflow");
        Self {
            max_item_id,
            seen_generation: AlignedBuffer::new_zeroed(len),
            candidate_generation: AlignedBuffer::new_zeroed(len),
            scores: AlignedBuffer::new_zeroed(len),
            source_masks: AlignedBuffer::new_zeroed(len),
            touched: Vec::new(),
            generation: 0,
        }
    }

    pub(crate) fn merge_filter_topk(
        &mut self,
        candidates: &[ScoredCandidate],
        seen_item_ids: &[u64],
        k: usize,
    ) -> Vec<ScoredCandidate> {
        if k == 0 {
            return Vec::new();
        }

        self.next_generation();
        let generation = self.generation;
        self.touched.clear();

        for &item_id in seen_item_ids {
            let Ok(index) = usize::try_from(item_id) else {
                continue;
            };
            if index <= self.max_item_id {
                self.seen_generation[index] = generation;
            }
        }

        for candidate in candidates {
            let Ok(index) = usize::try_from(candidate.item_id) else {
                continue;
            };
            if index > self.max_item_id || self.seen_generation[index] == generation {
                continue;
            }

            if self.candidate_generation[index] != generation {
                self.candidate_generation[index] = generation;
                self.scores[index] = candidate.score;
                self.source_masks[index] = candidate.source_mask;
                self.touched.push(candidate.item_id);
            } else {
                self.source_masks[index] |= candidate.source_mask;
                if candidate.score > self.scores[index] {
                    self.scores[index] = candidate.score;
                }
            }
        }

        let mut output: Vec<ScoredCandidate> = self
            .touched
            .iter()
            .map(|&item_id| {
                let index = item_id as usize;
                ScoredCandidate::new(item_id, self.scores[index], self.source_masks[index])
            })
            .collect();
        partial_topk_by(&mut output, k, scored_candidate_desc);
        output
    }

    fn next_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.seen_generation.as_mut_slice().fill(0);
            self.candidate_generation.as_mut_slice().fill(0);
            self.generation = 1;
        }
    }
}

pub(crate) fn merge_filter_topk(
    candidates: &[ScoredCandidate],
    seen_item_ids: &[u64],
    k: usize,
) -> Vec<ScoredCandidate> {
    if k == 0 {
        return Vec::new();
    }

    let max_candidate_id = candidates
        .iter()
        .map(|candidate| candidate.item_id)
        .max()
        .unwrap_or(0);
    let max_seen_id = seen_item_ids.iter().copied().max().unwrap_or(0);
    let max_item_id = max_candidate_id.max(max_seen_id);
    if max_item_id <= MAX_DENSE_ITEM_ID {
        return CandidateMergeWorkspace::new(max_item_id).merge_filter_topk(
            candidates,
            seen_item_ids,
            k,
        );
    }

    merge_filter_topk_hash(candidates, seen_item_ids, k)
}

fn merge_filter_topk_hash(
    candidates: &[ScoredCandidate],
    seen_item_ids: &[u64],
    k: usize,
) -> Vec<ScoredCandidate> {
    let seen: HashSet<u64> = seen_item_ids.iter().copied().collect();
    let mut merged: HashMap<u64, ScoredCandidate> = HashMap::new();

    for candidate in candidates {
        if seen.contains(&candidate.item_id) {
            continue;
        }

        merged
            .entry(candidate.item_id)
            .and_modify(|existing| {
                existing.source_mask |= candidate.source_mask;
                if candidate.score > existing.score {
                    existing.score = candidate.score;
                }
            })
            .or_insert(*candidate);
    }

    let mut output: Vec<ScoredCandidate> = merged.into_values().collect();
    partial_topk_by(&mut output, k, scored_candidate_desc);
    output
}

fn scored_candidate_desc(left: &ScoredCandidate, right: &ScoredCandidate) -> Ordering {
    right
        .score
        .partial_cmp(&left.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.item_id.cmp(&right.item_id))
}
