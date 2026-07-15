use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

const CACHE_LINE_ALIGNMENT: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub id: u32,
    pub score: f32,
    pub source_mask: u8,
}

pub struct AlignedBuffer<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
    _marker: PhantomData<T>,
}

impl<T: Copy> AlignedBuffer<T> {
    pub fn new_zeroed(len: usize) -> Self {
        let element_size = mem::size_of::<T>();
        let size = len
            .checked_mul(element_size)
            .expect("aligned buffer size overflow");
        let layout = Layout::from_size_align(size.max(1), CACHE_LINE_ALIGNMENT)
            .expect("valid aligned buffer layout");
        let raw_ptr = unsafe { alloc_zeroed(layout) } as *mut T;
        let ptr = NonNull::new(raw_ptr).unwrap_or_else(|| handle_alloc_error(layout));
        Self {
            ptr,
            len,
            layout,
            _marker: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[allow(dead_code)]
    pub fn ptr_alignment(&self) -> usize {
        self.ptr.as_ptr() as usize
    }
}

impl<T: Copy> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

impl<T: Copy> Deref for AlignedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Copy> DerefMut for AlignedBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

pub struct Workspace {
    max_id: usize,
    seen_generation: AlignedBuffer<u32>,
    candidate_generation: AlignedBuffer<u32>,
    scores: AlignedBuffer<f32>,
    source_masks: AlignedBuffer<u8>,
    touched: Vec<u32>,
    generation: u32,
}

impl Workspace {
    pub fn new(max_id: usize) -> Self {
        let len = max_id + 1;
        Self {
            max_id,
            seen_generation: AlignedBuffer::new_zeroed(len),
            candidate_generation: AlignedBuffer::new_zeroed(len),
            scores: AlignedBuffer::new_zeroed(len),
            source_masks: AlignedBuffer::new_zeroed(len),
            touched: Vec::new(),
            generation: 0,
        }
    }

    pub fn merge_filter_topk(
        &mut self,
        candidates: &[Candidate],
        seen_ids: &[u32],
        k: usize,
    ) -> Vec<Candidate> {
        if k == 0 {
            return Vec::new();
        }

        self.next_generation();
        let gen = self.generation;
        self.touched.clear();

        for &id in seen_ids {
            let index = id as usize;
            if index <= self.max_id {
                self.seen_generation[index] = gen;
            }
        }

        for candidate in candidates {
            let index = candidate.id as usize;
            if index > self.max_id || self.seen_generation[index] == gen {
                continue;
            }

            if self.candidate_generation[index] != gen {
                self.candidate_generation[index] = gen;
                self.scores[index] = candidate.score;
                self.source_masks[index] = candidate.source_mask;
                self.touched.push(candidate.id);
            } else {
                self.source_masks[index] |= candidate.source_mask;
                if candidate.score > self.scores[index] {
                    self.scores[index] = candidate.score;
                }
            }
        }

        let mut output: Vec<Candidate> = self
            .touched
            .iter()
            .map(|&id| {
                let index = id as usize;
                Candidate {
                    id,
                    score: self.scores[index],
                    source_mask: self.source_masks[index],
                }
            })
            .collect();

        topk_desc(&mut output, k);
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

    #[allow(dead_code)]
    pub fn seen_alignment(&self) -> usize {
        self.seen_generation.ptr_alignment()
    }

    #[allow(dead_code)]
    pub fn candidate_alignment(&self) -> usize {
        self.candidate_generation.ptr_alignment()
    }

    #[allow(dead_code)]
    pub fn score_alignment(&self) -> usize {
        self.scores.ptr_alignment()
    }

    #[allow(dead_code)]
    pub fn source_alignment(&self) -> usize {
        self.source_masks.ptr_alignment()
    }
}

#[allow(dead_code)]
pub fn merge_filter_topk(
    candidates: &[Candidate],
    seen_ids: &[u32],
    max_id: usize,
    k: usize,
) -> Vec<Candidate> {
    let mut workspace = Workspace::new(max_id);
    workspace.merge_filter_topk(candidates, seen_ids, k)
}

pub fn dot_scalar(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

pub fn dot_simd(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { dot_avx(left, right) };
        }
    }

    dot_scalar(left, right)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn dot_avx(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut index = 0usize;
    let len = left.len();
    let mut sum = _mm256_setzero_ps();

    while index + 8 <= len {
        let l = _mm256_loadu_ps(left.as_ptr().add(index));
        let r = _mm256_loadu_ps(right.as_ptr().add(index));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(l, r));
        index += 8;
    }

    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    let mut total: f32 = lanes.iter().sum();

    while index < len {
        total += *left.get_unchecked(index) * *right.get_unchecked(index);
        index += 1;
    }

    total
}

fn topk_desc(items: &mut Vec<Candidate>, k: usize) {
    fn cmp(left: &Candidate, right: &Candidate) -> std::cmp::Ordering {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    }

    if items.len() > k {
        items.select_nth_unstable_by(k, cmp);
        items.truncate(k);
    }
    items.sort_unstable_by(cmp);
}

#[allow(dead_code)]
pub fn naive_merge_filter_topk(
    candidates: &[Candidate],
    seen_ids: &[u32],
    max_id: usize,
    k: usize,
) -> Vec<Candidate> {
    use std::collections::{HashMap, HashSet};

    let seen: HashSet<u32> = seen_ids.iter().copied().collect();
    let mut merged: HashMap<u32, Candidate> = HashMap::new();
    for candidate in candidates {
        if candidate.id as usize > max_id || seen.contains(&candidate.id) {
            continue;
        }
        merged
            .entry(candidate.id)
            .and_modify(|existing| {
                existing.source_mask |= candidate.source_mask;
                if candidate.score > existing.score {
                    existing.score = candidate.score;
                }
            })
            .or_insert(*candidate);
    }

    let mut output: Vec<Candidate> = merged.into_values().collect();
    output.sort_unstable_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    });
    output.truncate(k);
    output
}
