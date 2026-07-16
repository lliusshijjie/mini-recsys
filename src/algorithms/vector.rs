//! Vector math kernels with scalar fallbacks.

pub(crate) fn dot_scalar(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

pub(crate) fn dot_simd(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { dot_avx(left, right) };
        }
    }

    dot_scalar(left, right)
}

#[cfg(test)]
pub(crate) fn cosine_similarity_scalar(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || left.len() != right.len() {
        return 0.0;
    }

    let left_norm = dot_scalar(left, left).sqrt();
    let right_norm = dot_scalar(right, right).sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }

    dot_scalar(left, right) / (left_norm * right_norm)
}

pub(crate) fn cosine_similarity_simd(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || left.len() != right.len() {
        return 0.0;
    }

    let left_norm = dot_simd(left, left).sqrt();
    let right_norm = dot_simd(right, right).sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }

    dot_simd(left, right) / (left_norm * right_norm)
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
        let left_values = _mm256_loadu_ps(left.as_ptr().add(index));
        let right_values = _mm256_loadu_ps(right.as_ptr().add(index));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(left_values, right_values));
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
