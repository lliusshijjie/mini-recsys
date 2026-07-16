//! Deterministic partial top-k helpers.

use std::cmp::Ordering;

pub(crate) fn partial_topk_by<T>(
    items: &mut Vec<T>,
    k: usize,
    compare: impl Fn(&T, &T) -> Ordering + Copy,
) {
    if k == 0 {
        items.clear();
        return;
    }

    if items.len() > k {
        items.select_nth_unstable_by(k, compare);
        items.truncate(k);
    }
    items.sort_unstable_by(compare);
}
