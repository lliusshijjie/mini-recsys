//! Request-level exposure policy for previously interacted items.

use crate::behavior::{BehaviorEvent, EventType};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

const HOUR_MS: u128 = 60 * 60 * 1000;
const DAY_MS: u128 = 24 * HOUR_MS;
const IMPRESSION_WINDOW_MS: u128 = DAY_MS;
const DISMISS_SUPPRESS_WINDOW_MS: u128 = 7 * DAY_MS;
const PURCHASE_SUPPRESS_WINDOW_MS: u128 = 30 * DAY_MS;
const IMPRESSION_BASE_DEBOOST: f32 = 0.06;
const IMPRESSION_REPEAT_DEBOOST: f32 = 0.03;
const IMPRESSION_MAX_DEBOOST: f32 = 0.18;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum ExposureDecision {
    Allow,
    Deboost(f32),
    Suppress,
}

#[derive(Debug, Clone, Copy, Default)]
struct ExposureState {
    impression_count: usize,
    suppress: bool,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ExposurePolicy {
    by_item: HashMap<u64, ExposureState>,
}

impl ExposurePolicy {
    pub(super) fn from_events(events: &[BehaviorEvent]) -> Self {
        Self::from_events_at(events, now_ms())
    }

    pub(super) fn from_events_at(events: &[BehaviorEvent], now_ms: u128) -> Self {
        let mut by_item: HashMap<u64, ExposureState> = HashMap::new();
        for event in events {
            let age_ms = now_ms.saturating_sub(event.timestamp_ms);
            let state = by_item.entry(event.item_id).or_default();
            match event.event_type {
                EventType::Impression if age_ms <= IMPRESSION_WINDOW_MS => {
                    state.impression_count += 1;
                }
                EventType::Dismiss if age_ms <= DISMISS_SUPPRESS_WINDOW_MS => {
                    state.suppress = true;
                }
                EventType::Purchase if age_ms <= PURCHASE_SUPPRESS_WINDOW_MS => {
                    state.suppress = true;
                }
                EventType::Click
                | EventType::Like
                | EventType::Impression
                | EventType::Dismiss
                | EventType::Purchase => {}
            }
        }

        Self { by_item }
    }

    pub(super) fn decision(&self, item_id: u64) -> ExposureDecision {
        let Some(state) = self.by_item.get(&item_id) else {
            return ExposureDecision::Allow;
        };
        if state.suppress {
            return ExposureDecision::Suppress;
        }
        if state.impression_count == 0 {
            return ExposureDecision::Allow;
        }

        let deboost = if state.impression_count >= 5 {
            IMPRESSION_MAX_DEBOOST
        } else {
            let repeat_count = state.impression_count.saturating_sub(1) as f32;
            IMPRESSION_BASE_DEBOOST + IMPRESSION_REPEAT_DEBOOST * repeat_count
        };
        ExposureDecision::Deboost(deboost)
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::{BehaviorEvent, EventType};

    const NOW_MS: u128 = 4_000_000_000;
    const HOUR_MS: u128 = 60 * 60 * 1000;
    const DAY_MS: u128 = 24 * HOUR_MS;

    fn event(item_id: u64, event_type: EventType, age_ms: u128) -> BehaviorEvent {
        let mut event = BehaviorEvent::new(1, item_id, event_type, "Books");
        event.timestamp_ms = NOW_MS - age_ms;
        event
    }

    #[test]
    fn single_recent_impression_deboosts_without_suppressing() {
        let policy =
            ExposurePolicy::from_events_at(&[event(10, EventType::Impression, HOUR_MS)], NOW_MS);

        assert_eq!(policy.decision(10), ExposureDecision::Deboost(0.06));
    }

    #[test]
    fn repeated_impressions_cap_the_deboost() {
        let policy = ExposurePolicy::from_events_at(
            &[
                event(10, EventType::Impression, HOUR_MS),
                event(10, EventType::Impression, 2 * HOUR_MS),
                event(10, EventType::Impression, 3 * HOUR_MS),
                event(10, EventType::Impression, 4 * HOUR_MS),
                event(10, EventType::Impression, 5 * HOUR_MS),
            ],
            NOW_MS,
        );

        assert_eq!(policy.decision(10), ExposureDecision::Deboost(0.18));
    }

    #[test]
    fn stale_impressions_do_not_affect_candidates() {
        let policy =
            ExposurePolicy::from_events_at(&[event(10, EventType::Impression, DAY_MS + 1)], NOW_MS);

        assert_eq!(policy.decision(10), ExposureDecision::Allow);
    }

    #[test]
    fn dismiss_suppresses_for_seven_days() {
        let active =
            ExposurePolicy::from_events_at(&[event(10, EventType::Dismiss, 6 * DAY_MS)], NOW_MS);
        let expired =
            ExposurePolicy::from_events_at(&[event(10, EventType::Dismiss, 8 * DAY_MS)], NOW_MS);

        assert_eq!(active.decision(10), ExposureDecision::Suppress);
        assert_eq!(expired.decision(10), ExposureDecision::Allow);
    }

    #[test]
    fn purchase_suppresses_for_thirty_days() {
        let active =
            ExposurePolicy::from_events_at(&[event(10, EventType::Purchase, 29 * DAY_MS)], NOW_MS);
        let expired =
            ExposurePolicy::from_events_at(&[event(10, EventType::Purchase, 31 * DAY_MS)], NOW_MS);

        assert_eq!(active.decision(10), ExposureDecision::Suppress);
        assert_eq!(expired.decision(10), ExposureDecision::Allow);
    }

    #[test]
    fn click_and_like_do_not_suppress_or_deboost() {
        let policy = ExposurePolicy::from_events_at(
            &[
                event(10, EventType::Click, HOUR_MS),
                event(11, EventType::Like, HOUR_MS),
            ],
            NOW_MS,
        );

        assert_eq!(policy.decision(10), ExposureDecision::Allow);
        assert_eq!(policy.decision(11), ExposureDecision::Allow);
    }
}
