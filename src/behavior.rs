//! Behavior events and lightweight user preference updates.

use crate::model::{Item, UserProfile};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

pub const RECENT_EVENT_LIMIT: usize = 100;
const MIN_WEIGHT: f32 = -1.0;
const MAX_WEIGHT: f32 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    Impression,
    Click,
    Like,
    Dismiss,
    Purchase,
}

impl EventType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Impression => "impression",
            Self::Click => "click",
            Self::Like => "like",
            Self::Dismiss => "dismiss",
            Self::Purchase => "purchase",
        }
    }

    fn preference_delta(self) -> (f32, f32) {
        match self {
            Self::Impression => (0.0, 0.0),
            Self::Click => (0.15, 0.20),
            Self::Like => (0.30, 0.40),
            Self::Dismiss => (-0.25, -0.50),
            Self::Purchase => (0.45, 0.0),
        }
    }
}

impl FromStr for EventType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "impression" => Ok(Self::Impression),
            "click" => Ok(Self::Click),
            "like" => Ok(Self::Like),
            "dismiss" => Ok(Self::Dismiss),
            "purchase" => Ok(Self::Purchase),
            _ => Err(format!("Unsupported event type: {}", value)),
        }
    }
}

impl fmt::Display for EventType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorEvent {
    pub uid: u64,
    pub item_id: u64,
    pub event_type: EventType,
    pub category: String,
    pub timestamp_ms: u128,
}

impl BehaviorEvent {
    pub fn new(uid: u64, item_id: u64, event_type: EventType, category: impl Into<String>) -> Self {
        Self {
            uid,
            item_id,
            event_type,
            category: category.into(),
            timestamp_ms: now_ms(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecentEvents {
    pub items: Vec<BehaviorEvent>,
}

impl RecentEvents {
    pub fn push(&mut self, event: BehaviorEvent) {
        self.items.push(event);
        if self.items.len() > RECENT_EVENT_LIMIT {
            let extra = self.items.len() - RECENT_EVENT_LIMIT;
            self.items.drain(0..extra);
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserPreferences {
    pub category_weights: HashMap<String, f32>,
    pub item_weights: HashMap<u64, f32>,
}

impl UserPreferences {
    pub fn from_profile(profile: &UserProfile) -> Self {
        let mut preferences = Self::default();
        for (category, weight) in &profile.category_weights {
            preferences.set_category_weight(category, weight * 0.6);
        }
        preferences
    }

    pub fn apply_event(&mut self, event_type: EventType, item: &Item) {
        let (category_delta, item_delta) = event_type.preference_delta();
        if category_delta != 0.0 {
            self.add_category_weight(&item.category, category_delta);
        }
        if item_delta != 0.0 {
            self.add_item_weight(item.id, item_delta);
        }
    }

    pub fn category_weight(&self, category: &str) -> f32 {
        self.category_weights
            .get(category)
            .copied()
            .unwrap_or_default()
    }

    pub fn item_weight(&self, item_id: u64) -> f32 {
        self.item_weights.get(&item_id).copied().unwrap_or_default()
    }

    pub fn set_category_weight(&mut self, category: impl Into<String>, weight: f32) {
        self.category_weights
            .insert(category.into(), clamp_weight(weight));
    }

    pub fn set_item_weight(&mut self, item_id: u64, weight: f32) {
        self.item_weights.insert(item_id, clamp_weight(weight));
    }

    fn add_category_weight(&mut self, category: &str, delta: f32) {
        let next = self.category_weight(category) + delta;
        self.set_category_weight(category, next);
    }

    fn add_item_weight(&mut self, item_id: u64, delta: f32) {
        let next = self.item_weight(item_id) + delta;
        self.set_item_weight(item_id, next);
    }
}

fn clamp_weight(weight: f32) -> f32 {
    weight.clamp(MIN_WEIGHT, MAX_WEIGHT)
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
    use crate::model::Item;
    use std::str::FromStr;

    fn item(id: u64, category: &str) -> Item {
        Item {
            id,
            name: format!("Item {}", id),
            category: category.to_string(),
            image_url: String::new(),
            price: 10.0,
            embedding: Vec::new(),
            popularity: 0.5,
        }
    }

    #[test]
    fn event_type_parses_supported_values_only() {
        assert_eq!(
            EventType::from_str("impression").unwrap(),
            EventType::Impression
        );
        assert_eq!(EventType::from_str("click").unwrap(), EventType::Click);
        assert_eq!(EventType::from_str("like").unwrap(), EventType::Like);
        assert_eq!(EventType::from_str("dismiss").unwrap(), EventType::Dismiss);
        assert_eq!(
            EventType::from_str("purchase").unwrap(),
            EventType::Purchase
        );
    }

    #[test]
    fn positive_events_increase_category_and_item_preference() {
        let mut preferences = UserPreferences::default();
        let item = item(7, "Books");

        preferences.apply_event(EventType::Click, &item);
        preferences.apply_event(EventType::Like, &item);

        assert!(preferences.category_weight("Books") > 0.0);
        assert!(preferences.item_weight(7) > preferences.category_weight("Books"));
    }

    #[test]
    fn dismiss_decreases_category_and_item_preference() {
        let mut preferences = UserPreferences::default();
        let item = item(3, "Electronics");

        preferences.apply_event(EventType::Dismiss, &item);

        assert!(preferences.category_weight("Electronics") < 0.0);
        assert!(preferences.item_weight(3) < 0.0);
    }

    #[test]
    fn purchase_increases_category_preference_without_item_boost() {
        let mut preferences = UserPreferences::default();
        let item = item(9, "Books");

        preferences.apply_event(EventType::Purchase, &item);

        assert!(preferences.category_weight("Books") > 0.0);
        assert_eq!(preferences.item_weight(9), 0.0);
    }

    #[test]
    fn preference_weights_are_clamped() {
        let mut preferences = UserPreferences::default();
        let item = item(1, "Home");

        for _ in 0..20 {
            preferences.apply_event(EventType::Like, &item);
        }
        assert_eq!(preferences.category_weight("Home"), 1.0);
        assert_eq!(preferences.item_weight(1), 1.0);

        for _ in 0..40 {
            preferences.apply_event(EventType::Dismiss, &item);
        }
        assert_eq!(preferences.category_weight("Home"), -1.0);
        assert_eq!(preferences.item_weight(1), -1.0);
    }

    #[test]
    fn recent_events_keep_latest_entries_in_order() {
        let mut events = RecentEvents::default();

        for item_id in 1..=105 {
            events.push(BehaviorEvent::new(
                1,
                item_id,
                EventType::Impression,
                "Books",
            ));
        }

        assert_eq!(events.items.len(), 100);
        assert_eq!(events.items.first().unwrap().item_id, 6);
        assert_eq!(events.items.last().unwrap().item_id, 105);
    }
}
