//! Storage layer: Sled embedded database wrapper.

use crate::behavior::{BehaviorEvent, RecentEvents, UserPreferences};
use crate::model::{Item, User};
use anyhow::{Context, Result};
use fastbloom_rs::{BloomFilter, FilterBuilder};
use sled::{Db, Tree};

/// Bloom filter parameters
const BLOOM_EXPECTED_ITEMS: u32 = 10000;
const BLOOM_FPR: f64 = 0.01;
/// Fixed hash count (derived from expected_items and fpr: k = -ln(fpr) / ln(2) ≈ 7).
const BLOOM_HASHES: u32 = 7;

pub struct Storage {
    _db: Db,
    users_tree: Tree,
    items_tree: Tree,
    history_tree: Tree,
    events_tree: Tree,
    preferences_tree: Tree,
}

impl Storage {
    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path).context("Failed to open sled database")?;
        let users_tree = db.open_tree("users").context("Failed to open users tree")?;
        let items_tree = db.open_tree("items").context("Failed to open items tree")?;
        let history_tree = db
            .open_tree("history")
            .context("Failed to open history tree")?;
        let events_tree = db
            .open_tree("events")
            .context("Failed to open events tree")?;
        let preferences_tree = db
            .open_tree("preferences")
            .context("Failed to open preferences tree")?;

        Ok(Self {
            _db: db,
            users_tree,
            items_tree,
            history_tree,
            events_tree,
            preferences_tree,
        })
    }

    fn u64_to_key(id: u64) -> [u8; 8] {
        id.to_be_bytes()
    }

    // ========== User CRUD ==========

    pub fn save_user(&self, user: &User) -> Result<()> {
        let key = Self::u64_to_key(user.id);
        let value = bincode::serialize(user).context("Failed to serialize user")?;
        self.users_tree
            .insert(key, value)
            .context("Failed to insert user")?;
        Ok(())
    }

    pub fn get_user(&self, uid: u64) -> Result<Option<User>> {
        let key = Self::u64_to_key(uid);
        match self.users_tree.get(key).context("Failed to get user")? {
            Some(bytes) => {
                let user = bincode::deserialize(&bytes).context("Failed to deserialize user")?;
                Ok(Some(user))
            }
            None => Ok(None),
        }
    }

    pub fn get_all_users(&self) -> Result<Vec<User>> {
        let mut users = Vec::new();
        for result in self.users_tree.iter() {
            let (_, value) = result.context("Failed to iterate users")?;
            let user: User = bincode::deserialize(&value).context("Failed to deserialize user")?;
            users.push(user);
        }
        Ok(users)
    }

    // ========== Item CRUD ==========

    pub fn save_item(&self, item: &Item) -> Result<()> {
        let key = Self::u64_to_key(item.id);
        let value = bincode::serialize(item).context("Failed to serialize item")?;
        self.items_tree
            .insert(key, value)
            .context("Failed to insert item")?;
        Ok(())
    }

    pub fn get_item(&self, id: u64) -> Result<Option<Item>> {
        let key = Self::u64_to_key(id);
        match self.items_tree.get(key).context("Failed to get item")? {
            Some(bytes) => {
                let item = bincode::deserialize(&bytes).context("Failed to deserialize item")?;
                Ok(Some(item))
            }
            None => Ok(None),
        }
    }

    pub fn iter_items(&self) -> impl Iterator<Item = Result<Item>> + '_ {
        self.items_tree.iter().map(|result| {
            let (_, value) = result.context("Failed to iterate items")?;
            bincode::deserialize(&value).context("Failed to deserialize item")
        })
    }

    pub fn items_count(&self) -> usize {
        self.items_tree.len()
    }

    pub fn users_count(&self) -> usize {
        self.users_tree.len()
    }

    // ========== Bloom Filter (user history deduplication) ==========

    /// Create a new empty bloom filter.
    fn new_bloom_filter() -> BloomFilter {
        FilterBuilder::new(BLOOM_EXPECTED_ITEMS as u64, BLOOM_FPR).build_bloom_filter()
    }

    /// Get a user's bloom filter (returns a fresh empty one if absent).
    pub fn get_user_filter(&self, uid: u64) -> Result<BloomFilter> {
        let key = Self::u64_to_key(uid);
        match self
            .history_tree
            .get(key)
            .context("Failed to get history")?
        {
            Some(bytes) => {
                // Restore bloom filter from raw bytes.
                Ok(BloomFilter::from_u8_array(&bytes, BLOOM_HASHES))
            }
            None => Ok(Self::new_bloom_filter()),
        }
    }

    /// Save a user's bloom filter.
    pub fn save_user_filter(&self, uid: u64, filter: &BloomFilter) -> Result<()> {
        let key = Self::u64_to_key(uid);
        let bytes = filter.get_u8_array();
        self.history_tree
            .insert(key, bytes)
            .context("Failed to save history")?;
        Ok(())
    }

    // ========== Behavior events and preferences ==========

    pub fn get_recent_events(&self, uid: u64) -> Result<RecentEvents> {
        let key = Self::u64_to_key(uid);
        match self
            .events_tree
            .get(key)
            .context("Failed to get recent events")?
        {
            Some(bytes) => {
                bincode::deserialize(&bytes).context("Failed to deserialize recent events")
            }
            None => Ok(RecentEvents::default()),
        }
    }

    pub fn append_user_event(&self, event: &BehaviorEvent) -> Result<()> {
        let key = Self::u64_to_key(event.uid);
        let mut events = self.get_recent_events(event.uid)?;
        events.push(event.clone());
        let value = bincode::serialize(&events).context("Failed to serialize recent events")?;
        self.events_tree
            .insert(key, value)
            .context("Failed to save recent events")?;
        Ok(())
    }

    pub fn get_user_preferences(&self, uid: u64) -> Result<UserPreferences> {
        let key = Self::u64_to_key(uid);
        match self
            .preferences_tree
            .get(key)
            .context("Failed to get user preferences")?
        {
            Some(bytes) => {
                bincode::deserialize(&bytes).context("Failed to deserialize user preferences")
            }
            None => Ok(UserPreferences::default()),
        }
    }

    pub fn save_user_preferences(&self, uid: u64, preferences: &UserPreferences) -> Result<()> {
        let key = Self::u64_to_key(uid);
        let value =
            bincode::serialize(preferences).context("Failed to serialize user preferences")?;
        self.preferences_tree
            .insert(key, value)
            .context("Failed to save user preferences")?;
        Ok(())
    }

    /// Force flush data to disk.
    pub fn flush(&self) -> Result<()> {
        self.users_tree
            .flush()
            .context("Failed to flush users tree")?;
        self.items_tree
            .flush()
            .context("Failed to flush items tree")?;
        self.history_tree
            .flush()
            .context("Failed to flush history tree")?;
        self.events_tree
            .flush()
            .context("Failed to flush events tree")?;
        self.preferences_tree
            .flush()
            .context("Failed to flush preferences tree")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::{BehaviorEvent, EventType, UserPreferences};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_db_path(name: &str) -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("mini-recsys-{}-{}", name, now))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn stores_recent_events_with_a_fixed_window() {
        let path = temp_db_path("events");
        let storage = Storage::new(&path).unwrap();

        for item_id in 1..=105 {
            storage
                .append_user_event(&BehaviorEvent::new(
                    1,
                    item_id,
                    EventType::Impression,
                    "Books",
                ))
                .unwrap();
        }

        let events = storage.get_recent_events(1).unwrap();
        assert_eq!(events.items.len(), 100);
        assert_eq!(events.items.first().unwrap().item_id, 6);
        assert_eq!(events.items.last().unwrap().item_id, 105);

        drop(storage);
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn stores_user_preferences() {
        let path = temp_db_path("preferences");
        let storage = Storage::new(&path).unwrap();
        let mut preferences = UserPreferences::default();
        preferences.set_category_weight("Books", 0.7);
        preferences.set_item_weight(42, -0.5);

        storage.save_user_preferences(1, &preferences).unwrap();

        let stored = storage.get_user_preferences(1).unwrap();
        assert_eq!(stored.category_weight("Books"), 0.7);
        assert_eq!(stored.item_weight(42), -0.5);

        drop(storage);
        let _ = fs::remove_dir_all(path);
    }
}
