//! 存储层 - Sled 嵌入式数据库封装

use anyhow::{Context, Result};
use fastbloom_rs::{BloomFilter, FilterBuilder, Membership};
use sled::{Db, Tree};
use crate::model::{User, Item};

/// Bloom Filter 参数
const BLOOM_EXPECTED_ITEMS: u32 = 10000;
const BLOOM_FPR: f64 = 0.01;
/// 固定 hash 函数数量 (根据 expected_items 和 fpr 计算: k = -ln(fpr) / ln(2) ≈ 7)
const BLOOM_HASHES: u32 = 7;

pub struct Storage {
    _db: Db,
    users_tree: Tree,
    items_tree: Tree,
    history_tree: Tree,
}

impl Storage {
    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path).context("Failed to open sled database")?;
        let users_tree = db.open_tree("users").context("Failed to open users tree")?;
        let items_tree = db.open_tree("items").context("Failed to open items tree")?;
        let history_tree = db.open_tree("history").context("Failed to open history tree")?;
        
        Ok(Self {
            _db: db,
            users_tree,
            items_tree,
            history_tree,
        })
    }

    fn u64_to_key(id: u64) -> [u8; 8] {
        id.to_be_bytes()
    }

    // ========== User CRUD ==========
    
    pub fn save_user(&self, user: &User) -> Result<()> {
        let key = Self::u64_to_key(user.id);
        let value = bincode::serialize(user).context("Failed to serialize user")?;
        self.users_tree.insert(key, value).context("Failed to insert user")?;
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
        self.items_tree.insert(key, value).context("Failed to insert item")?;
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

    // ========== Bloom Filter (用户历史去重) ==========

    /// 创建一个新的空 Bloom Filter
    fn new_bloom_filter() -> BloomFilter {
        FilterBuilder::new(BLOOM_EXPECTED_ITEMS as u64, BLOOM_FPR).build_bloom_filter()
    }

    /// 获取用户的 Bloom Filter（若不存在则返回新空过滤器）
    pub fn get_user_filter(&self, uid: u64) -> Result<BloomFilter> {
        let key = Self::u64_to_key(uid);
        match self.history_tree.get(key).context("Failed to get history")? {
            Some(bytes) => {
                // 从字节数组还原 BloomFilter
                Ok(BloomFilter::from_u8_array(&bytes, BLOOM_HASHES))
            }
            None => Ok(Self::new_bloom_filter()),
        }
    }

    /// 保存用户的 Bloom Filter
    pub fn save_user_filter(&self, uid: u64, filter: &BloomFilter) -> Result<()> {
        let key = Self::u64_to_key(uid);
        let bytes = filter.get_u8_array();
        self.history_tree.insert(key, bytes).context("Failed to save history")?;
        Ok(())
    }

    /// 强制刷新数据到磁盘
    pub fn flush(&self) -> Result<()> {
        self.users_tree.flush().context("Failed to flush users tree")?;
        self.items_tree.flush().context("Failed to flush items tree")?;
        self.history_tree.flush().context("Failed to flush history tree")?;
        Ok(())
    }
}
