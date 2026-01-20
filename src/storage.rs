//! 存储层 - Sled 嵌入式数据库封装

use anyhow::{Context, Result};
use sled::{Db, Tree};
use crate::model::{User, Item};

pub struct Storage {
    _db: Db,
    users_tree: Tree,
    items_tree: Tree,
}

impl Storage {
    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path).context("Failed to open sled database")?;
        let users_tree = db.open_tree("users").context("Failed to open users tree")?;
        let items_tree = db.open_tree("items").context("Failed to open items tree")?;
        
        Ok(Self {
            _db: db,
            users_tree,
            items_tree,
        })
    }

    /// 将 u64 转为 8 字节 BigEndian 用作 Key
    fn u64_to_key(id: u64) -> [u8; 8] {
        id.to_be_bytes()
    }

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

    /// 强制刷新数据到磁盘
    pub fn flush(&self) -> Result<()> {
        self.users_tree.flush().context("Failed to flush users tree")?;
        self.items_tree.flush().context("Failed to flush items tree")?;
        Ok(())
    }
}
