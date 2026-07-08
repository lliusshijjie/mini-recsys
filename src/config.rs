//! Runtime configuration loaded from environment variables.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const DEFAULT_PORT: u16 = 3000;
const DEFAULT_DATA_DIR: &str = "data";
const DEFAULT_MODEL_PATH: &str = "models/all-MiniLM-L6-v2.onnx";
const DEFAULT_TOKENIZER_PATH: &str = "models/tokenizer.json";
const DEFAULT_CORS_ORIGIN: &str = "http://localhost:5173";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppConfig {
    pub port: u16,
    pub data_dir: PathBuf,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub cors_origin: String,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        Self::from_getter(|key| std::env::var(key).ok())
    }

    pub fn from_getter<F>(get: F) -> Result<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        let port = match get("PORT") {
            Some(value) => value
                .parse::<u16>()
                .with_context(|| format!("Invalid PORT value: {}", value))?,
            None => DEFAULT_PORT,
        };

        Ok(Self {
            port,
            data_dir: PathBuf::from(get("DATA_DIR").unwrap_or_else(|| DEFAULT_DATA_DIR.into())),
            model_path: PathBuf::from(
                get("MODEL_PATH").unwrap_or_else(|| DEFAULT_MODEL_PATH.into()),
            ),
            tokenizer_path: PathBuf::from(
                get("TOKENIZER_PATH").unwrap_or_else(|| DEFAULT_TOKENIZER_PATH.into()),
            ),
            cors_origin: get("CORS_ORIGIN").unwrap_or_else(|| DEFAULT_CORS_ORIGIN.into()),
        })
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("db")
    }

    pub fn index_path(&self) -> PathBuf {
        self.data_dir.join("index.bin")
    }

    pub fn tantivy_path(&self) -> PathBuf {
        self.data_dir.join("tantivy_index")
    }

    pub fn db_path_str(&self) -> String {
        path_to_string(&self.db_path())
    }

    pub fn index_path_str(&self) -> String {
        path_to_string(&self.index_path())
    }

    pub fn tantivy_path_str(&self) -> String {
        path_to_string(&self.tantivy_path())
    }
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn config_from(values: &[(&str, &str)]) -> AppConfig {
        let map: HashMap<String, String> = values
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect();
        AppConfig::from_getter(|key| map.get(key).cloned()).unwrap()
    }

    #[test]
    fn defaults_match_local_development_paths() {
        let config = config_from(&[]);

        assert_eq!(config.port, 3000);
        assert_eq!(config.data_dir, PathBuf::from("data"));
        assert_eq!(config.db_path(), PathBuf::from("data").join("db"));
        assert_eq!(config.index_path(), PathBuf::from("data").join("index.bin"));
        assert_eq!(
            config.tantivy_path(),
            PathBuf::from("data").join("tantivy_index")
        );
        assert_eq!(
            config.model_path,
            PathBuf::from("models/all-MiniLM-L6-v2.onnx")
        );
        assert_eq!(
            config.tokenizer_path,
            PathBuf::from("models/tokenizer.json")
        );
        assert_eq!(config.cors_origin, "http://localhost:5173");
    }

    #[test]
    fn env_values_override_defaults() {
        let config = config_from(&[
            ("PORT", "8080"),
            ("DATA_DIR", "/var/lib/mini-recsys"),
            ("MODEL_PATH", "/models/model.onnx"),
            ("TOKENIZER_PATH", "/models/tokenizer.json"),
            ("CORS_ORIGIN", "https://example.test"),
        ]);

        assert_eq!(config.port, 8080);
        assert_eq!(config.data_dir.to_string_lossy(), "/var/lib/mini-recsys");
        assert_eq!(
            config.db_path().to_string_lossy(),
            "/var/lib/mini-recsys\\db"
        );
        assert_eq!(config.model_path.to_string_lossy(), "/models/model.onnx");
        assert_eq!(
            config.tokenizer_path.to_string_lossy(),
            "/models/tokenizer.json"
        );
        assert_eq!(config.cors_origin, "https://example.test");
    }

    #[test]
    fn invalid_port_returns_error() {
        let map = HashMap::from([("PORT".to_string(), "not-a-port".to_string())]);

        let error = AppConfig::from_getter(|key| map.get(key).cloned()).unwrap_err();

        assert!(error.to_string().contains("PORT"));
    }
}
