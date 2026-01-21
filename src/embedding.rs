//! Embedding 模块 - 使用 ONNX Runtime 进行语义向量化

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Value;
use ort::inputs;
use std::sync::Mutex;
use tokenizers::Tokenizer;

const MODEL_PATH: &str = "models/all-MiniLM-L6-v2.onnx";
const TOKENIZER_PATH: &str = "models/tokenizer.json";
const EMBEDDING_DIM: usize = 384;

pub struct EmbeddingModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl EmbeddingModel {
    pub fn new() -> Result<Self> {
        // 初始化 Session
        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(MODEL_PATH)
            .context("Failed to load ONNX model")?;

        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { session: Mutex::new(session), tokenizer })
    }

    /// 将文本编码为语义向量 (384 维)
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        // Step A: Tokenize
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

        let seq_len = input_ids.len();

        // Step B: 构建输入张量
        let input_ids_val = Value::from_array((vec![1usize, seq_len], input_ids))?;
        let attention_mask_val = Value::from_array((vec![1usize, seq_len], attention_mask.clone()))?;
        let token_type_ids_val = Value::from_array((vec![1usize, seq_len], token_type_ids))?;

        // Step C: 运行推理
        let mut session = self.session.lock().map_err(|_| anyhow::anyhow!("Failed to lock ONNX session"))?;
        let outputs = session.run(inputs![
            "input_ids" => input_ids_val,
            "attention_mask" => attention_mask_val,
            "token_type_ids" => token_type_ids_val,
        ])?;

        // Step D: Mean Pooling
        // ort 2.0 rc.9 try_extract_tensor 返回 (Shape, &[T])
        let (_, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;
        
        let hidden_states = Array2::from_shape_vec(
            (seq_len, EMBEDDING_DIM),
            output_data.to_vec()
        )?;

        let mask: Array1<f32> = attention_mask.iter().map(|&x| x as f32).collect();
        let mask_sum = mask.sum();

        let mut pooled = Array1::<f32>::zeros(EMBEDDING_DIM);
        for (i, &m) in mask.iter().enumerate() {
            if m > 0.0 {
                pooled = pooled + &hidden_states.row(i).to_owned();
            }
        }
        pooled = pooled / mask_sum;

        // Step E: L2 归一化
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = pooled.iter().map(|x| x / norm).collect();

        Ok(normalized)
    }

    pub fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}
