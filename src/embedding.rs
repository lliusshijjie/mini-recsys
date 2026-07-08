//! Embedding module: semantic vectorization via ONNX Runtime

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ort::inputs;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

const EMBEDDING_DIM: usize = 384;

pub struct EmbeddingModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl EmbeddingModel {
    pub fn new_with_paths(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        // Initialize session.
        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(model_path.as_ref())
            .context("Failed to load ONNX model")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Encode text into a semantic vector (384-dim).
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        // Step A: Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

        let seq_len = input_ids.len();

        // Step B: build input tensors
        let input_ids_val = Value::from_array((vec![1usize, seq_len], input_ids))?;
        let attention_mask_val =
            Value::from_array((vec![1usize, seq_len], attention_mask.clone()))?;
        let token_type_ids_val = Value::from_array((vec![1usize, seq_len], token_type_ids))?;

        // Step C: run inference
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock ONNX session"))?;
        let outputs = session.run(inputs![
            "input_ids" => input_ids_val,
            "attention_mask" => attention_mask_val,
            "token_type_ids" => token_type_ids_val,
        ])?;

        // Step D: Mean Pooling
        // ort 2.0 rc.9 try_extract_tensor returns (Shape, &[T])
        let (_, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        let hidden_states = Array2::from_shape_vec((seq_len, EMBEDDING_DIM), output_data.to_vec())?;

        let mask: Array1<f32> = attention_mask.iter().map(|&x| x as f32).collect();
        let mask_sum = mask.sum();

        let mut pooled = Array1::<f32>::zeros(EMBEDDING_DIM);
        for (i, &m) in mask.iter().enumerate() {
            if m > 0.0 {
                pooled += &hidden_states.row(i).to_owned();
            }
        }
        pooled /= mask_sum;

        // Step E: L2 normalization
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = pooled.iter().map(|x| x / norm).collect();

        Ok(normalized)
    }

    pub fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}
