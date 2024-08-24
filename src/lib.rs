use half::f16;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

pub type TensorVec = Vec<Vec<f32>>;
pub type ModelWeights = BTreeMap<String, Vec<f32>>;
const MODEL_PATH: &str = "models/model.safetensors";

#[derive(Debug, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub partial_rotary_factor: f64,
    pub qk_layernorm: bool,
}

pub fn get_weight(weight_name: &str) -> Vec<f32> {
    let weight_file = File::open(MODEL_PATH).unwrap();
    let buffer = unsafe { MmapOptions::new().map(&weight_file).unwrap() };
    let tensors = SafeTensors::deserialize(&buffer).unwrap();
    parse_tensor_data(tensors.tensor(weight_name).unwrap().data())
}

/// Converts the f16 weights to f32s
fn parse_tensor_data(data: &[u8]) -> Vec<f32> {
    data.chunks(2)
        .map(|chunk| f16::from_le_bytes(chunk.try_into().unwrap()))
        .map(|weight| weight.to_f32())
        .collect()
}
