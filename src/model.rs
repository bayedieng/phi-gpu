use phi_gpu::{self, ModelWeights, TensorVec};
use serde::Deserialize;

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

pub struct Model {
    config: Config,
}

impl Model {
    pub fn new() -> Self {
        let config = serde_json::from_str(include_str!("../models/config.json")).unwrap();

        Self { config }
    }

    fn embedding(&self, cfg: &Config, indices: &[usize]) -> TensorVec {
        let embedding = phi_gpu::get_weight("model.embed_tokens.weight");
        let mut embedding_matrix = vec![vec![0.; cfg.hidden_size]; cfg.vocab_size];
        embedding_matrix
            .iter_mut()
            .flatten()
            .zip(embedding)
            .for_each(|(y, x)| *y = x);
        let mut out_embedding = Vec::new();
        for embed_index in indices {
            out_embedding.push(embedding_matrix[embed_index.to_owned()].clone())
        }
        out_embedding
    }
    /// Takes in a usize slice as input due to starting with embeddings
    pub fn forward(&mut self, prompt_tokens: &[usize]) -> TensorVec {
        let out_embedding = self.embedding(&self.config, &prompt_tokens);
        out_embedding
    }
}
