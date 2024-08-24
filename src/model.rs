use phi_gpu::{self, Config, ModelWeights, TensorVec};

pub struct Model {
    config: Config,
}

impl Model {
    pub fn new(config: Config) -> Self {
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
