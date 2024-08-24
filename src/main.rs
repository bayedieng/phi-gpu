mod model;
use tokenizers::Tokenizer;
fn main() {
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").unwrap();
    let token_encoder = tokenizer
        .encode("Tell me about Julius Caesar", true)
        .unwrap();

    let prompt_tokens: Vec<usize> = token_encoder
        .get_ids()
        .iter()
        .map(|&token| token as usize)
        .collect();

    let mut model = model::Model::new();
    let out = model.forward(&prompt_tokens);
    println!("{:?}", &out[0][..8]);
}
