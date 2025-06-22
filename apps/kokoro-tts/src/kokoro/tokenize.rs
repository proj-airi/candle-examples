// https://github.com/lucasjinreal/Kokoros/blob/4c18036a174d28ddec3181c99395af332c32b28d/kokoros/src/tts/tokenize.rs

use crate::kokoro::vocab::{REVERSE_VOCAB, VOCAB};

/// Tokenizes the given phonemes string into a vector of token indices.
///
/// This function takes a text string as input and converts it into a vector of token indices
/// by looking up each character in the global `VOCAB` map and mapping it to the corresponding
/// token index. The resulting vector contains the token indices for the input text.
///
/// # Arguments
/// * `text` - The input text string to be tokenized.
///
/// # Returns
/// A vector of `i64` token indices representing the input text.
pub fn tokenize(phonemes: &str) -> Vec<i64> {
  phonemes
    .chars()
    .filter_map(|c| VOCAB.get(&c))
    .map(|&idx| idx as i64)
    .collect()
}

pub fn tokens_to_phonemes(tokens: &[i64]) -> String {
  tokens
    .iter()
    .filter_map(|&t| REVERSE_VOCAB.get(&(t as usize)))
    .collect()
}
