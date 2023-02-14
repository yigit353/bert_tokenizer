//! This crate is a Rust port of Google's BERT [GoogleBERT] WordPiece tokenizer.
//!
//! [GoogleBERT]: https://github.com/google-research/bert


use std::fs::File;
use std::io::{BufRead, BufReader};

use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

use indexmap::IndexMap;

pub type Vocab = IndexMap<String, u32>;
pub type Ids = Vec<u32>;
pub type InvVocab = IndexMap<u32, String>;

/// Load a vocabulary from a vocabulary file.
/// Not recommended to use this function directly, use `FullTokenizerBuilder::vocab_from_file` instead.
///
/// # Example
///
/// ```
/// use bert_tokenizer::{load_vocab, Vocab};
/// let vocab_file = "tests/cased_L-12_H-768_A-12/vocab.txt";
/// let vocab: Vocab = load_vocab(vocab_file);
/// assert_eq!(vocab.len(), 28996);
/// let vocab_file = "tests/uncased_L-12_H-768_A-12/vocab.txt";
/// let vocab: Vocab = load_vocab(vocab_file);
/// assert_eq!(vocab.len(), 30522);
/// ```
pub fn load_vocab(vocab_file: &str) -> Vocab {
    let mut vocab = IndexMap::new();
    let mut index = 0;

    let file = File::open(vocab_file).unwrap();
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let token = line.unwrap();
        if token.is_empty() {
            break;
        }
        vocab.insert(token.trim().to_owned(), index);
        index += 1;
    }

    vocab
}

fn convert_tokens_to_ids(vocab: &Vocab, tokens: &Vec<String>) -> Ids {
    let mut output = Ids::new();
    for token in tokens {
        let id = vocab.get(token).unwrap();
        output.push(*id);
    }
    output
}

fn convert_ids_to_tokens(inv_vocab: &InvVocab, ids: &Ids) -> Vec<String> {
    let mut output = Vec::new();
    for id in ids {
        let token = inv_vocab.get(id).unwrap();
        output.push(token.clone());
    }
    output
}

fn whitespace_tokenize(text: &str) -> Vec<&str> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }
    let tokens = text.split_whitespace().collect();
    tokens
}

fn run_strip_accents(text: &str) -> String {
    let mut output = String::new();
    for c in text.nfd() {
        if !c.is_mark_nonspacing() {
            output.push(c);
        }
    }
    output
}

fn run_split_on_punc(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut start_new_word = true;
    let mut output = vec![];
    while i < chars.len() {
        let char = chars[i];
        if is_punctuation(char) {
            output.push(char.to_string());
            start_new_word = true;
        } else {
            if start_new_word {
                output.push(String::new());
            }
            start_new_word = false;
            output.last_mut().unwrap().push(char);
        }
        i += 1;
    }

    output
}

fn is_punctuation(char: char) -> bool {
    char.is_ascii_punctuation() || char.is_punctuation()
}

fn is_whitespace(char: char) -> bool {
    match char {
        ' ' | '\t' | '\n' | '\r' => true,
        _ => char.is_whitespace(),
    }
}

fn is_control(char: char) -> bool {
    match char {
        '\t' | '\n' | '\r' => false,
        _ => char.is_control(),
    }
}

fn clean_text(text: &str) -> String {
    let mut output = String::new();
    for char in text.chars() {
        let cp = char as u32;
        if cp == 0 || cp == 0xfffd || is_control(char) {
            continue;
        }
        if is_whitespace(char) {
            output.push(' ');
        } else {
            output.push(char);
        }
    }
    output
}

fn tokenize_chinese_chars(text: &str) -> String {
    let mut output = vec![];
    for char in text.chars() {
        let cp = char as u32;
        if is_chinese_char(cp) {
            output.extend(vec![" ".to_string(), char.to_string(), " ".to_string()]);
        } else {
            output.push(char.to_string());
        }
    }
    output.join("")
}

fn is_chinese_char(cp: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

/// A trait for tokenizing text.
/// This trait is implemented by the `BasicTokenizer` and `WordPieceTokenizer`.
pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// A basic tokenizer that runs basic tokenization (punctuation splitting, lower casing, etc.).
/// By default, it does not lower case the input.
///
/// # Example
///
/// ```
/// use bert_tokenizer::{BasicTokenizer, Tokenizer};
/// let tokenizer = BasicTokenizer::default();
/// let tokens = tokenizer.tokenize("Hello, World!");
/// assert_eq!(tokens, vec!["Hello", ",", "World", "!"]);
/// ```
/// If you want to provide lower casing, you can use the `do_lower_case` method:
/// ```
/// use bert_tokenizer::{BasicTokenizer, Tokenizer};
/// let tokenizer = BasicTokenizer::do_lower_case(true).build();
/// let tokens = tokenizer.tokenize("Hello, World!");
/// assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
/// ```
#[derive(Default)]
pub struct BasicTokenizer {
    do_lower_case: bool,
    do_strip_accents: bool,
}

#[derive(Default)]
pub struct BasicTokenizerBuilder {
    do_lower_case: Option<bool>,
    do_strip_accents: Option<bool>,
}

impl BasicTokenizerBuilder {
    pub fn do_lower_case(mut self, do_lower_case: bool) -> Self {
        self.do_lower_case = Some(do_lower_case);
        self
    }

    pub fn do_strip_accents(mut self, do_strip_accents: bool) -> Self {
        self.do_strip_accents = Some(do_strip_accents);
        self
    }

    pub fn build(self) -> BasicTokenizer {
        BasicTokenizer {
            do_lower_case: self.do_lower_case.unwrap_or(false),
            do_strip_accents: self.do_strip_accents.unwrap_or(false),
        }
    }
}

impl BasicTokenizer {
    pub fn do_lower_case(do_lower_case: bool) -> BasicTokenizerBuilder {
        BasicTokenizerBuilder {
            do_lower_case: Some(do_lower_case),
            ..Default::default()
        }
    }
}

impl Tokenizer for BasicTokenizer {
    /// Apply basic tokenization (punctuation splitting, lower casing, etc.) to a piece of text.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to tokenize
    ///
    /// # Returns
    ///
    /// * `Vec<String>` - Vector of tokens
    ///
    /// # Example
    ///
    /// ```
    /// use bert_tokenizer::{BasicTokenizer, Tokenizer};
    ///
    /// let tokenizer = BasicTokenizer::default();
    /// let tokens = tokenizer.tokenize("Hello, World!");
    /// assert_eq!(tokens, vec!["Hello", ",", "World", "!"]);
    /// ```
    fn tokenize(&self, text: &str) -> Vec<String> {
        let text = clean_text(text);
        let text = tokenize_chinese_chars(&text);
        let orig_tokens = whitespace_tokenize(&text);
        let mut split_tokens = vec![];
        for token in orig_tokens {
            let token = if self.do_lower_case {
                token.to_lowercase()
            } else {
                token.to_string()
            };
            let token = if self.do_strip_accents {
                run_strip_accents(&token)
            } else {
                token
            };
            split_tokens.extend(run_split_on_punc(&token));
        }
        whitespace_tokenize(&split_tokens.join(" "))
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

/// A subword tokenizer that runs WordPiece tokenization algorithm.
///
/// # Example
///
/// ```
/// use bert_tokenizer::{Tokenizer, Vocab, WordPieceTokenizer};
///
/// let mut vocab = Vocab::new();
/// vocab.insert("hello".to_string(), 0);
/// vocab.insert("world".to_string(), 1);
/// vocab.insert("!".to_string(), 2);
/// vocab.insert("##!".to_string(), 3);
/// vocab.insert("##world".to_string(), 4);
/// vocab.insert("##hello".to_string(), 5);
///
/// let tokenizer = WordPieceTokenizer::new(vocab).build();
/// let tokens = tokenizer.tokenize("hello world!");
/// assert_eq!(tokens, vec!["hello", "world", "##!"]);
/// ```
pub struct WordPieceTokenizer {
    max_input_chars_per_word: u16,
    unk_token: String,
    vocab: Vocab,
}

#[derive(Default)]
pub struct WordPieceTokenizerBuilder {
    vocab: Vocab,
    unk_token: Option<String>,
    max_input_chars_per_word: Option<u16>,
}

impl WordPieceTokenizerBuilder {
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.unk_token = Some(unk_token);
        self
    }

    pub fn max_input_chars_per_word(mut self, max_input_chars_per_word: u16) -> Self {
        self.max_input_chars_per_word = Some(max_input_chars_per_word);
        self
    }

    pub fn build(self) -> WordPieceTokenizer {
        let unk_token = self.unk_token.unwrap_or("[UNK]".to_string());
        let max_input_chars_per_word = self.max_input_chars_per_word.unwrap_or(200);
        WordPieceTokenizer {
            max_input_chars_per_word,
            unk_token,
            vocab: self.vocab,
        }
    }
}

impl WordPieceTokenizer {
    pub fn new(vocab: Vocab) -> WordPieceTokenizerBuilder {
        WordPieceTokenizerBuilder {
            vocab,
            ..Default::default()
        }
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut output_tokens = vec![];
        for token in text.split_whitespace() {
            let chars: Vec<char> = token.chars().collect();
            if chars.len() as u16 > self.max_input_chars_per_word {
                output_tokens.push(self.unk_token.clone());
                continue;
            }

            let mut is_bad = false;
            let mut start = 0u16;
            let mut sub_tokens = vec![];
            while start < chars.len() as u16 {
                let mut end = chars.len() as u16;
                let mut cur_substr = None;
                while start < end {
                    let substr: String = chars[start as usize..end as usize].iter().collect();
                    let substr = if start > 0 {
                        format!("##{substr}")
                    } else {
                        substr
                    };
                    if self.vocab.contains_key(&substr) {
                        cur_substr = Some(substr);
                        break;
                    }
                    end -= 1;
                }
                if let Some(cur_substr) = cur_substr {
                    sub_tokens.push(cur_substr);
                } else {
                    is_bad = true;
                    break;
                }
                start = end;
            }

            if is_bad {
                output_tokens.push(self.unk_token.clone());
            } else {
                output_tokens.extend(sub_tokens);
            }
        }
        output_tokens
    }
}

/// A FullTokenizer that runs basic tokenization and WordPiece tokenization.
///
/// # Example
///
/// A full tokenizer can be built from a vocabulary as HashMap.
/// ```
/// use bert_tokenizer::{FullTokenizer, Tokenizer, Vocab};
///
/// let mut vocab = Vocab::new();
/// vocab.insert("hello".to_string(), 0);
/// vocab.insert("world".to_string(), 1);
/// vocab.insert("!".to_string(), 2);
/// vocab.insert(",".to_string(), 3);
/// vocab.insert("##,".to_string(), 4);
/// vocab.insert("##!".to_string(), 5);
/// vocab.insert("##world".to_string(), 6);
/// vocab.insert("##hello".to_string(), 7);
///
/// let tokenizer = FullTokenizer::new().vocab(vocab).do_lower_case(true).build();
/// let tokens = tokenizer.tokenize("Hello, World!");
/// assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
/// ```
/// Or from a vocabulary file.
/// ```
/// use bert_tokenizer::{FullTokenizer, Tokenizer};
/// let tokenizer = FullTokenizer::new().vocab_from_file("tests/cased_L-12_H-768_A-12/vocab.txt").build();
/// let tokens = tokenizer.tokenize("Hello, World!");
/// assert_eq!(tokens, vec!["Hello", ",", "World", "!"]);
/// ```
/// You can also specify whether to do lower case.
/// ```
/// use bert_tokenizer::{FullTokenizer, Tokenizer};
/// let tokenizer = FullTokenizer::new().vocab_from_file("tests/uncased_L-12_H-768_A-12/vocab.txt").do_lower_case(true).build();
/// let tokens = tokenizer.tokenize("Hello, World!");
/// assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
/// ```
pub struct FullTokenizer {
    inv_vocab: InvVocab,
    basic_tokenizer: BasicTokenizer,
    wordpiece_tokenizer: WordPieceTokenizer,
}

#[derive(Default)]
pub struct FullTokenizerBuilder {
    vocab: Option<Vocab>,
    do_lower_case: Option<bool>,
    do_strip_accents: Option<bool>,
}

impl FullTokenizerBuilder {
    pub fn vocab_from_file(&mut self, path: &str) -> &mut Self {
        self.vocab = Some(load_vocab(path));
        self
    }

    pub fn vocab(&mut self, vocab: Vocab) -> &mut Self {
        self.vocab = Some(vocab);
        self
    }

    pub fn do_lower_case(&mut self, do_lower_case: bool) -> &mut Self {
        self.do_lower_case = Some(do_lower_case);
        self
    }

    pub fn do_strip_accents(&mut self, do_strip_accents: bool) -> &mut Self {
        self.do_strip_accents = Some(do_strip_accents);
        self
    }

    pub fn build(&mut self) -> FullTokenizer {
        assert!(
            self.vocab.is_some(),
            "Vocab must be set directly or through a vocab file"
        );
        let inv_vocab = self
            .vocab
            .as_ref()
            .unwrap()
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        let basic_tokenizer = BasicTokenizer::do_lower_case(self.do_lower_case.unwrap_or(false))
            .do_strip_accents(self.do_strip_accents.unwrap_or(false))
            .build();
        let wordpiece_tokenizer = WordPieceTokenizer::new(self.vocab.take().unwrap()).build();

        FullTokenizer {
            inv_vocab,
            basic_tokenizer,
            wordpiece_tokenizer,
        }
    }
}

impl FullTokenizer {
    pub fn new() -> FullTokenizerBuilder {
        FullTokenizerBuilder::default()
    }

    /// Converts a sequence of tokens to a sequence of ids.
    ///
    /// # Example
    ///
    /// ```
    /// use bert_tokenizer::{FullTokenizer, Tokenizer, Vocab};
    ///
    /// let mut vocab = Vocab::new();
    /// vocab.insert("hello".to_string(), 0);
    /// vocab.insert("world".to_string(), 1);
    /// vocab.insert("!".to_string(), 2);
    /// vocab.insert("##!".to_string(), 3);
    /// vocab.insert("##world".to_string(), 4);
    /// vocab.insert("##hello".to_string(), 5);
    ///
    /// let tokenizer = FullTokenizer::new().vocab(vocab).build();
    /// let ids = tokenizer.convert_tokens_to_ids(&vec!["hello".to_string(), "!".to_string(), "world".to_string(), "##!".to_string()]);
    /// assert_eq!(ids, vec![0, 2, 1, 3]);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `tokens` - A sequence of tokens.
    ///
    /// # Returns
    ///
    /// A sequence of ids.
    pub fn convert_tokens_to_ids(&self, tokens: &Vec<String>) -> Vec<u32> {
        convert_tokens_to_ids(&self.wordpiece_tokenizer.vocab, tokens)
    }

    /// Converts a sequence of ids to a sequence of tokens.
    ///
    /// # Example
    ///
    /// ```
    /// use bert_tokenizer::{FullTokenizer, Tokenizer, Vocab};
    ///
    /// let mut vocab = Vocab::new();
    /// vocab.insert("hello".to_string(), 0);
    /// vocab.insert("world".to_string(), 1);
    /// vocab.insert("!".to_string(), 2);
    /// vocab.insert("##!".to_string(), 3);
    /// vocab.insert("##world".to_string(), 4);
    /// vocab.insert("##hello".to_string(), 5);
    ///
    /// let tokenizer = FullTokenizer::new().vocab(vocab).build();
    /// let tokens = tokenizer.convert_ids_to_tokens(&vec![0, 2, 4, 3]);
    /// assert_eq!(tokens, vec!["hello", "!", "##world", "##!"]);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `ids` - A sequence of ids.
    ///
    /// # Returns
    ///
    /// A sequence of tokens.
    pub fn convert_ids_to_tokens(&self, ids: &Ids) -> Vec<String> {
        convert_ids_to_tokens(&self.inv_vocab, ids)
    }

    /// Converts a sequence of subword tokens to a single text string.
    ///
    /// # Example
    ///
    /// ```
    /// use bert_tokenizer::{FullTokenizer, Tokenizer, Vocab};
    ///
    /// let mut vocab = Vocab::new();
    /// vocab.insert("hello".to_string(), 0);
    /// vocab.insert("world".to_string(), 1);
    /// vocab.insert("!".to_string(), 2);
    /// vocab.insert("##!".to_string(), 3);
    /// vocab.insert("##world".to_string(), 4);
    /// vocab.insert("##hello".to_string(), 5);
    ///
    /// let tokenizer = FullTokenizer::new().vocab(vocab).build();
    /// let text = "hello, world!";
    /// let tokens = tokenizer.tokenize(text);
    /// let text2 = tokenizer.convert_tokens_to_string(&tokens);
    /// println!("Before: {} -> Tokens: {:?} -> After: {}", text, tokens, text2);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `tokens` - A sequence of tokens.
    ///
    /// # Returns
    ///
    /// A single text string.
    pub fn convert_tokens_to_string(&self, tokens: &[String]) -> String {
        tokens.join(" ").replace(" ##", "")
    }

    /// Get subword tokens from the vocabulary.
    ///
    /// # Example
    ///
    /// ```
    /// use bert_tokenizer::{FullTokenizer, Tokenizer, Vocab};
    ///
    /// let mut vocab = Vocab::new();
    /// vocab.insert("hello".to_string(), 0);
    /// vocab.insert("world".to_string(), 1);
    /// vocab.insert("!".to_string(), 2);
    /// vocab.insert("##!".to_string(), 3);
    /// vocab.insert("##world".to_string(), 4);
    /// vocab.insert("##hello".to_string(), 5);
    ///
    /// let tokenizer = FullTokenizer::new().vocab(vocab).build();
    /// let tokens = tokenizer.get_vocab_words();
    /// assert_eq!(tokens, vec!["hello", "world", "!", "##!", "##world", "##hello"]);
    /// ```
    ///
    /// # Returns
    ///
    /// A sequence of subword tokens.
    pub fn get_vocab_words(&self) -> Vec<String> {
        self.wordpiece_tokenizer
            .vocab
            .keys()
            .map(|s| s.to_string())
            .collect()
    }
}

impl Tokenizer for FullTokenizer {
    /// Tokenize by applying basic and wordpiece tokenization.
    ///
    /// # Example
    ///
    /// ```
    /// use bert_tokenizer::{FullTokenizer, Tokenizer, Vocab};
    ///
    /// let mut vocab = Vocab::new();
    /// vocab.insert("hello".to_string(), 0);
    /// vocab.insert("world".to_string(), 1);
    /// vocab.insert("!".to_string(), 2);
    /// vocab.insert("##!".to_string(), 3);
    /// vocab.insert("##world".to_string(), 4);
    /// vocab.insert("##hello".to_string(), 5);
    ///
    /// let tokenizer = FullTokenizer::new().vocab(vocab).build();
    /// let text = "hello, world!";
    /// let tokens = tokenizer.tokenize(text);
    /// println!("Text: {} -> Tokens: {:?}", text, tokens);
    /// ```
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut split_tokens = vec![];
        for token in self.basic_tokenizer.tokenize(text) {
            for sub_token in self.wordpiece_tokenizer.tokenize(&token) {
                split_tokens.push(sub_token);
            }
        }

        split_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_split_on_punc() {
        let text = "test text!,^ 123";
        let split_on_punc = run_split_on_punc(text);
        assert_eq!(split_on_punc, vec!["test text", "!", ",", "^", " 123"]);
    }

    #[test]
    fn test_strip_accent() {
        let text = "Ragnarök";
        let stripped = run_strip_accents(text);
        assert_eq!(stripped, "Ragnarok");
    }
}