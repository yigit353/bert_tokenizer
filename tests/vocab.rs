use bert_tokenizer::{FullTokenizer, Tokenizer};

#[test]
fn test_full_tokenizer_uncased() {
    let vocab_file = "tests/uncased_L-12_H-768_A-12/vocab.txt";
    let do_lower_case = true;
    let tokenizer = FullTokenizer::new()
        .vocab_from_file(vocab_file)
        .do_lower_case(do_lower_case)
        .build();
    let tokens = tokenizer.tokenize("Hello world!");
    assert_eq!(tokens, vec!["hello", "world", "!"]);
    let ids = tokenizer.convert_tokens_to_ids(&tokens);
    let tokens = tokenizer.convert_ids_to_tokens(&ids);
    assert_eq!(tokens, vec!["hello", "world", "!"]);
    let text = tokenizer.convert_tokens_to_string(&tokens);
    assert_eq!(text, "hello world !");
}

#[test]
fn test_full_tokenizer_cased() {
    let vocab_file = "tests/cased_L-12_H-768_A-12/vocab.txt";
    let do_lower_case = false;
    let tokenizer = FullTokenizer::new()
        .vocab_from_file(vocab_file)
        .do_lower_case(do_lower_case)
        .build();
    let tokens = tokenizer.tokenize("Hello world!");
    assert_eq!(tokens, vec!["Hello", "world", "!"]);
    let ids = tokenizer.convert_tokens_to_ids(&tokens);
    let tokens = tokenizer.convert_ids_to_tokens(&ids);
    assert_eq!(tokens, vec!["Hello", "world", "!"]);
    let text = tokenizer.convert_tokens_to_string(&tokens);
    assert_eq!(text, "Hello world !");
}

#[test]
fn test_full_tokenizer_cased_strip_accents() {
    let vocab_file = "tests/cased_L-12_H-768_A-12/vocab.txt";
    let do_lower_case = false;
    let do_strip_accents = true;
    let tokenizer = FullTokenizer::new()
        .vocab_from_file(vocab_file)
        .do_lower_case(do_lower_case)
        .do_strip_accents(do_strip_accents)
        .build();
    let tokens = tokenizer.tokenize("Hello wörld!");
    assert_eq!(tokens, vec!["Hello", "world", "!"]);
    let ids = tokenizer.convert_tokens_to_ids(&tokens);
    let tokens = tokenizer.convert_ids_to_tokens(&ids);
    assert_eq!(tokens, vec!["Hello", "world", "!"]);
    let text = tokenizer.convert_tokens_to_string(&tokens);
    assert_eq!(text, "Hello world !");
}

#[test]
fn test_full_tokenizer_cased_no_strip_accents() {
    let vocab_file = "tests/cased_L-12_H-768_A-12/vocab.txt";
    let tokenizer = FullTokenizer::new()
        .vocab_from_file(vocab_file)
        .build();
    let tokens = tokenizer.tokenize("Hello wörld!");
    assert_eq!(tokens, vec!["Hello", "w", "##ö", "##rl", "##d", "!"]);
    let ids = tokenizer.convert_tokens_to_ids(&tokens);
    let tokens = tokenizer.convert_ids_to_tokens(&ids);
    assert_eq!(tokens,  vec!["Hello", "w", "##ö", "##rl", "##d", "!"]);
    let text = tokenizer.convert_tokens_to_string(&tokens);
    assert_eq!(text, "Hello wörld !");
}
