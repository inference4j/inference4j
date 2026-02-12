package io.github.inference4j.tokenizer;

public interface Tokenizer {
    EncodedInput encode(String text);
    EncodedInput encode(String text, int maxLength);
}
