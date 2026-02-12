package io.github.inference4j.preprocessing;

public interface Tokenizer {
    EncodedInput encode(String text);
    EncodedInput encode(String text, int maxLength);
}
