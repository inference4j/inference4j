package io.github.inference4j.tokenizer;

public record EncodedInput(
        long[] inputIds,
        long[] attentionMask,
        long[] tokenTypeIds
) {
}
