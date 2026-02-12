package io.github.inference4j.preprocessing;

public record EncodedInput(
        long[] inputIds,
        long[] attentionMask,
        long[] tokenTypeIds
) {
}
