package io.github.inference4j;

public interface GenerativeSession extends AutoCloseable {

    ForwardResult prefill(long[] tokenIds);
    ForwardResult decode(long tokenId);
    int cacheSequenceLength();
    void resetCache();

}
