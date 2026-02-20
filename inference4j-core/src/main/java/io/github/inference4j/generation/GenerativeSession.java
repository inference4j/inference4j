package io.github.inference4j.generation;

public interface GenerativeSession extends AutoCloseable {

    ForwardResult prefill(long[] tokenIds);
    ForwardResult decode(long tokenId);
    int cacheSequenceLength();
    void resetCache();

}
