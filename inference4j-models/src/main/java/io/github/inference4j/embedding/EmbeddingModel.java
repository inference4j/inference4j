package io.github.inference4j.embedding;

import java.util.List;

/**
 * Encodes text into dense vector embeddings.
 */
public interface EmbeddingModel extends AutoCloseable {

    float[] encode(String text);

    List<float[]> encodeBatch(List<String> texts);

    @Override
    void close();
}
