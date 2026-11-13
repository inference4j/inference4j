/*
 * Copyright 2026 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.inference4j.langchain4j;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.DimensionAwareEmbeddingModel;
import dev.langchain4j.model.embedding.request.EmbeddingInput;
import dev.langchain4j.model.embedding.request.EmbeddingInputType;
import dev.langchain4j.model.embedding.request.EmbeddingParameter;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import dev.langchain4j.model.embedding.request.EmbeddingRequestParameters;
import dev.langchain4j.model.embedding.response.EmbeddingResponse;
import dev.langchain4j.model.embedding.response.EmbeddingResponseMetadata;
import io.github.inference4j.nlp.SentenceTransformerEmbedder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A LangChain4j {@link dev.langchain4j.model.embedding.EmbeddingModel} backed by an inference4j
 * {@link SentenceTransformerEmbedder}, running in-process.
 *
 * <p>Asymmetric retrieval models expect one prefix on queries and another on documents. This adapter
 * reads LangChain4j's {@link EmbeddingInputType} from each request and applies the matching prefix, so
 * one model instance serves both ingestion and retrieval:
 *
 * <pre>{@code
 * var model = Inference4jEmbeddingModel.builder()
 *         .embedder(SentenceTransformerEmbedder.builder()
 *                 .modelId("inference4j/nomic-embed-text-v1.5").normalize().build())
 *         .queryPrefix("search_query: ")
 *         .documentPrefix("search_document: ")
 *         .build();
 *
 * EmbeddingStoreIngestor.builder().embeddingModel(model).embeddingStore(store)
 *         .embeddingInputType(EmbeddingInputType.DOCUMENT).build().ingest(documents);
 * EmbeddingStoreContentRetriever.builder().embeddingModel(model).embeddingStore(store)
 *         .embeddingInputType(EmbeddingInputType.QUERY).build();
 * }</pre>
 *
 * <p>Build the embedder <b>without</b> {@code textPrefix}: the adapter's prefixes replace it on every
 * call. With the embedder's {@code stride}, the prefix is repeated in every window.
 *
 * <p>A request without an input type is embedded as {@link Builder#defaultInputType(EmbeddingInputType)
 * the default type} ({@code DOCUMENT} unless set). When a prefix is configured, the first such request
 * logs a warning, because mixing prefixed and unprefixed vectors in one store silently hurts retrieval.
 *
 * <p>The {@code inputType} parameter is always accepted; without prefixes it changes nothing. Other
 * per-call parameters ({@code dimensions}, {@code modelName}) are rejected by LangChain4j: set
 * Matryoshka size on the embedder with {@code dimensions(int)}.
 *
 * <p>The adapter owns the embedder: {@link #close()} closes it. Thread-safe: one instance can serve a
 * whole application.
 */
public class Inference4jEmbeddingModel extends DimensionAwareEmbeddingModel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Inference4jEmbeddingModel.class);

    private final SentenceTransformerEmbedder embedder;
    private final String queryPrefix;
    private final String documentPrefix;
    private final EmbeddingInputType defaultInputType;
    private final Integer knownDimension;
    private final String modelName;
    private final AtomicBoolean warnedMissingInputType = new AtomicBoolean();

    private Inference4jEmbeddingModel(Builder builder) {
        this.embedder = Objects.requireNonNull(builder.embedder, "embedder is required");
        this.queryPrefix = builder.queryPrefix;
        this.documentPrefix = builder.documentPrefix;
        this.defaultInputType = builder.defaultInputType;
        this.knownDimension = builder.dimension;
        this.modelName = builder.modelName;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Embeds every input with the prefix for the request's input type. Embeddings come back in input
     * order. Token usage is not reported.
     */
    @Override
    public EmbeddingResponse doEmbed(EmbeddingRequest request) {
        List<String> texts = new ArrayList<>(request.inputs().size());
        for (EmbeddingInput input : request.inputs()) {
            texts.add(input.text());
        }
        List<float[]> vectors = embedder.encodeBatch(texts, prefixFor(request.inputType()));

        List<Embedding> embeddings = new ArrayList<>(vectors.size());
        for (float[] vector : vectors) {
            embeddings.add(Embedding.from(vector));
        }
        return EmbeddingResponse.builder()
                .embeddings(embeddings)
                .metadata(EmbeddingResponseMetadata.builder().modelName(modelName()).build())
                .build();
    }

    private String prefixFor(EmbeddingInputType requested) {
        EmbeddingInputType type = requested;
        if (type == null) {
            type = defaultInputType;
            if ((queryPrefix != null || documentPrefix != null) && warnedMissingInputType.compareAndSet(false, true)) {
                log.warn("Embedding request has no input type; embedding it as {}. Set embeddingInputType(...) on "
                        + "the EmbeddingStoreIngestor (DOCUMENT) and the EmbeddingStoreContentRetriever (QUERY) "
                        + "so queries and documents get their own prefix.", type);
            }
        }
        return type == EmbeddingInputType.QUERY ? queryPrefix : documentPrefix;
    }

    /**
     * {@code inputType} only. LangChain4j rejects any other per-call parameter before it reaches the model.
     */
    @Override
    public Set<EmbeddingParameter<?>> supportedParameters() {
        return Set.of(EmbeddingRequestParameters.INPUT_TYPE);
    }

    /**
     * The builder's {@code dimension}, or else the length of one embedding, computed once without a prefix.
     */
    @Override
    protected Integer knownDimension() {
        return knownDimension != null ? knownDimension : embedder.encode("dimension", null).length;
    }

    @Override
    public String modelName() {
        return modelName != null ? modelName : super.modelName();
    }

    /**
     * Closes the wrapped embedder.
     */
    @Override
    public void close() {
        embedder.close();
    }

    /**
     * Builder for {@link Inference4jEmbeddingModel}.
     */
    public static class Builder {

        private SentenceTransformerEmbedder embedder;
        private String queryPrefix;
        private String documentPrefix;
        private EmbeddingInputType defaultInputType = EmbeddingInputType.DOCUMENT;
        private Integer dimension;
        private String modelName;

        private Builder() {
        }

        /**
         * The embedder to run. Required. Build it without {@code textPrefix}; the adapter applies prefixes.
         */
        public Builder embedder(SentenceTransformerEmbedder embedder) {
            this.embedder = embedder;
            return this;
        }

        /**
         * Prefix for {@code QUERY} inputs, e.g. {@code "search_query: "} (nomic) or {@code "query: "} (E5).
         * Leave unset for none.
         */
        public Builder queryPrefix(String queryPrefix) {
            this.queryPrefix = queryPrefix;
            return this;
        }

        /**
         * Prefix for {@code DOCUMENT} inputs, e.g. {@code "search_document: "} (nomic) or
         * {@code "passage: "} (E5). Leave unset for none.
         */
        public Builder documentPrefix(String documentPrefix) {
            this.documentPrefix = documentPrefix;
            return this;
        }

        /**
         * Input type for requests that carry none. Defaults to {@code DOCUMENT}.
         */
        public Builder defaultInputType(EmbeddingInputType defaultInputType) {
            this.defaultInputType = Objects.requireNonNull(defaultInputType, "defaultInputType");
            return this;
        }

        /**
         * Embedding size reported by {@code dimension()}. Optional: without it, the adapter embeds one
         * short text the first time {@code dimension()} is called. Must match the embedder's output,
         * including its Matryoshka {@code dimensions(int)}.
         */
        public Builder dimension(int dimension) {
            if (dimension < 1) {
                throw new IllegalArgumentException("dimension must be at least 1, got " + dimension);
            }
            this.dimension = dimension;
            return this;
        }

        /**
         * Name reported by {@code modelName()} and in response metadata, e.g. the model id.
         */
        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * @throws NullPointerException if no embedder is set
         */
        public Inference4jEmbeddingModel build() {
            return new Inference4jEmbeddingModel(this);
        }
    }
}
