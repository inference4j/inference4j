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

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.embedding.request.EmbeddingInputType;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import dev.langchain4j.model.embedding.request.EmbeddingRequestParameters;
import dev.langchain4j.model.embedding.response.EmbeddingResponse;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import io.github.inference4j.nlp.SentenceTransformerEmbedder;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class Inference4jEmbeddingModelTest {

    private static final String QUERY_PREFIX = "search_query: ";
    private static final String DOCUMENT_PREFIX = "search_document: ";

    private SentenceTransformerEmbedder embedder;

    /**
     * Each text embeds to {@code [length, index]}, so tests can check which text landed where.
     */
    @BeforeEach
    void stubEmbedder() {
        embedder = mock(SentenceTransformerEmbedder.class);
        when(embedder.encodeBatch(anyList(), any())).thenAnswer(call -> {
            List<String> texts = call.getArgument(0);
            List<float[]> vectors = new ArrayList<>();
            for (int i = 0; i < texts.size(); i++) {
                vectors.add(new float[]{texts.get(i).length(), i});
            }
            return vectors;
        });
        when(embedder.encode(anyString(), any())).thenReturn(new float[]{1f, 2f, 3f});
    }

    private Inference4jEmbeddingModel prefixed() {
        return Inference4jEmbeddingModel.builder()
                .embedder(embedder)
                .queryPrefix(QUERY_PREFIX)
                .documentPrefix(DOCUMENT_PREFIX)
                .build();
    }

    private static EmbeddingRequest request(EmbeddingInputType type, String... texts) {
        EmbeddingRequest.Builder builder = EmbeddingRequest.builder().inputs(texts);
        if (type != null) {
            builder.parameters(EmbeddingRequestParameters.builder().inputType(type).build());
        }
        return builder.build();
    }

    // --- prefixes per input type ---

    @Test
    void queryInputsGetTheQueryPrefix() {
        prefixed().embed(request(EmbeddingInputType.QUERY, "what is rag?"));

        verify(embedder).encodeBatch(List.of("what is rag?"), QUERY_PREFIX);
    }

    @Test
    void documentInputsGetTheDocumentPrefix() {
        prefixed().embed(request(EmbeddingInputType.DOCUMENT, "rag retrieves context"));

        verify(embedder).encodeBatch(List.of("rag retrieves context"), DOCUMENT_PREFIX);
    }

    @Test
    void anUnsetPrefixMeansNoPrefixForThatSide() {
        var model = Inference4jEmbeddingModel.builder()
                .embedder(embedder)
                .queryPrefix("Represent this sentence for searching relevant passages: ")
                .build();

        model.embed(request(EmbeddingInputType.DOCUMENT, "a passage"));

        verify(embedder).encodeBatch(eq(List.of("a passage")), isNull());
    }

    @Test
    void withoutPrefixesTheInputTypeIsAcceptedAndIgnored() {
        var model = Inference4jEmbeddingModel.builder().embedder(embedder).build();

        model.embed(request(EmbeddingInputType.QUERY, "hello"));

        verify(embedder).encodeBatch(eq(List.of("hello")), isNull());
    }

    // --- default input type ---

    @Test
    void requestsWithoutAnInputTypeAreEmbeddedAsDocuments() {
        prefixed().embed(request(null, "no type"));

        verify(embedder).encodeBatch(List.of("no type"), DOCUMENT_PREFIX);
    }

    @Test
    void theDefaultInputTypeIsConfigurable() {
        var model = Inference4jEmbeddingModel.builder()
                .embedder(embedder)
                .queryPrefix(QUERY_PREFIX)
                .documentPrefix(DOCUMENT_PREFIX)
                .defaultInputType(EmbeddingInputType.QUERY)
                .build();

        model.embed("no type");

        verify(embedder).encodeBatch(List.of("no type"), QUERY_PREFIX);
    }

    // --- results ---

    @Test
    void embeddingsComeBackInInputOrder() {
        EmbeddingResponse response = prefixed().embed(request(EmbeddingInputType.DOCUMENT, "a", "bb", "ccc"));

        assertThat(response.embeddings()).extracting(Embedding::vector).containsExactly(
                new float[]{1f, 0f}, new float[]{2f, 1f}, new float[]{3f, 2f});
    }

    @Test
    void embedAllMapsSegmentsToEmbeddings() {
        List<Embedding> embeddings = prefixed()
                .embedAll(List.of(TextSegment.from("a"), TextSegment.from("bb")))
                .content();

        assertThat(embeddings).hasSize(2);
        assertThat(embeddings.get(1).vector()).containsExactly(2f, 1f);
    }

    @Test
    void modelNameIsReportedInResponseMetadata() {
        var model = Inference4jEmbeddingModel.builder()
                .embedder(embedder)
                .modelName("inference4j/nomic-embed-text-v1.5")
                .build();

        assertThat(model.modelName()).isEqualTo("inference4j/nomic-embed-text-v1.5");
        assertThat(model.embed(request(null, "x")).modelName()).isEqualTo("inference4j/nomic-embed-text-v1.5");
    }

    // --- dimension ---

    @Test
    void dimensionComesFromTheBuilderWithoutEmbedding() {
        var model = Inference4jEmbeddingModel.builder().embedder(embedder).dimension(256).build();

        assertThat(model.dimension()).isEqualTo(256);
        verify(embedder, never()).encode(anyString(), any());
    }

    @Test
    void dimensionIsProbedOnceWithoutAPrefixWhenNotGiven() {
        var model = prefixed();

        assertThat(model.dimension()).isEqualTo(3);
        assertThat(model.dimension()).isEqualTo(3);
        verify(embedder).encode("dimension", null);
    }

    @Test
    void dimensionMustBePositive() {
        assertThatThrownBy(() -> Inference4jEmbeddingModel.builder().dimension(0))
                .isInstanceOf(IllegalArgumentException.class);
    }

    // --- parameters and lifecycle ---

    @Test
    void perCallDimensionsAreRejected() {
        var request = EmbeddingRequest.builder()
                .inputs("x")
                .parameters(EmbeddingRequestParameters.builder().dimensions(128).build())
                .build();

        assertThatThrownBy(() -> prefixed().embed(request))
                .isInstanceOf(UnsupportedFeatureException.class)
                .hasMessageContaining("dimensions");
    }

    @Test
    void anEmbedderIsRequired() {
        assertThatThrownBy(() -> Inference4jEmbeddingModel.builder().build())
                .isInstanceOf(NullPointerException.class)
                .hasMessageContaining("embedder");
    }

    @Test
    void closeClosesTheEmbedder() {
        prefixed().close();

        verify(embedder).close();
    }

    // --- LangChain4j RAG components ---

    @Test
    void ingestorAndRetrieverApplyTheirOwnPrefixes() {
        var model = prefixed();
        var store = new InMemoryEmbeddingStore<TextSegment>();

        EmbeddingStoreIngestor.builder()
                .embeddingModel(model)
                .embeddingStore(store)
                .embeddingInputType(EmbeddingInputType.DOCUMENT)
                .build()
                .ingest(Document.from("Paris is the capital of France."));
        List<Content> found = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(model)
                .embeddingStore(store)
                .embeddingInputType(EmbeddingInputType.QUERY)
                .build()
                .retrieve(Query.from("capital of France?"));

        verify(embedder).encodeBatch(List.of("Paris is the capital of France."), DOCUMENT_PREFIX);
        verify(embedder).encodeBatch(List.of("capital of France?"), QUERY_PREFIX);
        assertThat(found).extracting(content -> content.textSegment().text())
                .containsExactly("Paris is the capital of France.");
    }
}
