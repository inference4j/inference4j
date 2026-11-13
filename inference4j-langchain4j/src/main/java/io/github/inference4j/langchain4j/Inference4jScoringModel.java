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

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import io.github.inference4j.nlp.SearchReranker;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A LangChain4j {@link ScoringModel} backed by an inference4j {@link SearchReranker} cross-encoder,
 * running in-process. Use it to rerank retrieved content:
 *
 * <pre>{@code
 * var scoringModel = new Inference4jScoringModel(MiniLMSearchReranker.builder().stride(64).build());
 *
 * var augmentor = DefaultRetrievalAugmentor.builder()
 *         .contentRetriever(retriever)
 *         .contentAggregator(ReRankingContentAggregator.builder()
 *                 .scoringModel(scoringModel)
 *                 .minScore(0.5)
 *                 .build())
 *         .build();
 * }</pre>
 *
 * <p>Scores come from the reranker unchanged. {@code MiniLMSearchReranker} returns a relevance
 * probability between 0 and 1, so {@code minScore} is a probability threshold.
 *
 * <p>Long segments follow the reranker's settings: with {@code stride}, a segment too long for one pass
 * is scored passage by passage with the full query, and the best passage score is kept (MaxP).
 *
 * <p>The adapter owns the reranker: {@link #close()} closes it. Thread-safe: one instance can serve a
 * whole application.
 */
public class Inference4jScoringModel implements ScoringModel, AutoCloseable {

    private final SearchReranker reranker;

    /**
     * @param reranker the cross-encoder to score with; closed by {@link #close()}
     */
    public Inference4jScoringModel(SearchReranker reranker) {
        this.reranker = Objects.requireNonNull(reranker, "reranker is required");
    }

    /**
     * Scores each segment's relevance to the query. Scores come back in segment order. Token usage is
     * not reported.
     */
    @Override
    public Response<List<Double>> scoreAll(List<TextSegment> segments, String query) {
        List<String> documents = new ArrayList<>(segments.size());
        for (TextSegment segment : segments) {
            documents.add(segment.text());
        }
        float[] scores = reranker.scoreBatch(query, documents);

        List<Double> result = new ArrayList<>(scores.length);
        for (float score : scores) {
            result.add((double) score);
        }
        return Response.from(result);
    }

    /**
     * Closes the wrapped reranker.
     */
    @Override
    public void close() {
        reranker.close();
    }
}
