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

package io.github.inference4j.nlp;

import io.github.inference4j.InferenceTask;

import java.util.List;

/**
 * Scores the relevance of text pairs using a cross-encoder architecture.
 *
 * <p>Cross-encoder models take a query-document pair as input and produce a single
 * relevance score. Unlike bi-encoders (which encode texts independently), cross-encoders
 * attend to both texts jointly, yielding higher accuracy at the cost of requiring
 * one forward pass per pair.
 *
 * <p>Common use case: re-ranking search results from a fast first-stage retriever
 * (e.g., BM25 or bi-encoder) to improve precision.
 *
 * @see MiniLMSearchReranker
 * @see QueryDocumentPair
 */
public interface SearchReranker extends InferenceTask<QueryDocumentPair, Float> {

    default float score(String query, String document) {
        return run(new QueryDocumentPair(query, document));
    }

    default float[] scoreBatch(String query, List<String> documents) {
        float[] scores = new float[documents.size()];
        for (int i = 0; i < documents.size(); i++) {
            scores[i] = score(query, documents.get(i));
        }
        return scores;
    }
}
