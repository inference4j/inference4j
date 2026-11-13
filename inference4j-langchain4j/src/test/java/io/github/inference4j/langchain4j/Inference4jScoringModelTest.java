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
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;
import dev.langchain4j.rag.query.Query;
import io.github.inference4j.nlp.QueryDocumentPair;
import io.github.inference4j.nlp.SearchReranker;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

class Inference4jScoringModelTest {

    /**
     * Scores a document by how many of the query's words it contains, divided by the query's word count,
     * and records the pairs it saw.
     */
    private static final class WordOverlapReranker implements SearchReranker {

        final List<QueryDocumentPair> seen = new ArrayList<>();
        boolean closed;

        @Override
        public Float run(QueryDocumentPair pair) {
            seen.add(pair);
            String[] words = pair.query().toLowerCase().split("\\W+");
            String document = pair.document().toLowerCase();
            int hits = 0;
            for (String word : words) {
                if (document.contains(word)) {
                    hits++;
                }
            }
            return (float) hits / words.length;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    @Test
    void scoresComeBackInSegmentOrder() {
        var model = new Inference4jScoringModel(new WordOverlapReranker());

        List<Double> scores = model.scoreAll(List.of(
                TextSegment.from("nothing relevant"),
                TextSegment.from("capital of France is Paris"),
                TextSegment.from("France")), "capital France").content();

        assertThat(scores).containsExactly(0.0, 1.0, 0.5);
    }

    @Test
    void queryAndDocumentReachTheRerankerInItsArgumentOrder() {
        var reranker = new WordOverlapReranker();
        var model = new Inference4jScoringModel(reranker);

        model.score("the document text", "the query");

        assertThat(reranker.seen).containsExactly(new QueryDocumentPair("the query", "the document text"));
    }

    @Test
    void floatScoresAreWidenedWithoutRescaling() {
        SearchReranker fixed = new SearchReranker() {
            @Override
            public Float run(QueryDocumentPair pair) {
                return 0.7310586f;
            }

            @Override
            public void close() {
            }
        };

        double score = new Inference4jScoringModel(fixed).score("doc", "query").content();

        assertThat(score).isCloseTo(0.7310586, within(1e-6));
    }

    @Test
    void noSegmentsGiveNoScores() {
        var reranker = new WordOverlapReranker();

        List<Double> scores = new Inference4jScoringModel(reranker).scoreAll(List.of(), "query").content();

        assertThat(scores).isEmpty();
        assertThat(reranker.seen).isEmpty();
    }

    @Test
    void aRerankerIsRequired() {
        assertThatThrownBy(() -> new Inference4jScoringModel(null))
                .isInstanceOf(NullPointerException.class)
                .hasMessageContaining("reranker");
    }

    @Test
    void closeClosesTheReranker() {
        var reranker = new WordOverlapReranker();

        new Inference4jScoringModel(reranker).close();

        assertThat(reranker.closed).isTrue();
    }

    @Test
    void reRankingContentAggregatorOrdersAndFiltersByScore() {
        var aggregator = ReRankingContentAggregator.builder()
                .scoringModel(new Inference4jScoringModel(new WordOverlapReranker()))
                .minScore(0.5)
                .build();
        Query query = Query.from("capital France");

        List<Content> ranked = aggregator.aggregate(Map.of(query, List.of(List.of(
                Content.from("nothing relevant"),
                Content.from("France"),
                Content.from("capital of France is Paris")))));

        assertThat(ranked).extracting(content -> content.textSegment().text())
                .containsExactly("capital of France is Paris", "France");
    }
}
