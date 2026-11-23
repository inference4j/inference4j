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

import io.github.inference4j.AbstractInferenceTask;
import io.github.inference4j.PreprocessResult;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.processing.TokenWindows;
import io.github.inference4j.processing.TruncationGuard;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.processing.Preprocessor;
import io.github.inference4j.Tensor;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.WordPieceTokenizer;
import io.github.inference4j.model.AbstractModelBuilder;

import java.util.Arrays;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Cross-encoder search reranker based on MiniLM architecture.
 *
 * <h2>Target model</h2>
 * <p>Designed for <a href="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2">
 * cross-encoder/ms-marco-MiniLM-L-6-v2</a>, a compact and fast cross-encoder trained
 * on MS MARCO passage ranking data.
 *
 * <p>The model directory should contain:
 * <ul>
 *   <li>{@code model.onnx} — the ONNX model file</li>
 *   <li>{@code vocab.txt} — WordPiece vocabulary</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (MiniLMSearchReranker reranker = MiniLMSearchReranker.builder().build()) {
 *     float score = reranker.score("What is Java?", "Java is a programming language.");
 *
 *     // Re-rank multiple documents
 *     float[] scores = reranker.scoreBatch("What is Java?", List.of(
 *         "Java is a programming language.",
 *         "Python is a programming language.",
 *         "The weather is nice today."
 *     ));
 * }
 * }</pre>
 *
 * @see SearchReranker
 */
public class MiniLMSearchReranker
        extends AbstractInferenceTask<QueryDocumentPair, Float>
        implements io.github.inference4j.nlp.SearchReranker {

    private static final String DEFAULT_MODEL_ID = "inference4j/ms-marco-MiniLM-L-6-v2";

    private static final int DEFAULT_MAX_LENGTH = 512;

    private final Tokenizer tokenizer;
    private final int maxLength;
    private final Integer stride;

    private MiniLMSearchReranker(InferenceSession session, Tokenizer tokenizer, int maxLength,
                                 TruncationGuard truncationGuard, Integer stride) {
        super(session,
                createPreprocessor(tokenizer, maxLength, session.inputNames(), truncationGuard),
                ctx -> {
                    Tensor outputTensor = ctx.outputs().values().iterator().next();
                    float[] logits = outputTensor.toFloats();
                    return toScore(logits[0]);
                });
        this.tokenizer = tokenizer;
        this.maxLength = maxLength;
        this.stride = stride;
    }

    /**
     * Scores a document's relevance to a query. With {@link Builder#stride(int)} set, a document
     * too long for one pass is scored passage by passage and the highest passage score is returned
     * ("MaxP"); the query is included in full with every passage.
     */
    @Override
    public float score(String query, String document) {
        if (stride == null) {
            return run(new QueryDocumentPair(query, document));
        }
        long[] queryIds = tokenizer.encode(query, Integer.MAX_VALUE).inputIds();     // [CLS] query [SEP]
        long[] documentIds = tokenizer.encode(document, Integer.MAX_VALUE).inputIds(); // [CLS] document [SEP]
        int queryTokens = queryIds.length - 2;
        int documentTokens = documentIds.length - 2;
        int passageSize = maxLength - 3 - queryTokens;                       // [CLS] q [SEP] passage [SEP]
        if (queryTokens + documentTokens + 3 <= maxLength || passageSize <= stride) {
            // Fits in one pass, or the query leaves no room for passages: the policy applies
            return run(new QueryDocumentPair(query, document));
        }
        float best = 0f;
        for (TokenWindows.Window passage : TokenWindows.plan(documentTokens, passageSize, stride)) {
            best = Math.max(best, scorePassage(queryIds, documentIds, passage));
        }
        return best;
    }

    private float scorePassage(long[] queryIds, long[] documentIds, TokenWindows.Window passage) {
        int queryTokens = queryIds.length - 2;
        int passageTokens = passage.end() - passage.start();
        long cls = queryIds[0];
        long sep = queryIds[queryIds.length - 1];

        long[] ids = new long[queryTokens + passageTokens + 3];
        long[] tokenTypes = new long[ids.length];
        ids[0] = cls;
        System.arraycopy(queryIds, 1, ids, 1, queryTokens);
        ids[1 + queryTokens] = sep;
        System.arraycopy(documentIds, 1 + passage.start(), ids, 2 + queryTokens, passageTokens);
        ids[ids.length - 1] = sep;
        Arrays.fill(tokenTypes, 2 + queryTokens, ids.length, 1L);           // second segment

        long[] attentionMask = new long[ids.length];
        Arrays.fill(attentionMask, 1L);
        long[] shape = {1, ids.length};
        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(ids, shape));
        inputs.put("attention_mask", Tensor.fromLongs(attentionMask, shape));
        if (session.inputNames().contains("token_type_ids")) {
            inputs.put("token_type_ids", Tensor.fromLongs(tokenTypes, shape));
        }
        float[] logits = session.run(inputs).values().iterator().next().toFloats();
        return toScore(logits[0]);
    }

    public static Builder builder() {
        return new Builder();
    }

    static float toScore(float logit) {
        return (float) (1.0 / (1.0 + Math.exp(-logit)));
    }

    private static Preprocessor<QueryDocumentPair, PreprocessResult> createPreprocessor(
            Tokenizer tokenizer, int maxLength, Set<String> expectedInputs,
            TruncationGuard truncationGuard) {
        return pair -> {
            EncodedInput encoded = tokenizer.encode(pair.query(), pair.document(), maxLength);
            truncationGuard.check(encoded, maxLength);
            long[] shape = {1, encoded.inputIds().length};

            Map<String, Tensor> inputs = new LinkedHashMap<>();
            inputs.put("input_ids", Tensor.fromLongs(encoded.inputIds(), shape));
            inputs.put("attention_mask", Tensor.fromLongs(encoded.attentionMask(), shape));
            if (expectedInputs.contains("token_type_ids")) {
                inputs.put("token_type_ids", Tensor.fromLongs(encoded.tokenTypeIds(), shape));
            }
            return PreprocessResult.of(inputs);
        };
    }

    public static class Builder extends AbstractModelBuilder<MiniLMSearchReranker, Builder> {
        private Tokenizer tokenizer;
        private int maxLength = DEFAULT_MAX_LENGTH;
        private TruncationPolicy truncation;
        private Integer stride;

        public Builder() {
            super(DEFAULT_MODEL_ID);
        }

        Builder session(InferenceSession session) {
            return useSession("model.onnx", session);
        }

        public Builder tokenizer(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }

        /**
         * What to do with input longer than {@code maxLength} tokens. Defaults to
         * {@link TruncationPolicy#TRUNCATE}, which keeps the first tokens and logs a warning.
         */
        public Builder truncation(TruncationPolicy truncation) {
            this.truncation = truncation;
            return this;
        }

        /**
         * Scores documents longer than {@code maxLength} in full with MaxP: the document is split
         * into passages that share {@code stride} tokens, each passage is scored together with the
         * whole query, and the highest score is returned. {@code 0} gives consecutive passages.
         * Off by default. When set, documents are not truncated, so
         * {@link #truncation(TruncationPolicy)} applies only to a query too long to leave room
         * for passages.
         */
        public Builder stride(int stride) {
            if (stride < 0) {
                throw new IllegalArgumentException("stride must not be negative, got " + stride);
            }
            this.stride = stride;
            return this;
        }

        @Override
        public MiniLMSearchReranker build() {
            List<String> required = tokenizer != null ? List.of("model.onnx") : List.of("model.onnx", "vocab.txt");
            return load(required, files -> {
                InferenceSession session = files.session("model.onnx");
                Tokenizer resolvedTokenizer = tokenizer != null
                        ? tokenizer : WordPieceTokenizer.fromVocabFile(files.file("vocab.txt"));
                return new MiniLMSearchReranker(session, resolvedTokenizer, maxLength,
                        new TruncationGuard("MiniLMSearchReranker", truncation), stride);
            });
        }
    }
}
