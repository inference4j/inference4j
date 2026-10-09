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
import io.github.inference4j.InferenceContext;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.PreprocessResult;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.TruncationGuard;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.processing.Preprocessor;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.TokenizerJsonParser;
import io.github.inference4j.tokenizer.UnigramTokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Punctuation, true-casing and sentence segmentation for raw, unpunctuated text — typically the
 * output of a speech recognizer such as {@code Wav2Vec2Recognizer}.
 *
 * <p>Wraps the punct-cap-seg models by
 * <a href="https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english">1-800-BAD-CODE</a>,
 * mirroring {@code PunctCapSegModelONNX} from the upstream {@code punctuators} package. In one
 * pass the model adds punctuation ({@code . , ?}), true-cases words including acronyms
 * ({@code us → U.S.}), and splits sentences.
 *
 * <pre>{@code
 * try (PunctCapSegModel model = PunctCapSegModel.builder().build()) {
 *     List<String> sentences = model.infer("marie curie moved to paris she won two nobel prizes");
 *     // ["Marie Curie moved to Paris.", "She won two Nobel Prizes."]
 * }
 * }</pre>
 *
 * <p>Input is lower-cased, stripped of punctuation (apostrophes are kept) and whitespace-collapsed
 * before tokenization. Input longer than {@code maxLength} tokens is handled according to
 * {@link Builder#truncation(TruncationPolicy)}.
 *
 * <p>Not thread-safe: the tokenizer's streaming decoder keeps state between calls.
 */
public class PunctCapSegModel extends AbstractInferenceTask<String, List<String>> {

    private static final String DEFAULT_MODEL_ID = "inference4j/punctuation-fullstop-truecase-english";
    private static final int DEFAULT_MAX_LENGTH = 256;

    // From the model's config.yaml
    private static final long BOS_ID = 1;
    private static final long EOS_ID = 2;
    private static final int ACRONYM = 1;
    private static final String[] POST_PUNCTUATION = {"", "", ".", ",", "?"};
    private static final int CAP_CHARS_PER_TOKEN = 16;

    private PunctCapSegModel(InferenceSession session, UnigramTokenizer tokenizer, int maxLength,
                             TruncationGuard truncationGuard) {
        super(session,
                createPreprocessor(tokenizer, maxLength, truncationGuard),
                ctx -> postProcess(ctx, tokenizer));
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Punctuates, true-cases and splits {@code text} into sentences.
     *
     * @return the sentences in order; empty if the input has no words
     */
    public List<String> infer(String text) {
        return run(text);
    }

    private static Preprocessor<String, PreprocessResult> createPreprocessor(
            UnigramTokenizer tokenizer, int maxLength, TruncationGuard truncationGuard) {
        return text -> {
            int maxPieces = maxLength - 2; // room for BOS and EOS
            EncodedInput encoded = tokenizer.encode(normalize(text), maxPieces);
            truncationGuard.check(encoded, maxPieces);
            long[] pieces = encoded.inputIds();
            long[] ids = new long[pieces.length + 2];
            ids[0] = BOS_ID;
            System.arraycopy(pieces, 0, ids, 1, pieces.length);
            ids[ids.length - 1] = EOS_ID;
            return PreprocessResult.of(Map.of("input_ids", Tensor.fromLongs(ids, new long[]{1, ids.length})));
        };
    }

    static List<String> postProcess(InferenceContext<String> ctx, UnigramTokenizer tokenizer) {
        long[] ids = ctx.preprocessed().get("input_ids").toLongs();
        long[] punctuation = ctx.outputs().get("post_preds").toLongs();
        boolean[] capitals = ctx.outputs().get("cap_preds").toBooleans();
        boolean[] sentenceEnds = ctx.outputs().get("seg_preds").toBooleans();

        List<String> sentences = new ArrayList<>();
        StringBuilder sentence = new StringBuilder();
        for (int t = 1; t < ids.length - 1; t++) {          // skip BOS and EOS
            // A word-initial token decodes with a leading space, which is the model's ▁ at index 0
            String token = tokenizer.decode((int) ids[t]);
            for (int c = 0; c < token.length(); c++) {
                char ch = token.charAt(c);
                boolean upper = c < CAP_CHARS_PER_TOKEN && capitals[t * CAP_CHARS_PER_TOKEN + c];
                sentence.append(upper ? Character.toUpperCase(ch) : ch);
                if (punctuation[t] == ACRONYM && ch != ' ') {
                    sentence.append('.');           // us → U.S.
                }
            }
            if (punctuation[t] < POST_PUNCTUATION.length) {
                sentence.append(POST_PUNCTUATION[(int) punctuation[t]]);
            }
            if (sentenceEnds[t]) {
                addSentence(sentences, sentence);
            }
        }
        addSentence(sentences, sentence);
        return sentences;
    }

    private static void addSentence(List<String> sentences, StringBuilder sentence) {
        String text = sentence.toString().strip();
        if (!text.isEmpty()) {
            sentences.add(text);
        }
        sentence.setLength(0);
    }

    /** Lower-cases, drops punctuation except apostrophes, and collapses whitespace. */
    static String normalize(String text) {
        String withoutPunctuation = text.replaceAll("[\\p{Punct}&&[^']]|[\\p{IsPunctuation}&&[^']]", "");
        return withoutPunctuation.strip().replaceAll("\\s+", " ").toLowerCase(Locale.ROOT);
    }

    public static class Builder {

        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private UnigramTokenizer tokenizer;
        private int maxLength = DEFAULT_MAX_LENGTH;
        private TruncationPolicy truncation;

        Builder session(InferenceSession session) {
            this.session = session;
            return this;
        }

        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder tokenizer(UnigramTokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        /** Maximum tokens per call including BOS and EOS; longer input is truncated. Default 256. */
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

        public PunctCapSegModel build() {
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                loadFromDirectory(source.resolve(id, List.of("model.onnx", "tokenizer.json")));
            }
            if (tokenizer == null) {
                throw new IllegalStateException("Tokenizer is required");
            }
            return new PunctCapSegModel(session, tokenizer, maxLength,
                    new TruncationGuard("PunctCapSegModel", truncation));
        }

        private void loadFromDirectory(Path dir) {
            Path modelPath = dir.resolve("model.onnx");
            Path tokenizerPath = dir.resolve("tokenizer.json");
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }
            if (!Files.exists(tokenizerPath)) {
                throw new ModelSourceException("Tokenizer file not found: " + tokenizerPath);
            }
            this.session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            if (this.tokenizer == null) {
                this.tokenizer = TokenizerJsonParser.parseUnigram(tokenizerPath).build();
            }
        }
    }
}
