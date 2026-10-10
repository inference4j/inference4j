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
import io.github.inference4j.InferenceContext;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.TokenWindows;
import io.github.inference4j.processing.TruncationGuard;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.preprocessing.text.ModelConfig;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.WordPieceTokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * BERT-based Named Entity Recognition (NER) model.
 *
 * <h2>Target models</h2>
 * <p>Designed for BERT-family models fine-tuned on token classification tasks such as
 * <a href="https://huggingface.co/dslim/distilbert-NER">dslim/distilbert-NER</a> and
 * <a href="https://huggingface.co/dslim/bert-base-NER">dslim/bert-base-NER</a>.
 * Both use IOB2 tagging with labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC,
 * B-MISC, I-MISC.
 *
 * <p>The model directory should contain:
 * <ul>
 *   <li>{@code model.onnx} — the ONNX model file</li>
 *   <li>{@code vocab.txt} — WordPiece vocabulary (cased)</li>
 *   <li>{@code config.json} — HuggingFace config with {@code id2label}</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (BertNerRecognizer ner = BertNerRecognizer.builder().build()) {
 *     List<NamedEntity> entities = ner.recognize("John works at Google in London.");
 *     // entities: [NamedEntity[text=John, label=PER, ...],
 *     //            NamedEntity[text=Google, label=ORG, ...],
 *     //            NamedEntity[text=London, label=LOC, ...]]
 * }
 * }</pre>
 *
 * @see NamedEntityRecognizer
 * @see NamedEntity
 */
public class BertNerRecognizer
        extends AbstractInferenceTask<String, List<NamedEntity>>
        implements NamedEntityRecognizer {

    private static final String DEFAULT_MODEL_ID = "inference4j/distilbert-NER";
    private static final int DEFAULT_MAX_LENGTH = 512;

    private final Tokenizer tokenizer;
    private final ModelConfig config;
    private final int maxLength;
    private final Integer stride;

    static final String WORD_IDS_KEY = "wordIds";

    private BertNerRecognizer(InferenceSession session, Tokenizer tokenizer,
                              ModelConfig config, int maxLength, TruncationGuard truncationGuard,
                              Integer stride) {
        super(session,
                createPreprocessor(tokenizer, maxLength, session.inputNames(), truncationGuard),
                ctx -> postProcess(ctx, config));
        this.tokenizer = tokenizer;
        this.config = config;
        this.maxLength = maxLength;
        this.stride = stride;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<NamedEntity> recognize(String text) {
        if (stride == null) {
            return run(text);
        }
        EncodedInput full = tokenizer.encode(text, Integer.MAX_VALUE);
        if (full.inputIds().length <= maxLength) {
            return run(text);
        }
        return recognizeInWindows(text, full);
    }

    /**
     * Runs the model over overlapping windows of the full token sequence, keeping each token's label
     * from the window where it is most central, then aggregates entities over the whole text.
     */
    private List<NamedEntity> recognizeInWindows(String text, EncodedInput full) {
        long[] ids = full.inputIds();               // [CLS] content… [SEP]
        int length = ids.length;
        String[] labels = new String[length];
        float[] scores = new float[length];

        for (TokenWindows.Window window : TokenWindows.plan(length - 2, maxLength - 2, stride)) {
            int windowTokens = window.end() - window.start();
            long[] windowIds = new long[windowTokens + 2];
            windowIds[0] = ids[0];
            System.arraycopy(ids, 1 + window.start(), windowIds, 1, windowTokens);
            windowIds[windowTokens + 1] = ids[length - 1];

            long[] attentionMask = new long[windowIds.length];
            Arrays.fill(attentionMask, 1L);       // single segment, no padding
            Tensor output = session.run(buildInputs(windowIds, attentionMask,
                            new long[windowIds.length], session.inputNames()))
                    .values().iterator().next();
            float[][] logits = output.squeeze(0).toFloats2D();
            for (int t = window.ownedStart(); t < window.ownedEnd(); t++) {
                labelToken(logits[1 + t - window.start()], config, labels, scores, 1 + t);
            }
        }
        return aggregateEntities(text, labels, scores, full.wordIds(), length);
    }

    static List<NamedEntity> postProcess(InferenceContext<String> ctx, ModelConfig config) {
        String originalText = ctx.input();
        Tensor outputTensor = ctx.outputs().values().iterator().next();

        float[][] tokenLogits = outputTensor.squeeze(0).toFloats2D();

        int[] wordIds = (int[]) ctx.metadata().get(WORD_IDS_KEY);

        int seqLen = tokenLogits.length;

        // Per-token: softmax → argmax → label
        String[] tokenLabels = new String[seqLen];
        float[] tokenScores = new float[seqLen];
        for (int t = 0; t < seqLen; t++) {
            labelToken(tokenLogits[t], config, tokenLabels, tokenScores, t);
        }

        return aggregateEntities(originalText, tokenLabels, tokenScores, wordIds, seqLen);
    }

    private static void labelToken(float[] logits, ModelConfig config,
                                   String[] labels, float[] scores, int position) {
        float[] probs = MathOps.softmax(logits);
        int bestIdx = argmax(probs);
        labels[position] = config.label(bestIdx);
        scores[position] = probs[bestIdx];
    }

    static List<NamedEntity> aggregateEntities(String originalText,
                                               String[] tokenLabels, float[] tokenScores,
                                               int[] wordIds, int seqLen) {
        // Build word-level labels: first subtoken wins
        List<WordLabel> wordLabels = new ArrayList<>();
        int prevWordId = -2;
        for (int t = 0; t < seqLen; t++) {
            int wordId = wordIds != null ? wordIds[t] : t;
            if (wordId == -1) {
                continue; // skip special tokens
            }
            if (wordId != prevWordId) {
                wordLabels.add(new WordLabel(wordId, tokenLabels[t], tokenScores[t]));
            }
            prevWordId = wordId;
        }

        // Split original text into words for offset reconstruction
        List<WordSpan> wordSpans = splitIntoWords(originalText);

        // Group B-I spans into entities
        List<NamedEntity> entities = new ArrayList<>();
        int i = 0;
        while (i < wordLabels.size()) {
            WordLabel wl = wordLabels.get(i);
            if (wl.label.startsWith("B-")) {
                String entityType = wl.label.substring(2);
                float scoreSum = wl.score;
                int count = 1;
                int startWord = wl.wordIndex;
                int endWord = wl.wordIndex;

                // Consume following I- tokens of the same type
                int j = i + 1;
                while (j < wordLabels.size()) {
                    WordLabel next = wordLabels.get(j);
                    if (next.label.equals("I-" + entityType)) {
                        scoreSum += next.score;
                        count++;
                        endWord = next.wordIndex;
                        j++;
                    } else {
                        break;
                    }
                }

                if (startWord < wordSpans.size() && endWord < wordSpans.size()) {
                    int charStart = wordSpans.get(startWord).start;
                    int charEnd = wordSpans.get(endWord).end;
                    String spanText = originalText.substring(charStart, charEnd);
                    entities.add(new NamedEntity(spanText, entityType, charStart, charEnd, scoreSum / count));
                }

                i = j;
            } else {
                i++;
            }
        }

        return entities;
    }

    static List<WordSpan> splitIntoWords(String text) {
        List<WordSpan> spans = new ArrayList<>();
        int i = 0;
        while (i < text.length()) {
            if (Character.isWhitespace(text.charAt(i))) {
                i++;
                continue;
            }
            int start = i;
            if (isPunctuation(text.charAt(i))) {
                spans.add(new WordSpan(start, start + 1));
                i++;
            } else {
                while (i < text.length() && !Character.isWhitespace(text.charAt(i)) && !isPunctuation(text.charAt(i))) {
                    i++;
                }
                spans.add(new WordSpan(start, i));
            }
        }
        return spans;
    }

    private static boolean isPunctuation(char c) {
        int type = Character.getType(c);
        return type == Character.CONNECTOR_PUNCTUATION
                || type == Character.DASH_PUNCTUATION
                || type == Character.END_PUNCTUATION
                || type == Character.FINAL_QUOTE_PUNCTUATION
                || type == Character.INITIAL_QUOTE_PUNCTUATION
                || type == Character.OTHER_PUNCTUATION
                || type == Character.START_PUNCTUATION;
    }

    private static int argmax(float[] values) {
        int bestIdx = 0;
        float bestVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > bestVal) {
                bestVal = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private static io.github.inference4j.processing.Preprocessor<String, PreprocessResult> createPreprocessor(
            Tokenizer tokenizer, int maxLength, Set<String> expectedInputs,
            TruncationGuard truncationGuard) {
        return text -> {
            EncodedInput encoded = tokenizer.encode(text, maxLength);
            truncationGuard.check(encoded, maxLength);
            return PreprocessResult.of(
                    buildInputs(encoded.inputIds(), encoded.attentionMask(), encoded.tokenTypeIds(), expectedInputs),
                    Map.of(WORD_IDS_KEY, encoded.wordIds()));
        };
    }

    private static Map<String, Tensor> buildInputs(long[] inputIds, long[] attentionMask,
                                                   long[] tokenTypeIds, Set<String> expectedInputs) {
        long[] shape = {1, inputIds.length};
        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(inputIds, shape));
        inputs.put("attention_mask", Tensor.fromLongs(attentionMask, shape));
        if (expectedInputs.contains("token_type_ids")) {
            inputs.put("token_type_ids", Tensor.fromLongs(tokenTypeIds, shape));
        }
        return inputs;
    }

    record WordLabel(int wordIndex, String label, float score) {
    }

    record WordSpan(int start, int end) {
    }

    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private Tokenizer tokenizer;
        private ModelConfig config;
        private int maxLength = DEFAULT_MAX_LENGTH;
        private TruncationPolicy truncation;
        private Integer stride;

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

        public Builder tokenizer(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder config(ModelConfig config) {
            this.config = config;
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
         * Processes input longer than {@code maxLength} in full, as overlapping windows that share
         * {@code stride} tokens; each token keeps its label from the window where it is most central.
         * A typical value is 128. Off by default. When set, input is never truncated, so
         * {@link #truncation(TruncationPolicy)} does not apply.
         *
         * @param stride tokens shared by consecutive windows, in {@code [1, maxLength - 2)}
         */
        public Builder stride(int stride) {
            this.stride = stride;
            return this;
        }

        public BertNerRecognizer build() {
            if (stride != null && (stride < 1 || stride >= maxLength - 2)) {
                throw new IllegalArgumentException(
                        "stride must be in [1, maxLength - 2), got " + stride + " for maxLength " + maxLength);
            }
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id, List.of("model.onnx", "vocab.txt", "config.json"));
                loadFromDirectory(dir);
            }
            if (tokenizer == null) {
                throw new IllegalStateException("Tokenizer is required");
            }
            if (config == null) {
                throw new IllegalStateException("ModelConfig is required");
            }
            return new BertNerRecognizer(session, tokenizer, config, maxLength,
                    new TruncationGuard("BertNerRecognizer", truncation), stride);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            Path vocabPath = dir.resolve("vocab.txt");
            Path configPath = dir.resolve("config.json");

            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }
            if (!Files.exists(vocabPath)) {
                throw new ModelSourceException("Vocabulary file not found: " + vocabPath);
            }
            if (!Files.exists(configPath)) {
                throw new ModelSourceException("Config file not found: " + configPath);
            }

            this.session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            try {
                if (this.tokenizer == null) {
                    this.tokenizer = WordPieceTokenizer.fromVocabFile(vocabPath, false);
                }
                if (this.config == null) {
                    this.config = ModelConfig.fromFile(configPath);
                }
            } catch (Exception e) {
                this.session.close();
                this.session = null;
                throw e;
            }
        }
    }
}
