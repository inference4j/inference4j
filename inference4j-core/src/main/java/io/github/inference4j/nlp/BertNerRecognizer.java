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
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.preprocessing.text.ModelConfig;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.WordPieceTokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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

    private BertNerRecognizer(InferenceSession session, Tokenizer tokenizer,
                              ModelConfig config, int maxLength,
                              ThreadLocal<int[]> wordIdsHolder) {
        super(session,
                createPreprocessor(tokenizer, maxLength, session.inputNames(), wordIdsHolder),
                ctx -> postProcess(ctx, config, wordIdsHolder));
        this.tokenizer = tokenizer;
        this.config = config;
        this.maxLength = maxLength;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<NamedEntity> recognize(String text) {
        return run(text);
    }

    static List<NamedEntity> postProcess(InferenceContext<String> ctx, ModelConfig config,
                                         ThreadLocal<int[]> wordIdsHolder) {
        String originalText = ctx.input();
        Tensor outputTensor = ctx.outputs().values().iterator().next();

        // Output shape: [1, seqLen, numLabels] → squeeze to [seqLen, numLabels]
        float[][] tokenLogits = outputTensor.squeeze(0).toFloats2D();

        int[] wordIds = wordIdsHolder.get();
        wordIdsHolder.remove();

        int seqLen = tokenLogits.length;

        // Per-token: softmax → argmax → label
        String[] tokenLabels = new String[seqLen];
        float[] tokenScores = new float[seqLen];
        for (int t = 0; t < seqLen; t++) {
            float[] probs = MathOps.softmax(tokenLogits[t]);
            int bestIdx = argmax(probs);
            tokenLabels[t] = config.label(bestIdx);
            tokenScores[t] = probs[bestIdx];
        }

        return aggregateEntities(originalText, tokenLabels, tokenScores, wordIds, seqLen);
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

    private static io.github.inference4j.processing.Preprocessor<String, Map<String, Tensor>> createPreprocessor(
            Tokenizer tokenizer, int maxLength, Set<String> expectedInputs,
            ThreadLocal<int[]> wordIdsHolder) {
        return text -> {
            EncodedInput encoded = tokenizer.encode(text, maxLength);
            wordIdsHolder.set(encoded.wordIds());
            long[] shape = {1, encoded.inputIds().length};
            Map<String, Tensor> inputs = new LinkedHashMap<>();
            inputs.put("input_ids", Tensor.fromLongs(encoded.inputIds(), shape));
            inputs.put("attention_mask", Tensor.fromLongs(encoded.attentionMask(), shape));
            if (expectedInputs.contains("token_type_ids")) {
                inputs.put("token_type_ids", Tensor.fromLongs(encoded.tokenTypeIds(), shape));
            }
            return inputs;
        };
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

        public BertNerRecognizer build() {
            ThreadLocal<int[]> wordIdsHolder = new ThreadLocal<>();
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
            return new BertNerRecognizer(session, tokenizer, config, maxLength, wordIdsHolder);
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
