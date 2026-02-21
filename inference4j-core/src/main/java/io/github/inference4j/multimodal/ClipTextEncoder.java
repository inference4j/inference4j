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

package io.github.inference4j.multimodal;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.nlp.TextEmbedder;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.tokenizer.BpeTokenizer;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * CLIP text encoder — maps text to 512-dimensional L2-normalized embeddings
 * in a shared image-text vector space.
 *
 * <p>CLIP (Contrastive Language-Image Pre-training) by
 * <a href="https://arxiv.org/abs/2103.00020">Radford et al. (2021)</a> trains
 * paired image and text encoders so that matching image-text pairs have high
 * cosine similarity. This wrapper runs the <strong>text encoder</strong> half.
 * Pair it with {@link ClipImageEncoder} for cross-modal retrieval or zero-shot
 * classification.
 *
 * <h2>Tokenization</h2>
 * <p>Uses byte-level BPE tokenization ({@link BpeTokenizer}) with CLIP's vocabulary.
 * The tokenizer is automatically loaded from {@code vocab.json} and {@code merges.txt}
 * in the model directory. BOS ({@code <|startoftext|>}) and EOS
 * ({@code <|endoftext|>}) tokens are added automatically, and sequences are padded
 * to 77 tokens.
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (ClipTextEncoder encoder = ClipTextEncoder.builder().build()) {
 *     float[] embedding = encoder.encode("a photo of a cat");
 *     // 512-dim L2-normalized vector
 * }
 * }</pre>
 *
 * <h2>Zero-shot classification</h2>
 * <pre>{@code
 * try (ClipImageEncoder imageEncoder = ClipImageEncoder.builder().build();
 *      ClipTextEncoder textEncoder = ClipTextEncoder.builder().build()) {
 *
 *     float[] imageEmb = imageEncoder.encode(ImageIO.read(Path.of("photo.jpg").toFile()));
 *     float[] catEmb = textEncoder.encode("a photo of a cat");
 *     float[] dogEmb = textEncoder.encode("a photo of a dog");
 *
 *     // Higher dot product = better match
 *     float catScore = dot(imageEmb, catEmb);
 *     float dogScore = dot(imageEmb, dogEmb);
 * }
 * }</pre>
 *
 * @see ClipImageEncoder
 * @see TextEmbedder
 */
public class ClipTextEncoder implements TextEmbedder {

    private static final String DEFAULT_MODEL_ID = "inference4j/clip-vit-base-patch32";
    private static final String TEXT_MODEL_FILE = "text_model.onnx";

    private static final Pattern CLIP_PATTERN = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE
    );

    private final InferenceSession session;
    private final Tokenizer tokenizer;

    private ClipTextEncoder(InferenceSession session, Tokenizer tokenizer) {
        this.session = session;
        this.tokenizer = tokenizer;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public float[] encode(String text) {
        EncodedInput encoded = tokenizer.encode(text);
        long[] shape = {1, encoded.inputIds().length};

        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(encoded.inputIds(), shape));
        inputs.put("attention_mask", Tensor.fromLongs(encoded.attentionMask(), shape));

        Map<String, Tensor> outputs = session.run(inputs);
        Tensor outputTensor = outputs.values().iterator().next();
        float[] embedding = outputTensor.toFloats();
        return MathOps.l2Normalize(embedding);
    }

    @Override
    public List<float[]> encodeBatch(List<String> texts) {
        List<float[]> results = new ArrayList<>(texts.size());
        for (String text : texts) {
            results.add(encode(text));
        }
        return results;
    }

    @Override
    public void close() {
        session.close();
    }

    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private Tokenizer tokenizer;

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

        public ClipTextEncoder build() {
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id, List.of(TEXT_MODEL_FILE, "vocab.json", "merges.txt"));
                loadFromDirectory(dir);
            }
            if (tokenizer == null) {
                throw new IllegalStateException("Tokenizer is required");
            }
            return new ClipTextEncoder(session, tokenizer);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve(TEXT_MODEL_FILE);
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Text model file not found: " + modelPath);
            }

            this.session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);

            try {
                if (this.tokenizer == null) {
                    Path vocabPath = dir.resolve("vocab.json");
                    Path mergesPath = dir.resolve("merges.txt");
                    if (!Files.exists(vocabPath)) {
                        throw new ModelSourceException("Vocabulary file not found: " + vocabPath);
                    }
                    if (!Files.exists(mergesPath)) {
                        throw new ModelSourceException("Merges file not found: " + mergesPath);
                    }
                    this.tokenizer = BpeTokenizer.builder(vocabPath, mergesPath)
                            .lowercase(true)
                            .endOfWordMarker("</w>")
                            .pattern(CLIP_PATTERN)
                            .bosToken("<|startoftext|>")
                            .eosToken("<|endoftext|>")
                            .pad(true)
                            .defaultMaxLength(77)
                            .build();
                }
            } catch (Exception e) {
                this.session.close();
                this.session = null;
                throw e;
            }
        }
    }
}
