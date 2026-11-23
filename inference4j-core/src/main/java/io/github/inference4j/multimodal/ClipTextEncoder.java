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
import io.github.inference4j.nlp.TextEmbedder;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.processing.TruncationGuard;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.tokenizer.BpeTokenizer;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.model.AbstractModelBuilder;
import io.github.inference4j.model.ModelFiles;

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
    private final TruncationGuard truncationGuard;

    private ClipTextEncoder(InferenceSession session, Tokenizer tokenizer, TruncationGuard truncationGuard) {
        this.session = session;
        this.tokenizer = tokenizer;
        this.truncationGuard = truncationGuard;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public float[] encode(String text) {
        EncodedInput encoded = tokenizer.encode(text);
        truncationGuard.check(encoded, encoded.inputIds().length);
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

    public static class Builder extends AbstractModelBuilder<ClipTextEncoder, Builder> {
        private Tokenizer tokenizer;
        private TruncationPolicy truncation;

        public Builder() {
            super(DEFAULT_MODEL_ID);
        }

        public Builder tokenizer(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        /**
         * What to do with input longer than CLIP's 77-token context. Defaults to
         * {@link TruncationPolicy#TRUNCATE}, which keeps the first tokens and logs a warning.
         */
        public Builder truncation(TruncationPolicy truncation) {
            this.truncation = truncation;
            return this;
        }

        @Override
        public ClipTextEncoder build() {
            List<String> required = tokenizer != null
                    ? List.of(TEXT_MODEL_FILE) : List.of(TEXT_MODEL_FILE, "vocab.json", "merges.txt");
            return load(required, files -> create(files, tokenizer, truncation));
        }
    }

    /**
     * Creates the encoder from {@code text_model.onnx} of a CLIP model and its BPE vocabulary.
     *
     * @param files      the model's files
     * @param tokenizer  the tokenizer, or {@code null} for the model's {@code vocab.json} and {@code merges.txt}
     * @param truncation the truncation policy, or {@code null} for the default
     */
    static ClipTextEncoder create(ModelFiles files, Tokenizer tokenizer, TruncationPolicy truncation) {
        InferenceSession session = files.session(TEXT_MODEL_FILE);
        Tokenizer resolved = tokenizer != null ? tokenizer
                : BpeTokenizer.builder(files.file("vocab.json"), files.file("merges.txt"))
                        .lowercase(true)
                        .endOfWordMarker("</w>")
                        .pattern(CLIP_PATTERN)
                        .bosToken("<|startoftext|>")
                        .eosToken("<|endoftext|>")
                        .pad(true)
                        .defaultMaxLength(77)
                        .build();
        return new ClipTextEncoder(session, resolved, new TruncationGuard("ClipTextEncoder", truncation));
    }
}
