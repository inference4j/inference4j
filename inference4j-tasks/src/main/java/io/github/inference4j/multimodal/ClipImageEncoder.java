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

import io.github.inference4j.AbstractInferenceTask;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.image.ImageTransformPipeline;
import io.github.inference4j.image.Interpolation;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.processing.Preprocessor;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.vision.ImageEmbedder;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * CLIP vision encoder — maps images to 512-dimensional L2-normalized embeddings
 * in a shared image-text vector space.
 *
 * <p>CLIP (Contrastive Language-Image Pre-training) by
 * <a href="https://arxiv.org/abs/2103.00020">Radford et al. (2021)</a> trains
 * paired image and text encoders so that matching image-text pairs have high
 * cosine similarity. This wrapper runs the <strong>vision encoder</strong> half.
 * Pair it with {@link ClipTextEncoder} for cross-modal retrieval or zero-shot
 * classification.
 *
 * <h2>Preprocessing</h2>
 * <ul>
 *   <li>Resize to 224×224, center crop</li>
 *   <li>Normalize with CLIP mean {@code [0.48145466, 0.4578275, 0.40821073]}
 *       and std {@code [0.26862954, 0.26130258, 0.27577711]}</li>
 *   <li>NCHW layout: {@code [1, 3, 224, 224]}</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (ClipImageEncoder encoder = ClipImageEncoder.builder().build()) {
 *     float[] embedding = encoder.encode(ImageIO.read(Path.of("cat.jpg").toFile()));
 *     // 512-dim L2-normalized vector
 * }
 * }</pre>
 *
 * @see ClipTextEncoder
 * @see ImageEmbedder
 */
public class ClipImageEncoder
        extends AbstractInferenceTask<BufferedImage, float[]>
        implements ImageEmbedder {

    private static final String DEFAULT_MODEL_ID = "inference4j/clip-vit-base-patch32";
    private static final String VISION_MODEL_FILE = "vision_model.onnx";
    private static final String DEFAULT_INPUT_NAME = "pixel_values";

    static final float[] CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    static final float[] CLIP_STD = {0.26862954f, 0.26130258f, 0.27577711f};

    private ClipImageEncoder(InferenceSession session, String inputName,
                             Preprocessor<BufferedImage, Tensor> imagePreprocessor) {
        super(session,
                image -> Map.of(inputName, imagePreprocessor.process(image)),
                ctx -> {
                    Tensor outputTensor = ctx.outputs().values().iterator().next();
                    float[] embedding = outputTensor.toFloats();
                    return MathOps.l2Normalize(embedding);
                });
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public float[] encode(BufferedImage image) {
        return run(image);
    }

    @Override
    public float[] encode(Path imagePath) {
        try {
            BufferedImage image = ImageIO.read(imagePath.toFile());
            return encode(image);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read image: " + imagePath, e);
        }
    }

    @Override
    public List<float[]> encodeBatch(List<BufferedImage> images) {
        List<float[]> results = new ArrayList<>(images.size());
        for (BufferedImage image : images) {
            results.add(encode(image));
        }
        return results;
    }

    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private Preprocessor<BufferedImage, Tensor> preprocessor;
        private String inputName;

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

        public Builder preprocessor(Preprocessor<BufferedImage, Tensor> preprocessor) {
            this.preprocessor = preprocessor;
            return this;
        }

        public ClipImageEncoder build() {
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id);
                loadFromDirectory(dir);
            }
            if (inputName == null) {
                inputName = DEFAULT_INPUT_NAME;
            }
            if (preprocessor == null) {
                preprocessor = ImageTransformPipeline.builder()
                        .interpolation(Interpolation.BICUBIC)
                        .resize(224, 224)
                        .centerCrop(224, 224)
                        .mean(CLIP_MEAN)
                        .std(CLIP_STD)
                        .build();
            }
            return new ClipImageEncoder(session, inputName, preprocessor);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve(VISION_MODEL_FILE);
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Vision model file not found: " + modelPath);
            }

            this.session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
        }
    }
}
