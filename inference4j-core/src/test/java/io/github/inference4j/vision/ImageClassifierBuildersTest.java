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
package io.github.inference4j.vision;

import io.github.inference4j.Tensor;
import io.github.inference4j.TestModels;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.preprocessing.image.Labels;
import io.github.inference4j.processing.OutputOperator;
import io.github.inference4j.processing.Preprocessor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * The options of the image classifier builders, checked against a model that returns its 1 × 3 × 32 × 32 input
 * unchanged, so each of the 3,072 input values is a class score (and its 1 × 32 × 32 × 3 channels-last twin).
 */
class ImageClassifierBuildersTest {

    private static final int CLASSES = 3 * 32 * 32;
    private static final long[] SHAPE = {1, 3, 32, 32};

    /** Puts 5 at index 7 and 0 everywhere else. */
    private static final Preprocessor<BufferedImage, Tensor> ONE_HOT = image -> {
        float[] values = new float[CLASSES];
        values[7] = 5f;
        return Tensor.fromFloats(values, SHAPE);
    };

    /** Multiplies every score by 10, which neither softmax nor identity would give. */
    private static final OutputOperator TIMES_TEN = values -> {
        float[] scaled = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            scaled[i] = values[i] * 10f;
        }
        return scaled;
    };

    @TempDir
    Path dir;

    private ModelSource source;

    @BeforeEach
    void setUp() throws IOException {
        try (InputStream model = getClass().getResourceAsStream("/detector-static-32.onnx")) {
            Files.copy(model, dir.resolve("model.onnx"));
        }
        source = id -> dir;
    }

    @Test
    void resNetUsesThePreprocessorOutputOperatorAndLabelsSetOnTheBuilder() {
        try (ResNetClassifier classifier = ResNetClassifier.builder().modelSource(source)
                .preprocessor(ONE_HOT).outputOperator(TIMES_TEN).labels(labels("custom")).build()) {
            assertTopIs(classifier.classify(image()), "custom-7", 50f);
        }
    }

    @Test
    void efficientNetUsesThePreprocessorOutputOperatorAndLabelsSetOnTheBuilder() {
        try (EfficientNetClassifier classifier = EfficientNetClassifier.builder().modelSource(source)
                .preprocessor(ONE_HOT).outputOperator(TIMES_TEN).labels(labels("custom")).build()) {
            assertTopIs(classifier.classify(image()), "custom-7", 50f);
        }
    }

    @Test
    void resNetAppliesSoftmaxWhenNoOutputOperatorIsSet() {
        try (ResNetClassifier classifier = ResNetClassifier.builder().modelSource(source)
                .preprocessor(ONE_HOT).labels(labels("custom")).build()) {
            List<Classification> results = classifier.classify(image());
            assertThat(results.get(0).label()).isEqualTo("custom-7");
            assertThat(results.get(0).confidence()).isLessThan(1f);
        }
    }

    @Test
    void efficientNetKeepsTheScoresWhenNoOutputOperatorIsSet() {
        try (EfficientNetClassifier classifier = EfficientNetClassifier.builder().modelSource(source)
                .preprocessor(ONE_HOT).labels(labels("custom")).build()) {
            assertTopIs(classifier.classify(image()), "custom-7", 5f);
        }
    }

    @Test
    void labelsTxtNextToTheModelIsUsedWhenNoLabelsAreSet() throws IOException {
        Files.write(dir.resolve("labels.txt"), names("file"));

        try (ResNetClassifier resNet = ResNetClassifier.builder().modelSource(source).preprocessor(ONE_HOT).build();
             EfficientNetClassifier efficientNet = EfficientNetClassifier.builder().modelSource(source)
                     .preprocessor(ONE_HOT).build()) {
            assertThat(resNet.classify(image()).get(0).label()).isEqualTo("file-7");
            assertThat(efficientNet.classify(image()).get(0).label()).isEqualTo("file-7");
        }
    }

    @Test
    void labelsSetOnTheBuilderWinOverLabelsTxt() throws IOException {
        Files.write(dir.resolve("labels.txt"), names("file"));

        try (ResNetClassifier resNet = ResNetClassifier.builder().modelSource(source).preprocessor(ONE_HOT)
                .labels(labels("custom")).build();
             EfficientNetClassifier efficientNet = EfficientNetClassifier.builder().modelSource(source)
                     .preprocessor(ONE_HOT).labels(labels("custom")).build()) {
            assertThat(resNet.classify(image()).get(0).label()).isEqualTo("custom-7");
            assertThat(efficientNet.classify(image()).get(0).label()).isEqualTo("custom-7");
        }
    }

    @Test
    void defaultPreprocessorResizesToTheModelInput() {
        for (Supplier<AbstractImageClassifier> build : List.<Supplier<AbstractImageClassifier>>of(
                () -> ResNetClassifier.builder().modelSource(source).labels(labels("custom")).build(),
                () -> EfficientNetClassifier.builder().modelSource(source).labels(labels("custom")).build())) {
            try (AbstractImageClassifier classifier = build.get()) {
                assertThat(classifier.classify(new BufferedImage(64, 48, BufferedImage.TYPE_INT_RGB), 3))
                        .hasSize(3);
            }
        }
    }

    @Test
    void closedClassifiersRejectClassify() {
        ResNetClassifier resNet = ResNetClassifier.builder().modelSource(source).labels(labels("custom")).build();
        EfficientNetClassifier efficientNet = EfficientNetClassifier.builder().modelSource(source)
                .labels(labels("custom")).build();
        resNet.close();
        efficientNet.close();

        assertThatThrownBy(() -> resNet.classify(image())).isInstanceOf(IllegalStateException.class);
        assertThatThrownBy(() -> efficientNet.classify(image())).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void defaultPreprocessorFollowsAChannelsLastModel(@TempDir Path nhwcDir) {
        // A blue top-left pixel is value 2 in channels-last order, but 2 * 32 * 32 in channels-first order
        BufferedImage image = image();
        image.setRGB(0, 0, 0x0000FF);

        try (EfficientNetClassifier classifier = EfficientNetClassifier.builder()
                .modelSource(TestModels.model(nhwcDir, "classifier-nhwc")).labels(labels("custom")).build()) {
            Classification top = classifier.classify(image, 1).get(0);

            assertThat(top.index()).isEqualTo(2);
            assertThat(top.confidence()).isCloseTo(1f, within(1e-5f));
        }
    }

    private static void assertTopIs(List<Classification> results, String label, float confidence) {
        assertThat(results).hasSize(5);
        assertThat(results.get(0).label()).isEqualTo(label);
        assertThat(results.get(0).index()).isEqualTo(7);
        assertThat(results.get(0).confidence()).isCloseTo(confidence, within(1e-4f));
    }

    private static BufferedImage image() {
        return new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
    }

    private static Labels labels(String prefix) {
        return Labels.of(names(prefix));
    }

    private static List<String> names(String prefix) {
        List<String> names = new ArrayList<>(CLASSES);
        for (int i = 0; i < CLASSES; i++) {
            names.add(prefix + "-" + i);
        }
        return names;
    }
}
