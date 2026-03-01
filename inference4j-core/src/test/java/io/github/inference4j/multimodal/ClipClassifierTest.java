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
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.processing.Preprocessor;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.vision.Classification;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class ClipClassifierTest {

    @Test
    void toClassifications_returnsTopKSortedByConfidence() {
        // Image embedding pointing toward label 0
        float[] imageEmb = {1.0f, 0.0f};

        // Label embeddings: label 0 is aligned, label 1 is orthogonal, label 2 is opposite
        float[][] labelEmbs = {
                {1.0f, 0.0f},  // dot = 1.0
                {0.0f, 1.0f},  // dot = 0.0
                {-1.0f, 0.0f}, // dot = -1.0
        };
        List<String> labels = List.of("cat", "dog", "car");

        List<Classification> results = io.github.inference4j.multimodal.ClipClassifier.toClassifications(imageEmb, labelEmbs, labels, 3);

        assertThat(results).hasSize(3);
        assertThat(results.get(0).label()).isEqualTo("cat");
        assertThat(results.get(0).index()).isEqualTo(0);
        assertThat(results.get(1).label()).isEqualTo("dog");
        assertThat(results.get(1).index()).isEqualTo(1);
        assertThat(results.get(2).label()).isEqualTo("car");
        assertThat(results.get(2).index()).isEqualTo(2);
    }

    @Test
    void toClassifications_topKLimitsResults() {
        float[] imageEmb = {1.0f, 0.0f};
        float[][] labelEmbs = {
                {1.0f, 0.0f},
                {0.0f, 1.0f},
                {-1.0f, 0.0f},
        };
        List<String> labels = List.of("cat", "dog", "car");

        List<Classification> results = ClipClassifier.toClassifications(imageEmb, labelEmbs, labels, 1);

        assertThat(results).hasSize(1);
        assertThat(results.get(0).label()).isEqualTo("cat");
    }

    @Test
    void toClassifications_confidencesSumToOne() {
        float[] imageEmb = MathOps.l2Normalize(new float[]{0.7f, 0.3f});
        float[][] labelEmbs = {
                MathOps.l2Normalize(new float[]{1.0f, 0.0f}),
                MathOps.l2Normalize(new float[]{0.0f, 1.0f}),
                MathOps.l2Normalize(new float[]{0.5f, 0.5f}),
        };
        List<String> labels = List.of("cat", "dog", "bird");

        List<Classification> results = ClipClassifier.toClassifications(imageEmb, labelEmbs, labels, 3);

        float sum = 0f;
        for (Classification c : results) {
            assertThat(c.confidence()).isGreaterThan(0f);
            assertThat(c.confidence()).isLessThanOrEqualTo(1f);
            sum += c.confidence();
        }
        assertThat(sum).isCloseTo(1.0f, within(1e-5f));
    }

    @Test
    void toClassifications_sortedDescending() {
        float[] imageEmb = MathOps.l2Normalize(new float[]{0.5f, 0.5f});
        float[][] labelEmbs = {
                MathOps.l2Normalize(new float[]{1.0f, 0.0f}),
                MathOps.l2Normalize(new float[]{0.0f, 1.0f}),
                MathOps.l2Normalize(new float[]{0.5f, 0.5f}),
                MathOps.l2Normalize(new float[]{-1.0f, 0.0f}),
        };
        List<String> labels = List.of("a", "b", "c", "d");

        List<Classification> results = ClipClassifier.toClassifications(imageEmb, labelEmbs, labels, 4);

        for (int i = 1; i < results.size(); i++) {
            assertThat(results.get(i - 1).confidence())
                    .as("Results not sorted descending at index " + i)
                    .isGreaterThanOrEqualTo(results.get(i).confidence());
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void classify_usesEncodersAndReturnsClassifications() {
        InferenceSession visionSession = mock(InferenceSession.class);
        InferenceSession textSession = mock(InferenceSession.class);
        Preprocessor<BufferedImage, Tensor> preprocessor = mock(Preprocessor.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        // Vision: mock preprocessor and session
        Tensor inputTensor = Tensor.fromFloats(new float[]{0.5f}, new long[]{1});
        when(preprocessor.process(any(BufferedImage.class))).thenReturn(inputTensor);

        // Vision session returns embedding pointing toward "cat"
        Tensor imageOutput = Tensor.fromFloats(new float[]{1.0f, 0.0f}, new long[]{1, 2});
        when(visionSession.run(any())).thenReturn(Map.of("image_embeds", imageOutput));

        // Text: mock tokenizer
        when(tokenizer.encode(anyString())).thenReturn(
                new EncodedInput(new long[]{1, 2}, new long[]{1, 1}, new long[]{0, 0}));

        // Text session returns different embeddings for each label
        Tensor catEmb = Tensor.fromFloats(new float[]{1.0f, 0.0f}, new long[]{1, 2});
        Tensor dogEmb = Tensor.fromFloats(new float[]{0.0f, 1.0f}, new long[]{1, 2});
        when(textSession.run(any()))
                .thenReturn(Map.of("text_embeds", catEmb))
                .thenReturn(Map.of("text_embeds", dogEmb));

        ClipImageEncoder imageEncoder = ClipImageEncoder.builder()
                .session(visionSession)
                .preprocessor(preprocessor)
                .build();

        ClipTextEncoder textEncoder = ClipTextEncoder.builder()
                .session(textSession)
                .tokenizer(tokenizer)
                .build();

        ClipClassifier classifier = ClipClassifier.builder()
                .imageEncoder(imageEncoder)
                .textEncoder(textEncoder)
                .build();

        BufferedImage image = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        List<Classification> results = classifier.classify(image, List.of("cat", "dog"));

        assertThat(results).hasSize(2);
        assertThat(results.get(0).label()).isEqualTo("cat");
    }

    @Test
    @SuppressWarnings("unchecked")
    void classify_withEmptyCandidateLabels_returnsEmpty() {
        InferenceSession visionSession = mock(InferenceSession.class);
        InferenceSession textSession = mock(InferenceSession.class);
        Preprocessor<BufferedImage, Tensor> preprocessor = mock(Preprocessor.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        ClipImageEncoder imageEncoder = ClipImageEncoder.builder()
                .session(visionSession)
                .preprocessor(preprocessor)
                .build();

        ClipTextEncoder textEncoder = ClipTextEncoder.builder()
                .session(textSession)
                .tokenizer(tokenizer)
                .build();

        ClipClassifier classifier = ClipClassifier.builder()
                .imageEncoder(imageEncoder)
                .textEncoder(textEncoder)
                .build();

        BufferedImage image = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        List<Classification> results = classifier.classify(image, List.of());

        assertThat(results).isEmpty();
    }

    @Test
    void close_delegatesToBothEncoders() {
        InferenceSession visionSession = mock(InferenceSession.class);
        InferenceSession textSession = mock(InferenceSession.class);
        @SuppressWarnings("unchecked")
        Preprocessor<BufferedImage, Tensor> preprocessor = mock(Preprocessor.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        ClipImageEncoder imageEncoder = ClipImageEncoder.builder()
                .session(visionSession)
                .preprocessor(preprocessor)
                .build();

        ClipTextEncoder textEncoder = ClipTextEncoder.builder()
                .session(textSession)
                .tokenizer(tokenizer)
                .build();

        ClipClassifier classifier = ClipClassifier.builder()
                .imageEncoder(imageEncoder)
                .textEncoder(textEncoder)
                .build();

        classifier.close();

        verify(visionSession).close();
        verify(textSession).close();
    }
}
