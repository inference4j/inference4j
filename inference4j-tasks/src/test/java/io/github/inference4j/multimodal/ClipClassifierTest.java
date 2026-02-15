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

import static org.junit.jupiter.api.Assertions.*;
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

        List<Classification> results = ClipClassifier.toClassifications(imageEmb, labelEmbs, labels, 3);

        assertEquals(3, results.size());
        assertEquals("cat", results.get(0).label());
        assertEquals(0, results.get(0).index());
        assertEquals("dog", results.get(1).label());
        assertEquals(1, results.get(1).index());
        assertEquals("car", results.get(2).label());
        assertEquals(2, results.get(2).index());
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

        assertEquals(1, results.size());
        assertEquals("cat", results.get(0).label());
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
            assertTrue(c.confidence() > 0f);
            assertTrue(c.confidence() <= 1f);
            sum += c.confidence();
        }
        assertEquals(1.0f, sum, 1e-5f);
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
            assertTrue(results.get(i - 1).confidence() >= results.get(i).confidence(),
                    "Results not sorted descending at index " + i);
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
                .labels("cat", "dog")
                .build();

        BufferedImage image = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        List<Classification> results = classifier.classify(image);

        assertEquals(2, results.size());
        assertEquals("cat", results.get(0).label());
    }

    @Test
    void close_delegatesToBothEncoders() {
        InferenceSession visionSession = mock(InferenceSession.class);
        InferenceSession textSession = mock(InferenceSession.class);
        @SuppressWarnings("unchecked")
        Preprocessor<BufferedImage, Tensor> preprocessor = mock(Preprocessor.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(tokenizer.encode(anyString())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));
        Tensor emb = Tensor.fromFloats(new float[]{1.0f}, new long[]{1, 1});
        when(textSession.run(any())).thenReturn(Map.of("text_embeds", emb));

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
                .labels("cat")
                .build();

        classifier.close();

        verify(visionSession).close();
        verify(textSession).close();
    }

    @Test
    void builder_missingLabels_throws() {
        assertThrows(IllegalStateException.class, () ->
                ClipClassifier.builder().build());
    }

    @Test
    void builder_emptyLabels_throws() {
        assertThrows(IllegalStateException.class, () ->
                ClipClassifier.builder()
                        .labels(List.of())
                        .build());
    }

    @Test
    void builder_defaultTopK_equalsLabelCount() {
        InferenceSession visionSession = mock(InferenceSession.class);
        InferenceSession textSession = mock(InferenceSession.class);
        @SuppressWarnings("unchecked")
        Preprocessor<BufferedImage, Tensor> preprocessor = mock(Preprocessor.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(tokenizer.encode(anyString())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));
        Tensor emb = Tensor.fromFloats(new float[]{1.0f}, new long[]{1, 1});
        when(textSession.run(any())).thenReturn(Map.of("text_embeds", emb));

        Tensor inputTensor = Tensor.fromFloats(new float[]{0.5f}, new long[]{1});
        when(preprocessor.process(any(BufferedImage.class))).thenReturn(inputTensor);
        Tensor imageOutput = Tensor.fromFloats(new float[]{1.0f}, new long[]{1, 1});
        when(visionSession.run(any())).thenReturn(Map.of("image_embeds", imageOutput));

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
                .labels("cat", "dog", "bird")
                .build();

        BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB);
        List<Classification> results = classifier.classify(image);

        // Default topK should be 3 (number of labels)
        assertEquals(3, results.size());
    }
}
