package io.github.inference4j.vision.classification;

import io.github.inference4j.OutputOperator;
import io.github.inference4j.image.Labels;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class EfficientNetTest {

    private static final Labels TEST_LABELS = Labels.of(List.of(
            "cat", "dog", "bird", "fish", "horse"
    ));

    @Test
    void postProcess_returnsTopKWithHighestProbability() {
        // EfficientNet outputs probabilities (softmax already applied in model)
        float[] probs = {0.05f, 0.50f, 0.20f, 0.05f, 0.20f};

        List<Classification> results = EfficientNet.postProcess(probs, TEST_LABELS, 3, OutputOperator.identity());

        assertEquals(3, results.size());
        assertEquals("dog", results.get(0).label());
        assertEquals(1, results.get(0).index());
        assertEquals(0.50f, results.get(0).confidence(), 1e-5f);
    }

    @Test
    void postProcess_preservesProbabilitiesWithoutSoftmax() {
        float[] probs = {0.10f, 0.40f, 0.20f, 0.05f, 0.25f};

        List<Classification> results = EfficientNet.postProcess(probs, TEST_LABELS, 5, OutputOperator.identity());

        float sum = 0f;
        for (Classification c : results) {
            assertTrue(c.confidence() > 0f);
            assertTrue(c.confidence() <= 1f);
            sum += c.confidence();
        }
        assertEquals(1.0f, sum, 1e-5f);
    }

    @Test
    void postProcess_sortedByConfidenceDescending() {
        float[] probs = {0.10f, 0.40f, 0.15f, 0.30f, 0.05f};

        List<Classification> results = EfficientNet.postProcess(probs, TEST_LABELS, 5, OutputOperator.identity());

        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i - 1).confidence() >= results.get(i).confidence(),
                    "Results not sorted descending at index " + i);
        }
    }

    @Test
    void postProcess_topKLargerThanClasses_returnsAll() {
        float[] probs = {0.05f, 0.10f, 0.15f, 0.20f, 0.50f};

        List<Classification> results = EfficientNet.postProcess(probs, TEST_LABELS, 10, OutputOperator.identity());

        assertEquals(5, results.size());
        assertEquals("horse", results.get(0).label());
    }

    @Test
    void postProcess_topKOne_returnsSingleResult() {
        float[] probs = {0.05f, 0.60f, 0.15f, 0.10f, 0.10f};

        List<Classification> results = EfficientNet.postProcess(probs, TEST_LABELS, 1, OutputOperator.identity());

        assertEquals(1, results.size());
        assertEquals("dog", results.get(0).label());
        assertEquals(1, results.get(0).index());
    }

    @Test
    void postProcess_highConfidencePreserved() {
        float[] probs = {0.95f, 0.02f, 0.01f, 0.01f, 0.01f};

        List<Classification> results = EfficientNet.postProcess(probs, TEST_LABELS, 1, OutputOperator.identity());

        assertEquals("cat", results.get(0).label());
        assertEquals(0, results.get(0).index());
        assertTrue(results.get(0).confidence() > 0.90f);
    }

    @Test
    void postProcess_equalProbabilitiesGiveEqualConfidence() {
        float[] probs = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

        List<Classification> results = EfficientNet.postProcess(probs, TEST_LABELS, 5, OutputOperator.identity());

        for (Classification c : results) {
            assertEquals(0.2f, c.confidence(), 1e-5f);
        }
    }
}
