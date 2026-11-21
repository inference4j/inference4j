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

import io.github.inference4j.exception.InferenceException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * The default methods of {@link ObjectDetector}, as a custom detector that implements only the abstract ones sees them.
 */
class ObjectDetectorTest {

    @TempDir
    Path dir;

    @Test
    void confidenceOnlyOverloadUsesTheDocumentedIouThreshold() {
        var detector = new RecordingDetector();
        BufferedImage image = new BufferedImage(4, 3, BufferedImage.TYPE_INT_RGB);

        detector.detect(image, 0.7f);

        assertThat(detector.calls).containsExactly("4x3 confidence=0.7 iou=0.45");
    }

    @Test
    void pathOverloadsReadTheFileAndDelegate() throws IOException {
        var detector = new RecordingDetector();
        Path file = dir.resolve("image.png");
        ImageIO.write(new BufferedImage(5, 2, BufferedImage.TYPE_INT_RGB), "png", file.toFile());

        detector.detect(file);
        detector.detect(file, 0.6f);
        detector.detect(file, 0.3f, 0.2f);

        assertThat(detector.calls).containsExactly(
                "5x2 default",
                "5x2 confidence=0.6 iou=0.45",
                "5x2 confidence=0.3 iou=0.2");
    }

    @Test
    void unreadableFileFailsWithAnInferenceException() throws IOException {
        Path file = Files.writeString(dir.resolve("not-an-image.png"), "text");

        assertThatThrownBy(() -> new RecordingDetector().detect(file))
                .isInstanceOf(InferenceException.class)
                .hasMessageContaining("not-an-image.png");
    }

    @Test
    void missingFileFailsWithAnInferenceException() {
        assertThatThrownBy(() -> new RecordingDetector().detect(dir.resolve("missing.png")))
                .isInstanceOf(InferenceException.class)
                .hasMessageContaining("missing.png");
    }

    private static final class RecordingDetector implements ObjectDetector {

        final List<String> calls = new ArrayList<>();

        @Override
        public List<Detection> detect(BufferedImage image) {
            calls.add(image.getWidth() + "x" + image.getHeight() + " default");
            return List.of();
        }

        @Override
        public List<Detection> detect(BufferedImage image, float confidenceThreshold, float iouThreshold) {
            calls.add(image.getWidth() + "x" + image.getHeight()
                    + " confidence=" + confidenceThreshold + " iou=" + iouThreshold);
            return List.of();
        }

        @Override
        public void close() {
        }
    }
}
