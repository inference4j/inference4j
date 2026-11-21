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

import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.preprocessing.image.Labels;
import io.github.inference4j.session.SessionConfigurer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Builder options shared by the detectors, on real models: where labels come from and maxDetections. Input-size
 * validation is covered by {@code ObjectDetectorBuildersTest} with a fixed-input model.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ObjectDetectorBuildersModelTest {

    private static final SessionConfigurer FOUR_THREADS = options -> options.setIntraOpNumThreads(4);
    private static final int COCO_CAT = 15;

    @TempDir
    Path tmp;

    private BufferedImage cat;

    @BeforeAll
    void setUp() throws IOException {
        cat = ImageIO.read(ObjectDetectorBuildersModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @Test
    void customDetrModelTakesItsLabelsFromConfigJson() throws IOException {
        ModelSource custom = customModel("inference4j/dfine-nano-coco", "custom-detr");
        Files.writeString(tmp.resolve("custom-detr/config.json"), configWithLabels(80));

        try (ObjectDetector detector = DetrDetector.builder().modelId("acme/detr").modelSource(custom)
                .sessionOptions(FOUR_THREADS).build()) {
            assertThat(detector.detect(cat)).extracting(Detection::label).contains("class-" + COCO_CAT);
        }
    }

    @Test
    void explicitLabelsWinOverTheModelDirectory() throws IOException {
        ModelSource custom = customModel("inference4j/dfine-nano-coco", "explicit-detr");
        Files.writeString(tmp.resolve("explicit-detr/config.json"), configWithLabels(80));

        try (ObjectDetector detector = DetrDetector.builder().modelId("acme/detr").modelSource(custom)
                .labels(Labels.coco()).sessionOptions(FOUR_THREADS).build()) {
            assertThat(detector.detect(cat)).extracting(Detection::label).contains("cat");
        }
    }

    @Test
    void customYoloModelTakesItsLabelsFromLabelsTxtUnlessSetExplicitly() throws IOException {
        ModelSource custom = customModel("inference4j/yolov8n", "custom-yolo");
        Files.write(tmp.resolve("custom-yolo/labels.txt"),
                IntStream.range(0, 80).mapToObj(i -> "label-" + i).toList());

        try (ObjectDetector fromFile = YoloV8Detector.builder().modelId("acme/yolo").modelSource(custom)
                .sessionOptions(FOUR_THREADS).build()) {
            assertThat(fromFile.detect(cat)).extracting(Detection::label).contains("label-" + COCO_CAT);
        }
        try (ObjectDetector explicit = YoloV8Detector.builder().modelId("acme/yolo").modelSource(custom)
                .labels(Labels.coco()).sessionOptions(FOUR_THREADS).build()) {
            assertThat(explicit.detect(cat)).extracting(Detection::label).contains("cat");
        }
    }

    @Test
    void presetsKeepCocoNamesAlthoughTheirConfigUsesOtherNames() {
        // The D-FINE mirrors' config.json says "motorbike" and "aeroplane"; presets use Labels.coco()
        try (ObjectDetector detector = DetrDetector.dfineNano().sessionOptions(FOUR_THREADS).build()) {
            assertThat(detector.detect(cat)).extracting(Detection::label).containsOnly("cat");
        }
    }

    @Test
    void maxDetectionsCapsTheResults() {
        try (ObjectDetector all = DetrDetector.dfineSmall().sessionOptions(FOUR_THREADS).build();
             ObjectDetector one = DetrDetector.dfineSmall().maxDetections(1).sessionOptions(FOUR_THREADS).build()) {
            List<Detection> everything = all.detect(cat);

            assertThat(everything).hasSizeGreaterThan(1);
            assertThat(one.detect(cat)).containsExactly(everything.get(0));
        }
    }

    /** A directory holding a copy of a mirror's model.onnx and nothing else, served for any model id. */
    private ModelSource customModel(String mirror, String name) throws IOException {
        Path source = HuggingFaceModelSource.defaultInstance().resolve(mirror, List.of("model.onnx"));
        Path dir = Files.createDirectories(tmp.resolve(name));
        Files.copy(source.resolve("model.onnx"), dir.resolve("model.onnx"));
        return id -> dir;
    }

    private static String configWithLabels(int count) {
        List<String> entries = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            entries.add("\"" + i + "\": \"class-" + i + "\"");
        }
        return "{\"id2label\": {" + entries.stream().collect(Collectors.joining(", ")) + "}}";
    }
}
