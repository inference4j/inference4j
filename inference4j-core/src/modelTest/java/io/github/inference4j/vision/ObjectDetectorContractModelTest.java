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

import io.github.inference4j.preprocessing.image.Labels;
import io.github.inference4j.session.SessionConfigurer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Runs the {@link ObjectDetector} swap contract against every detector preset, through the interface only.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ObjectDetectorContractModelTest {

    private static final SessionConfigurer FOUR_THREADS = options -> options.setIntraOpNumThreads(4);

    private Path catPath;
    private BufferedImage cat;

    @BeforeAll
    void setUp() throws Exception {
        catPath = Path.of(ObjectDetectorContractModelTest.class.getResource("/fixtures/cat.jpg").toURI());
        cat = ImageIO.read(catPath.toFile());
    }

    static Stream<Arguments> detectors() {
        return Stream.of(
                detector("D-FINE small", () -> DetrDetector.dfineSmall().sessionOptions(FOUR_THREADS).build()),
                detector("D-FINE nano", () -> DetrDetector.dfineNano().sessionOptions(FOUR_THREADS).build()),
                detector("D-FINE medium", () -> DetrDetector.dfineMedium().sessionOptions(FOUR_THREADS).build()),
                detector("D-FINE large", () -> DetrDetector.dfineLarge().sessionOptions(FOUR_THREADS).build()),
                detector("RT-DETRv2", () -> DetrDetector.rtDetrV2().sessionOptions(FOUR_THREADS).build()),
                detector("YOLOv8n", () -> YoloV8Detector.builder().sessionOptions(FOUR_THREADS).build()),
                detector("YOLO26n", () -> Yolo26Detector.builder().sessionOptions(FOUR_THREADS).build()),
                detector("YOLOX-Nano", () -> YoloXDetector.nano().sessionOptions(FOUR_THREADS).build()),
                detector("YOLOX-Tiny", () -> YoloXDetector.tiny().sessionOptions(FOUR_THREADS).build()),
                detector("YOLOX-S", () -> YoloXDetector.s().sessionOptions(FOUR_THREADS).build()),
                detector("PicoDet S", () -> PicoDetDetector.s().sessionOptions(FOUR_THREADS).build()),
                detector("PicoDet M", () -> PicoDetDetector.m().sessionOptions(FOUR_THREADS).build()));
    }

    private static Arguments detector(String name, Supplier<ObjectDetector> factory) {
        return Arguments.of(name, factory);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("detectors")
    void followsTheSwapContract(String name, Supplier<ObjectDetector> factory) {
        try (ObjectDetector detector = factory.get()) {
            List<Detection> detections = detector.detect(cat);

            assertThat(detections).as("finds the cat").anySatisfy(d -> assertThat(d.label()).isEqualTo("cat"));
            assertThat(detections).extracting(Detection::confidence)
                    .as("sorted by descending confidence")
                    .isSortedAccordingTo((a, b) -> Float.compare(b, a));
            Labels coco = Labels.coco();
            assertThat(detections).allSatisfy(d -> {
                assertThat(d.classIndex()).isBetween(0, coco.size() - 1);
                assertThat(d.label()).isEqualTo(coco.get(d.classIndex()));
                assertThat(d.box().x1()).isBetween(0f, d.box().x2());
                assertThat(d.box().x2()).isLessThanOrEqualTo(cat.getWidth());
                assertThat(d.box().y1()).isBetween(0f, d.box().y2());
                assertThat(d.box().y2()).isLessThanOrEqualTo(cat.getHeight());
            });

            assertThat(detector.detect(catPath)).as("Path and image inputs agree").isEqualTo(detections);

            List<Detection> confident = detector.detect(cat, 0.9f);
            assertThat(confident).allSatisfy(d -> assertThat(d.confidence()).isGreaterThanOrEqualTo(0.9f));
            assertThat(detector.detect(catPath, 0.9f)).isEqualTo(confident);
        }
    }
}
