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

import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.preprocessing.image.Labels;
import io.github.inference4j.session.SessionConfigurer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * The options every detector builder shares, checked through the public builders.
 */
class ObjectDetectorBuildersTest {

    /** Shared option names and parameter types; every detector builder must have them. */
    private static final Map<String, Class<?>> SHARED_OPTIONS = Map.of(
            "modelId", String.class,
            "modelSource", ModelSource.class,
            "sessionOptions", SessionConfigurer.class,
            "labels", Labels.class,
            "inputSize", int.class,
            "confidenceThreshold", float.class,
            "maxDetections", int.class);

    @TempDir
    Path dir;

    /** A model whose input is fixed at 1 × 3 × 32 × 32. */
    private ModelSource staticModel;

    @BeforeEach
    void setUp() throws IOException {
        try (InputStream model = getClass().getResourceAsStream("/detector-static-32.onnx")) {
            Files.copy(model, dir.resolve("model.onnx"));
        }
        staticModel = id -> dir;
    }

    /** A value for each shared option, to call the setters with. */
    private static final Map<String, Object> SAMPLE_VALUES = Map.of(
            "modelId", "acme/detector",
            "modelSource", (ModelSource) id -> Path.of(id),
            "sessionOptions", (SessionConfigurer) options -> { },
            "labels", Labels.coco(),
            "inputSize", 320,
            "confidenceThreshold", 0.4f,
            "maxDetections", 10);

    @Test
    void everyDetectorBuilderHasTheSharedOptions() throws Exception {
        for (Object builder : new Object[]{DetrDetector.builder(), YoloV8Detector.builder(), Yolo26Detector.builder(),
                YoloXDetector.builder(), PicoDetDetector.builder()}) {
            for (Map.Entry<String, Class<?>> option : SHARED_OPTIONS.entrySet()) {
                Method method = publicMethod(builder.getClass(), option.getKey(), option.getValue());
                String name = builder.getClass().getName() + "." + option.getKey();
                // Setters inherited from a generic base erase to the base type, so chaining is checked on the
                // returned object rather than on the declared return type
                assertThat(method.getReturnType()).as(name).isAssignableFrom(builder.getClass());
                assertThat(method.invoke(builder, SAMPLE_VALUES.get(option.getKey()))).as(name).isSameAs(builder);
            }
        }
    }

    @Test
    void maxDetectionsBelowOneIsRejected() {
        assertThatThrownBy(() -> DetrDetector.builder().maxDetections(0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("maxDetections");
        assertThatThrownBy(() -> YoloV8Detector.builder().maxDetections(0))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> Yolo26Detector.builder().maxDetections(-1))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void inputSizeThatConflictsWithAFixedModelInputFailsAtBuild() {
        assertThatThrownBy(() -> DetrDetector.builder().modelId("acme/static").modelSource(staticModel)
                .inputSize(64).build())
                .isInstanceOf(ModelLoadException.class)
                .hasMessageContaining("acme/static")
                .hasMessageContaining("fixed 32 × 32 input")
                .hasMessageContaining("inputSize(64)");
        assertThatThrownBy(() -> YoloV8Detector.builder().modelId("acme/static").modelSource(staticModel)
                .inputSize(64).build())
                .isInstanceOf(ModelLoadException.class);
        assertThatThrownBy(() -> Yolo26Detector.builder().modelId("acme/static").modelSource(staticModel)
                .inputSize(64).build())
                .isInstanceOf(ModelLoadException.class);
    }

    @Test
    void matchingOrUnsetInputSizeBuildsAgainstAFixedModelInput() {
        try (var detector = DetrDetector.builder().modelId("acme/static").modelSource(staticModel)
                .inputSize(32).build()) {
            assertThat(detector).isNotNull();
        }
        try (var detector = YoloV8Detector.builder().modelId("acme/static").modelSource(staticModel).build()) {
            assertThat(detector).isNotNull();
        }
        try (var detector = Yolo26Detector.builder().modelId("acme/static").modelSource(staticModel).build()) {
            assertThat(detector).isNotNull();
        }
    }

    private static Method publicMethod(Class<?> type, String name, Class<?> parameter) {
        try {
            return type.getMethod(name, parameter);
        } catch (NoSuchMethodException e) {
            throw new AssertionError(type.getName() + " has no public " + name + "(" + parameter.getSimpleName() + ")",
                    e);
        }
    }
}
