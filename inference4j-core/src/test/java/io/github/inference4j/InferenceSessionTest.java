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

package io.github.inference4j;

import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.session.SessionOptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.*;

class InferenceSessionTest {

    @TempDir
    Path tempDir;

    @Test
    void create_throwsModelLoadExceptionForNonexistentFile() {
        Path badPath = Path.of("/nonexistent/model.onnx");

        assertThatThrownBy(() -> InferenceSession.create(badPath))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains("/nonexistent/model.onnx"));
    }

    @Test
    void create_throwsModelLoadExceptionForInvalidFile() throws IOException {
        Path tempFile = tempDir.resolve("not-a-model.onnx");
        Files.writeString(tempFile, "this is not an onnx model");

        assertThatThrownBy(() -> InferenceSession.create(tempFile))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains(tempFile.toString()));
    }

    @Test
    void create_withSessionOptions_throwsForNonexistentFile() {
        Path badPath = Path.of("/nonexistent/model.onnx");
        SessionOptions options = SessionOptions.defaults();

        assertThatThrownBy(() -> InferenceSession.create(badPath, options))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains("/nonexistent/model.onnx"));
    }

    @Test
    void create_withSessionOptions_throwsForInvalidFile() throws IOException {
        Path tempFile = tempDir.resolve("bad-model.onnx");
        Files.writeString(tempFile, "garbage data");
        SessionOptions options = SessionOptions.defaults();

        assertThatThrownBy(() -> InferenceSession.create(tempFile, options))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains(tempFile.toString()));
    }

    @Test
    void create_withConfigurer_throwsForNonexistentFile() {
        Path badPath = Path.of("/nonexistent/model.onnx");

        assertThatThrownBy(() -> InferenceSession.create(badPath, opts -> { }))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains("/nonexistent/model.onnx"));
    }

    @Test
    void create_withConfigurer_throwsForInvalidFile() throws IOException {
        Path tempFile = tempDir.resolve("invalid.onnx");
        Files.writeString(tempFile, "not valid onnx");

        assertThatThrownBy(() -> InferenceSession.create(tempFile, opts -> { }))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains(tempFile.toString()));
    }

    @Test
    void create_modelLoadExceptionHasCause() {
        Path badPath = Path.of("/nonexistent/model.onnx");

        assertThatThrownBy(() -> InferenceSession.create(badPath))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getCause()).isNotNull());
    }

    @Test
    void create_withCustomSessionOptions_throwsForInvalidFile() throws IOException {
        Path tempFile = tempDir.resolve("custom-opts.onnx");
        Files.writeString(tempFile, "not onnx");

        SessionOptions options = SessionOptions.builder()
                .intraOpNumThreads(2)
                .interOpNumThreads(1)
                .build();

        assertThatThrownBy(() -> InferenceSession.create(tempFile, options))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains(tempFile.toString()));
    }

    @Test
    void runBooleanInputAndOutputRoundTrip() throws URISyntaxException {
        // test-bool-not.onnx: y = Not(x), x and y are BOOL [batch, n]
        Path model = Path.of(Objects.requireNonNull(
                InferenceSessionTest.class.getResource("/test-bool-not.onnx")).toURI());

        try (InferenceSession session = InferenceSession.create(model)) {
            assertThat(session.inputType("x")).isEqualTo(TensorType.BOOL);

            Map<String, Tensor> outputs = session.run(Map.of("x",
                    Tensor.fromBooleans(new boolean[]{true, false, false, true, true, false},
                            new long[]{2, 3})));

            Tensor y = outputs.get("y");
            assertThat(y.type()).isEqualTo(TensorType.BOOL);
            assertThat(y.shape()).isEqualTo(new long[]{2, 3});
            assertThat(y.toBooleans()).containsExactly(false, true, true, false, false, true);
        }
    }

    @Test
    void closeLetsARunInProgressFinish() throws Exception {
        InferenceSession session = InferenceSession.create(notModel());
        boolean[] input = pattern(3, 50_000_000);                 // large enough that the run takes a while
        Tensor tensor = Tensor.fromBooleans(input, new long[]{1, input.length});
        ExecutorService executor = Executors.newSingleThreadExecutor();
        CountDownLatch starting = new CountDownLatch(1);
        try {
            Future<boolean[]> run = executor.submit(() -> {
                starting.countDown();
                return session.run(Map.of("x", tensor)).get("y").toBooleans();
            });
            assertThat(starting.await(5, TimeUnit.SECONDS)).isTrue();
            Thread.sleep(20);                                     // let the run get under way
            session.close();

            boolean[] output;
            try {
                output = run.get(30, TimeUnit.SECONDS);
            } catch (ExecutionException e) {
                // On a slow machine close() can win the race before the run starts; that must fail cleanly.
                // Without the close lock, the run fails inside ONNX Runtime with a different message.
                assertThat(e.getCause()).isInstanceOf(IllegalStateException.class)
                        .hasMessage("The inference session was closed");
                return;
            }
            assertThat(output).hasSize(input.length);
            assertThat(output[0]).isNotEqualTo(input[0]);
            assertThat(output[input.length - 1]).isNotEqualTo(input[input.length - 1]);
        } finally {
            executor.shutdownNow();
        }
    }

    private static Path notModel() throws URISyntaxException {
        return Path.of(Objects.requireNonNull(InferenceSessionTest.class.getResource("/test-bool-not.onnx")).toURI());
    }

    private static boolean[] pattern(int seed, int length) {
        boolean[] values = new boolean[length];
        for (int i = 0; i < length; i++) {
            values[i] = ((i * 31 + seed) & 4) != 0;
        }
        return values;
    }

    @Test
    void concurrentRunsEachGetTheirOwnResult() throws Exception {
        int threads = 8;
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        try (InferenceSession session = InferenceSession.create(notModel())) {
            List<Future<Boolean>> results = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                int thread = t;
                results.add(executor.submit(() -> {
                    for (int i = 0; i < 300; i++) {
                        int length = 16 + ((thread * 300 + i) % 64) * 16;   // varied sizes exercise the pool
                        boolean[] input = pattern(thread * 1000 + i, length);
                        boolean[] output = session.run(Map.of("x", Tensor.fromBooleans(input, new long[]{1, length})))
                                .get("y").toBooleans();
                        for (int k = 0; k < length; k++) {
                            if (output[k] == input[k]) {
                                return false;
                            }
                        }
                    }
                    return true;
                }));
            }
            for (Future<Boolean> result : results) {
                assertThat(result.get()).as("every run returns Not(its own input)").isTrue();
            }
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void runAfterCloseFailsCleanly() throws Exception {
        InferenceSession session = InferenceSession.create(notModel());
        session.close();
        session.close();                                     // closing twice is harmless

        assertThatThrownBy(() -> session.run(Map.of("x", Tensor.fromBooleans(new boolean[]{true}, new long[]{1, 1}))))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("closed");
    }

    @Test
    void closeDuringConcurrentRunsEndsThemCleanly() throws Exception {
        int threads = 8;
        InferenceSession session = InferenceSession.create(notModel());
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        CountDownLatch running = new CountDownLatch(threads);
        try {
            List<Future<Throwable>> outcomes = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                outcomes.add(executor.submit(() -> {
                    running.countDown();
                    boolean[] input = pattern(7, 100_000);
                    try {
                        while (true) {
                            boolean[] output = session.run(Map.of("x", Tensor.fromBooleans(input, new long[]{1, input.length})))
                                    .get("y").toBooleans();
                            if (output[0] == input[0]) {
                                return new AssertionError("wrong result");
                            }
                        }
                    } catch (Throwable e) {
                        return e;
                    }
                }));
            }
            assertThat(running.await(5, TimeUnit.SECONDS)).isTrue();
            session.close();

            for (Future<Throwable> outcome : outcomes) {
                assertThat(outcome.get(10, TimeUnit.SECONDS)).isInstanceOf(IllegalStateException.class);
            }
        } finally {
            executor.shutdownNow();
        }
    }
}
