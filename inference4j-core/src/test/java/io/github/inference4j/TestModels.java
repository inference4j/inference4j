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

import io.github.inference4j.model.ModelSource;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

/**
 * Model directories for unit tests, built from the tiny ONNX fixtures in {@code src/test/resources/fixtures}
 * (see {@code generate.py} there), so tests load wrappers through their public builders.
 */
public final class TestModels {

    private TestModels() {
    }

    /**
     * Installs a fixture as {@code model.onnx} in {@code dir} and returns a source that resolves every model id to
     * {@code dir}.
     *
     * @param dir     the model directory, usually a {@code @TempDir}
     * @param fixture the fixture name, without {@code .onnx}
     * @return a source for {@code dir}
     */
    public static ModelSource model(Path dir, String fixture) {
        install(dir, fixture, "model.onnx");
        return source(dir);
    }

    /** A source that resolves every model id to {@code dir}. */
    public static ModelSource source(Path dir) {
        return id -> dir;
    }

    /**
     * Copies a fixture into {@code dir} under {@code fileName}.
     *
     * @param dir      the model directory
     * @param fixture  the fixture name, without {@code .onnx}
     * @param fileName the file name in {@code dir}
     * @return the installed file
     */
    public static Path install(Path dir, String fixture, String fileName) {
        String resource = "/fixtures/" + fixture + ".onnx";
        try (InputStream in = TestModels.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalArgumentException("No fixture " + resource);
            }
            Path target = dir.resolve(fileName);
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            return target;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Writes the CLIP tokenizer files: a {@code vocab.json} with {@code r</w>}, {@code g</w>} and {@code b</w>} as
     * ids 0, 1 and 2, the start and end tokens as 3 and 4, and a {@code merges.txt} with no merges.
     */
    public static void clipVocabulary(Path dir) {
        write(dir.resolve("vocab.json"),
                "{\"r</w>\": 0, \"g</w>\": 1, \"b</w>\": 2, \"<|startoftext|>\": 3, \"<|endoftext|>\": 4}");
        write(dir.resolve("merges.txt"), "#version: 0.2\n");
    }

    /** Writes a Wav2Vec2 {@code vocab.json}: {@code <pad>} (blank) 0, {@code a} 1, {@code b} 2, {@code c} 3, {@code |} 4. */
    public static Path wav2Vec2Vocabulary(Path dir) {
        return write(dir.resolve("vocab.json"), "{\"<pad>\": 0, \"a\": 1, \"b\": 2, \"c\": 3, \"|\": 4}");
    }

    /** The labels {@code L0} to {@code L(n-1)}. */
    public static String[] labels(int n) {
        List<String> labels = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            labels.add("L" + i);
        }
        return labels.toArray(new String[0]);
    }

    private static Path write(Path file, String content) {
        try {
            return Files.writeString(file, content);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
