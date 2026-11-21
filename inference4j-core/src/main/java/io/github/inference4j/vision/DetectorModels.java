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
import io.github.inference4j.preprocessing.image.Labels;
import io.github.inference4j.preprocessing.text.ModelConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Builder rules shared by the object detectors: where labels come from and which input size is used.
 */
final class DetectorModels {

    static final int DEFAULT_INPUT_SIZE = 640;
    static final int DEFAULT_MAX_DETECTIONS = 300;

    private DetectorModels() {
    }

    /**
     * Picks the labels, first match wins: labels set on the builder; COCO for a preset; {@code labels.txt} in the
     * model directory; {@code id2label} from its {@code config.json}; COCO.
     *
     * @param explicit labels set on the builder, or {@code null}
     * @param dir      the model directory, or {@code null} when the session was supplied directly
     * @param preset   whether the model is one of the wrapper's presets, which use COCO names
     */
    static Labels labels(Labels explicit, Path dir, boolean preset) {
        if (explicit != null) {
            return explicit;
        }
        if (preset || dir == null) {
            return Labels.coco();
        }
        Path labelsFile = dir.resolve("labels.txt");
        if (Files.exists(labelsFile)) {
            return Labels.fromFile(labelsFile);
        }
        Path configFile = dir.resolve("config.json");
        if (Files.exists(configFile)) {
            ModelConfig config = ModelConfig.fromFile(configFile);
            if (config.numLabels() > 0) {
                List<String> names = new ArrayList<>(config.numLabels());
                for (int i = 0; i < config.numLabels(); i++) {
                    names.add(config.label(i));
                }
                return Labels.of(names);
            }
        }
        return Labels.coco();
    }

    /**
     * Picks the input size. A model with a fixed square input uses that size, and a different explicit size fails;
     * a model with a flexible input uses the explicit size, or {@value #DEFAULT_INPUT_SIZE}.
     *
     * @param explicit   the size set on the builder, or {@code null}
     * @param fixedSize  the model's fixed input side, or a value below 1 when it is flexible or unknown
     * @param modelId    the model, for the error message
     * @throws ModelLoadException if the explicit size differs from the model's fixed size
     */
    static int inputSize(Integer explicit, int fixedSize, String modelId) {
        if (fixedSize > 0) {
            if (explicit != null && explicit != fixedSize) {
                throw new ModelLoadException("Model '" + modelId + "' has a fixed " + fixedSize + " × " + fixedSize
                        + " input; inputSize(" + explicit + ") cannot be used with it");
            }
            return fixedSize;
        }
        return explicit != null ? explicit : DEFAULT_INPUT_SIZE;
    }

    /**
     * The fixed side of a square image input of shape {@code [N, C, H, W]} or {@code [N, H, W, C]}, or -1 when the
     * height or width is dynamic.
     */
    static int fixedSize(long[] shape) {
        if (shape.length != 4) {
            return -1;
        }
        boolean channelsFirst = shape[1] == 3 || shape[1] == 1 || !(shape[3] == 3 || shape[3] == 1);
        long height = channelsFirst ? shape[2] : shape[1];
        long width = channelsFirst ? shape[3] : shape[2];
        return height > 0 && height == width ? (int) height : -1;
    }

    static int maxDetections(int value) {
        if (value < 1) {
            throw new IllegalArgumentException("maxDetections must be at least 1, got " + value);
        }
        return value;
    }
}
