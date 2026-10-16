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

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.session.SessionConfigurer;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Text line recognizer backed by PaddleOCR's CTC recognition models.
 *
 * <p>Reads one line of text per image, typically a crop of a {@link TextRegion} made with {@link QuadCrop}. The
 * line is resized to the model's 48-pixel height, keeping its aspect ratio, and decoded greedily: the most likely
 * character at each step, with repeats and blanks removed.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * try (var recognizer = PaddleTextRecognizer.small().build()) {        // PP-OCRv6 small, 21 MB (default)
 *     RecognizedText line = recognizer.recognize(lineImage);
 * }
 * try (var recognizer = PaddleTextRecognizer.tiny().build()) { ... }   // PP-OCRv6 tiny, 4.5 MB
 * }</pre>
 *
 * <p>The small model's dictionary has 18,708 characters: Chinese, Japanese, Latin scripts, symbols and more. The
 * tiny model's has 6,904. The model directory needs {@code model.onnx} and PaddleOCR's {@code inference.yml},
 * whose {@code PostProcess.character_dict} lists the characters.
 *
 * @see OcrPipeline
 */
public class PaddleTextRecognizer implements TextRecognizer {

    private static final String SMALL_MODEL_ID = "inference4j/pp-ocrv6-small-rec";
    private static final String TINY_MODEL_ID = "inference4j/pp-ocrv6-tiny-rec";
    private static final int HEIGHT = 48;
    private static final int MIN_WIDTH = 320;
    private static final int MAX_WIDTH = 3200;

    private final InferenceSession session;
    private final String inputName;
    private final String[] characters;   // index 0 is the CTC blank

    private PaddleTextRecognizer(InferenceSession session, String[] characters) {
        this.session = session;
        this.inputName = session.inputNames().iterator().next();
        this.characters = characters;
    }

    /** PP-OCRv6 small recognition model (21 MB), the default. */
    public static Builder small() {
        return new Builder().modelId(SMALL_MODEL_ID);
    }

    /** PP-OCRv6 tiny recognition model (4.5 MB). */
    public static Builder tiny() {
        return new Builder().modelId(TINY_MODEL_ID);
    }

    /** A builder for the default model; set {@link Builder#modelId(String)} for another PaddleOCR recognizer. */
    public static Builder builder() {
        return small();
    }

    @Override
    public RecognizedText recognize(BufferedImage lineImage) {
        float[][] probabilities = probabilities(lineImage);
        StringBuilder text = new StringBuilder();
        double confidenceSum = 0;
        int kept = 0;
        int previous = -1;
        for (float[] step : probabilities) {
            int best = 0;
            for (int c = 1; c < step.length; c++) {
                if (step[c] > step[best]) {
                    best = c;
                }
            }
            if (best != previous && best != 0) {
                text.append(characters[best]);
                confidenceSum += step[best];
                kept++;
            }
            previous = best;
        }
        return new RecognizedText(text.toString(), kept == 0 ? 0f : (float) (confidenceSum / kept));
    }

    @Override
    public void close() {
        session.close();
    }

    /** Per-step class probabilities for one line image, as PaddleOCR's {@code OCRReisizeNormImg} prepares it. */
    private float[][] probabilities(BufferedImage lineImage) {
        int width = lineImage.getWidth();
        int height = lineImage.getHeight();
        double ratio = (double) width / height;
        int inputWidth = (int) (HEIGHT * Math.max((double) MIN_WIDTH / HEIGHT, ratio));
        int resizedWidth;
        if (inputWidth > MAX_WIDTH) {
            inputWidth = MAX_WIDTH;
            resizedWidth = MAX_WIDTH;
        } else {
            resizedWidth = (int) Math.min(Math.ceil(HEIGHT * ratio), inputWidth);
        }
        resizedWidth = Math.max(1, resizedWidth);
        int[] resized = PaddleImages.resize(PaddleImages.pixels(lineImage), width, height, resizedWidth, HEIGHT);

        int plane = HEIGHT * inputWidth;
        float[] input = new float[3 * plane];   // zeros pad the right side
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < resizedWidth; x++) {
                int rgb = resized[y * resizedWidth + x];
                int i = y * inputWidth + x;
                // RGB, as paddleocr's predictor feeds it (inference.yml's "BGR" is not what reaches the model)
                input[i] = ((rgb >> 16) & 0xFF) / 127.5f - 1f;
                input[plane + i] = ((rgb >> 8) & 0xFF) / 127.5f - 1f;
                input[2 * plane + i] = (rgb & 0xFF) / 127.5f - 1f;
            }
        }
        Tensor output = session.run(Map.of(inputName,
                Tensor.fromFloats(input, new long[]{1, 3, HEIGHT, inputWidth}))).values().iterator().next();
        long[] shape = output.shape();
        int steps = (int) shape[1];
        int classes = (int) shape[2];
        float[] flat = output.toFloats();
        float[][] probabilities = new float[steps][classes];
        for (int t = 0; t < steps; t++) {
            System.arraycopy(flat, t * classes, probabilities[t], 0, classes);
        }
        return probabilities;
    }

    public static class Builder {
        private ModelSource modelSource;
        private String modelId = SMALL_MODEL_ID;
        private SessionConfigurer sessionConfigurer;

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        public PaddleTextRecognizer build() {
            ModelSource source = modelSource != null ? modelSource : HuggingFaceModelSource.defaultInstance();
            Path dir = source.resolve(modelId, List.of("model.onnx", "inference.yml"));
            Path modelPath = dir.resolve("model.onnx");
            Path configPath = dir.resolve("inference.yml");
            for (Path file : List.of(modelPath, configPath)) {
                if (!Files.exists(file)) {
                    throw new ModelSourceException("Model file not found: " + file);
                }
            }
            String[] characters = PaddleDictionary.read(configPath);
            InferenceSession session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            PaddleTextRecognizer recognizer = new PaddleTextRecognizer(session, characters);
            try {
                int classes = recognizer.probabilities(new BufferedImage(MIN_WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB))[0].length;
                if (classes != characters.length) {
                    throw new ModelLoadException("Model predicts " + classes + " classes but inference.yml defines "
                            + characters.length + " (blank + dictionary + space)");
                }
            } catch (RuntimeException e) {
                session.close();
                throw e;
            }
            return recognizer;
        }
    }

    /**
     * Reads the character list from PaddleOCR's {@code inference.yml}: {@code PostProcess.character_dict}, one
     * character per list item. Handles the YAML that PaddleOCR writes (plain and single- or double-quoted
     * scalars) without a YAML library.
     */
    static final class PaddleDictionary {

        private PaddleDictionary() {
        }

        static String[] read(Path inferenceYml) {
            List<String> lines;
            try {
                lines = Files.readAllLines(inferenceYml, StandardCharsets.UTF_8);
            } catch (IOException e) {
                throw new ModelSourceException("Failed to read " + inferenceYml + ": " + e.getMessage(), e);
            }
            List<String> dictionary = new ArrayList<>();
            boolean useSpace = true;
            boolean inPostProcess = false;
            boolean inDictionary = false;
            for (String line : lines) {
                if (!line.startsWith(" ") && !line.isBlank()) {
                    inPostProcess = line.startsWith("PostProcess:");
                    inDictionary = false;
                    continue;
                }
                if (!inPostProcess) {
                    continue;
                }
                if (line.startsWith("  character_dict:")) {
                    inDictionary = true;
                    continue;
                }
                if (inDictionary && line.startsWith("  - ")) {
                    dictionary.add(scalar(line.substring(4)));
                    continue;
                }
                inDictionary = false;
                if (line.startsWith("  use_space_char:")) {
                    useSpace = !line.substring("  use_space_char:".length()).trim().equalsIgnoreCase("false");
                }
            }
            if (dictionary.isEmpty()) {
                throw new ModelLoadException("No PostProcess.character_dict in " + inferenceYml);
            }
            List<String> characters = new ArrayList<>(dictionary.size() + 2);
            characters.add("");   // CTC blank
            characters.addAll(dictionary);
            if (useSpace) {
                characters.add(" ");
            }
            return characters.toArray(new String[0]);
        }

        private static String scalar(String raw) {
            // Only ASCII spaces: a dictionary entry may itself be Unicode whitespace, such as U+3000
            int end = raw.length();
            while (end > 0 && (raw.charAt(end - 1) == ' ' || raw.charAt(end - 1) == '\r')) {
                end--;
            }
            String value = raw.substring(0, end);
            if (value.length() >= 2 && value.startsWith("'") && value.endsWith("'")) {
                return value.substring(1, value.length() - 1).replace("''", "'");
            }
            if (value.length() >= 2 && value.startsWith("\"") && value.endsWith("\"")) {
                return unescape(value.substring(1, value.length() - 1));
            }
            return value;
        }

        private static String unescape(String s) {
            StringBuilder out = new StringBuilder();
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c != '\\' || i + 1 >= s.length()) {
                    out.append(c);
                    continue;
                }
                char e = s.charAt(++i);
                switch (e) {
                    case 'n' -> out.append('\n');
                    case 't' -> out.append('\t');
                    case 'r' -> out.append('\r');
                    case '0' -> out.append('\0');
                    case 'x' -> {
                        out.append((char) Integer.parseInt(s.substring(i + 1, i + 3), 16));
                        i += 2;
                    }
                    case 'u' -> {
                        out.append((char) Integer.parseInt(s.substring(i + 1, i + 5), 16));
                        i += 4;
                    }
                    case 'U' -> {
                        out.appendCodePoint(Integer.parseInt(s.substring(i + 1, i + 9), 16));
                        i += 8;
                    }
                    default -> out.append(e);   // \\, \", \/ and others stand for themselves
                }
            }
            return out.toString();
        }
    }
}
