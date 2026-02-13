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

package io.github.inference4j.examples;

import io.github.inference4j.audio.Transcription;
import io.github.inference4j.audio.Wav2Vec2;

import java.nio.file.Path;

/**
 * Demonstrates speech-to-text transcription with Wav2Vec2-CTC.
 *
 * Requires wav2vec2-base-960h ONNX model and a sample audio file — see inference4j-examples/README.md for setup.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SpeechToTextExample
 */
public class SpeechToTextExample {

    public static void main(String[] args) {
        String modelDir = "assets/models/wav2vec2-base-960h";
        String audioPath = "assets/audio/sample.wav";

        System.out.println("=== Wav2Vec2 Speech-to-Text ===");
        try (Wav2Vec2 model = Wav2Vec2.fromPretrained(modelDir)) {
            System.out.println("Wav2Vec2 loaded successfully.");
            System.out.println();

            long start = System.currentTimeMillis();
            Transcription result = model.transcribe(Path.of(audioPath));
            long elapsed = System.currentTimeMillis() - start;

            System.out.println("Transcription: \"" + result.text() + "\"");
            System.out.printf("Time: %d ms%n", elapsed);
        }
    }
}
