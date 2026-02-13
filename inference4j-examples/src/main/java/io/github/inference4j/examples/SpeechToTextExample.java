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
        String modelDir = "inference4j-examples/models/wav2vec2-base-960h";
        String audioPath = "inference4j-examples/audio/sample.wav";

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
