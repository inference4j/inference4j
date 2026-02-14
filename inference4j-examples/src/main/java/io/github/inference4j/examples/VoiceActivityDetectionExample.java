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

import io.github.inference4j.audio.SileroVadDetector;
import io.github.inference4j.audio.VoiceSegment;

import java.nio.file.Path;
import java.util.List;

/**
 * Demonstrates Voice Activity Detection (VAD) with Silero VAD.
 *
 * <p>Silero VAD identifies speech segments in audio, which is useful for:
 * <ul>
 *   <li>Preprocessing audio before speech-to-text to reduce processing time</li>
 *   <li>Detecting when a user starts/stops speaking in real-time applications</li>
 *   <li>Extracting speech segments from long recordings</li>
 * </ul>
 *
 * <p>Requires Silero VAD ONNX model — see inference4j-examples/README.md for setup.
 *
 * <p>Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.VoiceActivityDetectionExample
 */
public class VoiceActivityDetectionExample {

    public static void main(String[] args) {
        String audioPath = "assets/audio/sample.wav";

        System.out.println("=== Silero Voice Activity Detection ===");
        System.out.println();

        // Print audio file info
        Path audioFile = Path.of(audioPath);
        System.out.println("Audio file: " + audioFile.toAbsolutePath());
        System.out.println("File exists: " + java.nio.file.Files.exists(audioFile));
        try {
            if (java.nio.file.Files.exists(audioFile)) {
                System.out.println("File size: " + java.nio.file.Files.size(audioFile) + " bytes");
            }
        } catch (java.io.IOException e) {
            System.out.println("Could not read file size: " + e.getMessage());
        }
        System.out.println();

        try (SileroVadDetector vad = SileroVadDetector.builder().build()) {
            System.out.println("Silero VAD loaded successfully.");
            System.out.println();

            // Detect voice segments
            long start = System.currentTimeMillis();
            List<VoiceSegment> segments = vad.detect(audioFile);
            long elapsed = System.currentTimeMillis() - start;

            // Also get probabilities to check max
            float[] probs = vad.probabilities(audioFile);
            float maxProb = 0;
            for (float p : probs) {
                if (p > maxProb) maxProb = p;
            }
            System.out.printf("Max probability across all frames: %.4f%n", maxProb);
            System.out.println();

            System.out.printf("Detected %d speech segment(s):%n", segments.size());
            System.out.println();

            for (int i = 0; i < segments.size(); i++) {
                VoiceSegment segment = segments.get(i);
                System.out.printf("  Segment %d: %.2fs - %.2fs (duration: %.2fs, confidence: %.2f)%n",
                        i + 1,
                        segment.start(),
                        segment.end(),
                        segment.duration(),
                        segment.confidence());
            }

            System.out.println();
            System.out.printf("Processing time: %d ms%n", elapsed);
        }

        System.out.println();
        System.out.println("=== Custom Threshold Example ===");
        System.out.println();

        // Example with custom configuration
        try (SileroVadDetector vad = SileroVadDetector.builder()
                .threshold(0.7f)           // Higher threshold = more conservative detection
                .minSpeechDuration(0.3f)   // Ignore very short utterances
                .minSilenceDuration(0.15f) // Require more silence to end a segment
                .build()) {

            List<VoiceSegment> segments = vad.detect(Path.of(audioPath));

            System.out.printf("With higher threshold (0.7): %d segment(s)%n", segments.size());
            for (VoiceSegment segment : segments) {
                System.out.printf("  %.2fs - %.2fs (confidence: %.2f)%n",
                        segment.start(), segment.end(), segment.confidence());
            }
        }

        System.out.println();
        System.out.println("=== Probability Analysis Example ===");
        System.out.println();

        // Get raw probabilities for visualization
        try (SileroVadDetector vad = SileroVadDetector.builder().build()) {
            float[] probabilities = vad.probabilities(Path.of(audioPath));

            System.out.printf("Total frames analyzed: %d%n", probabilities.length);
            System.out.println();

            // Show first 20 frame probabilities
            System.out.println("First 20 frame probabilities:");
            for (int i = 0; i < Math.min(20, probabilities.length); i++) {
                float prob = probabilities[i];
                String bar = "=".repeat((int) (prob * 20));
                System.out.printf("  Frame %2d: %.3f |%-20s|%n", i, prob, bar);
            }
        }
    }
}

