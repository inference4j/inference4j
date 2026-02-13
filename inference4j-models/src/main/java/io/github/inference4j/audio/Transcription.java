package io.github.inference4j.audio;

/**
 * Result of a speech-to-text transcription.
 *
 * @param text the transcribed text
 */
public record Transcription(String text) {
}
