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

package io.github.inference4j.audio;

import io.github.inference4j.nlp.PunctCapSegModel;

import java.nio.file.Path;

/**
 * A {@link SpeechRecognizer} that adds punctuation, casing and sentence boundaries to another
 * recognizer's transcript.
 *
 * <p>Recognizers such as {@link Wav2Vec2Recognizer} produce upper-case text without punctuation
 * ({@code "MARIE CURIE MOVED TO PARIS"}). Wrapping them restores readable text
 * ({@code "Marie Curie moved to Paris."}) for display, embeddings, or cased models such as NER:
 *
 * <pre>{@code
 * try (SpeechRecognizer recognizer = new PunctuatedSpeechRecognizer(
 *         Wav2Vec2Recognizer.builder().build(),
 *         PunctCapSegModel.builder().build())) {
 *     String text = recognizer.transcribe(Path.of("speech.wav")).text();
 * }
 * }</pre>
 *
 * <p>Only {@link Transcription#text()} is formatted; segments are passed through unchanged.
 * Closing this recognizer closes both the wrapped recognizer and the model.
 *
 * <p>The model processes at most 256 tokens (roughly 200 words) per transcript; longer
 * transcripts are truncated. Transcribe long audio in shorter pieces, for example per voice
 * activity segment.
 */
public class PunctuatedSpeechRecognizer implements SpeechRecognizer {

    private final SpeechRecognizer recognizer;
    private final PunctCapSegModel model;

    public PunctuatedSpeechRecognizer(SpeechRecognizer recognizer, PunctCapSegModel model) {
        this.recognizer = recognizer;
        this.model = model;
    }

    @Override
    public Transcription transcribe(Path audioPath) {
        return punctuate(recognizer.transcribe(audioPath));
    }

    @Override
    public Transcription transcribe(float[] audioData, int sampleRate) {
        return punctuate(recognizer.transcribe(audioData, sampleRate));
    }

    private Transcription punctuate(Transcription transcription) {
        if (transcription.text() == null || transcription.text().isBlank()) {
            return transcription;
        }
        String text = String.join(" ", model.infer(transcription.text()));
        return new Transcription(text, transcription.segments());
    }

    @Override
    public void close() {
        try {
            recognizer.close();
        } finally {
            model.close();
        }
    }
}
