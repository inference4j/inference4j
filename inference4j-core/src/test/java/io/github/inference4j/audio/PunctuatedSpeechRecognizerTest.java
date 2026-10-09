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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PunctuatedSpeechRecognizerTest {

    private StubRecognizer recognizer;
    private PunctCapSegModel model;
    private PunctuatedSpeechRecognizer punctuated;

    @BeforeEach
    void setUp() {
        recognizer = new StubRecognizer();
        model = mock(PunctCapSegModel.class);
        punctuated = new PunctuatedSpeechRecognizer(recognizer, model);
    }

    @Test
    void formatsTranscriptFromPath() {
        recognizer.result = new Transcription("MARIE CURIE MOVED TO PARIS SHE WON");
        when(model.infer("MARIE CURIE MOVED TO PARIS SHE WON"))
                .thenReturn(List.of("Marie Curie moved to Paris.", "She won."));

        Transcription result = punctuated.transcribe(Path.of("speech.wav"));

        assertThat(result.text()).isEqualTo("Marie Curie moved to Paris. She won.");
        assertThat(recognizer.lastPath).isEqualTo(Path.of("speech.wav"));
    }

    @Test
    void formatsTranscriptFromSamples() {
        recognizer.result = new Transcription("HELLO THERE");
        when(model.infer("HELLO THERE")).thenReturn(List.of("Hello there."));

        Transcription result = punctuated.transcribe(new float[16_000], 16_000);

        assertThat(result.text()).isEqualTo("Hello there.");
        assertThat(recognizer.lastSampleRate).isEqualTo(16_000);
    }

    @Test
    void preservesSegments() {
        List<Transcription.Segment> segments = List.of(new Transcription.Segment("HELLO", 0f, 1f));
        recognizer.result = new Transcription("HELLO", segments);
        when(model.infer("HELLO")).thenReturn(List.of("Hello."));

        Transcription result = punctuated.transcribe(Path.of("speech.wav"));

        assertThat(result.segments()).isEqualTo(segments);
    }

    @Test
    void blankTranscriptSkipsModel() {
        recognizer.result = new Transcription("   ");

        Transcription result = punctuated.transcribe(Path.of("silence.wav"));

        assertThat(result.text()).isEqualTo("   ");
        verify(model, never()).infer(anyString());
    }

    @Test
    void closeClosesRecognizerAndModel() {
        punctuated.close();

        assertThat(recognizer.closed).isTrue();
        verify(model).close();
    }

    @Test
    void closeClosesModelEvenWhenRecognizerCloseFails() {
        recognizer.failOnClose = true;

        assertThatThrownBy(punctuated::close).isInstanceOf(IllegalStateException.class);
        verify(model).close();
    }

    private static final class StubRecognizer implements SpeechRecognizer {

        private Transcription result;
        private Path lastPath;
        private int lastSampleRate;
        private boolean closed;
        private boolean failOnClose;

        @Override
        public Transcription transcribe(Path audioPath) {
            lastPath = audioPath;
            return result;
        }

        @Override
        public Transcription transcribe(float[] audioData, int sampleRate) {
            lastSampleRate = sampleRate;
            return result;
        }

        @Override
        public void close() {
            closed = true;
            if (failOnClose) {
                throw new IllegalStateException("close failed");
            }
        }
    }
}
