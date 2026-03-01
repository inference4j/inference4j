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

package io.github.inference4j.generation;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class TokenStreamerTest {

    @Test
    void noStopSequences_flushesImmediately() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of(), streamed::add);

        streamer.accept("Hello");
        streamer.accept(" world");

        assertThat(streamed).isEqualTo(List.of("Hello", " world"));
        assertThat(streamer.isStopped()).isFalse();
        assertThat(streamer.getText()).isEqualTo("Hello world");
    }

    @Test
    void stopSequence_detected_trimmed_listenerNeverSeesIt() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("<|end|>"), streamed::add);

        streamer.accept("Hello");
        streamer.accept(" world");
        streamer.accept("<|end|>");

        assertThat(streamer.isStopped()).isTrue();
        assertThat(streamer.getText()).isEqualTo("Hello world");
        // Listener should never receive the stop sequence
        String joined = String.join("", streamed);
        assertThat(joined).isEqualTo("Hello world");
    }

    @Test
    void stopSequence_splitAcrossFragments() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("world"), streamed::add);

        streamer.accept("Hello wor");
        streamer.accept("ld");

        assertThat(streamer.isStopped()).isTrue();
        assertThat(streamer.getText()).isEqualTo("Hello ");
        String joined = String.join("", streamed);
        assertThat(joined).isEqualTo("Hello ");
    }

    @Test
    void partialMatch_notPrematurelyFlushed() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("world"), streamed::add);

        streamer.accept("Hello wor");
        // "wor" is a partial match for "world" — should be held in buffer
        assertThat(streamer.isStopped()).isFalse();

        // Now the rest doesn't form the stop sequence
        streamer.accept("k!");
        assertThat(streamer.isStopped()).isFalse();

        streamer.flush();
        assertThat(streamer.getText()).isEqualTo("Hello work!");
        String joined = String.join("", streamed);
        assertThat(joined).isEqualTo("Hello work!");
    }

    @Test
    void multipleStopSequences_firstMatchWins() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("<|end|>", "\n\n"), streamed::add);

        streamer.accept("Hello\n");
        streamer.accept("\nmore text");

        assertThat(streamer.isStopped()).isTrue();
        assertThat(streamer.getText()).isEqualTo("Hello");
    }

    @Test
    void flush_releasesHeldBuffer_whenNoStopMatched() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("xyz"), streamed::add);

        streamer.accept("abc");
        // "abc" fits within maxStopLength (3), so it's held
        assertThat(streamer.getText()).isEqualTo("");

        streamer.flush();
        assertThat(streamer.getText()).isEqualTo("abc");
        String joined = String.join("", streamed);
        assertThat(joined).isEqualTo("abc");
    }

    @Test
    void getText_accumulatesAllFlushedText() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("end"), streamed::add);

        streamer.accept("first ");
        streamer.accept("second ");
        streamer.accept("third");

        assertThat(streamer.isStopped()).isFalse();
        streamer.flush();
        assertThat(streamer.getText()).isEqualTo("first second third");
    }

    @Test
    void stopSequenceAtVeryStart() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("Hello"), streamed::add);

        streamer.accept("Hello world");

        assertThat(streamer.isStopped()).isTrue();
        assertThat(streamer.getText()).isEqualTo("");
        assertThat(streamed).isEmpty();
    }

    @Test
    void stopSequenceInMiddleOfFragment() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("STOP"), streamed::add);

        streamer.accept("beforeSTOPafter");

        assertThat(streamer.isStopped()).isTrue();
        assertThat(streamer.getText()).isEqualTo("before");
        String joined = String.join("", streamed);
        assertThat(joined).isEqualTo("before");
    }

    @Test
    void emptyFragmentsAreHandled() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of(), streamed::add);

        streamer.accept("");
        streamer.accept("Hello");
        streamer.accept("");

        assertThat(streamer.getText()).isEqualTo("Hello");
    }

    @Test
    void flushAfterStopIsNoop() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("end"), streamed::add);

        streamer.accept("the end");

        assertThat(streamer.isStopped()).isTrue();
        String textBefore = streamer.getText();

        streamer.flush();
        assertThat(streamer.getText()).isEqualTo(textBefore);
    }
}
