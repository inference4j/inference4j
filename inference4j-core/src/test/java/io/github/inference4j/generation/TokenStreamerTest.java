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

import static org.junit.jupiter.api.Assertions.*;

class TokenStreamerTest {

    @Test
    void noStopSequences_flushesImmediately() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of(), streamed::add);

        streamer.accept("Hello");
        streamer.accept(" world");

        assertEquals(List.of("Hello", " world"), streamed);
        assertFalse(streamer.isStopped());
        assertEquals("Hello world", streamer.getText());
    }

    @Test
    void stopSequence_detected_trimmed_listenerNeverSeesIt() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("<|end|>"), streamed::add);

        streamer.accept("Hello");
        streamer.accept(" world");
        streamer.accept("<|end|>");

        assertTrue(streamer.isStopped());
        assertEquals("Hello world", streamer.getText());
        // Listener should never receive the stop sequence
        String joined = String.join("", streamed);
        assertEquals("Hello world", joined);
    }

    @Test
    void stopSequence_splitAcrossFragments() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("world"), streamed::add);

        streamer.accept("Hello wor");
        streamer.accept("ld");

        assertTrue(streamer.isStopped());
        assertEquals("Hello ", streamer.getText());
        String joined = String.join("", streamed);
        assertEquals("Hello ", joined);
    }

    @Test
    void partialMatch_notPrematurelyFlushed() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("world"), streamed::add);

        streamer.accept("Hello wor");
        // "wor" is a partial match for "world" — should be held in buffer
        assertFalse(streamer.isStopped());

        // Now the rest doesn't form the stop sequence
        streamer.accept("k!");
        assertFalse(streamer.isStopped());

        streamer.flush();
        assertEquals("Hello work!", streamer.getText());
        String joined = String.join("", streamed);
        assertEquals("Hello work!", joined);
    }

    @Test
    void multipleStopSequences_firstMatchWins() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("<|end|>", "\n\n"), streamed::add);

        streamer.accept("Hello\n");
        streamer.accept("\nmore text");

        assertTrue(streamer.isStopped());
        assertEquals("Hello", streamer.getText());
    }

    @Test
    void flush_releasesHeldBuffer_whenNoStopMatched() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("xyz"), streamed::add);

        streamer.accept("abc");
        // "abc" fits within maxStopLength (3), so it's held
        assertEquals("", streamer.getText());

        streamer.flush();
        assertEquals("abc", streamer.getText());
        String joined = String.join("", streamed);
        assertEquals("abc", joined);
    }

    @Test
    void getText_accumulatesAllFlushedText() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("end"), streamed::add);

        streamer.accept("first ");
        streamer.accept("second ");
        streamer.accept("third");

        assertFalse(streamer.isStopped());
        streamer.flush();
        assertEquals("first second third", streamer.getText());
    }

    @Test
    void stopSequenceAtVeryStart() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("Hello"), streamed::add);

        streamer.accept("Hello world");

        assertTrue(streamer.isStopped());
        assertEquals("", streamer.getText());
        assertTrue(streamed.isEmpty());
    }

    @Test
    void stopSequenceInMiddleOfFragment() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("STOP"), streamed::add);

        streamer.accept("beforeSTOPafter");

        assertTrue(streamer.isStopped());
        assertEquals("before", streamer.getText());
        String joined = String.join("", streamed);
        assertEquals("before", joined);
    }

    @Test
    void emptyFragmentsAreHandled() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of(), streamed::add);

        streamer.accept("");
        streamer.accept("Hello");
        streamer.accept("");

        assertEquals("Hello", streamer.getText());
    }

    @Test
    void flushAfterStopIsNoop() {
        List<String> streamed = new ArrayList<>();
        TokenStreamer streamer = new TokenStreamer(Set.of("end"), streamed::add);

        streamer.accept("the end");

        assertTrue(streamer.isStopped());
        String textBefore = streamer.getText();

        streamer.flush();
        assertEquals(textBefore, streamer.getText());
    }
}
