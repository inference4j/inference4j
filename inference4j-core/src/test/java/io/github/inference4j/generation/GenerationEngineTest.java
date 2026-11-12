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

import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.InputTooLongException;
import io.github.inference4j.processing.TruncationGuard;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.TokenDecoder;
import io.github.inference4j.tokenizer.TokenStream;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.mockito.InOrder;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

class GenerationEngineTest {

    @Test
    void builder_requiresSession() {
        var builder = GenerationEngine.builder()
                .tokenizer(mock(Tokenizer.class))
                .decoder(mock(TokenDecoder.class))
                .eosTokenId(0);

        assertThatThrownBy(builder::build).isInstanceOf(NullPointerException.class);
    }

    @Test
    void builder_requiresTokenizer() {
        var builder = GenerationEngine.builder()
                .session(mock(GenerativeSession.class))
                .decoder(mock(TokenDecoder.class))
                .eosTokenId(0);

        assertThatThrownBy(builder::build).isInstanceOf(NullPointerException.class);
    }

    @Test
    void builder_requiresDecoder() {
        var builder = GenerationEngine.builder()
                .session(mock(GenerativeSession.class))
                .tokenizer(mock(Tokenizer.class))
                .eosTokenId(0);

        assertThatThrownBy(builder::build).isInstanceOf(NullPointerException.class);
    }

    @Test
    void builder_requiresAtLeastOneEosTokenId() {
        var builder = GenerationEngine.builder()
                .session(mock(GenerativeSession.class))
                .tokenizer(mock(Tokenizer.class))
                .decoder(mock(TokenDecoder.class));

        assertThatThrownBy(builder::build).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void generate_prefillsAndDecodesUntilEos() throws Exception {
        int eosTokenId = 50256;
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);

        when(tokenizer.encode(eq("Hello"), anyInt())).thenReturn(
                new EncodedInput(new long[]{15496}, new long[]{1}, new long[]{0}));

        // prefill returns logits that will greedily select token 318
        when(session.prefill(new long[]{15496})).thenReturn(
                new ForwardResult(logitsForToken(318, 50257)));

        // first decode returns logits selecting token 995
        when(session.decode(318)).thenReturn(
                new ForwardResult(logitsForToken(995, 50257)));

        // second decode returns logits selecting EOS
        when(session.decode(995)).thenReturn(
                new ForwardResult(logitsForToken(eosTokenId, 50257)));

        when(stream.next(318)).thenReturn(" is");
        when(stream.next(995)).thenReturn(" world");

        try (var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(eosTokenId)
                .maxNewTokens(100)
                .build()) {

            GenerationResult result = engine.generate("Hello");

            assertThat(result.text()).isEqualTo(" is world");
            assertThat(result.promptTokens()).isEqualTo(1);
            assertThat(result.generatedTokens()).isEqualTo(2);
            assertThat(result.duration()).isNotNull();
        } catch (Exception e) {
            fail(e);
        }

        verify(session, times(2)).resetCache();   // before prefill and after the last token
        verify(session).prefill(new long[]{15496});
        verify(session).decode(318);
        verify(session).decode(995);
        verify(session).close();
    }

    @Test
    void generate_streamsTokensToListener() {
        int eosTokenId = 0;
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);

        when(tokenizer.encode(eq("Hi"), anyInt())).thenReturn(
                new EncodedInput(new long[]{10}, new long[]{1}, new long[]{0}));

        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(1, 10)));
        when(session.decode(1)).thenReturn(new ForwardResult(logitsForToken(2, 10)));
        when(session.decode(2)).thenReturn(new ForwardResult(logitsForToken(eosTokenId, 10)));

        when(stream.next(1)).thenReturn("A");
        when(stream.next(2)).thenReturn("B");

        List<String> streamed = new ArrayList<>();

        try (var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(eosTokenId)
                .build()) {

            engine.generate("Hi", streamed::add);
        } catch (Exception e) {
            fail(e);
        }

        assertThat(streamed).isEqualTo(List.of("A", "B"));
    }

    @Test
    void generate_respectsMaxNewTokensLimit() {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);

        when(tokenizer.encode(eq("x"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        // Always return token 5 (never EOS)
        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(5, 10)));
        when(session.decode(anyLong())).thenReturn(new ForwardResult(logitsForToken(5, 10)));
        when(stream.next(5)).thenReturn("x");

        try (var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(999)
                .maxNewTokens(3)
                .build()) {

            GenerationResult result = engine.generate("x");

            assertThat(result.text()).isEqualTo("xxx");
            assertThat(result.generatedTokens()).isEqualTo(3);
        } catch (Exception e) {
            fail(e);
        }
    }

    @Test
    void generate_detectsAndTrimsStopSequence() {
        int eosTokenId = 999;
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);

        when(tokenizer.encode(eq("prompt"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(10, 20)));
        when(session.decode(10)).thenReturn(new ForwardResult(logitsForToken(11, 20)));
        when(session.decode(11)).thenReturn(new ForwardResult(logitsForToken(12, 20)));

        when(stream.next(10)).thenReturn("Hello");
        when(stream.next(11)).thenReturn(" world");
        when(stream.next(12)).thenReturn("!");

        List<String> streamed = new ArrayList<>();

        try (var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(eosTokenId)
                .stopSequence(" world")
                .maxNewTokens(100)
                .build()) {

            GenerationResult result = engine.generate("prompt", streamed::add);

            assertThat(result.text()).isEqualTo("Hello");
            // Streamed output must match final result — listener never sees stop sequence
            assertThat(String.join("", streamed)).isEqualTo(result.text());
        } catch (Exception e) {
            fail(e);
        }
    }

    @Test
    void generate_streamedOutputMatchesFinalResult_withStopSequence() {
        int eosTokenId = 999;
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);

        when(tokenizer.encode(eq("prompt"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        // Generate: "The" + " quick" + " brown" + " fox" + "<|end|>"
        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(10, 20)));
        when(session.decode(10)).thenReturn(new ForwardResult(logitsForToken(11, 20)));
        when(session.decode(11)).thenReturn(new ForwardResult(logitsForToken(12, 20)));
        when(session.decode(12)).thenReturn(new ForwardResult(logitsForToken(13, 20)));

        when(stream.next(10)).thenReturn("The");
        when(stream.next(11)).thenReturn(" quick");
        when(stream.next(12)).thenReturn(" brown");
        when(stream.next(13)).thenReturn(" fox<|end|>rest");

        List<String> streamed = new ArrayList<>();

        try (var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(eosTokenId)
                .stopSequence("<|end|>")
                .maxNewTokens(100)
                .build()) {

            GenerationResult result = engine.generate("prompt", streamed::add);

            assertThat(result.text()).isEqualTo("The quick brown fox");
            assertThat(String.join("", streamed)).isEqualTo(result.text());
        } catch (Exception e) {
            fail(e);
        }
    }

    @Test
    void generate_appliesChatTemplateWhenPresent() {
        int eosTokenId = 0;
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);
        ChatTemplate template = msg -> "<|user|>" + msg + "<|end|>";

        when(tokenizer.encode(eq("<|user|>Hi<|end|>"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1, 2, 3}, new long[]{1, 1, 1}, new long[]{0, 0, 0}));

        when(session.prefill(new long[]{1, 2, 3})).thenReturn(
                new ForwardResult(logitsForToken(eosTokenId, 10)));

        try (var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .chatTemplate(template)
                .eosTokenId(eosTokenId)
                .build()) {

            GenerationResult result = engine.generate("Hi");

            assertThat(result.text()).isEqualTo("");
            assertThat(result.promptTokens()).isEqualTo(3);
        } catch (Exception e) {
            fail(e);
        }

        verify(tokenizer).encode(eq("<|user|>Hi<|end|>"), anyInt());
    }

    @Test
    void generate_usesRawPromptWhenNoChatTemplate() {
        int eosTokenId = 0;
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);

        when(tokenizer.encode(eq("raw prompt"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        when(session.prefill(any())).thenReturn(
                new ForwardResult(logitsForToken(eosTokenId, 10)));

        try (var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(eosTokenId)
                .build()) {

            engine.generate("raw prompt");
        } catch (Exception e) {
            fail(e);
        }

        verify(tokenizer).encode(eq("raw prompt"), anyInt());
    }

    @Test
    void close_delegatesToSession() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);

        var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(0)
                .build();

        engine.close();

        verify(session).close();
    }

    @Test
    void generate_supportsMultipleEosTokenIds() {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);

        when(tokenizer.encode(eq("x"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        // prefill returns token 5, then decode returns second EOS token (200)
        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(5, 10)));
        when(session.decode(5)).thenReturn(new ForwardResult(logitsForToken(200, 300)));

        when(stream.next(5)).thenReturn("A");

        try (var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(100)
                .eosTokenId(200)
                .maxNewTokens(50)
                .build()) {

            GenerationResult result = engine.generate("x");

            assertThat(result.text()).isEqualTo("A");
            assertThat(result.generatedTokens()).isEqualTo(1);
        } catch (Exception e) {
            fail(e);
        }
    }

    /**
     * Creates a logits array where the given tokenId has the highest value.
     */
    // --- input length ---

    private static GenerationEngine.Builder immediateEosEngine(GenerativeSession session, Tokenizer tokenizer) {
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);
        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(7, 10)));
        return GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(7);
    }

    @Test
    void encodesWithoutLimitByDefault() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(eq("x"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        try (var engine = immediateEosEngine(session, tokenizer).build()) {
            engine.generate("x");
        }

        verify(tokenizer).encode("x", Integer.MAX_VALUE);
    }

    @Test
    void encodesWithMaxInputLength() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(eq("x"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        try (var engine = immediateEosEngine(session, tokenizer).maxInputLength(10).build()) {
            engine.generate("x");
        }

        verify(tokenizer).encode("x", 10);
    }

    @Test
    void reservesOneTokenForAppendedEos() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(eq("x"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        try (var engine = immediateEosEngine(session, tokenizer)
                .maxInputLength(10)
                .appendEosToInput(true)
                .build()) {
            engine.generate("x");
        }

        verify(tokenizer).encode("x", 9);
        verify(session).prefill(new long[]{1, 7});
    }

    @Test
    void prependsTokenAndReservesRoomForIt() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(eq("x"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        try (var engine = immediateEosEngine(session, tokenizer)
                .maxInputLength(10)
                .appendEosToInput(true)
                .prependToInput(0)
                .build()) {
            GenerationResult result = engine.generate("x");

            assertThat(result.promptTokens()).isEqualTo(3);
        }

        verify(tokenizer).encode("x", 8);
        verify(session).prefill(new long[]{0, 1, 7});
    }

    @Test
    void forcedBosTokenIsFedToTheModelButNotPartOfTheText() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);
        when(tokenizer.encode(eq("x"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));
        // Without forcing, the model would pick token 5 first
        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(5, 10)));
        when(session.decode(0)).thenReturn(new ForwardResult(logitsForToken(3, 10)));
        when(session.decode(3)).thenReturn(new ForwardResult(logitsForToken(7, 10)));
        when(stream.next(3)).thenReturn("Hi");

        try (var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(7)
                .forcedBosTokenId(0)
                .build()) {
            List<String> streamed = new ArrayList<>();
            GenerationResult result = engine.generate("x", streamed::add);

            assertThat(result.text()).isEqualTo("Hi");
            assertThat(streamed).containsExactly("Hi");
            assertThat(result.generatedTokens()).isEqualTo(1);
        }

        verify(session).decode(0);
        verify(session, never()).decode(5);
        verify(stream, never()).next(0);
    }

    @Test
    void failGuardRejectsTruncatedPrompt() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        // Tokenizer kept 2 of 50 tokens
        when(tokenizer.encode(eq("long prompt"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1, 2}, new long[]{1, 1}, new long[2], null, 50));

        try (var engine = immediateEosEngine(session, tokenizer)
                .maxInputLength(2)
                .truncationGuard(new TruncationGuard("TestGenerator", TruncationPolicy.FAIL))
                .build()) {
            assertThatThrownBy(() -> engine.generate("long prompt"))
                    .isInstanceOf(InputTooLongException.class)
                    .hasMessageContaining("TestGenerator");
        }
        verify(session, never()).prefill(any());
    }

    @Test
    void defaultGuardRunsTruncatedPrompt() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(eq("long prompt"), anyInt())).thenReturn(
                new EncodedInput(new long[]{1, 2}, new long[]{1, 1}, new long[2], null, 50));

        try (var engine = immediateEosEngine(session, tokenizer).maxInputLength(2).build()) {
            engine.generate("long prompt");
        }

        verify(session).prefill(new long[]{1, 2});
    }

    // --- keeping the end of over-long decoder-only prompts ---

    @Test
    void keepPromptEndKeepsTemplateAndEndOfMessage() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);

        try (var engine = immediateEosEngine(session, new CharTokenizer(false))
                .chatTemplate(msg -> "<" + msg + ">")
                .maxInputLength(6)
                .keepPromptEnd(true)
                .build()) {
            engine.generate("abcdefghij");
        }

        verify(session).prefill(CharTokenizer.ids("<ghij>"));
    }

    @Test
    void keepPromptEndKeepsTokensTheTokenizerAdds() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);

        try (var engine = immediateEosEngine(session, new CharTokenizer(true))
                .chatTemplate(msg -> "<" + msg + ">")
                .maxInputLength(6)
                .keepPromptEnd(true)
                .build()) {
            engine.generate("abcdefghij");
        }

        // BOS, the template's "<" and ">", and the last 3 characters of the message
        verify(session).prefill(CharTokenizer.withBos("<hij>"));
    }

    @Test
    void keepPromptEndWithoutTemplateKeepsTheEndOfThePrompt() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);

        try (var engine = immediateEosEngine(session, new CharTokenizer(true))
                .maxInputLength(4)
                .keepPromptEnd(true)
                .build()) {
            engine.generate("abcdefghij");
        }

        verify(session).prefill(CharTokenizer.withBos("hij"));
    }

    @Test
    void keepPromptEndFallsBackToTheEndWhenTheTemplateTransformsTheMessage() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);

        try (var engine = immediateEosEngine(session, new CharTokenizer(false))
                .chatTemplate(msg -> "<" + msg.toUpperCase() + ">")
                .maxInputLength(4)
                .keepPromptEnd(true)
                .build()) {
            engine.generate("abcdefghij");
        }

        verify(session).prefill(CharTokenizer.ids("HIJ>"));
    }

    @Test
    void keepPromptEndLeavesPromptsThatFitUnchanged() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);

        try (var engine = immediateEosEngine(session, new CharTokenizer(true))
                .chatTemplate(msg -> "<" + msg + ">")
                .maxInputLength(20)
                .keepPromptEnd(true)
                .build()) {
            engine.generate("abc");
        }

        verify(session).prefill(CharTokenizer.withBos("<abc>"));
    }

    @Test
    void withoutKeepPromptEndTheBeginningIsKept() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);

        try (var engine = immediateEosEngine(session, new CharTokenizer(false))
                .chatTemplate(msg -> "<" + msg + ">")
                .maxInputLength(6)
                .build()) {
            engine.generate("abcdefghij");
        }

        verify(session).prefill(CharTokenizer.ids("<abcde"));
    }

    @Test
    void keepPromptEndStillFailsUnderFailPolicy() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);

        try (var engine = immediateEosEngine(session, new CharTokenizer(false))
                .chatTemplate(msg -> "<" + msg + ">")
                .maxInputLength(6)
                .keepPromptEnd(true)
                .truncationGuard(new TruncationGuard("TestGenerator", TruncationPolicy.FAIL))
                .build()) {
            assertThatThrownBy(() -> engine.generate("abcdefghij")).isInstanceOf(InputTooLongException.class);
        }
        verify(session, never()).prefill(any());
    }

    /** One token per character (its code point), optionally preceded by BOS = 1; keeps the first tokens when cut. */
    private static final class CharTokenizer implements Tokenizer {
        private final boolean bos;

        CharTokenizer(boolean bos) {
            this.bos = bos;
        }

        static long[] ids(String text) {
            return text.chars().asLongStream().toArray();
        }

        static long[] withBos(String text) {
            long[] chars = ids(text);
            long[] out = new long[chars.length + 1];
            out[0] = 1;
            System.arraycopy(chars, 0, out, 1, chars.length);
            return out;
        }

        @Override
        public EncodedInput encode(String text) {
            return encode(text, Integer.MAX_VALUE);
        }

        @Override
        public EncodedInput encode(String text, int maxLength) {
            long[] all = bos ? withBos(text) : ids(text);
            long[] kept = java.util.Arrays.copyOf(all, Math.min(all.length, maxLength));
            long[] mask = new long[kept.length];
            java.util.Arrays.fill(mask, 1L);
            return new EncodedInput(kept, mask, new long[kept.length], null, all.length);
        }
    }

    private static float[] logitsForToken(int tokenId, int vocabSize) {
        float[] logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            logits[i] = -10.0f;
        }
        logits[tokenId] = 10.0f;
        return logits;
    }

    @Test
    void concurrentGenerateCallsRunOneAtATime() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);
        AtomicInteger active = new AtomicInteger();
        AtomicInteger maxActive = new AtomicInteger();

        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));
        when(session.prefill(any())).thenAnswer(invocation -> {
            maxActive.accumulateAndGet(active.incrementAndGet(), Math::max);
            return new ForwardResult(logitsForToken(5, 10));
        });
        when(session.decode(anyLong())).thenAnswer(invocation -> {
            Thread.sleep(2);                                  // widen the window for overlap
            active.decrementAndGet();
            return new ForwardResult(logitsForToken(9, 10));
        });
        when(stream.next(5)).thenReturn("x");

        int threads = 8;
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        try (var engine = GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(9)
                .build()) {
            List<Future<GenerationResult>> results = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                results.add(executor.submit(() -> engine.generate("x")));
            }
            for (Future<GenerationResult> result : results) {
                assertThat(result.get().text()).isEqualTo("x");
            }
        } finally {
            executor.shutdownNow();
        }
        assertThat(maxActive.get()).as("generations overlapping on one KV cache").isEqualTo(1);
    }

    @Test
    void freesTheCacheAfterEachGeneration() {
        GenerativeSession session = mock(GenerativeSession.class);
        var engine = oneTokenEngine(session);

        engine.generate("x");

        InOrder order = inOrder(session);
        order.verify(session).resetCache();
        order.verify(session).prefill(any());
        order.verify(session).resetCache();
    }

    @Test
    void aFailingListenerFreesTheCacheAndLeavesTheEngineUsable() {
        GenerativeSession session = mock(GenerativeSession.class);
        var engine = oneTokenEngine(session);

        assertThatThrownBy(() -> engine.generate("x", token -> {
            throw new IllegalArgumentException("listener failed");
        })).hasMessage("listener failed");

        verify(session, times(2)).resetCache();
        assertThat(engine.generate("x").text()).isEqualTo("x");
    }

    @Test
    void generateFromTheTokenListenerFails() {
        GenerativeSession session = mock(GenerativeSession.class);
        var engine = oneTokenEngine(session);

        assertThatThrownBy(() -> engine.generate("x", token -> engine.generate("y")))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("token listener");
    }

    @Test
    void generateAfterCloseFailsAndClosingTwiceClosesTheSessionOnce() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        var engine = oneTokenEngine(session);

        engine.close();
        engine.close();

        verify(session, times(1)).close();
        assertThatThrownBy(() -> engine.generate("x"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("closed");
    }

    @Test
    void aWaitingCallerCanBeInterrupted() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        var engine = oneTokenEngine(session);
        CountDownLatch generating = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<GenerationResult> holder = executor.submit(() -> engine.generate("x", token -> {
                generating.countDown();
                awaitQuietly(release);
            }));
            assertThat(generating.await(5, TimeUnit.SECONDS)).isTrue();

            AtomicReference<Throwable> failure = new AtomicReference<>();
            AtomicBoolean interruptKept = new AtomicBoolean();
            Thread waiter = new Thread(() -> {
                try {
                    engine.generate("y");
                } catch (Throwable e) {
                    failure.set(e);
                }
                interruptKept.set(Thread.currentThread().isInterrupted());
            });
            waiter.start();
            awaitParked(waiter);
            waiter.interrupt();
            waiter.join(5000);

            assertThat(failure.get()).isInstanceOf(InferenceException.class).hasMessageContaining("Interrupted");
            assertThat(interruptKept).isTrue();

            release.countDown();
            assertThat(holder.get().text()).isEqualTo("x");
        } finally {
            release.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void closeWaitsForTheGenerationInProgress() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        var engine = oneTokenEngine(session);
        CountDownLatch generating = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<GenerationResult> generation = executor.submit(() -> engine.generate("x", token -> {
                generating.countDown();
                awaitQuietly(release);
            }));
            assertThat(generating.await(5, TimeUnit.SECONDS)).isTrue();

            Thread closer = new Thread(() -> {
                try {
                    engine.close();
                } catch (Exception e) {
                    throw new IllegalStateException(e);
                }
            });
            closer.start();
            awaitParked(closer);
            verify(session, never()).close();

            release.countDown();
            closer.join(5000);
            assertThat(generation.get().text()).isEqualTo("x");
            InOrder order = inOrder(session);
            order.verify(session, times(2)).resetCache();
            order.verify(session).close();
        } finally {
            release.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void closeFromTheTokenListenerFails() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        var engine = oneTokenEngine(session);

        assertThatThrownBy(() -> engine.generate("x", token -> {
            try {
                engine.close();
            } catch (IllegalStateException e) {
                throw e;
            } catch (Exception e) {
                throw new IllegalStateException("unexpected", e);
            }
        })).isInstanceOf(IllegalStateException.class).hasMessageContaining("token listener");
        verify(session, never()).close();
    }

    @Test
    void aFailureWhileFreeingTheCacheDoesNotHideTheGenerationFailure() {
        GenerativeSession session = mock(GenerativeSession.class);
        var engine = oneTokenEngine(session);
        doNothing().doThrow(new IllegalStateException("reset failed")).when(session).resetCache();

        assertThatThrownBy(() -> engine.generate("x", token -> {
            throw new IllegalArgumentException("listener failed");
        })).hasMessage("listener failed")
                .satisfies(e -> assertThat(e.getSuppressed()).singleElement()
                        .extracting(Throwable::getMessage).isEqualTo("reset failed"));
    }

    /** Waits until {@code thread} is blocked on a lock. */
    private static void awaitParked(Thread thread) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (thread.getState() != Thread.State.WAITING && thread.getState() != Thread.State.TIMED_WAITING) {
            assertThat(System.nanoTime()).as("thread never blocked").isLessThan(deadline);
            Thread.sleep(1);
        }
    }

    /** An engine that generates the token "x" and then stops. */
    private static GenerationEngine oneTokenEngine(GenerativeSession session) {
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);
        TokenStream stream = streamOf(decoder);
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));
        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(5, 10)));
        when(session.decode(anyLong())).thenReturn(new ForwardResult(logitsForToken(9, 10)));
        when(stream.next(5)).thenReturn("x");
        return GenerationEngine.builder()
                .session(session)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .eosTokenId(9)
                .build();
    }

    private static void awaitQuietly(CountDownLatch latch) {
        try {
            latch.await(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** Stubs {@code decoder} to hand out one mock stream, which the test then programs token by token. */
    private static TokenStream streamOf(TokenDecoder decoder) {
        TokenStream stream = mock(TokenStream.class);
        when(decoder.newStream()).thenReturn(stream);
        when(stream.flush()).thenReturn("");
        return stream;
    }
}
