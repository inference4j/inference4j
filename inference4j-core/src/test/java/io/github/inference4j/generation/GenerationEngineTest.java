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

import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.TokenDecoder;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
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

        when(tokenizer.encode("Hello")).thenReturn(
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

        when(decoder.decode(318)).thenReturn(" is");
        when(decoder.decode(995)).thenReturn(" world");

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

        verify(session).resetCache();
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

        when(tokenizer.encode("Hi")).thenReturn(
                new EncodedInput(new long[]{10}, new long[]{1}, new long[]{0}));

        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(1, 10)));
        when(session.decode(1)).thenReturn(new ForwardResult(logitsForToken(2, 10)));
        when(session.decode(2)).thenReturn(new ForwardResult(logitsForToken(eosTokenId, 10)));

        when(decoder.decode(1)).thenReturn("A");
        when(decoder.decode(2)).thenReturn("B");

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

        when(tokenizer.encode("x")).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        // Always return token 5 (never EOS)
        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(5, 10)));
        when(session.decode(anyLong())).thenReturn(new ForwardResult(logitsForToken(5, 10)));
        when(decoder.decode(5)).thenReturn("x");

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

        when(tokenizer.encode("prompt")).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(10, 20)));
        when(session.decode(10)).thenReturn(new ForwardResult(logitsForToken(11, 20)));
        when(session.decode(11)).thenReturn(new ForwardResult(logitsForToken(12, 20)));

        when(decoder.decode(10)).thenReturn("Hello");
        when(decoder.decode(11)).thenReturn(" world");
        when(decoder.decode(12)).thenReturn("!");

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

        when(tokenizer.encode("prompt")).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        // Generate: "The" + " quick" + " brown" + " fox" + "<|end|>"
        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(10, 20)));
        when(session.decode(10)).thenReturn(new ForwardResult(logitsForToken(11, 20)));
        when(session.decode(11)).thenReturn(new ForwardResult(logitsForToken(12, 20)));
        when(session.decode(12)).thenReturn(new ForwardResult(logitsForToken(13, 20)));

        when(decoder.decode(10)).thenReturn("The");
        when(decoder.decode(11)).thenReturn(" quick");
        when(decoder.decode(12)).thenReturn(" brown");
        when(decoder.decode(13)).thenReturn(" fox<|end|>rest");

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
        ChatTemplate template = msg -> "<|user|>" + msg + "<|end|>";

        when(tokenizer.encode("<|user|>Hi<|end|>")).thenReturn(
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

        verify(tokenizer).encode("<|user|>Hi<|end|>");
    }

    @Test
    void generate_usesRawPromptWhenNoChatTemplate() {
        int eosTokenId = 0;
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);

        when(tokenizer.encode("raw prompt")).thenReturn(
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

        verify(tokenizer).encode("raw prompt");
    }

    @Test
    void close_delegatesToSession() throws Exception {
        GenerativeSession session = mock(GenerativeSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        TokenDecoder decoder = mock(TokenDecoder.class);

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

        when(tokenizer.encode("x")).thenReturn(
                new EncodedInput(new long[]{1}, new long[]{1}, new long[]{0}));

        // prefill returns token 5, then decode returns second EOS token (200)
        when(session.prefill(any())).thenReturn(new ForwardResult(logitsForToken(5, 10)));
        when(session.decode(5)).thenReturn(new ForwardResult(logitsForToken(200, 300)));

        when(decoder.decode(5)).thenReturn("A");

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
    private static float[] logitsForToken(int tokenId, int vocabSize) {
        float[] logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            logits[i] = -10.0f;
        }
        logits[tokenId] = 10.0f;
        return logits;
    }
}
