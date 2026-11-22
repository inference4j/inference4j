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

package io.github.inference4j.nlp;

import io.github.inference4j.generation.GenerationEngine;
import io.github.inference4j.processing.TokenLimits;

/**
 * The token budget of one task of an encoder-decoder wrapper: the engine's limits, with the task's prompt
 * prefix counted as part of the input.
 */
final class EngineTokenLimits implements TokenLimits {

    private final GenerationEngine engine;
    private final String promptPrefix;

    EngineTokenLimits(GenerationEngine engine, String promptPrefix) {
        this.engine = engine;
        this.promptPrefix = promptPrefix;
    }

    @Override
    public int maxInputTokens() {
        return engine.maxInputLength();
    }

    @Override
    public int maxOutputTokens() {
        return engine.maxNewTokens();
    }

    @Override
    public int countInputTokens(String text) {
        return engine.countInputTokens(promptPrefix + text);
    }
}
