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

import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.preprocessing.text.ModelConfig;

import java.util.Locale;

/**
 * The classes of an NLI model that zero-shot scoring compares: entailment and the class it is scored against.
 *
 * @param entailment    index of the entailment class
 * @param contradiction index of the class entailment is scored against when each label is judged on its own
 */
record NliLabels(int entailment, int contradiction) {

    /**
     * Finds both classes by name in {@code id2label}. Entailment is the class named "entail…". The opposing class is
     * the one named "contradict…"; a model without one, such as {@code {0: entailment, 1: not_entailment}}, falls
     * back to Hugging Face's choice: the last class when entailment is the first, otherwise the first. Looking it up
     * by name keeps a three-class model such as {@code {0: entailment, 1: contradiction, 2: neutral}} from being
     * scored against neutral.
     *
     * @throws ModelLoadException if no class is named "entail…"
     */
    static NliLabels from(ModelConfig config) {
        int entailment = -1;
        int contradiction = -1;
        for (int i = 0; i < config.numLabels(); i++) {
            String label = config.label(i).toLowerCase(Locale.ROOT);
            if (label.startsWith("entail")) {
                entailment = i;
            } else if (label.startsWith("contradict")) {
                contradiction = i;
            }
        }
        if (entailment < 0) {
            throw new ModelLoadException("No entailment label in config.json id2label; is this an NLI model?");
        }
        if (contradiction < 0) {
            contradiction = entailment == 0 ? config.numLabels() - 1 : 0;
        }
        return new NliLabels(entailment, contradiction);
    }
}
