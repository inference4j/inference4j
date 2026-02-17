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

package io.github.inference4j;

import java.util.List;

/**
 * An inference task that classifies an input against arbitrary candidate labels
 * provided at inference time (zero-shot classification).
 *
 * <p>Unlike {@link Classifier}, which classifies against a fixed set of categories
 * baked into the model, {@code ZeroShotClassifier} accepts candidate labels per call.
 * This is the natural API for models like CLIP where classification labels are
 * encoded at inference time rather than at training time.
 *
 * @param <I> the primary input type (e.g., {@code BufferedImage})
 * @param <C> the classification result type (e.g., {@code Classification})
 */
public interface ZeroShotClassifier<I, C> extends InferenceTask<ZeroShotInput<I>, List<C>> {

    /**
     * Classifies the input against the given candidate labels.
     *
     * @param input           the input to classify
     * @param candidateLabels the text labels to classify against
     * @return scored classifications, typically sorted by confidence descending
     */
    List<C> classify(I input, List<String> candidateLabels);

    @Override
    default List<C> run(ZeroShotInput<I> input) {
        return classify(input.input(), input.candidateLabels());
    }
}
