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

package io.github.inference4j.vision;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class PromptTest {

    @Test
    void pointIsSingleForegroundClick() {
        Prompt prompt = Prompt.point(10, 20);

        assertThat(prompt.clicks()).containsExactly(new Prompt.Click(new Point(10, 20), true));
        assertThat(prompt.box()).isEmpty();
    }

    @Test
    void boxHasNoClicks() {
        Prompt prompt = Prompt.box(1, 2, 3, 4);

        assertThat(prompt.clicks()).isEmpty();
        assertThat(prompt.box()).contains(new BoundingBox(1, 2, 3, 4));
    }

    @Test
    void clicksKeepInsertionOrderAndPolarity() {
        Prompt prompt = Prompt.point(1, 1).withBackground(2, 2).withPoint(3, 3);

        assertThat(prompt.clicks()).extracting(Prompt.Click::foreground).containsExactly(true, false, true);
    }

    @Test
    void withMethodsReturnNewPrompt() {
        Prompt original = Prompt.point(1, 1);

        Prompt extended = original.withBox(new BoundingBox(0, 0, 5, 5));

        assertThat(original.box()).isEmpty();
        assertThat(extended.box()).isPresent();
        assertThat(extended.clicks()).hasSize(1);
    }

    @Test
    void withBoxReplacesExistingBox() {
        Prompt prompt = Prompt.box(0, 0, 1, 1).withBox(new BoundingBox(2, 2, 3, 3));

        assertThat(prompt.box()).contains(new BoundingBox(2, 2, 3, 3));
    }
}
