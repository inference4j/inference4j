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

package io.github.inference4j.processing;

import io.github.inference4j.processing.TokenWindows.Window;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TokenWindowsTest {

    @Test
    void inputThatFitsIsOneWindowOwningEverything() {
        assertThat(TokenWindows.plan(100, 510, 128)).containsExactly(new Window(0, 100, 0, 100));
        assertThat(TokenWindows.plan(510, 510, 128)).containsExactly(new Window(0, 510, 0, 510));
    }

    @Test
    void emptyInputIsOneEmptyWindow() {
        assertThat(TokenWindows.plan(0, 510, 128)).containsExactly(new Window(0, 0, 0, 0));
    }

    @Test
    void overlappingWindowsSplitOwnershipAtTheMidpoint() {
        // windows of 10 overlapping by 4: [0,10) and [6,16); overlap [6,10), midpoint 8
        assertThat(TokenWindows.plan(16, 10, 4)).containsExactly(
                new Window(0, 10, 0, 8),
                new Window(6, 16, 8, 16));
    }

    @Test
    void lastWindowIsAlignedToTheEndAndFull() {
        // step 6: starts 0, 6, then 12 would end at 22 > 20, so the last window starts at 10
        List<Window> windows = TokenWindows.plan(20, 10, 4);

        assertThat(windows).extracting(Window::start).containsExactly(0, 6, 10);
        Window last = windows.get(windows.size() - 1);
        assertThat(last.end()).isEqualTo(20);
        assertThat(last.end() - last.start()).isEqualTo(10);
    }

    @Test
    void zeroStrideGivesBackToBackWindows() {
        assertThat(TokenWindows.plan(30, 10, 0)).containsExactly(
                new Window(0, 10, 0, 10),
                new Window(10, 20, 10, 20),
                new Window(20, 30, 20, 30));
    }

    @Test
    void invalidArgumentsAreRejected() {
        assertThatThrownBy(() -> TokenWindows.plan(-1, 10, 2)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> TokenWindows.plan(10, 0, 0)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> TokenWindows.plan(10, 5, 5)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> TokenWindows.plan(10, 5, -1)).isInstanceOf(IllegalArgumentException.class);
    }

    static Stream<Arguments> shapes() {
        return Stream.of(
                Arguments.of(11, 10, 0), Arguments.of(11, 10, 9), Arguments.of(1000, 510, 128),
                Arguments.of(1021, 510, 128), Arguments.of(5000, 254, 64), Arguments.of(37, 7, 3),
                Arguments.of(100, 2, 1), Arguments.of(513, 512, 511), Arguments.of(2048, 510, 0));
    }

    @ParameterizedTest
    @MethodSource("shapes")
    void ownedRangesAreContiguousAndCoverEveryTokenOnce(int tokens, int window, int stride) {
        List<Window> windows = TokenWindows.plan(tokens, window, stride);

        assertThat(windows.get(0).ownedStart()).isZero();
        assertThat(windows.get(windows.size() - 1).ownedEnd()).isEqualTo(tokens);
        for (int i = 1; i < windows.size(); i++) {
            assertThat(windows.get(i).ownedStart()).isEqualTo(windows.get(i - 1).ownedEnd());
        }
        int owned = windows.stream().mapToInt(w -> w.ownedEnd() - w.ownedStart()).sum();
        assertThat(owned).isEqualTo(tokens);
    }

    @ParameterizedTest
    @MethodSource("shapes")
    void windowsFitAndContainTheirOwnedTokens(int tokens, int window, int stride) {
        for (Window w : TokenWindows.plan(tokens, window, stride)) {
            assertThat(w.start()).isGreaterThanOrEqualTo(0);
            assertThat(w.end()).isLessThanOrEqualTo(tokens);
            assertThat(w.end() - w.start()).isLessThanOrEqualTo(window);
            assertThat(w.ownedStart()).isGreaterThanOrEqualTo(w.start());
            assertThat(w.ownedEnd()).isLessThanOrEqualTo(w.end());
            assertThat(w.ownedEnd()).isGreaterThan(w.ownedStart());
        }
    }

    @ParameterizedTest
    @MethodSource("shapes")
    void ownedTokensKeepHalfTheStrideAsContext(int tokens, int window, int stride) {
        List<Window> windows = TokenWindows.plan(tokens, window, stride);
        for (Window w : windows) {
            IntStream.range(w.ownedStart(), w.ownedEnd()).forEach(t -> {
                if (w.start() > 0) {
                    assertThat(t - w.start()).isGreaterThanOrEqualTo(stride / 2);
                }
                if (w.end() < tokens) {
                    assertThat(w.end() - 1 - t).isGreaterThanOrEqualTo(stride / 2 - 1);
                }
            });
        }
    }
}
