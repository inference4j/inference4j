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

import io.github.inference4j.exception.InferenceException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Uses test-bool-not.onnx: y = Not(x), x and y are BOOL [batch, n]. */
class NativeTensorsTest {

    private static final boolean[] VALUES = {true, false, false, true, true, false};

    private InferenceSession session;

    @BeforeEach
    void setUp() throws Exception {
        Path model = Path.of(Objects.requireNonNull(
                NativeTensorsTest.class.getResource("/test-bool-not.onnx")).toURI());
        session = InferenceSession.create(model);
    }

    @AfterEach
    void tearDown() {
        session.close();
    }

    private Map<String, Tensor> input() {
        return Map.of("x", Tensor.fromBooleans(VALUES, new long[]{2, 3}));
    }

    @Test
    void outputsStayNativeUntilCopied() {
        try (NativeTensors outputs = session.runNative(input(), NativeTensors.empty(), null)) {
            assertThat(outputs.names()).containsExactly("y");
            assertThat(outputs.copy("y").toBooleans()).containsExactly(false, true, true, false, false, true);
        }
    }

    @Test
    void anOutputFeedsTheNextRunWithoutLeavingNativeMemory() {
        try (NativeTensors first = session.runNative(input(), NativeTensors.empty(), null);
             NativeTensors second = session.runNative(Map.of(), first.rename(name -> "x"), null)) {
            // Not(Not(x)) == x
            assertThat(second.copy("y").toBooleans()).containsExactly(VALUES);
        }
    }

    @Test
    void selectAndRenameAreViewsOfTheSameMemory() {
        NativeTensors outputs = session.runNative(input(), NativeTensors.empty(), null);
        NativeTensors renamed = outputs.select(Set.of("y")).rename(name -> "x");

        assertThat(renamed.names()).containsExactly("x");
        assertThat(outputs.select(Set.of("missing")).names()).isEmpty();

        renamed.close();

        assertThat(outputs.isClosed()).isTrue();
        assertThatThrownBy(() -> outputs.copy("y")).isInstanceOf(InferenceException.class);
    }

    @Test
    void closedTensorsCannotBeUsedAsInputs() {
        NativeTensors outputs = session.runNative(input(), NativeTensors.empty(), null);
        NativeTensors asInput = outputs.rename(name -> "x");
        outputs.close();

        assertThatThrownBy(() -> session.runNative(Map.of(), asInput, null)).isInstanceOf(InferenceException.class);
    }

    @Test
    void copyingAnUnknownNameFails() {
        try (NativeTensors outputs = session.runNative(input(), NativeTensors.empty(), null)) {
            assertThatThrownBy(() -> outputs.copy("z")).isInstanceOf(InferenceException.class);
        }
    }

    @Test
    void emptyHasNoTensorsAndClosingItIsHarmless() {
        NativeTensors.empty().close();

        assertThat(NativeTensors.empty().names()).isEmpty();
        assertThat(NativeTensors.empty().isClosed()).isFalse();
    }

    @Test
    void outputTypeReportsTheModelsOutputType() {
        assertThat(session.outputType("y")).isEqualTo(TensorType.BOOL);
        assertThatThrownBy(() -> session.outputType("z")).isInstanceOf(InferenceException.class);
    }

    @Test
    void nativeInputsFromSeveralRunsFeedOneRun() throws Exception {
        try (InferenceSession and = InferenceSession.create(Path.of(Objects.requireNonNull(
                NativeTensorsTest.class.getResource("/test-bool-and.onnx")).toURI()));
             NativeTensors first = session.runNative(input(), NativeTensors.empty(), null);
             NativeTensors second = session.runNative(
                     Map.of("x", Tensor.fromBooleans(new boolean[]{false, false, true, true, false, true}, new long[]{2, 3})),
                     NativeTensors.empty(), null);
             NativeTensors result = and.runNative(Map.of(), null, first.rename(name -> "a"), second.rename(name -> "b"))) {
            // Not(VALUES) And Not({F,F,T,T,F,T})
            assertThat(result.copy("z").toBooleans()).containsExactly(false, true, false, false, false, false);
            assertThat(first.isClosed()).isFalse();
            assertThat(second.isClosed()).isFalse();
        }
    }

    @Test
    void javaAndNativeInputsCombine() throws Exception {
        try (InferenceSession and = InferenceSession.create(Path.of(Objects.requireNonNull(
                NativeTensorsTest.class.getResource("/test-bool-and.onnx")).toURI()));
             NativeTensors first = session.runNative(input(), NativeTensors.empty(), null);
             NativeTensors result = and.runNative(
                     Map.of("b", Tensor.fromBooleans(new boolean[]{true, true, true, false, false, false}, new long[]{2, 3})),
                     null, first.rename(name -> "a"))) {
            assertThat(result.copy("z").toBooleans()).containsExactly(false, true, true, false, false, false);
        }
    }

    @Test
    void anInputGivenTwiceFails() throws Exception {
        try (InferenceSession and = InferenceSession.create(Path.of(Objects.requireNonNull(
                NativeTensorsTest.class.getResource("/test-bool-and.onnx")).toURI()));
             NativeTensors first = session.runNative(input(), NativeTensors.empty(), null)) {
            NativeTensors asA = first.rename(name -> "a");

            assertThatThrownBy(() -> and.runNative(Map.of("a", input().get("x")), null, asA))
                    .isInstanceOf(InferenceException.class)
                    .hasMessageContaining("'a'");
        }
    }
}
