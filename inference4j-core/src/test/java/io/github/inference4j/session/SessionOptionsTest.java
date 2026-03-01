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

package io.github.inference4j.session;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class SessionOptionsTest {

    @Test
    void defaults_returnsNonNull() {
        SessionOptions options = SessionOptions.defaults();
        assertThat(options).isNotNull();
    }

    @Test
    void builder_returnsNonNull() {
        SessionOptions options = SessionOptions.builder().build();
        assertThat(options).isNotNull();
    }

    @Test
    void builder_customThreadCounts() {
        SessionOptions options = SessionOptions.builder()
                .intraOpNumThreads(4)
                .interOpNumThreads(2)
                .build();
        assertThat(options).isNotNull();
    }

    @Test
    void toOrtOptions_returnsNonNull() throws OrtException {
        SessionOptions options = SessionOptions.defaults();
        try (OrtSession.SessionOptions ortOptions = options.toOrtOptions()) {
            assertThat(ortOptions).isNotNull();
        }
    }

    @Test
    void toOrtOptions_customThreads() throws OrtException {
        SessionOptions options = SessionOptions.builder()
                .intraOpNumThreads(2)
                .interOpNumThreads(1)
                .build();
        try (OrtSession.SessionOptions ortOptions = options.toOrtOptions()) {
            assertThat(ortOptions).isNotNull();
        }
    }

    @Test
    void builder_fluentApi() {
        SessionOptions.Builder builder = SessionOptions.builder();
        assertThat(builder.intraOpNumThreads(4)).isSameAs(builder);
        assertThat(builder.interOpNumThreads(2)).isSameAs(builder);
    }

    @Test
    void builder_singleThread() throws OrtException {
        SessionOptions options = SessionOptions.builder()
                .intraOpNumThreads(1)
                .interOpNumThreads(1)
                .build();
        try (OrtSession.SessionOptions ortOptions = options.toOrtOptions()) {
            assertThat(ortOptions).isNotNull();
        }
    }
}
