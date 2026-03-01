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

package io.github.inference4j.exception;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class ExceptionTest {

    // --- InferenceException ---

    @Test
    void inferenceException_messageOnly() {
        InferenceException ex = new InferenceException("inference failed");
        assertThat(ex.getMessage()).isEqualTo("inference failed");
        assertThat(ex.getCause()).isNull();
    }

    @Test
    void inferenceException_messageAndCause() {
        RuntimeException cause = new RuntimeException("root cause");
        InferenceException ex = new InferenceException("inference failed", cause);
        assertThat(ex.getMessage()).isEqualTo("inference failed");
        assertThat(ex.getCause()).isSameAs(cause);
    }

    @Test
    void inferenceException_isRuntimeException() {
        assertThat(new InferenceException("test")).isInstanceOf(RuntimeException.class);
    }

    // --- ModelLoadException ---

    @Test
    void modelLoadException_messageOnly() {
        ModelLoadException ex = new ModelLoadException("cannot load model");
        assertThat(ex.getMessage()).isEqualTo("cannot load model");
        assertThat(ex.getCause()).isNull();
    }

    @Test
    void modelLoadException_messageAndCause() {
        Exception cause = new Exception("io error");
        ModelLoadException ex = new ModelLoadException("cannot load model", cause);
        assertThat(ex.getMessage()).isEqualTo("cannot load model");
        assertThat(ex.getCause()).isSameAs(cause);
    }

    @Test
    void modelLoadException_extendsInferenceException() {
        assertThat(new ModelLoadException("test")).isInstanceOf(InferenceException.class);
    }

    // --- ModelSourceException ---

    @Test
    void modelSourceException_messageOnly() {
        ModelSourceException ex = new ModelSourceException("source not found");
        assertThat(ex.getMessage()).isEqualTo("source not found");
        assertThat(ex.getCause()).isNull();
    }

    @Test
    void modelSourceException_messageAndCause() {
        Exception cause = new Exception("disk error");
        ModelSourceException ex = new ModelSourceException("source not found", cause);
        assertThat(ex.getMessage()).isEqualTo("source not found");
        assertThat(ex.getCause()).isSameAs(cause);
    }

    @Test
    void modelSourceException_extendsInferenceException() {
        assertThat(new ModelSourceException("test")).isInstanceOf(InferenceException.class);
    }

    // --- ModelDownloadException ---

    @Test
    void modelDownloadException_withStatusCode() {
        ModelDownloadException ex = new ModelDownloadException("404 not found", 404);
        assertThat(ex.getMessage()).isEqualTo("404 not found");
        assertThat(ex.statusCode()).isEqualTo(404);
        assertThat(ex.getCause()).isNull();
    }

    @Test
    void modelDownloadException_withCause() {
        Exception cause = new Exception("connection refused");
        ModelDownloadException ex = new ModelDownloadException("download failed", cause);
        assertThat(ex.getMessage()).isEqualTo("download failed");
        assertThat(ex.getCause()).isSameAs(cause);
        assertThat(ex.statusCode()).isEqualTo(-1);
    }

    @Test
    void modelDownloadException_extendsModelSourceException() {
        assertThat(new ModelDownloadException("test", 500)).isInstanceOf(ModelSourceException.class);
    }

    @Test
    void modelDownloadException_statusCodeVariousValues() {
        assertThat(new ModelDownloadException("ok", 200).statusCode()).isEqualTo(200);
        assertThat(new ModelDownloadException("forbidden", 403).statusCode()).isEqualTo(403);
        assertThat(new ModelDownloadException("server error", 500).statusCode()).isEqualTo(500);
    }

    // --- TensorConversionException ---

    @Test
    void tensorConversionException_messageOnly() {
        TensorConversionException ex = new TensorConversionException("bad tensor");
        assertThat(ex.getMessage()).isEqualTo("bad tensor");
        assertThat(ex.getCause()).isNull();
    }

    @Test
    void tensorConversionException_messageAndCause() {
        Exception cause = new Exception("buffer overflow");
        TensorConversionException ex = new TensorConversionException("bad tensor", cause);
        assertThat(ex.getMessage()).isEqualTo("bad tensor");
        assertThat(ex.getCause()).isSameAs(cause);
    }

    @Test
    void tensorConversionException_extendsInferenceException() {
        assertThat(new TensorConversionException("test")).isInstanceOf(InferenceException.class);
    }
}
