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

import static org.junit.jupiter.api.Assertions.*;

class ExceptionTest {

    // --- InferenceException ---

    @Test
    void inferenceException_messageOnly() {
        InferenceException ex = new InferenceException("inference failed");
        assertEquals("inference failed", ex.getMessage());
        assertNull(ex.getCause());
    }

    @Test
    void inferenceException_messageAndCause() {
        RuntimeException cause = new RuntimeException("root cause");
        InferenceException ex = new InferenceException("inference failed", cause);
        assertEquals("inference failed", ex.getMessage());
        assertSame(cause, ex.getCause());
    }

    @Test
    void inferenceException_isRuntimeException() {
        assertInstanceOf(RuntimeException.class, new InferenceException("test"));
    }

    // --- ModelLoadException ---

    @Test
    void modelLoadException_messageOnly() {
        ModelLoadException ex = new ModelLoadException("cannot load model");
        assertEquals("cannot load model", ex.getMessage());
        assertNull(ex.getCause());
    }

    @Test
    void modelLoadException_messageAndCause() {
        Exception cause = new Exception("io error");
        ModelLoadException ex = new ModelLoadException("cannot load model", cause);
        assertEquals("cannot load model", ex.getMessage());
        assertSame(cause, ex.getCause());
    }

    @Test
    void modelLoadException_extendsInferenceException() {
        assertInstanceOf(InferenceException.class, new ModelLoadException("test"));
    }

    // --- ModelSourceException ---

    @Test
    void modelSourceException_messageOnly() {
        ModelSourceException ex = new ModelSourceException("source not found");
        assertEquals("source not found", ex.getMessage());
        assertNull(ex.getCause());
    }

    @Test
    void modelSourceException_messageAndCause() {
        Exception cause = new Exception("disk error");
        ModelSourceException ex = new ModelSourceException("source not found", cause);
        assertEquals("source not found", ex.getMessage());
        assertSame(cause, ex.getCause());
    }

    @Test
    void modelSourceException_extendsInferenceException() {
        assertInstanceOf(InferenceException.class, new ModelSourceException("test"));
    }

    // --- ModelDownloadException ---

    @Test
    void modelDownloadException_withStatusCode() {
        ModelDownloadException ex = new ModelDownloadException("404 not found", 404);
        assertEquals("404 not found", ex.getMessage());
        assertEquals(404, ex.statusCode());
        assertNull(ex.getCause());
    }

    @Test
    void modelDownloadException_withCause() {
        Exception cause = new Exception("connection refused");
        ModelDownloadException ex = new ModelDownloadException("download failed", cause);
        assertEquals("download failed", ex.getMessage());
        assertSame(cause, ex.getCause());
        assertEquals(-1, ex.statusCode());
    }

    @Test
    void modelDownloadException_extendsModelSourceException() {
        assertInstanceOf(ModelSourceException.class, new ModelDownloadException("test", 500));
    }

    @Test
    void modelDownloadException_statusCodeVariousValues() {
        assertEquals(200, new ModelDownloadException("ok", 200).statusCode());
        assertEquals(403, new ModelDownloadException("forbidden", 403).statusCode());
        assertEquals(500, new ModelDownloadException("server error", 500).statusCode());
    }

    // --- TensorConversionException ---

    @Test
    void tensorConversionException_messageOnly() {
        TensorConversionException ex = new TensorConversionException("bad tensor");
        assertEquals("bad tensor", ex.getMessage());
        assertNull(ex.getCause());
    }

    @Test
    void tensorConversionException_messageAndCause() {
        Exception cause = new Exception("buffer overflow");
        TensorConversionException ex = new TensorConversionException("bad tensor", cause);
        assertEquals("bad tensor", ex.getMessage());
        assertSame(cause, ex.getCause());
    }

    @Test
    void tensorConversionException_extendsInferenceException() {
        assertInstanceOf(InferenceException.class, new TensorConversionException("test"));
    }
}
