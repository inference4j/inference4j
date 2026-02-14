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

/**
 * Thrown when a model file cannot be downloaded from a remote source.
 */
public class ModelDownloadException extends ModelSourceException {

    private final int statusCode;

    public ModelDownloadException(String message, int statusCode) {
        super(message);
        this.statusCode = statusCode;
    }

    public ModelDownloadException(String message, Throwable cause) {
        super(message, cause);
        this.statusCode = -1;
    }

    /**
     * Returns the HTTP status code from the failed download, or {@code -1}
     * if the failure was not HTTP-related (e.g., network error).
     */
    public int statusCode() {
        return statusCode;
    }
}
