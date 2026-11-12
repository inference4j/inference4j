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

package io.github.inference4j.audio;

import io.github.inference4j.exception.InferenceException;

import java.util.concurrent.locks.ReentrantLock;

/**
 * Serializes use of one {@link MoonshineModel} by a recognizer and its streams, and closes the model only
 * when no call is using it.
 */
final class MoonshineGuard {

    private final ReentrantLock lock = new ReentrantLock();
    private volatile boolean closed;

    /**
     * Waits for the model, then holds it until {@link #release()}.
     *
     * @throws IllegalStateException if the recognizer was closed
     * @throws InferenceException    if the thread is interrupted while waiting
     */
    void acquire() {
        try {
            lock.lockInterruptibly();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new InferenceException("Interrupted while waiting for another transcription to finish", e);
        }
        if (closed) {
            lock.unlock();
            throw new IllegalStateException("The speech recognizer was closed");
        }
    }

    void release() {
        lock.unlock();
    }

    boolean isClosed() {
        return closed;
    }

    /**
     * Waits for the call in progress, if any, then runs {@code closer} once.
     *
     * @throws IllegalStateException if called from a transcript listener, while this thread holds the model
     */
    void close(Runnable closer) {
        if (lock.isHeldByCurrentThread()) {
            throw new IllegalStateException("The speech recognizer cannot be closed from a transcript listener");
        }
        lock.lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            closer.run();
        } finally {
            lock.unlock();
        }
    }
}
