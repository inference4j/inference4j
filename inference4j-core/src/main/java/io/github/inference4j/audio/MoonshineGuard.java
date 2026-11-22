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

import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Lets a recognizer and its streams use one {@link MoonshineModel} at the same time, and closes the model
 * only when no call is using it.
 *
 * <p>Model work holds the read lock, so calls run in parallel; {@link #close(Runnable)} takes the write
 * lock, so it waits for them. No user code, such as a transcript listener, runs while the lock is held.
 */
final class MoonshineGuard {

    private final ReentrantReadWriteLock closeLock = new ReentrantReadWriteLock();
    private volatile boolean closed;

    /**
     * Holds the model open until {@link #release()}.
     *
     * @throws IllegalStateException if the recognizer was closed
     */
    void acquire() {
        if (!acquireIfOpen()) {
            throw new IllegalStateException("The speech recognizer was closed");
        }
    }

    /**
     * Holds the model open until {@link #release()}, unless the recognizer was closed.
     *
     * @return whether the model is held; when {@code false}, do not call {@link #release()}
     */
    boolean acquireIfOpen() {
        closeLock.readLock().lock();
        if (closed) {
            closeLock.readLock().unlock();
            return false;
        }
        return true;
    }

    void release() {
        closeLock.readLock().unlock();
    }

    boolean isClosed() {
        return closed;
    }

    /**
     * Waits for the calls in progress, then runs {@code closer} once. Later calls to {@link #acquire()}
     * throw.
     */
    void close(Runnable closer) {
        closeLock.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            closer.run();
        } finally {
            closeLock.writeLock().unlock();
        }
    }
}
