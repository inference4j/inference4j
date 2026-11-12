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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.TreeMap;

/**
 * A pool of direct {@link ByteBuffer}s for input tensors, keyed by capacity.
 *
 * <p>{@code lease(n)} returns the smallest pooled buffer that holds {@code n} bytes, as long as it is
 * at most {@value #MAX_OVERSIZE}x {@code n}; otherwise it allocates a new buffer, so a small input does
 * not tie up a large buffer.
 *
 * <p>The pool keeps at most {@value #MAX_POOLED_BUFFERS} buffers and {@link #MAX_POOLED_BYTES} bytes.
 * When it is over either limit, the largest buffers go first, so one long input does not leave its
 * buffers pinned for the life of the session. Buffers above {@link #MAX_BUFFER_BYTES} are never pooled:
 * allocating them costs little next to the inference they feed. Sizes are exact: a direct buffer's
 * {@code capacity()} is the off-heap memory it holds.
 *
 * <p>Thread-safe: an {@link InferenceSession} shared between threads leases and returns buffers
 * concurrently. The lock covers only map updates; new buffers are allocated outside it.
 */
class DirectBufferPool {

    static final int MAX_POOLED_BUFFERS = 32;
    static final long MAX_POOLED_BYTES = 32L * 1024 * 1024;
    static final int MAX_BUFFER_BYTES = 8 * 1024 * 1024;
    static final int MAX_OVERSIZE = 2;

    private final TreeMap<Integer, Deque<ByteBuffer>> pool = new TreeMap<>();
    private int totalBuffers;
    private long totalBytes;

    /**
     * Returns a direct {@link ByteBuffer} with at least {@code requiredBytes}
     * capacity, in native byte order, with position at 0.
     *
     * @param requiredBytes minimum capacity in bytes
     * @return a direct ByteBuffer in native byte order with position 0
     */
    ByteBuffer lease(int requiredBytes) {
        ByteBuffer pooled = takePooled(requiredBytes);
        if (pooled != null) {
            return pooled;
        }
        // Allocate outside the lock: allocateDirect zero-fills, and near the direct-memory limit it runs GC
        return ByteBuffer.allocateDirect(requiredBytes).order(ByteOrder.nativeOrder());
    }

    private synchronized ByteBuffer takePooled(int requiredBytes) {
        Map.Entry<Integer, Deque<ByteBuffer>> entry = pool.ceilingEntry(requiredBytes);
        if (entry == null || (long) entry.getKey() > (long) requiredBytes * MAX_OVERSIZE) {
            return null;
        }
        ByteBuffer buffer = remove(entry);
        buffer.clear();
        return buffer;
    }

    /**
     * Returns a buffer to the pool for future reuse. Ignores null, non-direct buffers and buffers
     * above {@link #MAX_BUFFER_BYTES}. If the pool is then over a limit, its largest buffers are dropped.
     *
     * @param buffer the buffer to return (may be null)
     */
    synchronized void returnBuffer(ByteBuffer buffer) {
        if (buffer == null || !buffer.isDirect() || buffer.capacity() > MAX_BUFFER_BYTES) {
            return;
        }
        pool.computeIfAbsent(buffer.capacity(), k -> new ArrayDeque<>()).add(buffer);
        totalBuffers++;
        totalBytes += buffer.capacity();
        while (totalBuffers > MAX_POOLED_BUFFERS || totalBytes > MAX_POOLED_BYTES) {
            remove(pool.lastEntry());
        }
    }

    /**
     * Removes all pooled buffers.
     */
    synchronized void clear() {
        pool.clear();
        totalBuffers = 0;
        totalBytes = 0;
    }

    /**
     * Returns the number of buffers currently held in the pool.
     */
    synchronized int size() {
        return totalBuffers;
    }

    /**
     * Returns the total capacity, in bytes, of the buffers currently held in the pool.
     */
    synchronized long pooledBytes() {
        return totalBytes;
    }

    private ByteBuffer remove(Map.Entry<Integer, Deque<ByteBuffer>> entry) {
        ByteBuffer buffer = entry.getValue().poll();
        if (entry.getValue().isEmpty()) {
            pool.remove(entry.getKey());
        }
        totalBuffers--;
        totalBytes -= buffer.capacity();
        return buffer;
    }
}
