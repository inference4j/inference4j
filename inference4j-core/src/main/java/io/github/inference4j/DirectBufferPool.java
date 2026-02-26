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
 * A simple pool of direct {@link ByteBuffer}s keyed by capacity.
 *
 * <p>Buffers are indexed in a {@link TreeMap} so that {@code lease(n)} returns
 * the smallest pooled buffer with capacity &ge; {@code n} in O(log n) time.
 * When the pool reaches {@link #MAX_POOLED_BUFFERS}, the smallest buffer is
 * evicted (it serves the fewest future requests).
 *
 * <p>Not thread-safe &mdash; intended for use within a single-threaded
 * {@link InferenceSession}.
 */
class DirectBufferPool {

    static final int MAX_POOLED_BUFFERS = 32;

    private final TreeMap<Integer, Deque<ByteBuffer>> pool = new TreeMap<>();
    private int totalBuffers;

    /**
     * Returns a direct {@link ByteBuffer} with at least {@code requiredBytes}
     * capacity, in native byte order, with position at 0.
     *
     * <p>If the pool contains a buffer with capacity &ge; {@code requiredBytes},
     * the smallest such buffer is removed from the pool and returned.
     * Otherwise a fresh direct buffer is allocated.
     *
     * @param requiredBytes minimum capacity in bytes
     * @return a direct ByteBuffer in native byte order with position 0
     */
    ByteBuffer lease(int requiredBytes) {
        Map.Entry<Integer, Deque<ByteBuffer>> entry = pool.ceilingEntry(requiredBytes);
        if (entry != null && !entry.getValue().isEmpty()) {
            ByteBuffer buffer = entry.getValue().poll();
            if (entry.getValue().isEmpty()) {
                pool.remove(entry.getKey());
            }
            totalBuffers--;
            buffer.clear();
            return buffer;
        }
        return ByteBuffer.allocateDirect(requiredBytes).order(ByteOrder.nativeOrder());
    }

    /**
     * Returns a buffer to the pool for future reuse. Ignores null and
     * non-direct buffers.
     *
     * <p>If the pool is at capacity, the smallest buffer is evicted.
     *
     * @param buffer the buffer to return (may be null)
     */
    void returnBuffer(ByteBuffer buffer) {
        if (buffer == null || !buffer.isDirect()) {
            return;
        }
        if (totalBuffers >= MAX_POOLED_BUFFERS) {
            evictSmallest();
        }
        pool.computeIfAbsent(buffer.capacity(), k -> new ArrayDeque<>()).add(buffer);
        totalBuffers++;
    }

    /**
     * Removes all pooled buffers.
     */
    void clear() {
        pool.clear();
        totalBuffers = 0;
    }

    /**
     * Returns the number of buffers currently held in the pool.
     */
    int size() {
        return totalBuffers;
    }

    private void evictSmallest() {
        Map.Entry<Integer, Deque<ByteBuffer>> first = pool.firstEntry();
        if (first == null) {
            return;
        }
        first.getValue().poll();
        if (first.getValue().isEmpty()) {
            pool.remove(first.getKey());
        }
        totalBuffers--;
    }
}
