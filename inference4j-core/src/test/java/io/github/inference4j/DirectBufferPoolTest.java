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

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;

import static org.assertj.core.api.Assertions.assertThat;

class DirectBufferPoolTest {

    @Test
    void lease_returnsDirect_nativeOrderBuffer() {
        var pool = new DirectBufferPool();
        ByteBuffer buffer = pool.lease(64);

        assertThat(buffer.isDirect()).isTrue();
        assertThat(buffer.order()).isEqualTo(ByteOrder.nativeOrder());
        assertThat(buffer.capacity()).isGreaterThanOrEqualTo(64);
    }

    @Test
    void lease_afterReturn_reusesSameBuffer() {
        var pool = new DirectBufferPool();
        ByteBuffer original = pool.lease(128);
        pool.returnBuffer(original);

        ByteBuffer reused = pool.lease(128);

        assertThat(reused).isSameAs(original);
    }

    @Test
    void lease_returnsLargerBuffer_whenExactSizeUnavailable() {
        var pool = new DirectBufferPool();
        ByteBuffer large = pool.lease(256);
        pool.returnBuffer(large);

        ByteBuffer result = pool.lease(150);

        assertThat(result).isSameAs(large);
        assertThat(result.capacity()).isEqualTo(256);
    }

    @Test
    void lease_allocatesNew_whenPoolEmpty() {
        var pool = new DirectBufferPool();

        ByteBuffer first = pool.lease(64);
        ByteBuffer second = pool.lease(64);

        assertThat(first).isNotSameAs(second);
        assertThat(pool.size()).isEqualTo(0);
    }

    @Test
    void returnBuffer_incrementsSize() {
        var pool = new DirectBufferPool();
        ByteBuffer buf1 = pool.lease(64);
        ByteBuffer buf2 = pool.lease(128);

        pool.returnBuffer(buf1);
        assertThat(pool.size()).isEqualTo(1);

        pool.returnBuffer(buf2);
        assertThat(pool.size()).isEqualTo(2);
    }

    @Test
    void returnBuffer_dropsTheLargest_whenOverTheBufferCount() {
        var pool = new DirectBufferPool();
        ByteBuffer[] buffers = new ByteBuffer[DirectBufferPool.MAX_POOLED_BUFFERS];
        for (int i = 0; i < buffers.length; i++) {
            buffers[i] = pool.lease((i + 1) * 16);
        }
        ByteBuffer extra = pool.lease(16);
        for (ByteBuffer buf : buffers) {
            pool.returnBuffer(buf);
        }
        assertThat(pool.size()).isEqualTo(DirectBufferPool.MAX_POOLED_BUFFERS);

        pool.returnBuffer(extra);

        assertThat(pool.size()).isEqualTo(DirectBufferPool.MAX_POOLED_BUFFERS);
        int largest = DirectBufferPool.MAX_POOLED_BUFFERS * 16;
        assertThat(pool.lease(largest)).isNotSameAs(buffers[buffers.length - 1]);
        assertThat(pool.lease(16)).isSameAs(buffers[0]);
    }

    @Test
    void returnBuffer_dropsTheLargest_whenOverTheByteCap() {
        var pool = new DirectBufferPool();
        int big = DirectBufferPool.MAX_BUFFER_BYTES;                    // 8 MB: four fill the 32 MB cap
        List<ByteBuffer> bigOnes = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            bigOnes.add(pool.lease(big));
        }
        bigOnes.forEach(pool::returnBuffer);
        assertThat(pool.pooledBytes()).isEqualTo(DirectBufferPool.MAX_POOLED_BYTES);

        ByteBuffer small = pool.lease(1024);
        pool.returnBuffer(small);

        assertThat(pool.size()).isEqualTo(4);                            // three big ones and the small one
        assertThat(pool.pooledBytes()).isEqualTo(3L * big + 1024);
        assertThat(pool.lease(1024)).isSameAs(small);
    }

    @Test
    void returnBuffer_neverPoolsBuffersAboveTheSizeLimit() {
        var pool = new DirectBufferPool();

        pool.returnBuffer(pool.lease(DirectBufferPool.MAX_BUFFER_BYTES + 1));

        assertThat(pool.size()).isZero();
        assertThat(pool.pooledBytes()).isZero();
    }

    @Test
    void lease_doesNotHandOutABufferMoreThanTwiceTheRequest() {
        var pool = new DirectBufferPool();
        ByteBuffer large = pool.lease(1024);
        pool.returnBuffer(large);

        ByteBuffer small = pool.lease(100);

        assertThat(small).isNotSameAs(large);
        assertThat(small.capacity()).isEqualTo(100);
        assertThat(pool.size()).isEqualTo(1);
        assertThat(pool.lease(512)).isSameAs(large);                      // exactly 2x still qualifies
    }

    @Test
    void pooledBytesTracksLeasesReturnsAndClear() {
        var pool = new DirectBufferPool();
        ByteBuffer a = pool.lease(64);
        ByteBuffer b = pool.lease(128);

        pool.returnBuffer(a);
        pool.returnBuffer(b);
        assertThat(pool.pooledBytes()).isEqualTo(192);

        pool.lease(128);
        assertThat(pool.pooledBytes()).isEqualTo(64);

        pool.clear();
        assertThat(pool.pooledBytes()).isZero();
    }

    @Test
    void returnBuffer_ignoresNull() {
        var pool = new DirectBufferPool();
        pool.returnBuffer(null);

        assertThat(pool.size()).isEqualTo(0);
    }

    @Test
    void returnBuffer_ignoresNonDirect() {
        var pool = new DirectBufferPool();
        ByteBuffer heapBuffer = ByteBuffer.allocate(64);
        pool.returnBuffer(heapBuffer);

        assertThat(pool.size()).isEqualTo(0);
    }

    @Test
    void clear_removesAllBuffers() {
        var pool = new DirectBufferPool();
        pool.returnBuffer(pool.lease(64));
        pool.returnBuffer(pool.lease(128));
        pool.returnBuffer(pool.lease(256));
        assertThat(pool.size()).isEqualTo(3);

        pool.clear();

        assertThat(pool.size()).isEqualTo(0);
    }

    @Test
    void lease_bufferHasZeroPosition() {
        var pool = new DirectBufferPool();
        ByteBuffer buffer = pool.lease(64);
        // Simulate writing to the buffer
        buffer.putFloat(1.0f);
        pool.returnBuffer(buffer);

        ByteBuffer reused = pool.lease(64);

        assertThat(reused.position()).isEqualTo(0);
    }

    @Test
    void lease_viewBufferIsDirect() {
        var pool = new DirectBufferPool();
        ByteBuffer buffer = pool.lease(64);

        assertThat(buffer.asFloatBuffer().isDirect()).isTrue();
        assertThat(buffer.asLongBuffer().isDirect()).isTrue();
        assertThat(buffer.asShortBuffer().isDirect()).isTrue();
    }

    @Test
    void lease_doesNotReuseBuffer_whenAllPooledAreTooSmall() {
        var pool = new DirectBufferPool();
        ByteBuffer small = pool.lease(32);
        pool.returnBuffer(small);

        ByteBuffer large = pool.lease(64);

        assertThat(large).isNotSameAs(small);
        assertThat(large.capacity()).isGreaterThanOrEqualTo(64);
        // Small buffer should still be in the pool
        assertThat(pool.size()).isEqualTo(1);
    }

    @Test
    void lease_prefersSmallerAdequateBuffer() {
        var pool = new DirectBufferPool();
        ByteBuffer buf128 = pool.lease(128);
        ByteBuffer buf256 = pool.lease(256);
        ByteBuffer buf512 = pool.lease(512);

        pool.returnBuffer(buf128);
        pool.returnBuffer(buf256);
        pool.returnBuffer(buf512);

        ByteBuffer result = pool.lease(200);

        assertThat(result).isSameAs(buf256);
        assertThat(pool.size()).isEqualTo(2);
    }

    @Test
    void concurrentLeasesNeverHandOutTheSameBuffer() throws Exception {
        var pool = new DirectBufferPool();
        Set<ByteBuffer> inUse = Collections.synchronizedSet(Collections.newSetFromMap(new IdentityHashMap<>()));
        int threads = 8;
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        try {
            List<Future<Boolean>> results = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                results.add(executor.submit(() -> {
                    for (int i = 0; i < 5_000; i++) {
                        ByteBuffer buffer = pool.lease(64 << ThreadLocalRandom.current().nextInt(4));
                        if (!inUse.add(buffer)) {
                            return false;           // another thread holds this buffer
                        }
                        inUse.remove(buffer);
                        pool.returnBuffer(buffer);
                    }
                    return true;
                }));
            }
            for (Future<Boolean> result : results) {
                assertThat(result.get()).as("no buffer leased to two threads at once").isTrue();
            }
        } finally {
            executor.shutdownNow();
        }
        assertThat(pool.size()).isBetween(0, DirectBufferPool.MAX_POOLED_BUFFERS);
        assertThat(pool.pooledBytes()).isBetween(0L, DirectBufferPool.MAX_POOLED_BYTES);
    }
}
