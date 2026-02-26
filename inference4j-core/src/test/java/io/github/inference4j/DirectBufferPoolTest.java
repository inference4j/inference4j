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

        ByteBuffer result = pool.lease(100);

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
    void returnBuffer_evictsSmallest_whenAtCapacity() {
        var pool = new DirectBufferPool();

        // Fill pool to capacity
        ByteBuffer[] buffers = new ByteBuffer[DirectBufferPool.MAX_POOLED_BUFFERS];
        for (int i = 0; i < buffers.length; i++) {
            buffers[i] = pool.lease((i + 1) * 16);
        }
        for (ByteBuffer buf : buffers) {
            pool.returnBuffer(buf);
        }
        assertThat(pool.size()).isEqualTo(DirectBufferPool.MAX_POOLED_BUFFERS);

        // Return one more - should evict the smallest (16 bytes)
        ByteBuffer extra = pool.lease(1024);
        pool.returnBuffer(extra);
        assertThat(pool.size()).isEqualTo(DirectBufferPool.MAX_POOLED_BUFFERS);

        // The smallest (16 bytes) was evicted, so requesting 16 should give a new buffer
        ByteBuffer small = pool.lease(16);
        assertThat(small).isNotSameAs(buffers[0]);
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
}
