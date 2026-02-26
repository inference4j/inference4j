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

package io.github.inference4j.http;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.net.http.HttpResponse.BodySubscriber;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Flow;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ProgressSubscriberTest {

    @Mock
    BodySubscriber<String> delegate;

    @Mock
    ProgressListener listener;

    @Mock
    Flow.Subscription subscription;

    @Test
    void onNext_tracksBytesAndNotifiesListener() {
        var subscriber = new ProgressSubscriber<>(delegate, 1000L, listener);
        var buffers = List.of(ByteBuffer.wrap(new byte[256]));

        subscriber.onNext(buffers);

        verify(listener).onProgress(256L, 1000L);
        verify(delegate).onNext(buffers);
    }

    @Test
    void onNext_multipleCalls_accumulatesBytesCorrectly() {
        var subscriber = new ProgressSubscriber<>(delegate, 1000L, listener);

        subscriber.onNext(List.of(ByteBuffer.wrap(new byte[200])));
        subscriber.onNext(List.of(ByteBuffer.wrap(new byte[300])));
        subscriber.onNext(List.of(ByteBuffer.wrap(new byte[500])));

        var bytesArgs = ArgumentCaptor.forClass(Long.class);
        var totalArgs = ArgumentCaptor.forClass(Long.class);
        verify(listener, org.mockito.Mockito.times(3)).onProgress(bytesArgs.capture(), totalArgs.capture());

        assertThat(bytesArgs.getAllValues()).containsExactly(200L, 500L, 1000L);
        assertThat(totalArgs.getAllValues()).containsExactly(1000L, 1000L, 1000L);
    }

    @Test
    void onNext_multipleBuffersInSingleCall_sumsAllRemaining() {
        var subscriber = new ProgressSubscriber<>(delegate, 1000L, listener);
        var buffers = List.of(
                ByteBuffer.wrap(new byte[100]),
                ByteBuffer.wrap(new byte[150]),
                ByteBuffer.wrap(new byte[50])
        );

        subscriber.onNext(buffers);

        verify(listener).onProgress(300L, 1000L);
    }

    @Test
    void onSubscribe_delegatesToWrapped() {
        var subscriber = new ProgressSubscriber<>(delegate, 1000L, listener);

        subscriber.onSubscribe(subscription);

        verify(delegate).onSubscribe(subscription);
    }

    @Test
    void onError_delegatesToWrapped() {
        var subscriber = new ProgressSubscriber<>(delegate, 1000L, listener);
        var error = new RuntimeException("connection lost");

        subscriber.onError(error);

        verify(delegate).onError(error);
    }

    @Test
    void onComplete_delegatesToWrapped() {
        var subscriber = new ProgressSubscriber<>(delegate, 1000L, listener);

        subscriber.onComplete();

        verify(delegate).onComplete();
    }

    @Test
    void getBody_delegatesToWrapped() {
        var expected = CompletableFuture.completedFuture("result");
        when(delegate.getBody()).thenReturn(expected);
        var subscriber = new ProgressSubscriber<>(delegate, 1000L, listener);

        var body = subscriber.getBody();

        assertThat(body).isSameAs(expected);
    }
}
