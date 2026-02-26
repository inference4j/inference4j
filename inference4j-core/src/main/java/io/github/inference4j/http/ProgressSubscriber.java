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

import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Flow;
import java.net.http.HttpResponse.BodySubscriber;

/**
 * A {@link BodySubscriber} decorator that tracks download progress.
 *
 * <p>Wraps an existing subscriber and intercepts each {@link #onNext} call to
 * accumulate the number of bytes received, notifying a {@link ProgressListener}
 * after every chunk. All other subscriber methods are forwarded to the delegate unchanged.
 *
 * <p>Example usage with {@link java.net.http.HttpClient}:
 * <pre>{@code
 * HttpResponse.BodyHandler<Path> handler = responseInfo -> {
 *     long total = responseInfo.headers()
 *             .firstValueAsLong("Content-Length").orElse(-1L);
 *     BodySubscriber<Path> downstream = HttpResponse.BodySubscribers.ofFile(target);
 *     return new ProgressSubscriber<>(downstream, total, new DownloadProgressBar("model.onnx"));
 * };
 * }</pre>
 *
 * @param <T> the response body type
 * @see ProgressListener
 */
public class ProgressSubscriber<T> implements BodySubscriber<T> {

    private final BodySubscriber<T> delegate;
    private final long totalBytes;
    private final ProgressListener listener;
    private long bytesReceived = 0;

    /**
     * Creates a new progress-tracking subscriber.
     *
     * @param delegate   the subscriber to forward data to
     * @param totalBytes expected total size in bytes (from {@code Content-Length}), or {@code -1} if unknown
     * @param listener   callback invoked after each chunk with cumulative progress
     */
    public ProgressSubscriber(BodySubscriber<T> delegate, long totalBytes, ProgressListener listener) {
        this.delegate = delegate;
        this.totalBytes = totalBytes;
        this.listener = listener;
    }

    @Override
    public CompletionStage<T> getBody() {
        return delegate.getBody();
    }

    @Override
    public void onSubscribe(Flow.Subscription subscription) {
        delegate.onSubscribe(subscription);
    }

    @Override
    public void onNext(List<ByteBuffer> item) {
        long bytes = item.stream().mapToLong(ByteBuffer::remaining).sum();
        bytesReceived += bytes;
        listener.onProgress(bytesReceived, totalBytes);
        delegate.onNext(item);
    }

    @Override
    public void onError(Throwable throwable) {
        delegate.onError(throwable);
    }

    @Override
    public void onComplete() {
        delegate.onComplete();
    }
}
