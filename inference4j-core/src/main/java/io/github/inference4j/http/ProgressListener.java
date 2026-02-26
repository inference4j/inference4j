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

/**
 * Callback for tracking download progress.
 *
 * @see ProgressSubscriber
 * @see DownloadProgressBar
 */
@FunctionalInterface
public interface ProgressListener {

    /**
     * Called when new bytes have been received.
     *
     * @param bytesDownloaded cumulative number of bytes downloaded so far
     * @param totalBytes      expected total size in bytes, or {@code 0} if unknown
     */
    void onProgress(long bytesDownloaded, long totalBytes);
}
