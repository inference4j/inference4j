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

import io.github.inference4j.exception.ModelDownloadException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

/**
 * A {@link ModelSource} that downloads model files from HuggingFace Hub and caches
 * them locally.
 *
 * <p>On first use, model files are downloaded from
 * {@code https://huggingface.co/{repoId}/resolve/main/{filename}} and stored in a
 * local cache directory. Subsequent calls return the cached path immediately without
 * any network requests.
 *
 * <p>Cache directory resolution priority:
 * <ol>
 *   <li>Constructor parameter</li>
 *   <li>System property {@code inference4j.cache.dir}</li>
 *   <li>Environment variable {@code INFERENCE4J_CACHE_DIR}</li>
 *   <li>Default: {@code ~/.cache/inference4j/}</li>
 * </ol>
 *
 * <p>Example usage:
 * <pre>{@code
 * // Uses default cache directory
 * ModelSource hf = HuggingFaceModelSource.defaultInstance();
 * Path dir = hf.resolve("inference4j/silero-vad");
 *
 * // Custom cache directory
 * ModelSource hf = new HuggingFaceModelSource(Path.of("/tmp/models"));
 * }</pre>
 */
public class HuggingFaceModelSource implements ModelSource {

    private static final Logger logger = LoggerFactory.getLogger(HuggingFaceModelSource.class);

    private static final String HF_BASE_URL = "https://huggingface.co";

    private static final List<String> DEFAULT_FILES = List.of(
            "model.onnx", "vocab.txt", "vocab.json",
            "config.json", "labels.txt", "silero_vad.onnx"
    );

    private static volatile HuggingFaceModelSource instance;

    private final Path cacheDir;
    private final HttpClient httpClient;
    private final ConcurrentHashMap<String, ReentrantLock> repoLocks = new ConcurrentHashMap<>();

    /**
     * Creates a new source with the default cache directory.
     */
    public HuggingFaceModelSource() {
        this(resolveDefaultCacheDir());
    }

    /**
     * Creates a new source with a custom cache directory.
     *
     * @param cacheDir the directory where downloaded models will be cached
     */
    public HuggingFaceModelSource(Path cacheDir) {
        this.cacheDir = cacheDir;
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
    }

    /**
     * Returns a shared default instance using the default cache directory.
     */
    public static HuggingFaceModelSource defaultInstance() {
        if (instance == null) {
            synchronized (HuggingFaceModelSource.class) {
                if (instance == null) {
                    instance = new HuggingFaceModelSource();
                }
            }
        }
        return instance;
    }

    @Override
    public Path resolve(String repoId) {
        Path repoDir = cacheDir.resolve(repoId);

        // Cache hit: if model.onnx already exists, return immediately
        if (Files.exists(repoDir.resolve("model.onnx"))
                || Files.exists(repoDir.resolve("silero_vad.onnx"))) {
            logger.debug("Cache hit for {}", repoId);
            return repoDir;
        }

        // Acquire per-repo lock to prevent concurrent duplicate downloads
        ReentrantLock lock = repoLocks.computeIfAbsent(repoId, k -> new ReentrantLock());
        lock.lock();
        try {
            // Double-check after acquiring lock
            if (Files.exists(repoDir.resolve("model.onnx"))
                    || Files.exists(repoDir.resolve("silero_vad.onnx"))) {
                logger.debug("Cache hit for {} (after lock)", repoId);
                return repoDir;
            }

            Files.createDirectories(repoDir);
            logger.info("Downloading model files for {} to {}", repoId, repoDir);

            for (String filename : DEFAULT_FILES) {
                downloadFile(repoId, filename, repoDir);
            }

            logger.info("Model download complete for {}", repoId);
            return repoDir;
        } catch (IOException e) {
            throw new ModelDownloadException(
                    "Failed to create cache directory: " + repoDir, e);
        } finally {
            lock.unlock();
        }
    }

    private void downloadFile(String repoId, String filename, Path targetDir) {
        URI uri = URI.create(HF_BASE_URL + "/" + repoId + "/resolve/main/" + filename);
        Path targetFile = targetDir.resolve(filename);
        Path tmpFile = targetDir.resolve(filename + ".tmp");

        HttpRequest request = HttpRequest.newBuilder(uri)
                .GET()
                .build();

        try {
            HttpResponse<Path> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofFile(tmpFile));

            int status = response.statusCode();
            if (status == 404) {
                // Silently skip — not all repos have all files
                Files.deleteIfExists(tmpFile);
                return;
            }
            if (status < 200 || status >= 300) {
                Files.deleteIfExists(tmpFile);
                throw new ModelDownloadException(
                        "Failed to download " + uri + " (HTTP " + status + ")", status);
            }

            // Atomic move from tmp to final location
            Files.move(tmpFile, targetFile, StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
            logger.debug("Downloaded {}", filename);

        } catch (ModelDownloadException e) {
            throw e;
        } catch (IOException | InterruptedException e) {
            // Clean up tmp file on failure
            try {
                Files.deleteIfExists(tmpFile);
            } catch (IOException ignored) {
            }
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new ModelDownloadException(
                    "Failed to download " + uri + ": " + e.getMessage(), e);
        }
    }

    private static Path resolveDefaultCacheDir() {
        String sysProp = System.getProperty("inference4j.cache.dir");
        if (sysProp != null && !sysProp.isBlank()) {
            return Path.of(sysProp);
        }

        String envVar = System.getenv("INFERENCE4J_CACHE_DIR");
        if (envVar != null && !envVar.isBlank()) {
            return Path.of(envVar);
        }

        return Path.of(System.getProperty("user.home"), ".cache", "inference4j");
    }
}
