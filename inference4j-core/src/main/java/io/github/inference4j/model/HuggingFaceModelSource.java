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

package io.github.inference4j.model;

import io.github.inference4j.exception.ModelDownloadException;
import io.github.inference4j.exception.ModelSourceException;
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
import java.util.ArrayList;
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

    /**
     * Resolves a model by repo ID without a file list.
     *
     * <p>This overload exists for backward compatibility with custom {@link ModelSource}
     * lambdas. Callers that know which files they need should prefer
     * {@link #resolve(String, List)} instead.
     *
     * @throws ModelSourceException if this is called on a repo that has never been
     *         downloaded — the file list is required for the initial download
     */
    @Override
    public Path resolve(String repoId) {
        Path repoDir = cacheDir.resolve(repoId);
        if (Files.isDirectory(repoDir)) {
            logger.debug("Cache hit for {}", repoId);
            return repoDir;
        }
        throw new ModelSourceException(
                "No cached model found for '" + repoId + "' and no required files specified. "
                        + "Use resolve(modelId, requiredFiles) to trigger download.");
    }

    @Override
    public Path resolve(String repoId, List<String> requiredFiles) {
        Path repoDir = cacheDir.resolve(repoId);

        // Cache hit: all required files already exist
        if (allFilesPresent(repoDir, requiredFiles)) {
            logger.debug("Cache hit for {}", repoId);
            return repoDir;
        }

        // Acquire per-repo lock to prevent concurrent duplicate downloads
        ReentrantLock lock = repoLocks.computeIfAbsent(repoId, k -> new ReentrantLock());
        lock.lock();
        try {
            // Double-check after acquiring lock
            if (allFilesPresent(repoDir, requiredFiles)) {
                logger.debug("Cache hit for {} (after lock)", repoId);
                return repoDir;
            }

            Files.createDirectories(repoDir);
            logger.info("Downloading model files for {} to {}", repoId, repoDir);

            List<String> missing = missingFiles(repoDir, requiredFiles);
            for (String filename : missing) {
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

    /**
     * Resolves a model subdirectory, downloading all files in it if not cached.
     *
     * <p>Genai models on HuggingFace are organized in subdirectories within the
     * repository. This method lists the files in the subdirectory via the HuggingFace
     * API and downloads any that are missing from the local cache.
     *
     * @param repoId       the HuggingFace repository ID
     * @param subdirectory the subdirectory path within the repo
     * @return path to the local subdirectory containing the model files
     */
    @Override
    public Path resolve(String repoId, String subdirectory) {
        Path subDir = cacheDir.resolve(repoId).resolve(subdirectory);

        // Cache hit: directory exists and is non-empty
        if (Files.isDirectory(subDir) && isNonEmpty(subDir)) {
            logger.debug("Cache hit for {}/{}", repoId, subdirectory);
            return subDir;
        }

        ReentrantLock lock = repoLocks.computeIfAbsent(
                repoId + "/" + subdirectory, k -> new ReentrantLock());
        lock.lock();
        try {
            // Double-check after lock
            if (Files.isDirectory(subDir) && isNonEmpty(subDir)) {
                logger.debug("Cache hit for {}/{} (after lock)", repoId, subdirectory);
                return subDir;
            }

            Files.createDirectories(subDir);
            logger.info("Downloading genai model {}/{} to {}", repoId, subdirectory, subDir);

            List<String> files = listRemoteFiles(repoId, subdirectory);
            for (String filename : files) {
                if (!Files.exists(subDir.resolve(filename))) {
                    downloadFile(repoId, subdirectory + "/" + filename,
                            subDir, filename);
                }
            }

            logger.info("Model download complete for {}/{}", repoId, subdirectory);
            return subDir;
        } catch (IOException e) {
            throw new ModelDownloadException(
                    "Failed to create cache directory: " + subDir, e);
        } finally {
            lock.unlock();
        }
    }

    private static boolean allFilesPresent(Path dir, List<String> files) {
        if (!Files.isDirectory(dir)) {
            return false;
        }
        for (String file : files) {
            if (!Files.exists(dir.resolve(file))) {
                return false;
            }
        }
        return true;
    }

    private static List<String> missingFiles(Path dir, List<String> files) {
        List<String> missing = new ArrayList<>();
        for (String file : files) {
            if (!Files.exists(dir.resolve(file))) {
                missing.add(file);
            }
        }
        return missing;
    }

    private void downloadFile(String repoId, String filename, Path targetDir) {
        downloadFile(repoId, filename, targetDir, filename);
    }

    private void downloadFile(String repoId, String remotePath,
                              Path targetDir, String targetFilename) {
        URI uri = URI.create(HF_BASE_URL + "/" + repoId + "/resolve/main/" + remotePath);
        Path targetFile = targetDir.resolve(targetFilename);
        Path tmpFile = targetDir.resolve(targetFilename + ".tmp");

        HttpRequest request = HttpRequest.newBuilder(uri).GET().build();

        try {
            logger.debug("Downloading {} ...", remotePath);
            HttpResponse<Path> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofFile(tmpFile));

            int status = response.statusCode();
            if (status < 200 || status >= 300) {
                Files.deleteIfExists(tmpFile);
                throw new ModelDownloadException(
                        "Failed to download " + uri + " (HTTP " + status + ")", status);
            }

            Files.move(tmpFile, targetFile, StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
            logger.debug("Downloaded {}", targetFilename);

        } catch (ModelDownloadException e) {
            throw e;
        } catch (IOException | InterruptedException e) {
            try { Files.deleteIfExists(tmpFile); } catch (IOException ignored) { }
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new ModelDownloadException(
                    "Failed to download " + uri + ": " + e.getMessage(), e);
        }
    }

    private static boolean isNonEmpty(Path dir) {
        try (var stream = Files.list(dir)) {
            return stream.findFirst().isPresent();
        } catch (IOException e) {
            return false;
        }
    }

    private List<String> listRemoteFiles(String repoId, String subdirectory) {
        URI uri = URI.create(HF_BASE_URL + "/api/models/" + repoId
                + "/tree/main/" + subdirectory);
        HttpRequest request = HttpRequest.newBuilder(uri).GET().build();

        try {
            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new ModelDownloadException(
                        "Failed to list files at " + uri + " (HTTP " + response.statusCode() + ")",
                        response.statusCode());
            }
            return parseFileList(response.body());
        } catch (ModelDownloadException e) {
            throw e;
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new ModelDownloadException(
                    "Failed to list files at " + uri + ": " + e.getMessage(), e);
        }
    }

    /**
     * Parses the HuggingFace tree API JSON response to extract filenames.
     * The response is an array of objects, each with a "path" and "type" field.
     * We only want entries where type is "file".
     */
    List<String> parseFileList(String json) {
        List<String> files = new ArrayList<>();
        int idx = 0;
        while ((idx = json.indexOf("\"type\"", idx)) != -1) {
            int typeStart = json.indexOf('"', json.indexOf(':', idx) + 1) + 1;
            int typeEnd = json.indexOf('"', typeStart);
            String type = json.substring(typeStart, typeEnd);
            idx = typeEnd;

            if (!"file".equals(type)) {
                continue;
            }

            // Find the "path" field near this entry — search backward since
            // in HF API response, "path" comes before "type" in each object
            int pathIdx = json.lastIndexOf("\"path\"", idx);
            if (pathIdx != -1) {
                int pathStart = json.indexOf('"', json.indexOf(':', pathIdx) + 1) + 1;
                int pathEnd = json.indexOf('"', pathStart);
                String fullPath = json.substring(pathStart, pathEnd);
                // Extract just the filename (strip the subdirectory prefix)
                int lastSlash = fullPath.lastIndexOf('/');
                String filename = lastSlash >= 0 ? fullPath.substring(lastSlash + 1) : fullPath;
                files.add(filename);
            }
        }
        return files;
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
