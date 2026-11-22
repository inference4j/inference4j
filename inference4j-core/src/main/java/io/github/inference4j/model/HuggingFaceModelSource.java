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
import io.github.inference4j.http.DownloadProgressBar;
import io.github.inference4j.http.ProgressListener;
import io.github.inference4j.http.ProgressSubscriber;
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
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

/**
 * A {@link ModelSource} that downloads model files from HuggingFace Hub and caches
 * them locally, keeping the cache up to date with the Hub.
 *
 * <p>On first use, model files are downloaded from
 * {@code https://huggingface.co/{repoId}/resolve/{revision}/{filename}} and stored in a
 * local cache directory, with a {@code .inference4j-revision.json} file recording the
 * repo's commit and each file's id.
 *
 * <p><b>Updates.</b> At most once per {@linkplain Builder#checkInterval(Duration) check interval}
 * (a day by default), resolving a cached repo asks the Hub for its current revision: one small
 * API call. Files whose id changed are downloaded again; unchanged files are kept, so a README
 * edit never re-downloads a model. A cache written before these checks existed is verified by
 * hashing the local files once. When the Hub cannot be reached, the cached files are used and a
 * warning is logged.
 *
 * <p><b>Offline.</b> With {@code INFERENCE4J_OFFLINE=true}, {@code -Dinference4j.offline=true},
 * {@code HF_HUB_OFFLINE=1} or {@link Builder#offline(boolean)}, the Hub is never contacted:
 * cached files are used as they are, and missing files fail to resolve.
 *
 * <p>Cache directory resolution priority:
 * <ol>
 *   <li>Constructor parameter</li>
 *   <li>System property {@code inference4j.cache.dir}</li>
 *   <li>Environment variable {@code INFERENCE4J_CACHE_DIR}</li>
 *   <li>Default: {@code ~/.cache/inference4j/}</li>
 * </ol>
 * The Hub endpoint is {@code https://huggingface.co} unless {@code -Dinference4j.hf.endpoint} or
 * {@code HF_ENDPOINT} names a mirror.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Uses default cache directory
 * ModelSource hf = HuggingFaceModelSource.defaultInstance();
 * Path dir = hf.resolve("inference4j/silero-vad");
 *
 * // Custom cache directory, checked for updates every hour
 * ModelSource hf = HuggingFaceModelSource.builder()
 *         .cacheDir(Path.of("/tmp/models"))
 *         .checkInterval(Duration.ofHours(1))
 *         .build();
 * }</pre>
 *
 * <p>Thread-safe. Several processes may share a cache directory: files and the revision record
 * are replaced atomically.
 */
public class HuggingFaceModelSource implements ModelSource {

    private static final Logger logger = LoggerFactory.getLogger(HuggingFaceModelSource.class);

    private static final String DEFAULT_ENDPOINT = "https://huggingface.co";
    private static final Duration DEFAULT_CHECK_INTERVAL = Duration.ofDays(1);
    private static final Duration API_TIMEOUT = Duration.ofSeconds(5);

    private static volatile HuggingFaceModelSource instance;

    private final Path cacheDir;
    private final String endpoint;
    private final boolean offline;
    private final Duration checkInterval;
    private final Clock clock;
    private final HttpClient httpClient;
    private final ConcurrentHashMap<String, ReentrantLock> repoLocks = new ConcurrentHashMap<>();
    private final Set<String> warnedUnreachable = ConcurrentHashMap.newKeySet();
    /** When a check last failed per repo, so an unreachable Hub is not asked again until the interval passes. */
    private final ConcurrentHashMap<String, Instant> failedChecks = new ConcurrentHashMap<>();

    /**
     * Creates a new source with the default cache directory, endpoint, check interval and
     * offline setting.
     */
    public HuggingFaceModelSource() {
        this(builder());
    }

    /**
     * Creates a new source with a custom cache directory, and the default endpoint, check
     * interval and offline setting.
     *
     * @param cacheDir the directory where downloaded models will be cached
     */
    public HuggingFaceModelSource(Path cacheDir) {
        this(builder().cacheDir(cacheDir));
    }

    private HuggingFaceModelSource(Builder builder) {
        this.cacheDir = builder.cacheDir != null ? builder.cacheDir : resolveDefaultCacheDir();
        String base = builder.endpoint != null ? builder.endpoint.toString() : defaultEndpoint();
        this.endpoint = base.endsWith("/") ? base.substring(0, base.length() - 1) : base;
        this.offline = builder.offline != null ? builder.offline : defaultOffline();
        this.checkInterval = builder.checkInterval != null ? builder.checkInterval : defaultCheckInterval();
        this.clock = builder.clock;
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(API_TIMEOUT)
                .build();
    }

    /** Returns a builder for a source with custom settings. */
    public static Builder builder() {
        return new Builder();
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
     * Resolves a model by repo ID without a file list. Returns the cached directory as is, without
     * checking for updates.
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
        Map<String, Path> files = new LinkedHashMap<>();
        for (String file : requiredFiles) {
            files.put(file, repoDir.resolve(file));
        }
        withRepoLock(repoId, () -> sync(repoId, repoDir, files));
        return repoDir;
    }

    /**
     * Resolves a model subdirectory, downloading all files in it if not cached and keeping them up
     * to date like {@link #resolve(String, List)}.
     *
     * <p>Genai models on HuggingFace are organized in subdirectories within the
     * repository. The files to fetch are those the Hub lists under the subdirectory.
     *
     * @param repoId       the HuggingFace repository ID
     * @param subdirectory the subdirectory path within the repo
     * @return path to the local subdirectory containing the model files
     */
    @Override
    public Path resolve(String repoId, String subdirectory) {
        Path repoDir = cacheDir.resolve(repoId);
        Path subDir = repoDir.resolve(subdirectory);
        withRepoLock(repoId, () -> {
            CacheRecord record = CacheRecord.read(repoDir);
            boolean cached = Files.isDirectory(subDir) && isNonEmpty(subDir);
            if (cached && !checkDue(repoId, record)) {
                logger.debug("Cache hit for {}/{}", repoId, subdirectory);
                return;
            }
            if (offline) {
                if (cached) {
                    return;
                }
                throw offlineMiss(repoId + "/" + subdirectory);
            }
            Revision revision = fetchRevision(repoId);
            if (revision == null) {
                if (cached) {
                    warnUnreachable(repoId);
                    return;
                }
                throw new ModelDownloadException("Cannot list the files of " + repoId + "/" + subdirectory
                        + ": the Hub could not be reached", 0);
            }
            String prefix = subdirectory.endsWith("/") ? subdirectory : subdirectory + "/";
            Map<String, Path> files = new LinkedHashMap<>();
            for (String remotePath : revision.files().keySet()) {
                if (remotePath.startsWith(prefix) && remotePath.indexOf('/', prefix.length()) < 0) {
                    files.put(remotePath, subDir.resolve(remotePath.substring(prefix.length())));
                }
            }
            if (files.isEmpty()) {
                throw new ModelSourceException("No files found under " + repoId + "/" + subdirectory);
            }
            update(repoId, repoDir, files, record, revision);
        });
        return subDir;
    }

    /** Brings {@code files} (remote path to local file) up to date, checking the Hub when due. */
    private void sync(String repoId, Path repoDir, Map<String, Path> files) {
        CacheRecord record = CacheRecord.read(repoDir);
        boolean cached = files.values().stream().allMatch(Files::exists);
        if (cached && !checkDue(repoId, record)) {
            logger.debug("Cache hit for {}", repoId);
            return;
        }
        if (offline) {
            if (cached) {
                return;
            }
            throw offlineMiss(repoId);
        }
        Revision revision = fetchRevision(repoId);
        if (revision == null) {
            if (cached) {
                warnUnreachable(repoId);
                return;
            }
            // The API may fail where downloads work: fetch what is missing, unverified
            createDirectories(repoDir);
            logger.info("Downloading model files for {} to {}", repoId, repoDir);
            files.forEach((remotePath, local) -> {
                if (!Files.exists(local)) {
                    downloadFile(repoId, "main", remotePath, local);
                }
            });
            return;
        }
        update(repoId, repoDir, files, record, revision);
    }

    /** Downloads every missing or changed file at the Hub's revision, then records it. */
    private void update(String repoId, Path repoDir, Map<String, Path> files, CacheRecord record, Revision revision) {
        Map<String, String> ids = new LinkedHashMap<>(record != null ? record.files() : Map.of());
        List<String> changed = new ArrayList<>();
        List<String> missing = new ArrayList<>();
        for (Map.Entry<String, Path> file : files.entrySet()) {
            String remotePath = file.getKey();
            Path local = file.getValue();
            Revision.RemoteFile remote = revision.files().get(remotePath);
            if (!Files.exists(local)) {
                missing.add(remotePath);
            } else if (remote != null) {
                String localId = ids.get(remotePath);
                if (localId == null) {
                    localId = remote.lfs() ? FileHashes.sha256(local) : FileHashes.gitBlobId(local);
                }
                if (remote.id().equals(localId)) {
                    ids.put(remotePath, localId);
                } else {
                    changed.add(remotePath);
                }
            }
        }
        if (!changed.isEmpty()) {
            logger.info("{} changed on the Hub (revision {}); updating {}", repoId, revision.commit(), changed);
        }
        if (!missing.isEmpty()) {
            createDirectories(repoDir);
            logger.info("Downloading model files for {} to {}", repoId, repoDir);
        }
        for (String remotePath : concat(missing, changed)) {
            Path local = files.get(remotePath);
            createDirectories(local.getParent());
            downloadFile(repoId, revision.commit(), remotePath, local);
            Revision.RemoteFile remote = revision.files().get(remotePath);
            if (remote != null) {
                ids.put(remotePath, remote.id());
            }
        }
        createDirectories(repoDir);
        new CacheRecord(revision.commit(), clock.instant(), ids).write(repoDir);
    }

    private boolean checkDue(String repoId, CacheRecord record) {
        Instant now = clock.instant();
        Instant failed = failedChecks.get(repoId);
        if (failed != null && now.isBefore(failed.plus(checkInterval))) {
            return false;
        }
        return record == null || !now.isBefore(record.checkedAt().plus(checkInterval));
    }

    /** The repo's current commit and file ids, or {@code null} when the Hub cannot answer. */
    private Revision fetchRevision(String repoId) {
        URI uri = URI.create(endpoint + "/api/models/" + repoId + "/revision/main?blobs=true");
        HttpRequest request = HttpRequest.newBuilder(uri).timeout(API_TIMEOUT).GET().build();
        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                logger.debug("Revision check of {} failed: HTTP {}", repoId, response.statusCode());
                return null;
            }
            return Revision.parse(response.body());
        } catch (IOException e) {
            logger.debug("Revision check of {} failed: {}", repoId, e.toString());
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    private void warnUnreachable(String repoId) {
        failedChecks.put(repoId, clock.instant());
        if (warnedUnreachable.add(repoId)) {
            logger.warn("Could not check {} for updates on the Hub; using the cached files", repoId);
        }
    }

    private static ModelSourceException offlineMiss(String what) {
        return new ModelSourceException("Offline mode is on and " + what + " is not fully cached");
    }

    private void withRepoLock(String repoId, Runnable action) {
        ReentrantLock lock = repoLocks.computeIfAbsent(repoId, k -> new ReentrantLock());
        lock.lock();
        try {
            action.run();
        } finally {
            lock.unlock();
        }
    }

    private static void createDirectories(Path dir) {
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
            throw new ModelDownloadException("Failed to create cache directory: " + dir, e);
        }
    }

    private static List<String> concat(List<String> a, List<String> b) {
        List<String> all = new ArrayList<>(a);
        all.addAll(b);
        return all;
    }

    private void downloadFile(String repoId, String revision, String remotePath, Path targetFile) {
        URI uri = URI.create(endpoint + "/" + repoId + "/resolve/" + revision + "/" + remotePath);
        Path tmpFile = targetFile.resolveSibling(targetFile.getFileName() + ".tmp");

        HttpRequest request = HttpRequest.newBuilder(uri).GET().build();

        try {
            logger.debug("Downloading {} ...", remotePath);

            HttpResponse<Path> response = httpClient.send(request,
                    progressBodyHandler(
                        HttpResponse.BodyHandlers.ofFile(tmpFile),
                        new DownloadProgressBar(remotePath)
                    ));

            int status = response.statusCode();
            if (status < 200 || status >= 300) {
                Files.deleteIfExists(tmpFile);
                throw new ModelDownloadException(
                        "Failed to download " + uri + " (HTTP " + status + ")", status);
            }

            Files.move(tmpFile, targetFile, StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
            logger.debug("Downloaded {}", remotePath);

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

    static <T> HttpResponse.BodyHandler<T> progressBodyHandler(HttpResponse.BodyHandler<T> delegate, ProgressListener listener) {
        return responseInfo -> {
            long contentLength = responseInfo.headers()
                .firstValueAsLong("Content-Length").orElse(-1L);

            HttpResponse.BodySubscriber<T> delegateSubscriber = delegate.apply(responseInfo);
            return new ProgressSubscriber<>(delegateSubscriber, contentLength, listener);
        };
    }

    private static boolean isNonEmpty(Path dir) {
        try (var stream = Files.list(dir)) {
            return stream.findFirst().isPresent();
        } catch (IOException e) {
            return false;
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

    private static String defaultEndpoint() {
        String value = setting("inference4j.hf.endpoint", "HF_ENDPOINT");
        return value != null ? value : DEFAULT_ENDPOINT;
    }

    private static boolean defaultOffline() {
        String value = setting("inference4j.offline", "INFERENCE4J_OFFLINE");
        if (value != null) {
            return Boolean.parseBoolean(value);
        }
        String hf = System.getenv("HF_HUB_OFFLINE");
        return hf != null && (hf.equals("1") || hf.equalsIgnoreCase("true"));
    }

    private static Duration defaultCheckInterval() {
        String value = setting("inference4j.cache.check-interval", "INFERENCE4J_CACHE_CHECK_INTERVAL");
        if (value == null) {
            return DEFAULT_CHECK_INTERVAL;
        }
        try {
            return requireNotNegative(Duration.parse(value));
        } catch (DateTimeParseException e) {
            throw new IllegalArgumentException("Invalid cache check interval '" + value
                    + "': use an ISO-8601 duration such as PT1H or P1D", e);
        }
    }

    private static Duration requireNotNegative(Duration interval) {
        if (interval.isNegative()) {
            throw new IllegalArgumentException("The cache check interval must not be negative, got " + interval);
        }
        return interval;
    }

    private static String setting(String property, String environmentVariable) {
        String value = System.getProperty(property);
        if (value == null || value.isBlank()) {
            value = System.getenv(environmentVariable);
        }
        return value == null || value.isBlank() ? null : value.strip();
    }

    /** Settings for a {@link HuggingFaceModelSource}; each defaults as described on the class. */
    public static final class Builder {
        private Path cacheDir;
        private URI endpoint;
        private Boolean offline;
        private Duration checkInterval;
        private Clock clock = Clock.systemUTC();

        private Builder() {
        }

        /** The cache directory. */
        public Builder cacheDir(Path cacheDir) {
            this.cacheDir = Objects.requireNonNull(cacheDir, "cacheDir");
            return this;
        }

        /** The Hub endpoint, such as a mirror. Defaults to {@code https://huggingface.co}. */
        public Builder endpoint(URI endpoint) {
            this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
            return this;
        }

        /** When {@code true}, never contacts the Hub: cached files are used as they are. */
        public Builder offline(boolean offline) {
            this.offline = offline;
            return this;
        }

        /**
         * How long a cached repo is trusted before the Hub is asked for updates again. Defaults to a
         * day; {@link Duration#ZERO} checks on every resolve.
         *
         * @throws IllegalArgumentException if {@code checkInterval} is negative
         */
        public Builder checkInterval(Duration checkInterval) {
            this.checkInterval = requireNotNegative(Objects.requireNonNull(checkInterval, "checkInterval"));
            return this;
        }

        /** The clock that times the check interval. Defaults to the system clock. */
        public Builder clock(Clock clock) {
            this.clock = Objects.requireNonNull(clock, "clock");
            return this;
        }

        public HuggingFaceModelSource build() {
            return new HuggingFaceModelSource(this);
        }
    }
}
