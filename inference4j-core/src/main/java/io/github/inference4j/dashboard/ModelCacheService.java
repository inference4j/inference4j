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
package io.github.inference4j.dashboard;

import io.github.inference4j.exception.ModelDownloadException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.CachedModel;
import io.github.inference4j.model.CachedSnapshot;
import io.github.inference4j.model.EvictionResult;
import io.github.inference4j.model.ModelCache;
import io.github.inference4j.model.UpdateCheck;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.FileStore;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

/**
 * The operations a web tier offers over a {@link ModelCache}, independent of any web framework: the Spring Boot
 * actuator endpoint and dashboard call it, and other framework integrations can expose the same operations.
 *
 * <p>On top of {@link ModelCache} it adds what a web request needs: a summary with the disk usage of the cache's
 * volume, a Hub check reused for a minute so repeated requests do not call the Hub each time, updates that run in
 * the background one at a time, and the mapping of a failure to an HTTP status and a message that leaves out the
 * cache directory.
 *
 * <p>Thread-safe. It owns the thread that runs updates; {@link #close()} stops it.
 */
public final class ModelCacheService implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(ModelCacheService.class);

    /** How long a check's answer is reused. */
    private static final Duration CHECK_REUSE = Duration.ofMinutes(1);

    private final ModelCache cache;

    private final ExecutorService updates;

    private final Map<String, UpdateCheck> checks = new ConcurrentHashMap<>();

    private final Set<String> pendingUpdates = ConcurrentHashMap.newKeySet();

    /** Creates the service over {@code cache}. */
    public ModelCacheService(ModelCache cache) {
        this.cache = Objects.requireNonNull(cache, "cache");
        this.updates = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "inference4j-model-updates");
            thread.setDaemon(true);
            return thread;
        });
    }

    /** The cache the service runs on. */
    public ModelCache cache() {
        return cache;
    }

    /** Totals over every cached repo, and the disk usage of the volume that holds the cache. */
    public CacheSummary summary() {
        List<CachedModel> models = cache.list();
        int snapshots = 0;
        int inUse = 0;
        int updatesAvailable = 0;
        int stale = 0;
        long sizeBytes = 0;
        long supersededBytes = 0;
        for (CachedModel model : models) {
            sizeBytes += model.sizeBytes();
            updatesAvailable += model.updateAvailable() ? 1 : 0;
            stale += model.stale() ? 1 : 0;
            for (CachedSnapshot snapshot : model.snapshots()) {
                snapshots++;
                if (snapshot.leases() > 0) {
                    inUse++;
                }
                else if (!snapshot.current()) {
                    supersededBytes += snapshot.sizeBytes();
                }
            }
        }
        return new CacheSummary(models.size(), snapshots, inUse, updatesAvailable, stale, pendingUpdates.size(),
                sizeBytes, supersededBytes, diskUsage().orElse(null));
    }

    /** Every cached repo, read from local state only. */
    public List<ModelView> models() {
        return cache.list().stream().map(model -> view(model, null)).toList();
    }

    /**
     * One cached repo, read from local state only; empty when it is not cached.
     *
     * @throws IllegalArgumentException if {@code repoId} is not a valid repo id
     */
    public Optional<ModelView> model(String repoId) {
        return cache.find(repoId).map(model -> view(model, null));
    }

    /**
     * One cached repo, after asking the Hub whether it changed; empty when it is not cached. A check younger than
     * a minute, made while the repo had the same revision, is reused.
     *
     * @throws IllegalArgumentException if {@code repoId} is not a valid repo id
     * @throws ModelSourceException   if the source is offline
     * @throws ModelDownloadException if the Hub cannot be reached
     */
    public Optional<ModelView> check(String repoId) {
        Optional<CachedModel> found = cache.find(repoId);
        if (found.isEmpty()) {
            return Optional.empty();
        }
        UpdateCheck result = check(found.get());
        // The check records its answer, so read the repo again
        CachedModel model = cache.find(repoId).orElse(found.get());
        return Optional.of(view(model, result));
    }

    /**
     * Starts bringing a cached repo up to date with the Hub, in the background; empty when it is not cached.
     * Updates run one at a time. A failed update is logged, and the repo keeps its current snapshot.
     *
     * @throws IllegalArgumentException if {@code repoId} is not a valid repo id
     * @throws IllegalStateException    if the service is closed
     */
    public Optional<UpdateStatus> startUpdate(String repoId) {
        Optional<CachedModel> found = cache.find(repoId);
        if (found.isEmpty()) {
            return Optional.empty();
        }
        String id = found.get().repoId();
        if (!pendingUpdates.add(id)) {
            return Optional.of(UpdateStatus.ALREADY_RUNNING);
        }
        try {
            updates.execute(() -> update(id));
        }
        catch (RejectedExecutionException ex) {
            pendingUpdates.remove(id);
            throw new IllegalStateException("The model cache service is shutting down", ex);
        }
        return Optional.of(UpdateStatus.ACCEPTED);
    }

    /** Whether an update started here is still running for {@code repoId}. */
    public boolean updating(String repoId) {
        return pendingUpdates.contains(repoId);
    }

    /**
     * Deletes a repo's snapshots that are not in use; empty when it is not cached.
     *
     * @throws IllegalArgumentException if {@code repoId} is not a valid repo id
     */
    public Optional<EvictionResult> evict(String repoId) {
        if (cache.find(repoId).isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(cache.evict(repoId));
    }

    /**
     * Deletes one snapshot of a repo, unless it is in use; empty when the repo has no snapshot at
     * {@code revision}.
     *
     * @throws IllegalArgumentException if {@code repoId} or {@code revision} is not valid
     */
    public Optional<EvictionResult> evict(String repoId, String revision) {
        Optional<CachedModel> found = cache.find(repoId);
        boolean cached = found.isPresent()
                && found.get().snapshots().stream().anyMatch(s -> s.revision().equals(revision));
        // Rejects an invalid revision before answering that it is not cached
        EvictionResult result = cache.evict(repoId, revision);
        return cached ? Optional.of(result) : Optional.empty();
    }

    /** Deletes every repo's superseded snapshots that are not in use, and unfinished downloads. */
    public EvictionResult prune() {
        return cache.prune();
    }

    /**
     * The HTTP status and message for a failure of one of this service's operations: 400 for an invalid repo id
     * or revision, 409 when the source cannot do it in its current state (offline, or the repo evicted
     * meanwhile), 502 when the Hub fails, 503 when the service is closed.
     *
     * @throws RuntimeException {@code ex} itself, when it is not one of those failures
     */
    public CacheError error(RuntimeException ex) {
        if (ex instanceof IllegalArgumentException) {
            return new CacheError(400, safeMessage(ex));
        }
        if (ex instanceof ModelDownloadException) {
            logger.warn("Model cache operation failed: {}", safeMessage(ex));
            return new CacheError(502, safeMessage(ex));
        }
        if (ex instanceof ModelSourceException) {
            return new CacheError(409, safeMessage(ex));
        }
        if (ex instanceof IllegalStateException && ex.getCause() instanceof RejectedExecutionException) {
            return new CacheError(503, safeMessage(ex));
        }
        throw ex;
    }

    /** Stops the update thread, waiting up to five seconds for a running update. */
    @Override
    public void close() {
        updates.shutdown();
        try {
            if (!updates.awaitTermination(5, TimeUnit.SECONDS)) {
                updates.shutdownNow();
            }
        }
        catch (InterruptedException ex) {
            updates.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    private UpdateCheck check(CachedModel model) {
        UpdateCheck last = checks.get(model.repoId());
        if (last != null && Objects.equals(last.cachedRevision(), model.revision())
                && last.checkedAt().isAfter(Instant.now().minus(CHECK_REUSE))) {
            return last;
        }
        UpdateCheck result = cache.check(model.repoId());
        checks.put(model.repoId(), result);
        return result;
    }

    private void update(String repoId) {
        try {
            CachedModel model = cache.update(repoId);
            logger.info("Updated {} to revision {}", repoId, model.revision());
        }
        catch (RuntimeException ex) {
            logger.warn("Update of {} failed: {}", repoId, safeMessage(ex));
        }
        finally {
            checks.remove(repoId);
            pendingUpdates.remove(repoId);
        }
    }

    private ModelView view(CachedModel model, UpdateCheck check) {
        return new ModelView(model.repoId(), model.revision(), model.lastChecked(), model.remoteRevision(),
                model.updateAvailable(), model.stale(), pendingUpdates.contains(model.repoId()), model.snapshots(),
                model.sizeBytes(), check != null ? ModelCheck.of(check) : null);
    }

    /** The disk usage of the volume that holds the cache directory, or its nearest existing parent. */
    private Optional<DiskUsage> diskUsage() {
        Path dir = cache.directory().toAbsolutePath();
        while (dir != null && !Files.exists(dir)) {
            dir = dir.getParent();
        }
        if (dir == null) {
            return Optional.empty();
        }
        try {
            FileStore store = Files.getFileStore(dir);
            return Optional.of(new DiskUsage(store.getTotalSpace(), store.getUsableSpace()));
        }
        catch (IOException | SecurityException ex) {
            logger.debug("Cannot read the disk usage of the model cache: {}", ex.toString());
            return Optional.empty();
        }
    }

    /** The exception's message, with the cache directory left out. */
    private String safeMessage(RuntimeException ex) {
        String message = Objects.requireNonNullElse(ex.getMessage(), ex.getClass().getSimpleName());
        String directory = cache.directory().toString();
        return directory.isEmpty() ? message : message.replace(directory, "<cache>");
    }
}
