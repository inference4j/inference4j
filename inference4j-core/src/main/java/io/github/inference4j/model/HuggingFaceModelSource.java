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
import io.github.inference4j.model.CacheRegistry.RepoLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
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
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A {@link ModelSource} that downloads model files from HuggingFace Hub and caches
 * them locally, keeping the cache up to date with the Hub.
 *
 * <p><b>Snapshots.</b> Files are downloaded from
 * {@code https://huggingface.co/{repoId}/resolve/{commit}/{filename}} into a snapshot of the repo at that
 * commit, under {@code {cacheDir}/hub/models--{org}--{name}/snapshots/{commit}/}. An update writes a new
 * snapshot next to the current one and switches {@code refs/main} to it once it is complete. The files that did
 * not change are moved into it from the old snapshot, or copied when a model in this JVM uses the old snapshot or
 * it is younger than the {@linkplain Builder#minimumSnapshotAge minimum age}; they are never hard links, which
 * ONNX Runtime refuses for external data. A model that is loaded keeps reading the snapshot it was loaded from,
 * so an update never mixes files of two revisions.
 *
 * <p><b>Updates.</b> At most once per {@linkplain Builder#checkInterval(Duration) check interval}
 * (a day by default), opening a cached repo asks the Hub for its current revision: one small
 * API call. When cached files changed, the changed files are downloaded into a new snapshot; when none did,
 * nothing is downloaded, so a README edit never re-downloads a model. When the Hub cannot be reached, or an
 * update fails, the cached snapshot is used and a warning is logged.
 *
 * <p><b>Leases and pruning.</b> {@link #open(String, List)} returns a {@link ResolvedModel} that keeps its
 * snapshot on disk until closed; each {@link io.github.inference4j.InferenceSession} created from it keeps
 * its own. With {@link PruneMode#AUTO} (the default), the snapshots an update supersedes are deleted unless a
 * model in this JVM uses them or they are younger than the {@linkplain Builder#minimumSnapshotAge minimum age},
 * which protects a model another process is loading. {@link #cache()} lists, checks, updates and evicts cached
 * models.
 *
 * <p><b>Older caches.</b> Files cached in the older flat layout, {@code {cacheDir}/{org}/{name}/}, are moved
 * into snapshots without downloading them again: online, when their hash matches the Hub's; offline, as a
 * snapshot of {@value ModelRevision#UNKNOWN} revision, checked at the next online open. An older version of the
 * library sharing the cache downloads them again. A snapshot file that has other hard links, as caches written by
 * earlier builds have, is given a private copy when it is opened. A cache directory that is not writable is read
 * in place, without update checks or eviction protection.
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
 * HuggingFaceModelSource hf = HuggingFaceModelSource.defaultInstance();
 * try (ResolvedModel model = hf.open("inference4j/silero-vad", List.of("model.onnx"))) {
 *     InferenceSession session = InferenceSession.create(model, "model.onnx");
 * }
 *
 * // Custom cache directory, checked for updates every hour
 * HuggingFaceModelSource hf = HuggingFaceModelSource.builder()
 *         .cacheDir(Path.of("/tmp/models"))
 *         .checkInterval(Duration.ofHours(1))
 *         .build();
 * }</pre>
 *
 * <p>Thread-safe. Sources on the same cache directory share their locks and leases, and several processes
 * may share a cache directory: each repo is updated under an exclusive file lock.
 */
public class HuggingFaceModelSource implements ModelSource {

    private static final Logger logger = LoggerFactory.getLogger(HuggingFaceModelSource.class);

    private static final String DEFAULT_ENDPOINT = "https://huggingface.co";
    private static final Duration DEFAULT_CHECK_INTERVAL = Duration.ofDays(1);
    private static final Duration DEFAULT_MINIMUM_SNAPSHOT_AGE = Duration.ofMinutes(10);
    private static final Duration API_TIMEOUT = Duration.ofSeconds(5);
    private static final String UNKNOWN = ModelRevision.UNKNOWN;

    private static volatile HuggingFaceModelSource instance;

    private final Path cacheDir;
    private final String endpoint;
    private final boolean offline;
    private final Duration checkInterval;
    private final Clock clock;
    private final PruneMode pruneMode;
    private final Duration minimumSnapshotAge;
    private final HttpClient httpClient;
    private final CacheRegistry registry;
    private final HubModelCache cache;
    private final AtomicBoolean warnedReadOnly = new AtomicBoolean();

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
        this.cacheDir = (builder.cacheDir != null ? builder.cacheDir : resolveDefaultCacheDir()).normalize();
        String base = builder.endpoint != null ? builder.endpoint.toString() : defaultEndpoint();
        this.endpoint = base.endsWith("/") ? base.substring(0, base.length() - 1) : base;
        this.offline = builder.offline != null ? builder.offline : defaultOffline();
        this.checkInterval = builder.checkInterval != null ? builder.checkInterval : defaultCheckInterval();
        this.clock = builder.clock;
        this.pruneMode = builder.pruneMode != null ? builder.pruneMode : defaultPruneMode();
        this.minimumSnapshotAge = builder.minimumSnapshotAge;
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(API_TIMEOUT)
                .build();
        this.registry = CacheRegistry.of(cacheDir);
        this.cache = new HubModelCache(this);
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

    /** The cache this source downloads into, to list, check, update and evict cached models. */
    public ModelCache cache() {
        return cache;
    }

    /**
     * Resolves a cached repo without a file list: the directory of its current snapshot, adopting files cached
     * in the older flat layout if needed, without checking for updates.
     *
     * <p>The directory is not protected from eviction or pruning once this returns; prefer
     * {@link #open(String)}, which keeps it until closed.
     *
     * @throws ModelSourceException if this is called on a repo that has never been
     *         downloaded — the file list is required for the initial download
     */
    @Override
    public Path resolve(String repoId) {
        try (ResolvedModel model = open(repoId)) {
            return model.directory();
        }
    }

    /**
     * Resolves {@code requiredFiles} of a repo, downloading and updating them like {@link #open(String, List)},
     * and returns the directory of the snapshot holding them.
     *
     * <p>The directory is not protected from eviction or pruning once this returns: a later update may prune
     * it. Prefer {@link #open(String, List)}, which keeps it until closed.
     */
    @Override
    public Path resolve(String repoId, List<String> requiredFiles) {
        try (ResolvedModel model = open(repoId, requiredFiles)) {
            return model.directory();
        }
    }

    /**
     * Resolves a model subdirectory, downloading and updating its files like {@link #open(String, String)},
     * and returns its directory in the snapshot holding it.
     *
     * <p>The directory is not protected from eviction or pruning once this returns; prefer
     * {@link #open(String, String)}, which keeps it until closed.
     *
     * @param repoId       the HuggingFace repository ID
     * @param subdirectory the subdirectory path within the repo
     * @return path to the local subdirectory containing the model files
     */
    @Override
    public Path resolve(String repoId, String subdirectory) {
        try (ResolvedModel model = open(repoId, subdirectory)) {
            return model.directory();
        }
    }

    /**
     * Opens a cached repo without a file list: its current snapshot, adopting files cached in the older flat
     * layout if needed, without checking for updates.
     *
     * @throws ModelSourceException if the repo was never downloaded
     */
    @Override
    public ResolvedModel open(String repoId) {
        return open(repoId, Request.ANY);
    }

    /**
     * Opens {@code requiredFiles} of a repo, downloading the missing ones and, when a check is due, updating
     * changed ones into a new snapshot. The returned handle keeps its snapshot on disk until it is closed.
     *
     * @throws IllegalArgumentException if {@code repoId} or a file path is not valid, or would leave the cache
     * @throws ModelSourceException     if the files cannot be resolved, as when offline and not cached
     * @throws ModelDownloadException   if a download fails and the files are not cached
     */
    @Override
    public ResolvedModel open(String repoId, List<String> requiredFiles) {
        List<String> files = new ArrayList<>();
        for (String file : requiredFiles) {
            files.add(RepoLayout.validatePath(file));
        }
        return open(repoId, new Request(List.copyOf(files), null));
    }

    /**
     * Opens a subdirectory of a repo, such as a genai model's, with every file the Hub lists directly in it,
     * downloading and updating them like {@link #open(String, List)}.
     *
     * @return a handle whose {@linkplain ResolvedModel#directory() directory} is the subdirectory
     */
    @Override
    public ResolvedModel open(String repoId, String subdirectory) {
        String sub = subdirectory.endsWith("/") ? subdirectory.substring(0, subdirectory.length() - 1) : subdirectory;
        return open(repoId, new Request(null, RepoLayout.validatePath(sub)));
    }

    /** What a caller asked for: some files, a subdirectory, or (neither) whatever is cached. */
    record Request(List<String> files, String subdirectory) {

        static final Request ANY = new Request(null, null);

        boolean any() {
            return files == null && subdirectory == null;
        }

        boolean satisfiedBy(Path dir) {
            if (files != null) {
                return files.stream().allMatch(f -> Files.isRegularFile(dir.resolve(f)));
            }
            if (subdirectory != null) {
                return !CacheFiles.listShallow(dir.resolve(subdirectory)).isEmpty();
            }
            return Files.isDirectory(dir);
        }

        Path directory(Path dir) {
            return subdirectory != null ? dir.resolve(subdirectory) : dir;
        }
    }

    private ResolvedModel open(String repoId, Request request) {
        RepoLayout repo = RepoLayout.of(cacheDir, repoId);
        CacheRegistry.Repo entry = registry.repo(repo.repoId);
        RepoLock lock;
        try {
            lock = lock(repo, entry);
        } catch (IOException e) {
            return openReadOnly(repo, request, e);
        }
        try (lock) {
            String revision = sync(repo, entry, request, false);
            repairLinks(repo, entry, revision, request);
            entry.acquire(revision);
            return new CacheRegistry.Lease(entry, revision, request.directory(repo.snapshot(revision)));
        }
    }

    RepoLock lock(RepoLayout repo, CacheRegistry.Repo entry) throws IOException {
        return RepoLock.acquire(entry, repo.lockFile);
    }

    /** Serves files in place from a cache this process cannot write to, unmanaged. */
    private ResolvedModel openReadOnly(RepoLayout repo, Request request, IOException cause) {
        String current = repo.readRef();
        Path dir = null;
        if (current != null && request.satisfiedBy(repo.snapshot(current))) {
            dir = repo.snapshot(current);
        } else if (repo.legacy != null && request.satisfiedBy(repo.legacy)) {
            dir = repo.legacy;
        }
        if (dir == null) {
            throw new ModelSourceException("Cannot lock " + repo.repoId + " in the model cache " + cacheDir
                    + ": " + cause.getMessage(), cause);
        }
        if (warnedReadOnly.compareAndSet(false, true)) {
            logger.warn("The model cache {} is not writable ({}); reading cached models in place, without update "
                    + "checks or eviction protection", cacheDir, cause.toString());
        }
        return ResolvedModel.unmanaged(request.directory(dir));
    }

    /**
     * Brings the files {@code request} needs up to date, under the repo lock, and returns the snapshot holding
     * them. With {@code force}, checks the Hub whatever the interval and fails instead of falling back.
     */
    String sync(RepoLayout repo, CacheRegistry.Repo entry, Request request, boolean force) {
        String current = repo.readRef();
        if (request.any() && !force) {
            if (current != null) {
                return current;
            }
            String adopted = adoptLegacy(repo, request, null);
            if (adopted != null) {
                return adopted;
            }
            throw new ModelSourceException(
                    "No cached model found for '" + repo.repoId + "' and no required files specified. "
                            + "Use resolve(modelId, requiredFiles) to trigger download.");
        }
        boolean complete = current != null && request.satisfiedBy(repo.snapshot(current));
        boolean due = force || checkDue(entry, repo.readState(), current);
        if (complete && !due) {
            logger.debug("Cache hit for {}", repo.repoId);
            return current;
        }
        if (offline) {
            if (force) {
                throw new ModelSourceException("Offline mode is on: cannot update " + repo.repoId);
            }
            if (complete) {
                return current;
            }
            String adopted = adoptLegacy(repo, request, current);
            if (adopted != null) {
                return adopted;
            }
            throw offlineMiss(repo.repoId + (request.subdirectory() != null ? "/" + request.subdirectory() : ""));
        }
        if (!due && current != null && request.files() != null) {
            // A file requested later comes from the cached revision; updates wait for the next check
            addFiles(repo, current, missing(request.files(), repo.snapshot(current)));
            return current;
        }
        Revision remote = fetchRevision(repo.repoId);
        if (remote == null) {
            if (force) {
                throw new ModelDownloadException("Cannot check " + repo.repoId
                        + " for updates: the Hub could not be reached", 0);
            }
            if (complete) {
                warnUnreachable(entry);
                return current;
            }
            entry.failedCheck = clock.instant();
            return syncWithoutApi(repo, request, current);
        }
        return applyRevision(repo, entry, request, current, complete, remote, force);
    }

    private boolean checkDue(CacheRegistry.Repo entry, RepoState state, String current) {
        Instant now = clock.instant();
        Instant failed = entry.failedCheck;
        if (failed != null && now.isBefore(failed.plus(checkInterval))) {
            return false;
        }
        return state == null || UNKNOWN.equals(current) || !state.remoteChangedFiles().isEmpty()
                || !now.isBefore(state.checkedAt().plus(checkInterval));
    }

    /** Moves the repo to the Hub's revision when cached files changed; otherwise records the check. */
    private String applyRevision(RepoLayout repo, CacheRegistry.Repo entry, Request request, String current,
                                 boolean complete, Revision remote, boolean force) {
        String target = remote.commit();
        Map<String, String> hashes = new HashMap<>();
        Path currentDir = current != null ? repo.snapshot(current) : null;
        SnapshotManifest currentManifest = current != null ? repo.readManifest(current) : null;
        List<String> cachedFiles = currentDir != null ? CacheFiles.list(currentDir) : List.of();

        Set<String> subdirectories = new LinkedHashSet<>(
                currentManifest != null ? currentManifest.subdirectories() : Set.of());
        if (request.subdirectory() != null) {
            subdirectories.add(request.subdirectory());
        }
        Set<String> wanted = new LinkedHashSet<>();
        if (request.files() != null) {
            wanted.addAll(request.files());
        }
        wanted.addAll(cachedFiles);
        for (String subdirectory : subdirectories) {
            List<String> listed = listing(remote, subdirectory);
            if (listed.isEmpty() && subdirectory.equals(request.subdirectory())) {
                throw new ModelSourceException("No files found under " + repo.repoId + "/" + subdirectory);
            }
            wanted.addAll(listed);
        }
        List<String> changed = changedFiles(repo, current, cachedFiles, currentManifest, remote, hashes);
        Instant now = clock.instant();
        try {
            if (current != null && !current.equals(target) && !UNKNOWN.equals(current) && changed.isEmpty()
                    && missing(wanted, currentDir).isEmpty()) {
                // Nothing cached changed and nothing is missing: keep the snapshot, record the check
                if (!subdirectories.equals(currentManifest != null ? currentManifest.subdirectories() : Set.of())) {
                    addFiles(repo, current, Set.of(), subdirectories);
                }
                writeState(repo, now, target, List.of());
                entry.failedCheck = null;
                return current;
            }
            if (!changed.isEmpty()) {
                logger.info("{} changed on the Hub (revision {}); updating {}", repo.repoId, target, changed);
            }
            buildSnapshot(repo, entry, target, wanted, request, remote, subdirectories, hashes);
        } catch (ModelSourceException e) {
            if (force || !complete) {
                throw e;
            }
            entry.failedCheck = now;
            writeState(repo, now, target, changed);
            logger.warn("Could not update {} to revision {}; using the cached revision {}: {}",
                    repo.repoId, target, current, e.getMessage());
            return current;
        }
        repo.writeRef(target);
        writeState(repo, now, target, List.of());
        entry.failedCheck = null;
        if (!target.equals(current) && pruneMode == PruneMode.AUTO) {
            try {
                EvictionResult pruned = cache.removeSuperseded(repo, entry, false);
                if (!pruned.removed().isEmpty()) {
                    logger.debug("Pruned {} of {}", pruned.removed(), repo.repoId);
                }
            } catch (RuntimeException e) {
                logger.warn("Could not prune the old snapshots of {}: {}", repo.repoId, e.toString());
            }
        }
        return target;
    }

    /** The cached files of {@code current} whose content differs from {@code remote}'s. */
    List<String> changedFiles(RepoLayout repo, String current, List<String> cachedFiles,
                              SnapshotManifest manifest, Revision remote, Map<String, String> hashes) {
        List<String> changed = new ArrayList<>();
        for (String path : cachedFiles) {
            Revision.RemoteFile file = remote.files().get(path);
            if (file != null && !file.id().equals(localId(repo.snapshot(current).resolve(path), manifest, path,
                    file.lfs(), hashes))) {
                changed.add(path);
            }
        }
        return changed;
    }

    /**
     * Fills snapshot {@code target} with {@code wanted} files at {@code remote}'s commit, then records it. Files
     * another snapshot or the flat layout already holds are reused: copied out of a snapshot that may still be in
     * use, otherwise moved, once every download succeeded, so that a failed update leaves the current snapshot
     * whole.
     */
    private void buildSnapshot(RepoLayout repo, CacheRegistry.Repo entry, String target, Set<String> wanted,
                               Request request, Revision remote, Set<String> subdirectories,
                               Map<String, String> hashes) {
        Path dir = repo.snapshot(target);
        Map<String, SnapshotManifest> manifests = new HashMap<>();
        List<String> others = repo.snapshotNames().stream().filter(n -> !n.equals(target)).toList();
        Map<String, Revision.RemoteFile> placed = new LinkedHashMap<>();
        List<Reusable> moves = new ArrayList<>();
        boolean announced = false;
        try {
            for (String path : wanted) {
                Revision.RemoteFile file = remote.files().get(path);
                boolean requested = request.files() != null && request.files().contains(path);
                if (file == null && !requested) {
                    continue;   // deleted on the Hub
                }
                Path local = dir.resolve(path);
                if (!Files.isRegularFile(local)) {
                    Reusable reusable = reusable(repo, entry, path, file, others, manifests, hashes);
                    if (reusable == null) {
                        if (!announced) {
                            logger.info("Downloading model files for {} to {}", repo.repoId, dir);
                            announced = true;
                        }
                        download(repo, target, path, local);
                    } else if (reusable.movable()) {
                        moves.add(reusable);
                    } else {
                        CacheFiles.copy(reusable.source(), local, repo.tmp());
                    }
                }
                placed.put(path, file);
            }
            move(repo, dir, moves);
        } catch (IOException e) {
            throw new ModelSourceException("Failed to place the files of " + repo.repoId + " in " + dir + ": "
                    + e.getMessage(), e);
        }
        Map<String, SnapshotManifest.Entry> entries = new LinkedHashMap<>();
        placed.forEach((path, file) -> entries.put(path, new SnapshotManifest.Entry(file != null ? file.id() : null,
                file != null && file.lfs(), CacheFiles.size(dir.resolve(path)))));
        SnapshotManifest manifest = repo.readManifest(target);
        if (manifest == null) {
            manifest = SnapshotManifest.empty(target, clock.instant());
        }
        manifest.with(entries, subdirectories).write(repo.manifestFile(target));
    }

    /**
     * A file to reuse: its path in the repo, where it is, the snapshot holding it ({@code null} for the flat
     * layout), and whether it may be moved rather than copied.
     */
    private record Reusable(String path, Path source, String snapshot, boolean movable) {
    }

    /**
     * Where another snapshot or the flat layout holds {@code file} with the Hub's id, or {@code null}. A snapshot's
     * file may be moved when the snapshot could be evicted: no model in this JVM uses it and it is older than the
     * minimum age, which protects a load in progress in another process. Flat-layout files are always moved.
     */
    private Reusable reusable(RepoLayout repo, CacheRegistry.Repo entry, String path, Revision.RemoteFile file,
                              List<String> others, Map<String, SnapshotManifest> manifests,
                              Map<String, String> hashes) {
        if (file == null) {
            return null;
        }
        Instant now = clock.instant();
        for (String other : others) {
            Path candidate = repo.snapshot(other).resolve(path);
            if (Files.isRegularFile(candidate)) {
                SnapshotManifest manifest = manifests.computeIfAbsent(other, repo::readManifest);
                if (file.id().equals(localId(candidate, manifest, path, file.lfs(), hashes))) {
                    boolean movable = entry.leases(other) == 0
                            && !now.isBefore(repo.createdAt(other).plus(minimumSnapshotAge));
                    return new Reusable(path, candidate, other, movable);
                }
            }
        }
        if (repo.legacy != null) {
            Path legacy = repo.legacy.resolve(path);
            if (Files.isRegularFile(legacy) && file.id().equals(localId(legacy, null, path, file.lfs(), hashes))) {
                logger.debug("Adopting {} of {} from the flat cache layout", path, repo.repoId);
                return new Reusable(path, legacy, null, true);
            }
        }
        return null;
    }

    /**
     * Moves {@code moves} into the snapshot {@code dir}, copying where a move fails, and removes what moved from
     * the manifests of the snapshots it left. Those snapshots are superseded and were never in use, so they are
     * not served again: only the current snapshot is, and the caller points {@code refs/main} at {@code dir}.
     */
    private void move(RepoLayout repo, Path dir, List<Reusable> moves) throws IOException {
        Map<String, Set<String>> movedOut = new LinkedHashMap<>();
        for (Reusable reusable : moves) {
            Path local = dir.resolve(reusable.path());
            if (CacheFiles.move(reusable.source(), local)) {
                if (reusable.snapshot() != null) {
                    movedOut.computeIfAbsent(reusable.snapshot(), k -> new LinkedHashSet<>()).add(reusable.path());
                } else {
                    CacheFiles.deleteEmptyDirectories(reusable.source().getParent(), cacheDir);
                }
            } else {
                CacheFiles.copy(reusable.source(), local, repo.tmp());
                if (reusable.snapshot() == null) {
                    deleteAdopted(reusable.source());
                }
            }
        }
        movedOut.forEach((snapshot, paths) -> {
            SnapshotManifest manifest = repo.readManifest(snapshot);
            if (manifest != null) {
                manifest.without(paths).write(repo.manifestFile(snapshot));
            }
        });
    }

    /** Deletes a flat-layout file now held by a snapshot, and the directories it leaves empty; best effort. */
    private void deleteAdopted(Path legacy) {
        try {
            Files.deleteIfExists(legacy);
            CacheFiles.deleteEmptyDirectories(legacy.getParent(), cacheDir);
        } catch (IOException e) {
            logger.debug("Cannot delete {}, which prune() retries: {}", legacy, e.toString());
        }
    }

    /**
     * Gives each file {@code request} opens in snapshot {@code revision} a single hard link, as ONNX Runtime
     * requires of external data: caches written by earlier builds linked files across snapshots and the flat
     * layout. A link from the flat layout is removed, since the snapshot adopted that file; any other gets the
     * file a private copy. Skipped where link counts cannot be read, as on Windows, and for a snapshot that a model
     * in this JVM uses. The caller holds the repo lock.
     */
    private void repairLinks(RepoLayout repo, CacheRegistry.Repo entry, String revision, Request request) {
        Path dir = repo.snapshot(revision);
        List<String> paths;
        if (request.files() != null) {
            paths = request.files();
        } else if (request.subdirectory() != null) {
            paths = CacheFiles.listShallow(dir.resolve(request.subdirectory())).stream()
                    .map(name -> request.subdirectory() + "/" + name).toList();
        } else {
            paths = CacheFiles.list(dir);
        }
        for (String path : paths) {
            Path file = dir.resolve(path);
            if (CacheFiles.linkCount(file) <= 1) {
                continue;
            }
            if (entry.leases(revision) > 0) {
                if (entry.warnedLinkedInUse.compareAndSet(false, true)) {
                    logger.warn("{} has hard-linked files in snapshot {}, which a loaded model uses, so they are "
                            + "left as they are; ONNX Runtime may refuse their external data", repo.repoId, revision);
                }
                return;
            }
            if (entry.reportedRepair.compareAndSet(false, true)) {
                logger.info("Replacing the hard-linked files of {} in {}, which ONNX Runtime refuses as external "
                        + "data", repo.repoId, dir);
            }
            try {
                if (repo.legacy != null) {
                    Path legacy = repo.legacy.resolve(path);
                    if (Files.isRegularFile(legacy) && Files.isSameFile(file, legacy)) {
                        deleteAdopted(legacy);
                    }
                }
                if (CacheFiles.linkCount(file) > 1) {
                    CacheFiles.copy(file, file, repo.tmp());
                }
            } catch (IOException e) {
                logger.warn("Could not replace the hard-linked {} with a copy: {}", file, e.toString());
            }
        }
    }

    /** The Hub id of a local file: recorded in {@code manifest}, else computed (and remembered in {@code hashes}). */
    private static String localId(Path file, SnapshotManifest manifest, String path, boolean lfs,
                                  Map<String, String> hashes) {
        String recorded = manifest != null ? manifest.id(path, lfs) : null;
        if (recorded != null) {
            return recorded;
        }
        return hashes.computeIfAbsent(file.toAbsolutePath() + "|" + lfs,
                k -> lfs ? FileHashes.sha256(file) : FileHashes.gitBlobId(file));
    }

    /** Adds files to the existing snapshot {@code revision}, downloaded at that revision, ids unrecorded. */
    private void addFiles(RepoLayout repo, String revision, Set<String> files) {
        addFiles(repo, revision, files, Set.of());
    }

    private void addFiles(RepoLayout repo, String revision, Set<String> files, Set<String> subdirectories) {
        Map<String, SnapshotManifest.Entry> entries = new LinkedHashMap<>();
        Path dir = repo.snapshot(revision);
        if (!files.isEmpty()) {
            logger.info("Downloading model files for {} to {}", repo.repoId, dir);
        }
        for (String path : files) {
            Path local = dir.resolve(path);
            download(repo, UNKNOWN.equals(revision) ? "main" : revision, path, local);
            entries.put(path, new SnapshotManifest.Entry(null, false, CacheFiles.size(local)));
        }
        if (!entries.isEmpty() || !subdirectories.isEmpty()) {
            SnapshotManifest manifest = repo.readManifest(revision);
            if (manifest == null) {
                manifest = SnapshotManifest.empty(UNKNOWN.equals(revision) ? null : revision, clock.instant());
            }
            manifest.with(entries, subdirectories).write(repo.manifestFile(revision));
        }
    }

    private static Set<String> missing(Collection<String> files, Path dir) {
        Set<String> missing = new LinkedHashSet<>();
        for (String file : files) {
            if (!Files.isRegularFile(dir.resolve(file))) {
                missing.add(file);
            }
        }
        return missing;
    }

    /**
     * When the revision API fails but downloads may work: adopts the flat layout, or fetches the missing files at
     * the cached revision, or at {@code main} with the commit the download reports.
     */
    private String syncWithoutApi(RepoLayout repo, Request request, String current) {
        if (request.subdirectory() != null) {
            String adopted = adoptLegacy(repo, request, current);
            if (adopted != null) {
                return adopted;
            }
            throw new ModelDownloadException("Cannot list the files of " + repo.repoId + "/"
                    + request.subdirectory() + ": the Hub could not be reached", 0);
        }
        String adopted = adoptLegacy(repo, request, current);
        if (adopted != null) {
            return adopted;
        }
        Set<String> missing = current != null
                ? missing(request.files(), repo.snapshot(current)) : new LinkedHashSet<>(request.files());
        String revision = current;
        if (revision == null && missing.isEmpty()) {
            throw new ModelSourceException("No files requested for " + repo.repoId);
        }
        if (revision == null) {
            // The first download names the commit the files come from
            String first = missing.iterator().next();
            Fetched fetched = fetch(repo, "main", first);
            revision = fetched.commit() != null ? fetched.commit() : UNKNOWN;
            try {
                Path local = repo.snapshot(revision).resolve(first);
                Files.createDirectories(local.getParent());
                Files.move(fetched.file(), local, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException e) {
                throw new ModelDownloadException("Failed to store " + first + ": " + e.getMessage(), e);
            } finally {
                deleteQuietly(fetched.file());
            }
            missing.remove(first);
            Map<String, SnapshotManifest.Entry> entries = new LinkedHashMap<>();
            entries.put(first, new SnapshotManifest.Entry(null, false,
                    CacheFiles.size(repo.snapshot(revision).resolve(first))));
            SnapshotManifest.empty(UNKNOWN.equals(revision) ? null : revision, clock.instant())
                    .with(entries, Set.of()).write(repo.manifestFile(revision));
        }
        addFiles(repo, revision, missing);
        if (current == null) {
            repo.writeRef(revision);
        }
        return revision;
    }

    /**
     * Adopts {@code request}'s files from the flat layout into the {@value ModelRevision#UNKNOWN} snapshot, moving
     * them; {@code null} if neither holds them or the current snapshot has a known commit.
     */
    private String adoptLegacy(RepoLayout repo, Request request, String current) {
        if (repo.legacy == null || (current != null && !UNKNOWN.equals(current))
                || !Files.isDirectory(repo.legacy)) {
            return null;
        }
        List<String> files;
        if (request.files() != null) {
            files = request.files();
        } else if (request.subdirectory() != null) {
            files = CacheFiles.listShallow(repo.legacy.resolve(request.subdirectory())).stream()
                    .map(name -> request.subdirectory() + "/" + name).toList();
        } else {
            files = CacheFiles.list(repo.legacy).stream()
                    .filter(f -> !f.endsWith(".tmp") && !f.endsWith(".inference4j-revision.json")).toList();
        }
        Path dir = repo.snapshot(UNKNOWN);
        // A file adopted earlier is in the snapshot, no longer in the flat layout
        if (files.isEmpty() || !files.stream().allMatch(f -> Files.isRegularFile(dir.resolve(f))
                || Files.isRegularFile(repo.legacy.resolve(f)))) {
            return null;
        }
        Map<String, SnapshotManifest.Entry> entries = new LinkedHashMap<>();
        try {
            for (String file : files) {
                Path local = dir.resolve(file);
                if (!Files.isRegularFile(local)) {
                    Path legacy = repo.legacy.resolve(file);
                    if (CacheFiles.move(legacy, local)) {
                        CacheFiles.deleteEmptyDirectories(legacy.getParent(), cacheDir);
                    } else {
                        CacheFiles.copy(legacy, local, repo.tmp());
                        deleteAdopted(legacy);
                    }
                }
                entries.put(file, new SnapshotManifest.Entry(null, false, CacheFiles.size(local)));
            }
        } catch (IOException e) {
            throw new ModelSourceException("Failed to adopt the cached files of " + repo.repoId + ": "
                    + e.getMessage(), e);
        }
        SnapshotManifest manifest = repo.readManifest(UNKNOWN);
        if (manifest == null) {
            manifest = SnapshotManifest.empty(null, clock.instant());
        }
        manifest.with(entries, request.subdirectory() != null ? Set.of(request.subdirectory()) : Set.of())
                .write(repo.manifestFile(UNKNOWN));
        if (current == null) {
            repo.writeRef(UNKNOWN);
            logger.info("Adopted the cached files of {} from {}; they will be checked against the Hub when it can "
                    + "be reached", repo.repoId, repo.legacy);
        }
        return UNKNOWN;
    }

    /** The files the Hub lists directly in {@code subdirectory}. */
    private static List<String> listing(Revision remote, String subdirectory) {
        String prefix = subdirectory + "/";
        return remote.files().keySet().stream()
                .filter(p -> p.startsWith(prefix) && p.indexOf('/', prefix.length()) < 0)
                .filter(RepoLayout::isValidPath)
                .sorted()
                .toList();
    }

    private void writeState(RepoLayout repo, Instant checkedAt, String remoteCommit, List<String> changed) {
        new RepoState(repo.repoId, checkedAt, remoteCommit, changed).write(repo.stateFile());
    }

    /** The repo's current commit and file ids, or {@code null} when the Hub cannot answer. */
    Revision fetchRevision(String repoId) {
        URI uri = URI.create(endpoint + "/api/models/" + repoId + "/revision/main?blobs=true");
        HttpRequest request = HttpRequest.newBuilder(uri).timeout(API_TIMEOUT).GET().build();
        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                logger.debug("Revision check of {} failed: HTTP {}", repoId, response.statusCode());
                return null;
            }
            Revision revision = Revision.parse(response.body());
            RepoLayout.validateRevision(revision.commit());
            return revision;
        } catch (IOException | IllegalArgumentException e) {
            logger.debug("Revision check of {} failed: {}", repoId, e.toString());
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    private void warnUnreachable(CacheRegistry.Repo entry) {
        entry.failedCheck = clock.instant();
        if (entry.warnedUnreachable.compareAndSet(false, true)) {
            logger.warn("Could not check {} for updates on the Hub; using the cached files", entry.repoId);
        }
    }

    private static ModelSourceException offlineMiss(String what) {
        return new ModelSourceException("Offline mode is on and " + what + " is not fully cached");
    }

    /** A downloaded file in the repo's {@code tmp/}, and the commit the Hub said it belongs to, if it said. */
    private record Fetched(Path file, String commit) {
    }

    /** Downloads {@code remotePath} at {@code revision} to {@code target}, which appears complete or not at all. */
    private void download(RepoLayout repo, String revision, String remotePath, Path target) {
        Fetched fetched = fetch(repo, revision, remotePath);
        try {
            Files.createDirectories(target.getParent());
            Files.move(fetched.file(), target, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new ModelDownloadException("Failed to store " + target + ": " + e.getMessage(), e);
        } finally {
            deleteQuietly(fetched.file());
        }
    }

    private Fetched fetch(RepoLayout repo, String revision, String remotePath) {
        URI uri = URI.create(endpoint + "/" + repo.repoId + "/resolve/" + revision + "/" + remotePath);
        Path tmpFile = repo.tmp().resolve(UUID.randomUUID() + ".part");
        HttpRequest request = HttpRequest.newBuilder(uri).GET().build();
        try {
            Files.createDirectories(repo.tmp());
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
            logger.debug("Downloaded {}", remotePath);
            return new Fetched(tmpFile, repoCommit(response));
        } catch (ModelDownloadException e) {
            throw e;
        } catch (IOException | InterruptedException | UncheckedIOException e) {
            deleteQuietly(tmpFile);
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new ModelDownloadException(
                    "Failed to download " + uri + ": " + e.getMessage(), e);
        }
    }

    /** The commit the Hub reports in {@code X-Repo-Commit}, on this response or a redirect before it. */
    private static String repoCommit(HttpResponse<?> response) {
        for (Optional<? extends HttpResponse<?>> r = Optional.of(response); r.isPresent();
             r = r.get().previousResponse()) {
            Optional<String> commit = r.get().headers().firstValue("X-Repo-Commit");
            if (commit.isPresent()) {
                try {
                    return RepoLayout.validateRevision(commit.get().strip());
                } catch (IllegalArgumentException e) {
                    return null;
                }
            }
        }
        return null;
    }

    private static void deleteQuietly(Path file) {
        try {
            Files.deleteIfExists(file);
        } catch (IOException ignored) {
            // a leftover in tmp/, removed by the next prune
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

    Path cacheDir() {
        return cacheDir;
    }

    boolean offline() {
        return offline;
    }

    Clock clock() {
        return clock;
    }

    Duration minimumSnapshotAge() {
        return minimumSnapshotAge;
    }

    CacheRegistry registry() {
        return registry;
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
            return requireNotNegative(Duration.parse(value), "cache check interval");
        } catch (DateTimeParseException e) {
            throw new IllegalArgumentException("Invalid cache check interval '" + value
                    + "': use an ISO-8601 duration such as PT1H or P1D", e);
        }
    }

    private static PruneMode defaultPruneMode() {
        String value = setting("inference4j.cache.prune", "INFERENCE4J_CACHE_PRUNE");
        if (value == null) {
            return PruneMode.AUTO;
        }
        try {
            return PruneMode.valueOf(value.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid cache prune mode '" + value + "': use auto or manual", e);
        }
    }

    private static Duration requireNotNegative(Duration duration, String what) {
        if (duration.isNegative()) {
            throw new IllegalArgumentException("The " + what + " must not be negative, got " + duration);
        }
        return duration;
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
        private PruneMode pruneMode;
        private Duration minimumSnapshotAge = DEFAULT_MINIMUM_SNAPSHOT_AGE;

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
            this.checkInterval = requireNotNegative(Objects.requireNonNull(checkInterval, "checkInterval"),
                    "cache check interval");
            return this;
        }

        /**
         * What happens to the snapshots an update supersedes. Defaults to the system property
         * {@code inference4j.cache.prune}, else the environment variable {@code INFERENCE4J_CACHE_PRUNE}, else
         * {@link PruneMode#AUTO}.
         */
        public Builder pruneMode(PruneMode pruneMode) {
            this.pruneMode = Objects.requireNonNull(pruneMode, "pruneMode");
            return this;
        }

        /**
         * How old a snapshot must be before pruning or eviction may delete it, counted from its creation.
         * Leases only cover models in this JVM, so this protects a snapshot another process sharing the cache
         * may still be loading. Defaults to 10 minutes.
         *
         * @throws IllegalArgumentException if {@code minimumSnapshotAge} is negative
         */
        public Builder minimumSnapshotAge(Duration minimumSnapshotAge) {
            this.minimumSnapshotAge = requireNotNegative(
                    Objects.requireNonNull(minimumSnapshotAge, "minimumSnapshotAge"), "minimum snapshot age");
            return this;
        }

        /** The clock that times the check interval and snapshot ages. Defaults to the system clock. */
        public Builder clock(Clock clock) {
            this.clock = Objects.requireNonNull(clock, "clock");
            return this;
        }

        public HuggingFaceModelSource build() {
            return new HuggingFaceModelSource(this);
        }
    }
}
