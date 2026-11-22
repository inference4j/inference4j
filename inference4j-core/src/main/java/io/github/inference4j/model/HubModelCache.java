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
import io.github.inference4j.model.CacheRegistry.RepoLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.function.Function;

/** The {@link ModelCache} of a {@link HuggingFaceModelSource}'s cache directory. */
final class HubModelCache implements ModelCache {

    private static final Logger logger = LoggerFactory.getLogger(HubModelCache.class);

    private final HuggingFaceModelSource source;

    HubModelCache(HuggingFaceModelSource source) {
        this.source = source;
    }

    @Override
    public Path directory() {
        return source.cacheDir();
    }

    @Override
    public List<CachedModel> list() {
        Path hub = source.cacheDir().resolve(RepoLayout.HUB);
        List<CachedModel> models = new ArrayList<>();
        for (String name : CacheFiles.directories(hub)) {
            if (!name.startsWith(RepoLayout.PREFIX)) {
                continue;
            }
            RepoLayout repo = layoutOf(hub.resolve(name));
            if (repo != null) {
                describe(repo).ifPresent(models::add);
            }
        }
        models.sort(Comparator.comparing(CachedModel::repoId));
        return models;
    }

    @Override
    public Optional<CachedModel> find(String repoId) {
        return describe(RepoLayout.of(source.cacheDir(), repoId));
    }

    @Override
    public UpdateCheck check(String repoId) {
        RepoLayout repo = RepoLayout.of(source.cacheDir(), repoId);
        CacheRegistry.Repo entry = source.registry().repo(repo.repoId);
        try (RepoLock lock = lock(repo, entry)) {
            String current = repo.readRef();
            if (current == null) {
                throw new ModelSourceException(repoId + " is not cached");
            }
            if (source.offline()) {
                throw new ModelSourceException("Offline mode is on: cannot check " + repoId + " for updates");
            }
            Revision remote = source.fetchRevision(repo.repoId);
            if (remote == null) {
                throw new ModelDownloadException("Cannot check " + repoId
                        + " for updates: the Hub could not be reached", 0);
            }
            List<String> changed = source.changedFiles(repo, current, CacheFiles.list(repo.snapshot(current)),
                    repo.readManifest(current), remote, new HashMap<>());
            Instant now = source.clock().instant();
            new RepoState(repo.repoId, now, remote.commit(), changed).write(repo.stateFile());
            entry.failedCheck = null;
            return new UpdateCheck(repo.repoId, current, remote.commit(), changed, now);
        }
    }

    @Override
    public CachedModel update(String repoId) {
        RepoLayout repo = RepoLayout.of(source.cacheDir(), repoId);
        CacheRegistry.Repo entry = source.registry().repo(repo.repoId);
        try (RepoLock lock = lock(repo, entry)) {
            if (repo.readRef() == null) {
                throw new ModelSourceException(repoId + " is not cached");
            }
            source.sync(repo, entry, HuggingFaceModelSource.Request.ANY, true);
            return describe(repo).orElseThrow();
        }
    }

    @Override
    public EvictionResult evict(String repoId) {
        RepoLayout repo = RepoLayout.of(source.cacheDir(), repoId);
        CacheRegistry.Repo entry = source.registry().repo(repo.repoId);
        try (RepoLock lock = lock(repo, entry)) {
            return remove(repo, entry, repo.snapshotNames(), true);
        }
    }

    @Override
    public EvictionResult evict(String repoId, String revision) {
        RepoLayout repo = RepoLayout.of(source.cacheDir(), repoId);
        RepoLayout.validateRevision(revision);
        CacheRegistry.Repo entry = source.registry().repo(repo.repoId);
        try (RepoLock lock = lock(repo, entry)) {
            return remove(repo, entry, repo.snapshotNames().contains(revision) ? List.of(revision) : List.of(),
                    false);
        }
    }

    @Override
    public EvictionResult prune() {
        Path hub = source.cacheDir().resolve(RepoLayout.HUB);
        List<ModelRevision> removed = new ArrayList<>();
        List<ModelRevision> inUse = new ArrayList<>();
        long freed = 0;
        for (String name : CacheFiles.directories(hub)) {
            if (!name.startsWith(RepoLayout.PREFIX)) {
                continue;
            }
            RepoLayout repo = layoutOf(hub.resolve(name));
            if (repo == null) {
                continue;
            }
            CacheRegistry.Repo entry = source.registry().repo(repo.repoId);
            try (RepoLock lock = lock(repo, entry)) {
                EvictionResult result = removeSuperseded(repo, entry, true);
                removed.addAll(result.removed());
                inUse.addAll(result.inUse());
                freed += result.freedBytes();
            }
        }
        return new EvictionResult(removed, inUse, freed);
    }

    /**
     * Removes the repo's snapshots other than the current one; with {@code legacyToo}, also the flat-layout
     * files adopted into its snapshots. The caller holds the repo lock.
     */
    EvictionResult removeSuperseded(RepoLayout repo, CacheRegistry.Repo entry, boolean legacyToo) {
        String current = repo.readRef();
        List<String> superseded = repo.snapshotNames().stream().filter(n -> !n.equals(current)).toList();
        return remove(repo, entry, superseded, legacyToo && current != null);
    }

    /** Deletes the {@code revisions} not in use; the caller holds the repo lock. */
    private EvictionResult remove(RepoLayout repo, CacheRegistry.Repo entry, Collection<String> revisions,
                                  boolean legacyToo) {
        String current = repo.readRef();
        Instant now = source.clock().instant();
        Set<String> adoptedPaths = new LinkedHashSet<>();
        if (legacyToo) {
            for (String name : repo.snapshotNames()) {
                adoptedPaths.addAll(CacheFiles.list(repo.snapshot(name)));
            }
        }
        List<ModelRevision> removed = new ArrayList<>();
        List<ModelRevision> inUse = new ArrayList<>();
        Map<Object, Long> removedBytes = new HashMap<>();
        for (String revision : revisions) {
            ModelRevision id = new ModelRevision(repo.repoId, revision);
            if (entry.leases(revision) > 0 || now.isBefore(repo.createdAt(revision).plus(source.minimumSnapshotAge()))) {
                inUse.add(id);
                continue;
            }
            Path snapshot = repo.snapshot(revision);
            Map<Object, Long> bytes = sizes(snapshot, CacheFiles.list(snapshot));
            Path trash = repo.trash().resolve(revision + "-" + UUID.randomUUID());
            try {
                Files.createDirectories(repo.trash());
                Files.move(snapshot, trash, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException e) {
                // Open files keep it in place on some systems, such as Windows
                logger.debug("Cannot move {} to the trash: {}", snapshot, e.toString());
                inUse.add(id);
                continue;
            }
            if (revision.equals(current)) {
                repo.deleteRef();
            }
            try {
                Files.deleteIfExists(repo.manifestFile(revision));
            } catch (IOException e) {
                logger.debug("Cannot delete {}: {}", repo.manifestFile(revision), e.toString());
            }
            removedBytes.putAll(bytes);
            removed.add(id);
        }
        if (legacyToo && repo.legacy != null) {
            removedBytes.putAll(removeAdopted(repo, adoptedPaths));
        }
        CacheFiles.deleteRecursively(repo.trash());
        CacheFiles.deleteRecursively(repo.tmp());
        Set<Object> kept = new HashSet<>();
        for (String name : repo.snapshotNames()) {
            kept.addAll(sizes(repo.snapshot(name), CacheFiles.list(repo.snapshot(name))).keySet());
        }
        if (repo.legacy != null) {
            kept.addAll(sizes(repo.legacy, CacheFiles.list(repo.legacy)).keySet());
        }
        long freed = removedBytes.entrySet().stream().filter(e -> !kept.contains(e.getKey()))
                .mapToLong(Map.Entry::getValue).sum();
        if (repo.snapshotNames().isEmpty() && repo.readRef() == null) {
            CacheFiles.deleteRecursively(repo.root);
        }
        return new EvictionResult(removed, inUse, freed);
    }

    /** Deletes the flat-layout files at {@code paths}, which snapshots adopted; their sizes by file key. */
    private static Map<Object, Long> removeAdopted(RepoLayout repo, Set<String> paths) {
        Map<Object, Long> bytes = new HashMap<>();
        for (String path : paths) {
            Path file = repo.legacy.resolve(path);
            if (!Files.isRegularFile(file)) {
                continue;
            }
            Object key = CacheFiles.key(file);
            long size = CacheFiles.size(file);
            try {
                Files.delete(file);
                bytes.put(key, size);
                CacheFiles.deleteEmptyDirectories(file.getParent(), repo.legacy.getParent());
            } catch (IOException e) {
                logger.debug("Cannot delete {}: {}", file, e.toString());
            }
        }
        // The org directory, when the repo was its last
        Path cacheDir = repo.root.getParent().getParent();
        CacheFiles.deleteEmptyDirectories(repo.legacy, cacheDir);
        return bytes;
    }

    private static Map<Object, Long> sizes(Path dir, List<String> files) {
        Map<Object, Long> sizes = new LinkedHashMap<>();
        for (String file : files) {
            Path path = dir.resolve(file);
            sizes.put(CacheFiles.key(path), CacheFiles.size(path));
        }
        return sizes;
    }

    private Optional<CachedModel> describe(RepoLayout repo) {
        List<String> names = repo.snapshotNames();
        String current = repo.readRef();
        if (names.isEmpty() && current == null) {
            return Optional.empty();
        }
        RepoState state = repo.readState();
        Set<Object> counted = new HashSet<>();
        long total = 0;
        List<CachedSnapshot> snapshots = new ArrayList<>();
        for (String name : names) {
            Path dir = repo.snapshot(name);
            SnapshotManifest manifest = repo.readManifest(name);
            Function<String, String> ids = path -> {
                if (manifest == null) {
                    return null;
                }
                SnapshotManifest.Entry entry = manifest.files().get(path);
                return entry != null ? entry.id() : null;
            };
            List<CachedFile> files = new ArrayList<>();
            long size = 0;
            for (String path : CacheFiles.list(dir)) {
                Path file = dir.resolve(path);
                long bytes = CacheFiles.size(file);
                files.add(new CachedFile(path, ids.apply(path), bytes));
                size += bytes;
                if (counted.add(CacheFiles.key(file))) {
                    total += bytes;
                }
            }
            snapshots.add(new CachedSnapshot(name, name.equals(current),
                    source.registry().leases(repo.repoId, name), files, size));
        }
        return Optional.of(new CachedModel(repo.repoId, current,
                state != null ? state.checkedAt() : null,
                state != null ? state.remoteCommit() : null,
                state != null && !state.remoteChangedFiles().isEmpty(),
                snapshots, total));
    }

    /** The layout of the repo directory {@code dir}, or {@code null} if its name maps to no valid repo id. */
    private RepoLayout layoutOf(Path dir) {
        try {
            RepoLayout repo = RepoLayout.of(source.cacheDir(), RepoLayout.repoIdOf(dir));
            return repo.root.getFileName().equals(dir.getFileName()) ? repo : null;
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    private RepoLock lock(RepoLayout repo, CacheRegistry.Repo entry) {
        try {
            return source.lock(repo, entry);
        } catch (IOException e) {
            throw new ModelSourceException("Cannot lock " + repo.repoId + " in the model cache "
                    + source.cacheDir() + ": " + e.getMessage(), e);
        }
    }
}
