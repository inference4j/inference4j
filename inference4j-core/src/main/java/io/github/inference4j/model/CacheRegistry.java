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

import java.io.IOException;
import java.io.InterruptedIOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The in-JVM state of each cache directory, shared by every source on it: per repo, the lock that serializes
 * sync and eviction, and how many open handles use each snapshot.
 */
final class CacheRegistry {

    private static final ConcurrentHashMap<Path, CacheRegistry> REGISTRIES = new ConcurrentHashMap<>();

    private final ConcurrentHashMap<String, Repo> repos = new ConcurrentHashMap<>();

    private CacheRegistry() {
    }

    /** The registry of {@code cacheDir}, however the path to it is spelled. */
    static CacheRegistry of(Path cacheDir) {
        return REGISTRIES.computeIfAbsent(canonical(cacheDir), k -> new CacheRegistry());
    }

    Repo repo(String repoId) {
        return repos.computeIfAbsent(repoId, Repo::new);
    }

    /** How many handles use {@code revision} of {@code repoId}. */
    int leases(String repoId, String revision) {
        Repo repo = repos.get(repoId);
        return repo == null ? 0 : repo.leases(revision);
    }

    /** The real path of the nearest existing ancestor, followed by the rest, so that links resolve alike. */
    static Path canonical(Path dir) {
        Path absolute = dir.toAbsolutePath().normalize();
        Path existing = absolute;
        while (existing != null && !Files.exists(existing)) {
            existing = existing.getParent();
        }
        if (existing == null) {
            return absolute;
        }
        try {
            return existing.toRealPath().resolve(existing.relativize(absolute));
        } catch (IOException e) {
            return absolute;
        }
    }

    /** One repo's lock, leases and check bookkeeping. */
    static final class Repo {

        final String repoId;
        final ReentrantLock lock = new ReentrantLock();
        /** The exclusive file lock, held while {@link #lock} is; guarded by it. */
        FileChannel channel;
        FileLock fileLock;
        /** When a check last failed, so an unreachable Hub is not asked again until the interval passes. */
        volatile Instant failedCheck;
        final AtomicBoolean warnedUnreachable = new AtomicBoolean();
        private final Map<String, Integer> leases = new HashMap<>();

        private Repo(String repoId) {
            this.repoId = repoId;
        }

        /** Takes a lease on {@code revision}; callers hold {@link #lock}, so eviction never misses it. */
        synchronized void acquire(String revision) {
            leases.merge(revision, 1, Integer::sum);
        }

        synchronized void release(String revision) {
            leases.computeIfPresent(revision, (k, n) -> n > 1 ? n - 1 : null);
        }

        synchronized int leases(String revision) {
            return leases.getOrDefault(revision, 0);
        }
    }

    /** A lease on one snapshot, released once. */
    static final class Lease implements ResolvedModel {

        private final Repo repo;
        private final String revision;
        private final Path directory;
        private final AtomicBoolean closed = new AtomicBoolean();

        /** Wraps a lease the caller already took on {@code revision}. */
        Lease(Repo repo, String revision, Path directory) {
            this.repo = repo;
            this.revision = revision;
            this.directory = directory;
        }

        @Override
        public Path directory() {
            return directory;
        }

        @Override
        public Optional<ModelRevision> revision() {
            return Optional.of(new ModelRevision(repo.repoId, revision));
        }

        @Override
        public ResolvedModel retain() {
            // This lease keeps the count above zero, so no lock is needed: eviction cannot be deciding meanwhile
            synchronized (repo) {
                if (closed.get()) {
                    throw new IllegalStateException("The resolved model was closed");
                }
                repo.acquire(revision);
            }
            return new Lease(repo, revision, directory);
        }

        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) {
                repo.release(revision);
            }
        }

        @Override
        public String toString() {
            return "ResolvedModel[" + repo.repoId + "@" + revision + ", " + directory + "]";
        }
    }

    /** The repo lock: the in-JVM lock, then an exclusive file lock so that other processes wait too. */
    static final class RepoLock implements AutoCloseable {

        private final Repo repo;

        private RepoLock(Repo repo) {
            this.repo = repo;
        }

        /**
         * Locks {@code repo}, waiting for other threads and processes.
         *
         * @throws IOException if the lock file cannot be created, as in a read-only cache
         */
        static RepoLock acquire(Repo repo, Path lockFile) throws IOException {
            repo.lock.lock();
            if (repo.lock.getHoldCount() > 1) {
                return new RepoLock(repo);
            }
            try {
                Files.createDirectories(lockFile.getParent());
                FileChannel channel = FileChannel.open(lockFile, StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE);
                try {
                    repo.fileLock = lockFile(channel);
                    repo.channel = channel;
                } catch (IOException | RuntimeException e) {
                    channel.close();
                    throw e;
                }
            } catch (IOException | RuntimeException e) {
                repo.lock.unlock();
                throw e;
            }
            return new RepoLock(repo);
        }

        private static FileLock lockFile(FileChannel channel) throws IOException {
            long backoff = 10;
            while (true) {
                try {
                    return channel.lock();
                } catch (OverlappingFileLockException e) {
                    // Held through another class loader's registry in this JVM
                    try {
                        Thread.sleep(backoff);
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        throw new InterruptedIOException("Interrupted waiting for " + channel);
                    }
                    backoff = Math.min(backoff * 2, 500);
                }
            }
        }

        @Override
        public void close() {
            try {
                if (repo.lock.getHoldCount() == 1 && repo.channel != null) {
                    try {
                        repo.fileLock.release();
                    } catch (IOException ignored) {
                        // closing the channel releases it too
                    }
                    try {
                        repo.channel.close();
                    } catch (IOException ignored) {
                        // nothing left to release
                    }
                    repo.channel = null;
                    repo.fileLock = null;
                }
            } finally {
                repo.lock.unlock();
            }
        }
    }
}
