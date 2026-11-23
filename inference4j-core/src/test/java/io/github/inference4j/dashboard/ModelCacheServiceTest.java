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
import io.github.inference4j.model.CachedFile;
import io.github.inference4j.model.CachedModel;
import io.github.inference4j.model.CachedSnapshot;
import io.github.inference4j.model.EvictionResult;
import io.github.inference4j.model.ModelCache;
import io.github.inference4j.model.ModelRevision;
import io.github.inference4j.model.UpdateCheck;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ModelCacheServiceTest {

    @TempDir
    Path cacheDir;

    private ModelCacheService service;

    @AfterEach
    void close() {
        if (service != null) {
            service.close();
        }
    }

    @Test
    void summaryAddsUpTheRepos() {
        FakeCache cache = new FakeCache(cacheDir);
        cache.put(new CachedModel("org/a", "a2", null, null, true,
                List.of(snapshot("a2", true, 1, 300), snapshot("a1", false, 0, 200)), 500));
        cache.put(new CachedModel("org/b", "b1", null, null, false,
                List.of(snapshot("b1", true, 0, 100), snapshot("b0", false, 2, 50)), 150));
        service = new ModelCacheService(cache);

        CacheSummary summary = service.summary();

        assertThat(summary.models()).isEqualTo(2);
        assertThat(summary.snapshots()).isEqualTo(4);
        assertThat(summary.snapshotsInUse()).isEqualTo(2);
        assertThat(summary.updatesAvailable()).isEqualTo(1);
        assertThat(summary.sizeBytes()).isEqualTo(650);
        // Only the superseded snapshot no model reads
        assertThat(summary.supersededBytes()).isEqualTo(200);
        assertThat(summary.updating()).isZero();
        assertThat(summary.disk()).isNotNull();
        assertThat(summary.disk().totalBytes()).isPositive();
        assertThat(summary.disk().usableBytes()).isBetween(0L, summary.disk().totalBytes());
    }

    @Test
    void summaryReadsTheDiskOfTheNearestExistingParent() {
        service = new ModelCacheService(new FakeCache(cacheDir.resolve("not/created/yet")));

        assertThat(service.summary().disk()).isNotNull();
        assertThat(service.summary().models()).isZero();
    }

    @Test
    void modelIsEmptyWhenNotCached() {
        FakeCache cache = new FakeCache(cacheDir);
        cache.put(model("org/a", "a1"));
        service = new ModelCacheService(cache);

        assertThat(service.model("org/a")).map(ModelView::revision).contains("a1");
        assertThat(service.model("org/missing")).isEmpty();
        assertThat(service.check("org/missing")).isEmpty();
        assertThat(service.startUpdate("org/missing")).isEmpty();
        assertThat(service.evict("org/missing")).isEmpty();
        assertThat(cache.checks.get()).isZero();
    }

    @Test
    void checkIsReusedForAMinute() {
        FakeCache cache = new FakeCache(cacheDir);
        cache.put(model("org/a", "a1"));
        cache.changed = List.of("model.onnx");
        service = new ModelCacheService(cache);

        ModelView first = service.check("org/a").orElseThrow();
        ModelView second = service.check("org/a").orElseThrow();

        assertThat(cache.checks.get()).isEqualTo(1);
        assertThat(first.check().updateAvailable()).isTrue();
        assertThat(first.check().changedFiles()).containsExactly("model.onnx");
        assertThat(second.check()).isEqualTo(first.check());
    }

    @Test
    void checkAsksAgainOnceTheRevisionChanged() {
        FakeCache cache = new FakeCache(cacheDir);
        cache.put(model("org/a", "a1"));
        service = new ModelCacheService(cache);

        service.check("org/a");
        cache.put(model("org/a", "a2"));
        service.check("org/a");

        assertThat(cache.checks.get()).isEqualTo(2);
    }

    @Test
    void updateRunsInTheBackgroundOnceAtATime() throws Exception {
        FakeCache cache = new FakeCache(cacheDir);
        cache.put(model("org/a", "a1"));
        cache.updateGate = new CountDownLatch(1);
        service = new ModelCacheService(cache);

        assertThat(service.startUpdate("org/a")).contains(UpdateStatus.ACCEPTED);
        assertThat(service.startUpdate("org/a")).contains(UpdateStatus.ALREADY_RUNNING);
        assertThat(service.updating("org/a")).isTrue();
        assertThat(service.model("org/a").orElseThrow().updating()).isTrue();
        assertThat(service.summary().updating()).isEqualTo(1);

        cache.updateGate.countDown();
        cache.updated.await(5, TimeUnit.SECONDS);
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (service.updating("org/a") && System.nanoTime() < deadline) {
            Thread.sleep(10);
        }

        assertThat(service.updating("org/a")).isFalse();
        assertThat(cache.updates.get()).isEqualTo(1);
    }

    @Test
    void startUpdateFailsOnceClosed() {
        FakeCache cache = new FakeCache(cacheDir);
        cache.put(model("org/a", "a1"));
        service = new ModelCacheService(cache);
        service.close();

        assertThatThrownBy(() -> service.startUpdate("org/a"))
            .isInstanceOf(IllegalStateException.class)
            .satisfies(ex -> assertThat(service.error((RuntimeException) ex).status()).isEqualTo(503));
        assertThat(service.updating("org/a")).isFalse();
    }

    @Test
    void evictOfARevisionNotCachedIsEmpty() {
        FakeCache cache = new FakeCache(cacheDir);
        cache.put(model("org/a", "a1"));
        service = new ModelCacheService(cache);

        assertThat(service.evict("org/a", "0123abcd")).isEmpty();
        assertThat(service.evict("org/a", "a1")).isPresent();
    }

    @Test
    void errorMapsFailuresToStatuses() {
        service = new ModelCacheService(new FakeCache(cacheDir));
        String dir = cacheDir.toString();

        assertThat(service.error(new IllegalArgumentException("bad id"))).isEqualTo(new CacheError(400, "bad id"));
        assertThat(service.error(new ModelSourceException("Offline mode is on")).status()).isEqualTo(409);
        assertThat(service.error(new ModelDownloadException("Cannot reach " + dir + "/x", 503)))
            .isEqualTo(new CacheError(502, "Cannot reach <cache>/x"));
        IllegalStateException other = new IllegalStateException("bug");
        assertThatThrownBy(() -> service.error(other)).isSameAs(other);
    }

    private static CachedModel model(String repoId, String revision) {
        return new CachedModel(repoId, revision, null, null, false, List.of(snapshot(revision, true, 0, 10)), 10);
    }

    private static CachedSnapshot snapshot(String revision, boolean current, int leases, long size) {
        return new CachedSnapshot(revision, current, leases, List.of(new CachedFile("model.onnx", revision, size)),
                size);
    }

    /** An in-memory {@link ModelCache}. */
    private static final class FakeCache implements ModelCache {

        final Path directory;

        final Map<String, CachedModel> models = new ConcurrentHashMap<>();

        final AtomicInteger checks = new AtomicInteger();

        final AtomicInteger updates = new AtomicInteger();

        final CountDownLatch updated = new CountDownLatch(1);

        volatile List<String> changed = List.of();

        volatile CountDownLatch updateGate = new CountDownLatch(0);

        FakeCache(Path directory) {
            this.directory = directory;
        }

        void put(CachedModel model) {
            models.put(model.repoId(), model);
        }

        @Override
        public Path directory() {
            return directory;
        }

        @Override
        public List<CachedModel> list() {
            List<CachedModel> list = new ArrayList<>(models.values());
            list.sort(Comparator.comparing(CachedModel::repoId));
            return list;
        }

        @Override
        public Optional<CachedModel> find(String repoId) {
            return Optional.ofNullable(models.get(repoId));
        }

        @Override
        public UpdateCheck check(String repoId) {
            checks.incrementAndGet();
            return new UpdateCheck(repoId, models.get(repoId).revision(), "remote", changed, Instant.now());
        }

        @Override
        public CachedModel update(String repoId) {
            try {
                updateGate.await(5, TimeUnit.SECONDS);
            }
            catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
            updates.incrementAndGet();
            updated.countDown();
            return models.get(repoId);
        }

        @Override
        public EvictionResult evict(String repoId) {
            CachedModel removed = models.remove(repoId);
            return new EvictionResult(List.of(new ModelRevision(repoId, removed.revision())), List.of(),
                    removed.sizeBytes());
        }

        @Override
        public EvictionResult evict(String repoId, String revision) {
            return new EvictionResult(List.of(), List.of(), 0);
        }

        @Override
        public EvictionResult prune() {
            return new EvictionResult(List.of(), List.of(), 0);
        }
    }
}
