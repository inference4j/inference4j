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

import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.TestSupport.MutableClock;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ModelCacheTest {

    private static final String REPO = "org/model";
    private static final String OTHER = "org/other";
    private static final List<String> FILES = List.of("model.onnx", "tokenizer.json");

    @TempDir
    Path cacheDir;

    private FakeHub hub;
    private MutableClock clock;

    @BeforeEach
    void setUp() throws IOException {
        hub = new FakeHub(REPO, Set.of("model.onnx"))
                .put("model.onnx", "weights v1")
                .put("tokenizer.json", "tokenizer v1")
                .addRepo(OTHER, Set.of())
                .put(OTHER, "model.onnx", "other weights");
        clock = new MutableClock();
    }

    @AfterEach
    void tearDown() {
        hub.close();
    }

    private HuggingFaceModelSource.Builder builder() {
        return HuggingFaceModelSource.builder().cacheDir(cacheDir).endpoint(hub.endpoint()).offline(false)
                .clock(clock).minimumSnapshotAge(Duration.ZERO);
    }

    private HuggingFaceModelSource source() {
        return builder().build();
    }

    @Test
    void listReportsRevisionFilesSizeAndLeases() {
        HuggingFaceModelSource source = source();
        source.resolve(OTHER, List.of("model.onnx"));
        try (ResolvedModel model = source.open(REPO, FILES)) {
            List<CachedModel> models = source.cache().list();

            assertThat(models).extracting(CachedModel::repoId).containsExactly(REPO, OTHER);
            CachedModel cached = models.get(0);
            assertThat(cached.revision()).isEqualTo(hub.commit());
            assertThat(cached.lastChecked()).isEqualTo(clock.now);
            assertThat(cached.sizeBytes()).isEqualTo("weights v1".length() + "tokenizer v1".length());
            assertThat(cached.updateAvailable()).isFalse();
            assertThat(cached.stale()).isFalse();
            CachedSnapshot snapshot = cached.snapshots().get(0);
            assertThat(snapshot.current()).isTrue();
            assertThat(snapshot.leases()).isEqualTo(1);
            assertThat(snapshot.files()).extracting(CachedFile::path, CachedFile::sizeBytes)
                    .containsExactly(org.assertj.core.groups.Tuple.tuple("model.onnx", 10L),
                            org.assertj.core.groups.Tuple.tuple("tokenizer.json", 12L));
            assertThat(snapshot.files().get(0).id()).hasSize(64);   // the LFS SHA-256
        }
        assertThat(source.cache().find(REPO).orElseThrow().snapshots().get(0).leases()).isZero();
        assertThat(source.cache().directory()).isEqualTo(cacheDir);
    }

    @Test
    void findIsEmptyForAnUncachedRepo() {
        assertThat(source().cache().find("org/nothing")).isEmpty();
        assertThat(source().cache().list()).isEmpty();
    }

    @Test
    void checkReportsChangedFilesWithoutDownloading() {
        HuggingFaceModelSource source = source();
        String cachedRevision = hub.commit();
        source.resolve(REPO, FILES);
        hub.put("tokenizer.json", "tokenizer v2").put("README.md", "card");

        UpdateCheck check = source.cache().check(REPO);

        assertThat(check.cachedRevision()).isEqualTo(cachedRevision);
        assertThat(check.remoteRevision()).isEqualTo(hub.commit());
        assertThat(check.changedFiles()).containsExactly("tokenizer.json");
        assertThat(check.updateAvailable()).isTrue();
        assertThat(check.checkedAt()).isEqualTo(clock.now);
        assertThat(hub.downloads("tokenizer.json")).isEqualTo(1);
        CachedModel cached = source.cache().find(REPO).orElseThrow();
        assertThat(cached.updateAvailable()).isTrue();
        assertThat(cached.remoteRevision()).isEqualTo(hub.commit());
    }

    @Test
    void aCheckedUpdateIsFetchedByTheNextOpen() throws IOException {
        HuggingFaceModelSource source = source();
        source.resolve(REPO, FILES);
        hub.put("tokenizer.json", "tokenizer v2");
        source.cache().check(REPO);

        Path dir = source.resolve(REPO, FILES);

        assertThat(Files.readString(dir.resolve("tokenizer.json"))).isEqualTo("tokenizer v2");
    }

    @Test
    void checkFailsForAnUncachedRepo() {
        assertThatThrownBy(() -> source().cache().check(REPO)).isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("not cached");
    }

    @Test
    void updateDownloadsTheLatestRevision() throws IOException {
        HuggingFaceModelSource source = source();
        source.resolve(REPO, FILES);
        hub.put("model.onnx", "weights v2");

        CachedModel updated = source.cache().update(REPO);

        assertThat(updated.revision()).isEqualTo(hub.commit());
        assertThat(updated.updateAvailable()).isFalse();
        assertThat(updated.snapshots()).extracting(CachedSnapshot::revision).containsExactly(hub.commit());
        assertThat(Files.readString(source.resolve(REPO, FILES).resolve("model.onnx"))).isEqualTo("weights v2");
        assertThat(hub.downloads("tokenizer.json")).isEqualTo(1);
    }

    @Test
    void updateWithAnUnreachableHubFails() {
        HuggingFaceModelSource source = source();
        source.resolve(REPO, FILES);
        hub.revisionStatus = 503;

        assertThatThrownBy(() -> source.cache().update(REPO)).isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("could not be reached");
    }

    @Test
    void autoPruningRemovesTheSupersededSnapshot() {
        HuggingFaceModelSource source = source();
        String first = hub.commit();
        source.resolve(REPO, FILES);
        hub.put("model.onnx", "weights v2");

        source.cache().update(REPO);

        assertThat(cacheDir.resolve("hub/models--org--model/snapshots").resolve(first)).doesNotExist();
    }

    @Test
    void pruneRemovesSupersededSnapshots() {
        HuggingFaceModelSource source = builder().pruneMode(PruneMode.MANUAL).build();
        String first = hub.commit();
        source.resolve(REPO, FILES);
        hub.put("model.onnx", "weights v2");
        source.cache().update(REPO);
        assertThat(source.cache().find(REPO).orElseThrow().snapshots()).hasSize(2);

        EvictionResult result = source.cache().prune();

        assertThat(result.removed()).containsExactly(new ModelRevision(REPO, first));
        assertThat(result.freedBytes()).isEqualTo("weights v1".length());   // the tokenizer moved to the new one
        assertThat(source.cache().find(REPO).orElseThrow().snapshots()).extracting(CachedSnapshot::revision)
                .containsExactly(hub.commit());
    }

    @Test
    void pruneAfterAdoptionFreesNothing() throws IOException {
        Path flat = cacheDir.resolve(REPO);
        Files.createDirectories(flat);
        Files.writeString(flat.resolve("model.onnx"), "weights v1");
        Files.writeString(flat.resolve("tokenizer.json"), "tokenizer v1");
        HuggingFaceModelSource source = source();
        Path dir = source.resolve(REPO, FILES);

        EvictionResult result = source.cache().prune();

        assertThat(result.freedBytes()).isZero();   // adoption moved the flat files into the snapshot
        assertThat(cacheDir.resolve("org")).doesNotExist();
        assertThat(Files.readString(dir.resolve("model.onnx"))).isEqualTo("weights v1");
    }

    @Test
    void aLoadedOldRevisionIsReportedStale() {
        HuggingFaceModelSource source = source();
        try (ResolvedModel loaded = source.open(REPO, FILES)) {
            hub.put("model.onnx", "weights v2");
            source.cache().update(REPO);

            CachedModel cached = source.cache().find(REPO).orElseThrow();

            assertThat(cached.stale()).isTrue();
            assertThat(cached.snapshots()).hasSize(2);
        }
        assertThat(source.cache().find(REPO).orElseThrow().stale()).isFalse();
    }

    @Test
    void evictOneRevision() {
        HuggingFaceModelSource source = builder().pruneMode(PruneMode.MANUAL).build();
        String first = hub.commit();
        source.resolve(REPO, FILES);
        hub.put("model.onnx", "weights v2");
        source.cache().update(REPO);

        EvictionResult result = source.cache().evict(REPO, first);

        assertThat(result.removed()).containsExactly(new ModelRevision(REPO, first));
        assertThat(source.cache().find(REPO).orElseThrow().revision()).isEqualTo(hub.commit());
    }

    @Test
    void rejectsRepoIdsThatEscapeTheCache() {
        HuggingFaceModelSource source = source();
        for (String repoId : List.of("../escape", "org/..", "..", "org/../../x", "a/b/c", "", "/abs", "org\\x")) {
            assertThatThrownBy(() -> source.open(repoId, FILES)).as(repoId)
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> source.cache().find(repoId)).as(repoId)
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> source.cache().evict(repoId)).as(repoId)
                    .isInstanceOf(IllegalArgumentException.class);
        }
        assertThatThrownBy(() -> source.open(REPO, List.of("../model.onnx")))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> source.open(REPO, "cpu/../../x")).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> source.cache().evict(REPO, "../x")).isInstanceOf(IllegalArgumentException.class);
        assertThat(hub.requests).isEmpty();
    }

    @Test
    void defaultCacheIsTheDefaultSourcesCache() {
        assertThat(ModelCache.defaultCache()).isSameAs(HuggingFaceModelSource.defaultInstance().cache());
    }
}
