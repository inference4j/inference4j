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

import io.github.inference4j.InferenceSession;
import io.github.inference4j.model.TestSupport.MutableClock;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assumptions.assumeThat;

class HuggingFaceModelSourceSnapshotTest {

    private static final String REPO = "org/model";
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
                .put("config.json", "config v1");
        clock = new MutableClock();
    }

    @AfterEach
    void tearDown() {
        hub.close();
    }

    private HuggingFaceModelSource.Builder builder() {
        return HuggingFaceModelSource.builder().cacheDir(cacheDir).endpoint(hub.endpoint()).offline(false)
                .clock(clock);
    }

    private HuggingFaceModelSource source() {
        return builder().build();
    }

    /** No minimum age, so eviction only depends on leases. */
    private HuggingFaceModelSource evictingSource() {
        return builder().minimumSnapshotAge(Duration.ZERO).build();
    }

    private HuggingFaceModelSource offlineSource() {
        return builder().offline(true).build();
    }

    private Path snapshot(String revision) {
        return cacheDir.resolve("hub/models--org--model/snapshots").resolve(revision);
    }

    private void seedFlatCache(String weights, String tokenizer) throws IOException {
        Path flat = cacheDir.resolve(REPO);
        Files.createDirectories(flat);
        Files.writeString(flat.resolve("model.onnx"), weights);
        Files.writeString(flat.resolve("tokenizer.json"), tokenizer);
    }

    /** Every file under {@code dir} has a single hard link, as ONNX Runtime requires of external data. */
    private static void assertSingleLinks(Path dir) throws IOException {
        List<Path> files;
        try (Stream<Path> walk = Files.walk(dir)) {
            files = walk.filter(Files::isRegularFile).toList();
        }
        assertThat(files).isNotEmpty();
        for (Path file : files) {
            assertThat(links(file)).as(file.toString()).isEqualTo(1);
        }
    }

    private static int links(Path file) throws IOException {
        return (Integer) Files.getAttribute(file, "unix:nlink");
    }

    private static void assumeLinkCounts() {
        assumeThat(FileSystems.getDefault().supportedFileAttributeViews()).contains("unix");
    }

    private int leases(HuggingFaceModelSource source, String revision) {
        return source.cache().find(REPO).orElseThrow().snapshots().stream()
                .filter(s -> s.revision().equals(revision)).findFirst().orElseThrow().leases();
    }

    // Snapshots

    @Test
    void anUpdateLeavesTheLoadedSnapshotUntouched() throws IOException {
        HuggingFaceModelSource source = source();
        try (ResolvedModel loaded = source.open(REPO, FILES)) {
            hub.put("model.onnx", "weights v2");
            clock.now = clock.now.plus(Duration.ofDays(2));

            try (ResolvedModel updated = source.open(REPO, FILES)) {
                assertThat(updated.directory()).isNotEqualTo(loaded.directory());
                assertThat(Files.readString(updated.file("model.onnx"))).isEqualTo("weights v2");
                assertThat(updated.revision()).contains(new ModelRevision(REPO, hub.commit()));
            }
            assertThat(Files.readString(loaded.file("model.onnx"))).isEqualTo("weights v1");
        }
    }

    @Test
    void unchangedFilesAreReusedNotDownloaded() throws IOException {
        HuggingFaceModelSource source = builder().pruneMode(PruneMode.MANUAL).build();
        Path first = source.resolve(REPO, FILES);
        hub.put("model.onnx", "weights v2");
        clock.now = clock.now.plus(Duration.ofDays(2));

        Path second = source.resolve(REPO, FILES);

        assertThat(hub.downloads("tokenizer.json")).isEqualTo(1);
        assertThat(Files.readString(second.resolve("tokenizer.json"))).isEqualTo("tokenizer v1");
        assertThat(Files.readString(first.resolve("model.onnx"))).isEqualTo("weights v1");
    }

    @Test
    void anUpdateMovesUnchangedFilesOutOfAnUnusedSnapshot() throws IOException {
        assumeLinkCounts();
        HuggingFaceModelSource source = source();
        String first = hub.commit();
        source.resolve(REPO, FILES);
        hub.put("model.onnx", "weights v2");
        clock.now = clock.now.plus(Duration.ofDays(2));

        try (ResolvedModel updated = source.open(REPO, FILES)) {
            assertThat(Files.readString(updated.file("tokenizer.json"))).isEqualTo("tokenizer v1");
            assertThat(Files.readString(updated.file("model.onnx"))).isEqualTo("weights v2");
            assertSingleLinks(updated.directory());
        }
        assertThat(hub.downloads("tokenizer.json")).isEqualTo(1);
        assertThat(snapshot(first)).doesNotExist();   // auto-pruned
        assertThat(source.cache().find(REPO).orElseThrow().snapshots()).extracting(CachedSnapshot::revision)
                .containsExactly(hub.commit());
    }

    @Test
    void aSnapshotLeftByMovedFilesIsNeverServedAgain() throws IOException {
        HuggingFaceModelSource source = builder().pruneMode(PruneMode.MANUAL).build();
        String first = hub.commit();
        source.resolve(REPO, FILES);
        hub.put("model.onnx", "weights v2");
        clock.now = clock.now.plus(Duration.ofDays(2));
        source.resolve(REPO, FILES);

        CachedSnapshot old = source.cache().find(REPO).orElseThrow().snapshots().stream()
                .filter(s -> s.revision().equals(first)).findFirst().orElseThrow();
        assertThat(old.current()).isFalse();
        assertThat(old.files()).extracting(CachedFile::path).containsExactly("model.onnx");
        assertThat(snapshot(first).resolve("tokenizer.json")).doesNotExist();
        try (ResolvedModel model = offlineSource().open(REPO, FILES)) {
            assertThat(model.revision()).contains(new ModelRevision(REPO, hub.commit()));
        }
        assertThat(source.cache().prune().removed()).containsExactly(new ModelRevision(REPO, first));
    }

    @Test
    void anUpdateCopiesUnchangedFilesOfASnapshotInUse() throws IOException {
        assumeLinkCounts();
        HuggingFaceModelSource source = source();
        try (ResolvedModel loaded = source.open(REPO, FILES)) {
            hub.put("model.onnx", "weights v2");
            clock.now = clock.now.plus(Duration.ofDays(2));

            try (ResolvedModel updated = source.open(REPO, FILES)) {
                assertThat(Files.readString(updated.file("tokenizer.json"))).isEqualTo("tokenizer v1");
                assertSingleLinks(updated.directory());
            }
            assertThat(Files.readString(loaded.file("tokenizer.json"))).isEqualTo("tokenizer v1");
            assertThat(Files.readString(loaded.file("model.onnx"))).isEqualTo("weights v1");
            assertSingleLinks(loaded.directory());
        }
        assertThat(hub.downloads("tokenizer.json")).isEqualTo(1);
    }

    @Test
    void anUpdateCopiesUnchangedFilesOfAYoungSnapshot() throws IOException {
        HuggingFaceModelSource source = builder().checkInterval(Duration.ZERO).build();
        Path first = source.resolve(REPO, FILES);
        hub.put("model.onnx", "weights v2");
        clock.now = clock.now.plus(Duration.ofMinutes(1));

        Path second = source.resolve(REPO, FILES);

        assertThat(second).isNotEqualTo(first);
        assertThat(Files.readString(second.resolve("tokenizer.json"))).isEqualTo("tokenizer v1");
        assertThat(Files.readString(first.resolve("tokenizer.json"))).isEqualTo("tokenizer v1");   // kept whole
        assertThat(hub.downloads("tokenizer.json")).isEqualTo(1);
    }

    @Test
    void aFileRequestedLaterIsFetchedAtTheCachedRevision() throws IOException {
        HuggingFaceModelSource source = source();
        String cachedRevision = hub.commit();
        source.resolve(REPO, List.of("model.onnx"));
        hub.put("config.json", "config v2");
        clock.now = clock.now.plus(Duration.ofHours(1));

        Path dir = source.resolve(REPO, List.of("model.onnx", "config.json"));

        assertThat(dir).isEqualTo(snapshot(cachedRevision));
        assertThat(Files.readString(dir.resolve("config.json"))).isEqualTo("config v1");
        assertThat(hub.downloadsAt(cachedRevision, "config.json")).isEqualTo(1);
        assertThat(hub.revisionChecks()).isEqualTo(1);
    }

    @Test
    void aFileRequestedWhenACheckIsDueComesFromTheHubRevision() throws IOException {
        HuggingFaceModelSource source = builder().pruneMode(PruneMode.MANUAL).build();
        Path first = source.resolve(REPO, FILES);
        hub.put("extra.json", "added later");
        clock.now = clock.now.plus(Duration.ofDays(2));

        Path dir = source.resolve(REPO, List.of("model.onnx", "tokenizer.json", "extra.json"));

        assertThat(dir).isEqualTo(snapshot(hub.commit()));
        assertThat(Files.readString(dir.resolve("extra.json"))).isEqualTo("added later");
        assertThat(Files.readString(dir.resolve("model.onnx"))).isEqualTo("weights v1");
        assertThat(first).isNotEqualTo(dir);
        assertThat(hub.downloads("model.onnx")).isEqualTo(1);
    }

    @Test
    void aFailedDownloadLeavesTheCurrentRevisionInPlace() throws IOException {
        HuggingFaceModelSource source = source();
        String cachedRevision = hub.commit();
        source.resolve(REPO, FILES);
        hub.put("model.onnx", "weights v2");
        hub.failDownload("model.onnx", true);
        clock.now = clock.now.plus(Duration.ofDays(2));

        List<String> warnings = TestSupport.warnings(HuggingFaceModelSource.class, () -> {
            Path dir = source.resolve(REPO, FILES);
            assertThat(dir).isEqualTo(snapshot(cachedRevision));
        });

        assertThat(warnings).singleElement().asString().startsWith("Could not update org/model");
        CachedModel model = source.cache().find(REPO).orElseThrow();
        assertThat(model.revision()).isEqualTo(cachedRevision);
        assertThat(model.updateAvailable()).isTrue();
        assertThat(Files.readString(snapshot(cachedRevision).resolve("model.onnx"))).isEqualTo("weights v1");

        hub.failDownload("model.onnx", false);
        clock.now = clock.now.plus(Duration.ofDays(2));
        Path dir = source.resolve(REPO, FILES);

        assertThat(Files.readString(dir.resolve("model.onnx"))).isEqualTo("weights v2");
    }

    @Test
    void anInterruptedUpdateIsResumed() throws IOException {
        HuggingFaceModelSource source = source();
        source.resolve(REPO, FILES);
        hub.put("model.onnx", "weights v2").put("tokenizer.json", "tokenizer v2");
        hub.failDownload("tokenizer.json", true);
        clock.now = clock.now.plus(Duration.ofDays(2));
        TestSupport.warnings(HuggingFaceModelSource.class, () -> source.resolve(REPO, FILES));
        hub.failDownload("tokenizer.json", false);
        clock.now = clock.now.plus(Duration.ofDays(2));

        Path dir = source.resolve(REPO, FILES);

        assertThat(dir).isEqualTo(snapshot(hub.commit()));
        assertThat(Files.readString(dir.resolve("tokenizer.json"))).isEqualTo("tokenizer v2");
        assertThat(hub.downloadsAt(hub.commit(), "model.onnx")).isEqualTo(1);
    }

    // Migration from the flat layout

    @Test
    void aFlatCacheIsAdoptedWithoutDownloading() throws IOException {
        seedFlatCache("weights v1", "tokenizer v1");

        Path dir = source().resolve(REPO, FILES);

        assertThat(dir).isEqualTo(snapshot(hub.commit()));
        assertThat(hub.count("/" + REPO + "/resolve/")).isZero();
        assertThat(Files.readString(dir.resolve("model.onnx"))).isEqualTo("weights v1");
        assertThat(cacheDir.resolve(REPO).resolve("model.onnx")).doesNotExist();   // moved, not linked
        assertThat(cacheDir.resolve("org")).doesNotExist();
    }

    @Test
    void aFlatCacheAdoptedOnlineHasSingleLinks() throws IOException {
        assumeLinkCounts();
        seedFlatCache("weights v1", "tokenizer v1");

        assertSingleLinks(source().resolve(REPO, FILES));
    }

    @Test
    void aFlatCacheAdoptedOfflineHasSingleLinks() throws IOException {
        assumeLinkCounts();
        seedFlatCache("weights v1", "tokenizer v1");

        try (ResolvedModel model = offlineSource().open(REPO, FILES)) {
            assertSingleLinks(model.directory());
        }
        assertThat(cacheDir.resolve(REPO)).doesNotExist();
    }

    @Test
    void filesAdoptedOfflineOneRequestAtATimeAreAllAdopted() throws IOException {
        seedFlatCache("weights v1", "tokenizer v1");
        HuggingFaceModelSource source = offlineSource();
        source.resolve(REPO, List.of("model.onnx"));

        Path dir = source.resolve(REPO, FILES);

        assertThat(dir).isEqualTo(snapshot(ModelRevision.UNKNOWN));
        assertThat(Files.readString(dir.resolve("tokenizer.json"))).isEqualTo("tokenizer v1");
        assertThat(Files.readString(dir.resolve("model.onnx"))).isEqualTo("weights v1");
    }

    @Test
    void aFlatCacheIsAdoptedOfflineAsUnknownRevision() throws IOException {
        seedFlatCache("weights v1", "tokenizer v1");

        try (ResolvedModel model = offlineSource().open(REPO, FILES)) {
            assertThat(model.directory()).isEqualTo(snapshot(ModelRevision.UNKNOWN));
            assertThat(model.revision()).contains(new ModelRevision(REPO, ModelRevision.UNKNOWN));
            assertThat(Files.readString(model.file("tokenizer.json"))).isEqualTo("tokenizer v1");
        }
        assertThat(hub.requests).isEmpty();
    }

    @Test
    void anUnknownRevisionIsPromotedAtTheNextCheck() throws IOException {
        seedFlatCache("weights v1", "tokenizer v1");
        offlineSource().resolve(REPO, FILES);

        Path dir = source().resolve(REPO, FILES);

        assertThat(dir).isEqualTo(snapshot(hub.commit()));
        assertThat(source().cache().find(REPO).orElseThrow().revision()).isEqualTo(hub.commit());
        assertThat(hub.count("/" + REPO + "/resolve/")).isZero();
    }

    @Test
    void aReadOnlyFlatCacheIsServedInPlace() throws IOException {
        seedFlatCache("weights v1", "tokenizer v1");
        assumeThat(cacheDir.toFile().setWritable(false)).isTrue();
        try {
            assumeThat(Files.isWritable(cacheDir)).isFalse();   // not when running as root
            HuggingFaceModelSource source = offlineSource();

            List<String> warnings = TestSupport.warnings(HuggingFaceModelSource.class, () -> {
                try (ResolvedModel model = source.open(REPO, FILES)) {
                    assertThat(model.directory()).isEqualTo(cacheDir.resolve(REPO));
                    assertThat(model.revision()).isEmpty();
                }
            });

            assertThat(warnings).singleElement().asString().contains("is not writable");
        } finally {
            cacheDir.toFile().setWritable(true);
        }
    }

    // Hard links left by earlier builds

    @Test
    void openingRepairsAFileLinkedFromTheFlatLayout() throws IOException {
        assumeLinkCounts();
        seedFlatCache("weights v1", "tokenizer v1");
        HuggingFaceModelSource source = offlineSource();
        Path dir = source.resolve(REPO, FILES);
        Path flat = cacheDir.resolve(REPO).resolve("model.onnx");
        Files.createDirectories(flat.getParent());
        Files.createLink(flat, dir.resolve("model.onnx"));
        assertThat(links(dir.resolve("model.onnx"))).isEqualTo(2);

        try (ResolvedModel model = source.open(REPO, FILES)) {
            assertThat(links(model.file("model.onnx"))).isEqualTo(1);
            assertThat(Files.readString(model.file("model.onnx"))).isEqualTo("weights v1");
            assertSingleLinks(model.directory());
        }
        assertThat(flat).doesNotExist();   // the adopted flat copy is dropped, not duplicated
    }

    @Test
    void openingRepairsAFileWithAnotherHardLink() throws IOException {
        assumeLinkCounts();
        seedFlatCache("weights v1", "tokenizer v1");
        HuggingFaceModelSource source = offlineSource();
        Path dir = source.resolve(REPO, FILES);
        Path elsewhere = cacheDir.resolve("elsewhere.onnx");
        Files.createLink(elsewhere, dir.resolve("model.onnx"));

        try (ResolvedModel model = source.open(REPO, FILES)) {
            assertThat(links(model.file("model.onnx"))).isEqualTo(1);
            assertThat(Files.readString(model.file("model.onnx"))).isEqualTo("weights v1");
        }
        assertThat(links(elsewhere)).isEqualTo(1);
        assertThat(Files.readString(elsewhere)).isEqualTo("weights v1");
    }

    @Test
    void aLinkedFileOfASnapshotInUseIsLeftAlone() throws IOException {
        assumeLinkCounts();
        seedFlatCache("weights v1", "tokenizer v1");
        HuggingFaceModelSource source = offlineSource();
        try (ResolvedModel loaded = source.open(REPO, FILES)) {
            Files.createLink(cacheDir.resolve("elsewhere.onnx"), loaded.file("model.onnx"));

            List<String> warnings = TestSupport.warnings(HuggingFaceModelSource.class, () -> {
                try (ResolvedModel again = source.open(REPO, FILES)) {
                    assertThat(again.directory()).isEqualTo(loaded.directory());
                }
            });

            assertThat(warnings).singleElement().asString().contains("hard-linked");
            assertThat(links(loaded.file("model.onnx"))).isEqualTo(2);
        }
        try (ResolvedModel model = source.open(REPO, FILES)) {
            assertThat(links(model.file("model.onnx"))).isEqualTo(1);
        }
    }

    // Leases

    @Test
    void evictSkipsAnOpenModel() {
        HuggingFaceModelSource source = evictingSource();
        try (ResolvedModel model = source.open(REPO, FILES)) {
            EvictionResult result = source.cache().evict(REPO);

            assertThat(result.removed()).isEmpty();
            assertThat(result.inUse()).containsExactly(new ModelRevision(REPO, hub.commit()));
            assertThat(model.file("model.onnx")).exists();
        }
    }

    @Test
    void evictRemovesTheSnapshotOnceClosed() {
        HuggingFaceModelSource source = evictingSource();
        Path dir;
        try (ResolvedModel model = source.open(REPO, FILES)) {
            dir = model.directory();
        }

        EvictionResult result = source.cache().evict(REPO);

        assertThat(result.removed()).containsExactly(new ModelRevision(REPO, hub.commit()));
        assertThat(result.freedBytes()).isEqualTo("weights v1".length() + "tokenizer v1".length());
        assertThat(dir).doesNotExist();
        assertThat(source.cache().find(REPO)).isEmpty();
    }

    @Test
    void aYoungSnapshotIsNotEvicted() {
        HuggingFaceModelSource source = source();
        source.resolve(REPO, FILES);

        assertThat(source.cache().evict(REPO).inUse()).hasSize(1);

        clock.now = clock.now.plus(Duration.ofMinutes(11));
        assertThat(source.cache().evict(REPO).removed()).hasSize(1);
    }

    @Test
    void aSessionKeepsItsSnapshotAfterTheBuilderHandleCloses() throws IOException {
        try (InputStream onnx = getClass().getResourceAsStream("/test-bool-not.onnx")) {
            hub.put(REPO, "model.onnx", onnx.readAllBytes());
        }
        HuggingFaceModelSource source = evictingSource();
        InferenceSession session;
        try (ResolvedModel model = source.open(REPO, List.of("model.onnx"))) {
            session = InferenceSession.create(model, "model.onnx");
        }

        assertThat(session.modelRevision()).contains(new ModelRevision(REPO, hub.commit()));
        assertThat(source.cache().evict(REPO).inUse()).hasSize(1);

        session.close();
        assertThat(source.cache().evict(REPO).removed()).hasSize(1);
    }

    @Test
    void closingTwiceReleasesOnce() {
        HuggingFaceModelSource source = source();
        ResolvedModel first = source.open(REPO, FILES);
        ResolvedModel second = first.retain();

        first.close();
        first.close();

        assertThat(leases(source, hub.commit())).isEqualTo(1);
        second.close();
        assertThat(leases(source, hub.commit())).isZero();
    }

    @Test
    void twoSourcesOnOneDirectoryShareLeases() throws IOException {
        HuggingFaceModelSource one = evictingSource();
        Path alias = cacheDir.resolve("alias");
        try {
            Files.createSymbolicLink(alias, cacheDir);
        } catch (UnsupportedOperationException | IOException e) {
            alias = cacheDir;   // no symbolic links here: the same spelling still shares
        }
        HuggingFaceModelSource other = HuggingFaceModelSource.builder()
                .cacheDir(alias).offline(true).clock(clock).minimumSnapshotAge(Duration.ZERO).build();

        try (ResolvedModel model = one.open(REPO, FILES)) {
            assertThat(other.cache().evict(REPO).inUse()).containsExactly(new ModelRevision(REPO, hub.commit()));
        }
        assertThat(other.cache().evict(REPO).removed()).hasSize(1);
    }

    // Concurrency

    @Test
    void evictWaitsForARunningResolve() throws Exception {
        HuggingFaceModelSource source = evictingSource();
        FakeHub.Gate gate = hub.holdDownloads();
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<Path> resolve = executor.submit(() -> source.resolve(REPO, FILES));
            gate.awaitWaiting();
            Future<EvictionResult> evict = executor.submit(() -> source.cache().evict(REPO));
            Thread.sleep(200);

            assertThat(evict.isDone()).isFalse();
            gate.release();
            Path dir = resolve.get(10, TimeUnit.SECONDS);

            assertThat(evict.get(10, TimeUnit.SECONDS).removed()).containsExactly(
                    new ModelRevision(REPO, hub.commit()));
            assertThat(dir).doesNotExist();
        } finally {
            gate.release();
            executor.shutdownNow();
        }
    }

    @Test
    void concurrentResolvesDownloadOnce() throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(8);
        try {
            List<CompletableFuture<Path>> resolves = new ArrayList<>();
            for (int i = 0; i < 8; i++) {
                HuggingFaceModelSource source = source();
                resolves.add(CompletableFuture.supplyAsync(() -> source.resolve(REPO, FILES), executor));
            }
            for (CompletableFuture<Path> resolve : resolves) {
                assertThat(resolve.get(10, TimeUnit.SECONDS)).isEqualTo(snapshot(hub.commit()));
            }
        } finally {
            executor.shutdownNow();
        }

        assertThat(hub.downloads("model.onnx")).isEqualTo(1);
        assertThat(hub.revisionChecks()).isEqualTo(1);
    }
}
