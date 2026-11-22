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

class HuggingFaceModelSourceRevisionTest {

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
                .put("README.md", "card v1");
        clock = new MutableClock();
    }

    @AfterEach
    void tearDown() {
        hub.close();
    }

    private HuggingFaceModelSource source() {
        return HuggingFaceModelSource.builder().cacheDir(cacheDir).endpoint(hub.endpoint()).offline(false)
                .clock(clock).build();
    }

    private Path snapshot(String revision) {
        return cacheDir.resolve("hub/models--org--model/snapshots").resolve(revision);
    }

    private CachedModel cached() {
        return source().cache().find(REPO).orElseThrow();
    }

    @Test
    void aFirstResolveDownloadsAtTheHubRevisionAndRecordsIt() throws IOException {
        Path dir = source().resolve(REPO, FILES);

        assertThat(dir).isEqualTo(snapshot(hub.commit()));
        assertThat(Files.readString(dir.resolve("model.onnx"))).isEqualTo("weights v1");
        assertThat(hub.downloadsAt(hub.commit(), "model.onnx")).isEqualTo(1);
        assertThat(hub.downloadsAt(hub.commit(), "tokenizer.json")).isEqualTo(1);
        CachedModel model = cached();
        assertThat(model.revision()).isEqualTo(hub.commit());
        assertThat(model.lastChecked()).isEqualTo(clock.now);
        assertThat(model.snapshots()).singleElement().satisfies(snapshot -> assertThat(snapshot.files())
                .extracting(CachedFile::path).containsExactly("model.onnx", "tokenizer.json"));
    }

    @Test
    void aCachedRepoIsNotCheckedAgainWithinTheInterval() {
        source().resolve(REPO, FILES);
        clock.now = clock.now.plus(Duration.ofHours(23));

        source().resolve(REPO, FILES);

        assertThat(hub.revisionChecks()).isEqualTo(1);
    }

    @Test
    void onlyChangedFilesAreDownloadedAgainOnceTheIntervalHasPassed() throws IOException {
        source().resolve(REPO, FILES);
        hub.put("tokenizer.json", "tokenizer v2");
        clock.now = clock.now.plus(Duration.ofHours(25));

        Path dir = source().resolve(REPO, FILES);

        assertThat(dir).isEqualTo(snapshot(hub.commit()));
        assertThat(Files.readString(dir.resolve("tokenizer.json"))).isEqualTo("tokenizer v2");
        assertThat(Files.readString(dir.resolve("model.onnx"))).isEqualTo("weights v1");
        assertThat(hub.downloadsAt(hub.commit(), "tokenizer.json")).isEqualTo(1);
        assertThat(hub.count("/" + REPO + "/resolve/")).isEqualTo(3);   // two first downloads, one update
        assertThat(cached().revision()).isEqualTo(hub.commit());
    }

    @Test
    void aNewCommitThatChangesNoCachedFileDownloadsNothing() {
        Path first = source().resolve(REPO, FILES);
        hub.put("README.md", "card v2");
        clock.now = clock.now.plus(Duration.ofDays(2));

        Path dir = source().resolve(REPO, FILES);

        assertThat(dir).isEqualTo(first);
        assertThat(hub.revisionChecks()).isEqualTo(2);
        assertThat(hub.count("/" + REPO + "/resolve/")).isEqualTo(2);
        CachedModel model = cached();
        assertThat(model.remoteRevision()).isEqualTo(hub.commit());
        assertThat(model.revision()).isNotEqualTo(hub.commit());
        assertThat(model.lastChecked()).isEqualTo(clock.now);
        assertThat(model.updateAvailable()).isFalse();
    }

    @Test
    void aStaleFlatCacheFileIsDownloadedAgain() throws IOException {
        // As left by an earlier version: correct weights (an LFS file), a stale tokenizer
        Path flat = cacheDir.resolve(REPO);
        Files.createDirectories(flat);
        Files.writeString(flat.resolve("model.onnx"), "weights v1");
        Files.writeString(flat.resolve("tokenizer.json"), "tokenizer v0");

        Path dir = source().resolve(REPO, FILES);

        assertThat(Files.readString(dir.resolve("tokenizer.json"))).isEqualTo("tokenizer v1");
        assertThat(hub.downloads("tokenizer.json")).isEqualTo(1);
        assertThat(hub.downloads("model.onnx")).isZero();
    }

    @Test
    void anUnreachableHubFallsBackToTheCacheWithOneWarning() throws IOException {
        source().resolve(REPO, FILES);
        hub.revisionStatus = 503;
        clock.now = clock.now.plus(Duration.ofDays(2));
        HuggingFaceModelSource source = source();

        List<String> warnings = TestSupport.warnings(HuggingFaceModelSource.class, () -> {
            source.resolve(REPO, FILES);
            source.resolve(REPO, FILES);
        });

        assertThat(Files.readString(source.resolve(REPO, FILES).resolve("tokenizer.json"))).isEqualTo("tokenizer v1");
        assertThat(warnings).containsExactly("Could not check org/model for updates on the Hub; using the cached files");
        assertThat(hub.revisionChecks()).isEqualTo(2);   // the first resolve, then one failed check
    }

    @Test
    void offlineNeverContactsTheHub() {
        source().resolve(REPO, FILES);
        clock.now = clock.now.plus(Duration.ofDays(2));
        HuggingFaceModelSource offline = HuggingFaceModelSource.builder().cacheDir(cacheDir)
                .endpoint(hub.endpoint()).offline(true).clock(clock).build();

        offline.resolve(REPO, FILES);

        assertThat(hub.revisionChecks()).isEqualTo(1);
        assertThatThrownBy(() -> offline.resolve(REPO, List.of("other.onnx")))
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("Offline");
    }

    @Test
    void aZeroIntervalChecksOnEveryResolve() {
        HuggingFaceModelSource source = HuggingFaceModelSource.builder().cacheDir(cacheDir)
                .endpoint(hub.endpoint()).offline(false).clock(clock).checkInterval(Duration.ZERO).build();

        source.resolve(REPO, FILES);
        source.resolve(REPO, FILES);

        assertThat(hub.revisionChecks()).isEqualTo(2);
    }

    @Test
    void missingFilesAreDownloadedAtTheReportedCommitWhenTheRevisionApiFails() throws IOException {
        hub.revisionStatus = 500;

        Path dir = source().resolve(REPO, FILES);

        assertThat(Files.readString(dir.resolve("tokenizer.json"))).isEqualTo("tokenizer v1");
        assertThat(dir).isEqualTo(snapshot(hub.commit()));
        assertThat(hub.downloadsAt("main", "model.onnx")).isEqualTo(1);
        assertThat(hub.downloadsAt(hub.commit(), "tokenizer.json")).isEqualTo(1);
        assertThat(cached().lastChecked()).isNull();
    }

    @Test
    void withoutACommitHeaderTheRevisionIsUnknown() {
        hub.revisionStatus = 500;
        hub.commitHeader = false;

        Path dir = source().resolve(REPO, FILES);

        assertThat(dir).isEqualTo(snapshot(ModelRevision.UNKNOWN));
        assertThat(cached().revision()).isEqualTo(ModelRevision.UNKNOWN);
    }

    @Test
    void subdirectoryFilesComeFromTheRevisionListing() throws IOException {
        hub.put("cpu/genai_config.json", "{}").put("cpu/model.onnx", "cpu weights").put("gpu/model.onnx", "gpu");

        Path dir = source().resolve(REPO, "cpu");

        assertThat(dir).isEqualTo(snapshot(hub.commit()).resolve("cpu"));
        assertThat(Files.readString(dir.resolve("model.onnx"))).isEqualTo("cpu weights");
        assertThat(dir.resolve("genai_config.json")).exists();
        assertThat(snapshot(hub.commit()).resolve("gpu")).doesNotExist();

        hub.put("cpu/model.onnx", "cpu weights v2");
        clock.now = clock.now.plus(Duration.ofDays(2));
        Path updated = source().resolve(REPO, "cpu");

        assertThat(updated).isEqualTo(snapshot(hub.commit()).resolve("cpu"));
        assertThat(Files.readString(updated.resolve("model.onnx"))).isEqualTo("cpu weights v2");
        assertThat(hub.downloads("cpu/genai_config.json")).isEqualTo(1);
    }

    @Test
    void rejectsANegativeInterval() {
        assertThatThrownBy(() -> HuggingFaceModelSource.builder().checkInterval(Duration.ofHours(-1)))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsANegativeMinimumSnapshotAge() {
        assertThatThrownBy(() -> HuggingFaceModelSource.builder().minimumSnapshotAge(Duration.ofMinutes(-1)))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
