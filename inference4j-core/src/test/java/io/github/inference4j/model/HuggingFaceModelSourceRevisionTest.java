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

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.github.inference4j.exception.ModelSourceException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
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

    /** A clock the test moves forward. */
    static final class MutableClock extends Clock {
        Instant now = Instant.parse("2026-11-22T10:00:00Z");

        @Override
        public ZoneOffset getZone() {
            return ZoneOffset.UTC;
        }

        @Override
        public Clock withZone(java.time.ZoneId zone) {
            return this;
        }

        @Override
        public Instant instant() {
            return now;
        }
    }

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

    private Path cached(String file) {
        return cacheDir.resolve(REPO).resolve(file);
    }

    @Test
    void aFirstResolveDownloadsAtTheHubRevisionAndRecordsIt() throws IOException {
        Path dir = source().resolve(REPO, FILES);

        assertThat(dir).isEqualTo(cacheDir.resolve(REPO));
        assertThat(Files.readString(cached("model.onnx"))).isEqualTo("weights v1");
        assertThat(hub.downloads("model.onnx")).isEqualTo(1);
        assertThat(hub.downloads("tokenizer.json")).isEqualTo(1);
        CacheRecord record = CacheRecord.read(dir);
        assertThat(record.commit()).isEqualTo(hub.commit());
        assertThat(record.checkedAt()).isEqualTo(clock.now);
        assertThat(record.files()).containsOnlyKeys("model.onnx", "tokenizer.json");
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

        source().resolve(REPO, FILES);

        assertThat(Files.readString(cached("tokenizer.json"))).isEqualTo("tokenizer v2");
        assertThat(hub.downloads("tokenizer.json")).isEqualTo(1);
        assertThat(hub.count("/" + REPO + "/resolve/")).isEqualTo(3);   // two first downloads, one update
        assertThat(CacheRecord.read(cacheDir.resolve(REPO)).commit()).isEqualTo(hub.commit());
    }

    @Test
    void aNewCommitThatChangesNoCachedFileDownloadsNothing() {
        source().resolve(REPO, FILES);
        hub.put("README.md", "card v2");
        clock.now = clock.now.plus(Duration.ofDays(2));

        source().resolve(REPO, FILES);

        assertThat(hub.revisionChecks()).isEqualTo(2);
        assertThat(hub.count("/" + REPO + "/resolve/")).isEqualTo(2);
        CacheRecord record = CacheRecord.read(cacheDir.resolve(REPO));
        assertThat(record.commit()).isEqualTo(hub.commit());
        assertThat(record.checkedAt()).isEqualTo(clock.now);
    }

    @Test
    void aCacheWithoutARecordIsVerifiedByHashing() throws IOException {
        // As left by an earlier version: correct weights (an LFS file), a stale tokenizer
        Files.createDirectories(cacheDir.resolve(REPO));
        Files.writeString(cached("model.onnx"), "weights v1");
        Files.writeString(cached("tokenizer.json"), "tokenizer v0");

        source().resolve(REPO, FILES);

        assertThat(Files.readString(cached("tokenizer.json"))).isEqualTo("tokenizer v1");
        assertThat(hub.downloads("tokenizer.json")).isEqualTo(1);
        assertThat(hub.downloads("model.onnx")).isZero();
    }

    @Test
    void anUnreachableHubFallsBackToTheCacheWithOneWarning() throws IOException {
        source().resolve(REPO, FILES);
        hub.revisionStatus = 503;
        clock.now = clock.now.plus(Duration.ofDays(2));
        HuggingFaceModelSource source = source();

        List<String> warnings = warnings(() -> {
            source.resolve(REPO, FILES);
            source.resolve(REPO, FILES);
        });

        assertThat(Files.readString(cached("tokenizer.json"))).isEqualTo("tokenizer v1");
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
    void missingFilesAreStillDownloadedWhenTheRevisionApiFails() throws IOException {
        hub.revisionStatus = 500;

        source().resolve(REPO, FILES);

        assertThat(Files.readString(cached("tokenizer.json"))).isEqualTo("tokenizer v1");
        assertThat(CacheRecord.read(cacheDir.resolve(REPO))).isNull();
    }

    @Test
    void subdirectoryFilesComeFromTheRevisionListing() throws IOException {
        hub.put("cpu/genai_config.json", "{}").put("cpu/model.onnx", "cpu weights").put("gpu/model.onnx", "gpu");

        Path dir = source().resolve(REPO, "cpu");

        assertThat(dir).isEqualTo(cacheDir.resolve(REPO).resolve("cpu"));
        assertThat(Files.readString(dir.resolve("model.onnx"))).isEqualTo("cpu weights");
        assertThat(dir.resolve("genai_config.json")).exists();
        assertThat(cacheDir.resolve(REPO).resolve("gpu")).doesNotExist();

        hub.put("cpu/model.onnx", "cpu weights v2");
        clock.now = clock.now.plus(Duration.ofDays(2));
        source().resolve(REPO, "cpu");

        assertThat(Files.readString(dir.resolve("model.onnx"))).isEqualTo("cpu weights v2");
        assertThat(hub.downloads("cpu/genai_config.json")).isZero();
    }

    @Test
    void rejectsANegativeInterval() {
        assertThatThrownBy(() -> HuggingFaceModelSource.builder().checkInterval(Duration.ofHours(-1)))
                .isInstanceOf(IllegalArgumentException.class);
    }

    private static List<String> warnings(Runnable action) {
        Logger logger = (Logger) LoggerFactory.getLogger(HuggingFaceModelSource.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
        try {
            action.run();
        } finally {
            logger.detachAppender(appender);
            appender.stop();
        }
        return appender.list.stream().filter(e -> e.getLevel() == Level.WARN)
                .map(ILoggingEvent::getFormattedMessage).toList();
    }
}
