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

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * Inspects and manages the models a {@link HuggingFaceModelSource} keeps on disk.
 *
 * <p>Each cached repo has one or more snapshots, one per commit; the current snapshot is the one models load.
 * Snapshots a loaded model still uses are never deleted, nor are snapshots younger than the source's
 * {@linkplain HuggingFaceModelSource.Builder#minimumSnapshotAge minimum age}.
 *
 * <pre>{@code
 * ModelCache cache = ModelCache.defaultCache();
 * for (CachedModel model : cache.list()) {
 *     System.out.println(model.repoId() + " " + model.revision() + " " + model.sizeBytes());
 * }
 * if (cache.check("inference4j/silero-vad").updateAvailable()) {
 *     cache.update("inference4j/silero-vad");
 * }
 * }</pre>
 *
 * <p>Thread-safe.
 */
public interface ModelCache {

    /** The default source's cache: {@code HuggingFaceModelSource.defaultInstance().cache()}. */
    static ModelCache defaultCache() {
        return HuggingFaceModelSource.defaultInstance().cache();
    }

    /** The cache directory. */
    Path directory();

    /** Every cached repo, read from local state only. */
    List<CachedModel> list();

    /**
     * A cached repo, read from local state only.
     *
     * @throws IllegalArgumentException if {@code repoId} is not a valid repo id
     */
    Optional<CachedModel> find(String repoId);

    /**
     * Asks the Hub whether a cached repo changed, without downloading anything. The result is recorded, so
     * {@link CachedModel#updateAvailable()} reflects it, and the next load fetches the changed files.
     *
     * @throws io.github.inference4j.exception.ModelSourceException   if the repo is not cached, or the source is
     *                                                                offline
     * @throws io.github.inference4j.exception.ModelDownloadException if the Hub cannot be reached
     */
    UpdateCheck check(String repoId);

    /**
     * Brings a cached repo up to date with the Hub now, whatever the check interval, downloading its changed
     * files into a new snapshot. Models already loaded keep their snapshot; models loaded afterwards read the
     * new one.
     *
     * @return the repo after the update
     * @throws io.github.inference4j.exception.ModelSourceException   if the repo is not cached, or the source is
     *                                                                offline
     * @throws io.github.inference4j.exception.ModelDownloadException if the Hub cannot be reached or a download
     *                                                                fails
     */
    CachedModel update(String repoId);

    /**
     * Deletes a repo's snapshots, except those in use, and the files of the older flat cache layout it was
     * adopted from. Models loaded afterwards download it again.
     */
    EvictionResult evict(String repoId);

    /** Deletes one snapshot of a repo, unless it is in use. */
    EvictionResult evict(String repoId, String revision);

    /**
     * Deletes every repo's superseded snapshots that are not in use, unfinished downloads, and the files of the
     * older flat cache layout that were adopted into snapshots.
     */
    EvictionResult prune();
}
