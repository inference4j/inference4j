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

import io.github.inference4j.model.CachedModel;
import io.github.inference4j.model.CachedSnapshot;

import java.time.Instant;
import java.util.List;

/**
 * A cached repo as a web tier shows it: {@link CachedModel}'s components, plus whether it is
 * {@linkplain CachedModel#stale() stale}, whether an update started by the {@link ModelCacheService} is still
 * running, and the Hub check when one was asked for.
 *
 * @param repoId          the Hub repo id
 * @param revision        the current snapshot's commit, or {@code null} when it has none
 * @param lastChecked     when the Hub was last asked for the repo's revision, or {@code null} if never
 * @param remoteRevision  the Hub's commit at that check, or {@code null} if never checked
 * @param updateAvailable whether that check found cached files that changed and have not been downloaded yet
 * @param stale           whether a model in use reads an older version than a model loaded now would
 * @param updating        whether an update is running in the background
 * @param snapshots       its snapshots, the current one included
 * @param sizeBytes       the disk space its snapshots use
 * @param check           the Hub check, or {@code null} when none was asked for
 */
public record ModelView(String repoId, String revision, Instant lastChecked, String remoteRevision,
                        boolean updateAvailable, boolean stale, boolean updating, List<CachedSnapshot> snapshots,
                        long sizeBytes, ModelCheck check) {

    public ModelView {
        snapshots = List.copyOf(snapshots);
    }
}
