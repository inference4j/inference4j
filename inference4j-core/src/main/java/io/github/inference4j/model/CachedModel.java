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

import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A repo in the cache, as recorded locally.
 *
 * @param repoId          the Hub repo id
 * @param revision        the current snapshot's commit, or {@code null} when it has none
 * @param lastChecked     when the Hub was last asked for the repo's revision, or {@code null} if never
 * @param remoteRevision  the Hub's commit at that check, or {@code null} if never checked
 * @param updateAvailable whether that check found cached files that changed and have not been downloaded yet
 * @param snapshots       its snapshots, the current one included
 * @param sizeBytes       the disk space its snapshots use, counting each linked file once
 */
public record CachedModel(String repoId, String revision, Instant lastChecked, String remoteRevision,
                          boolean updateAvailable, List<CachedSnapshot> snapshots, long sizeBytes) {

    public CachedModel {
        snapshots = List.copyOf(snapshots);
    }

    /**
     * Whether a model still in use reads an older snapshot whose files differ from the current one, so that
     * it serves an older version of the model than a model loaded now would.
     */
    public boolean stale() {
        CachedSnapshot current = snapshots.stream().filter(CachedSnapshot::current).findFirst().orElse(null);
        if (current == null) {
            return false;
        }
        Map<String, String> currentIds = new HashMap<>();
        current.files().forEach(f -> currentIds.put(f.path(), f.id()));
        for (CachedSnapshot snapshot : snapshots) {
            if (snapshot.current() || snapshot.leases() == 0) {
                continue;
            }
            for (CachedFile file : snapshot.files()) {
                if (currentIds.containsKey(file.path()) && !Objects.equals(currentIds.get(file.path()), file.id())) {
                    return true;
                }
            }
        }
        return false;
    }
}
