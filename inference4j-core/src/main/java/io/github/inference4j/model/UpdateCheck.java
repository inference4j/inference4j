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
import java.util.List;

/**
 * What the Hub reported when a cached repo was checked for updates.
 *
 * @param repoId         the Hub repo id
 * @param cachedRevision the current snapshot's commit
 * @param remoteRevision the Hub's current commit
 * @param changedFiles   the cached files whose content differs on the Hub
 * @param checkedAt      when the check ran
 */
public record UpdateCheck(String repoId, String cachedRevision, String remoteRevision, List<String> changedFiles,
                          Instant checkedAt) {

    public UpdateCheck {
        changedFiles = List.copyOf(changedFiles);
    }

    /** Whether {@link ModelCache#update(String)} would download changed files. */
    public boolean updateAvailable() {
        return !changedFiles.isEmpty();
    }
}
