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

import io.github.inference4j.model.UpdateCheck;

import java.time.Instant;
import java.util.List;

/**
 * {@link UpdateCheck}'s components, plus whether an update would download anything.
 *
 * @param cachedRevision  the cached commit
 * @param remoteRevision  the Hub's commit
 * @param changedFiles    the cached files that changed on the Hub
 * @param checkedAt       when the Hub was asked
 * @param updateAvailable whether any cached file changed
 */
public record ModelCheck(String cachedRevision, String remoteRevision, List<String> changedFiles,
                         Instant checkedAt, boolean updateAvailable) {

    public ModelCheck {
        changedFiles = List.copyOf(changedFiles);
    }

    static ModelCheck of(UpdateCheck check) {
        return new ModelCheck(check.cachedRevision(), check.remoteRevision(), check.changedFiles(),
                check.checkedAt(), check.updateAvailable());
    }
}
