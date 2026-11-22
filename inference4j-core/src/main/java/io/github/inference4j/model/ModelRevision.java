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

import java.util.Objects;

/**
 * The revision of a Hub repo that a loaded model was read from.
 *
 * @param repoId the Hub repo id, such as {@code inference4j/silero-vad}
 * @param commit the commit the cached files belong to, or {@value #UNKNOWN} when the files were adopted
 *               from an older cache without a recorded commit
 */
public record ModelRevision(String repoId, String commit) {

    /** The commit of files adopted from an older cache before their commit was checked against the Hub. */
    public static final String UNKNOWN = "unknown";

    public ModelRevision {
        Objects.requireNonNull(repoId, "repoId");
        Objects.requireNonNull(commit, "commit");
    }
}
