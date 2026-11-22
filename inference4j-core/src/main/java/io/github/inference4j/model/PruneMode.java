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

/**
 * What happens to the snapshots an update replaces.
 *
 * <p>Set with {@link HuggingFaceModelSource.Builder#pruneMode(PruneMode)}, the system property
 * {@code inference4j.cache.prune} or the environment variable {@code INFERENCE4J_CACHE_PRUNE}
 * ({@code auto} or {@code manual}). Defaults to {@link #AUTO}.
 */
public enum PruneMode {

    /**
     * After each update, the repo's superseded snapshots are deleted, except those a loaded model still uses
     * and those younger than the {@linkplain HuggingFaceModelSource.Builder#minimumSnapshotAge minimum age}.
     */
    AUTO,

    /** Superseded snapshots stay on disk until {@link ModelCache#prune()} or {@link ModelCache#evict} removes them. */
    MANUAL
}
