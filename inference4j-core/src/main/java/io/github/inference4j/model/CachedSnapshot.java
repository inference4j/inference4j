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

import java.util.List;

/**
 * One snapshot of a cached repo: its files at one commit.
 *
 * @param revision  the commit, or {@value ModelRevision#UNKNOWN} for files adopted from an older cache
 * @param current   whether models loaded now read this snapshot
 * @param leases    how many open handles in this JVM keep it in use, such as loaded models
 * @param files     its files
 * @param sizeBytes the size of its files
 */
public record CachedSnapshot(String revision, boolean current, int leases, List<CachedFile> files, long sizeBytes) {

    public CachedSnapshot {
        files = List.copyOf(files);
    }
}
