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

/**
 * Totals over every cached repo.
 *
 * @param models           the cached repos
 * @param snapshots        their snapshots, current and superseded
 * @param snapshotsInUse   the snapshots that loaded models read
 * @param updatesAvailable the repos whose last Hub check found changed files not downloaded yet
 * @param stale            the repos with a model in use that reads an older version than a model loaded now
 *                         would
 * @param updating         the updates running in the background
 * @param sizeBytes        the disk space the cached repos use
 * @param supersededBytes  the disk space of superseded snapshots not in use, which a prune deletes
 * @param disk             the volume that holds the cache, or {@code null} when its usage cannot be read
 */
public record CacheSummary(int models, int snapshots, int snapshotsInUse, int updatesAvailable, int stale,
                           int updating, long sizeBytes, long supersededBytes, DiskUsage disk) {
}
