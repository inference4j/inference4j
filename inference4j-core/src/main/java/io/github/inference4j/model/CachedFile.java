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
 * A file in a cached snapshot.
 *
 * @param path      the file's path in the repo, with {@code /} separators
 * @param id        the id the Hub gives the file (its SHA-256 when stored in LFS, its git blob id otherwise), or
 *                  {@code null} when it was not recorded, as for files adopted from an older cache
 * @param sizeBytes the file's size
 */
public record CachedFile(String path, String id, long sizeBytes) {
}
