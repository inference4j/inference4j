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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * What a snapshot holds, kept in {@code manifests/{revision}.json}: its commit, when it was created, each file's
 * Hub id and size, and the subdirectories that were requested whole. Advisory: a file without a recorded id is
 * verified by hashing it.
 *
 * @param commit         the commit, or {@code null} for files adopted from an older cache
 * @param createdAt      when the snapshot was created, which the minimum age before deletion counts from
 * @param files          the files by repo path
 * @param subdirectories the subdirectories requested whole, re-listed at each update
 */
record SnapshotManifest(String commit, Instant createdAt, Map<String, Entry> files, Set<String> subdirectories) {

    /** A file: its Hub id ({@code null} if not known), whether that id is an LFS SHA-256, and its size. */
    record Entry(String id, boolean lfs, long size) {
    }

    private static final Logger logger = LoggerFactory.getLogger(SnapshotManifest.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    SnapshotManifest {
        files = Map.copyOf(files);
        subdirectories = Set.copyOf(subdirectories);
    }

    static SnapshotManifest empty(String commit, Instant createdAt) {
        return new SnapshotManifest(commit, createdAt, Map.of(), Set.of());
    }

    SnapshotManifest with(Map<String, Entry> added, Set<String> addedSubdirectories) {
        Map<String, Entry> allFiles = new LinkedHashMap<>(files);
        allFiles.putAll(added);
        Set<String> allSubdirectories = new LinkedHashSet<>(subdirectories);
        allSubdirectories.addAll(addedSubdirectories);
        return new SnapshotManifest(commit, createdAt, allFiles, allSubdirectories);
    }

    /** The recorded id of {@code path} if it was recorded with the same kind of id; {@code null} otherwise. */
    String id(String path, boolean lfs) {
        Entry entry = files.get(path);
        return entry != null && entry.lfs() == lfs ? entry.id() : null;
    }

    /** The manifest in {@code file}; {@code null} if there is none or it cannot be read. */
    static SnapshotManifest read(Path file) {
        if (!Files.isRegularFile(file)) {
            return null;
        }
        try {
            JsonNode root = MAPPER.readTree(file.toFile());
            Map<String, Entry> files = new LinkedHashMap<>();
            root.path("files").fields().forEachRemaining(e -> files.put(e.getKey(), new Entry(
                    e.getValue().path("id").asText(null), e.getValue().path("lfs").asBoolean(),
                    e.getValue().path("size").asLong())));
            List<String> subdirectories = new ArrayList<>();
            root.path("subdirectories").forEach(n -> subdirectories.add(n.asText()));
            return new SnapshotManifest(root.path("commit").asText(null),
                    Instant.parse(root.path("createdAt").asText()), files, new LinkedHashSet<>(subdirectories));
        } catch (IOException | RuntimeException e) {
            logger.debug("Ignoring unreadable {}: {}", file, e.toString());
            return null;
        }
    }

    void write(Path file) {
        ObjectNode root = MAPPER.createObjectNode()
                .put("commit", commit)
                .put("createdAt", createdAt.toString());
        ObjectNode filesNode = root.putObject("files");
        files.entrySet().stream().sorted(Map.Entry.comparingByKey()).forEach(e -> filesNode.putObject(e.getKey())
                .put("id", e.getValue().id())
                .put("lfs", e.getValue().lfs())
                .put("size", e.getValue().size()));
        subdirectories.stream().sorted().forEach(root.putArray("subdirectories")::add);
        CacheFiles.writeJson(file, root);
    }
}
