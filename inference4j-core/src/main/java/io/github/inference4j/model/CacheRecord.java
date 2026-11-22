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
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * What a cached repo holds: the Hub commit it was last checked against, when, and the id of each cached
 * file (its git blob id, or its SHA-256 for files stored in LFS). Kept in {@value #FILE_NAME} in the
 * repo's cache directory.
 */
record CacheRecord(String commit, Instant checkedAt, Map<String, String> files) {

    static final String FILE_NAME = ".inference4j-revision.json";

    private static final Logger logger = LoggerFactory.getLogger(CacheRecord.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    CacheRecord {
        files = Map.copyOf(files);
    }

    /** Reads the record of {@code repoDir}; {@code null} if there is none or it cannot be read. */
    static CacheRecord read(Path repoDir) {
        Path file = repoDir.resolve(FILE_NAME);
        if (!Files.exists(file)) {
            return null;
        }
        try {
            JsonNode root = MAPPER.readTree(file.toFile());
            Map<String, String> files = new LinkedHashMap<>();
            root.path("files").fields().forEachRemaining(e -> files.put(e.getKey(), e.getValue().asText()));
            return new CacheRecord(root.path("commit").asText(), Instant.parse(root.path("checkedAt").asText()),
                    files);
        } catch (IOException | RuntimeException e) {
            logger.debug("Ignoring unreadable {}: {}", file, e.toString());
            return null;
        }
    }

    /** Writes the record atomically, so another process never reads half of it. */
    void write(Path repoDir) {
        ObjectNode root = MAPPER.createObjectNode()
                .put("commit", commit)
                .put("checkedAt", checkedAt.toString());
        ObjectNode filesNode = root.putObject("files");
        files.forEach(filesNode::put);
        Path file = repoDir.resolve(FILE_NAME);
        Path tmp = repoDir.resolve(FILE_NAME + ".tmp");
        try {
            MAPPER.writerWithDefaultPrettyPrinter().writeValue(tmp.toFile(), root);
            Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            // Only costs a re-check next time
            logger.warn("Could not write {}: {}", file, e.toString());
        }
    }
}
