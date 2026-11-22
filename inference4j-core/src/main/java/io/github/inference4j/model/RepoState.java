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
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * What a cached repo last learned from the Hub, kept in {@code repo.json}: when it was checked, the Hub's commit
 * then, and which cached files differed from the current snapshot.
 */
record RepoState(String repoId, Instant checkedAt, String remoteCommit, List<String> remoteChangedFiles) {

    private static final Logger logger = LoggerFactory.getLogger(RepoState.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    RepoState {
        remoteChangedFiles = List.copyOf(remoteChangedFiles);
    }

    /** The state in {@code file}; {@code null} if there is none or it cannot be read. */
    static RepoState read(Path file) {
        if (!Files.isRegularFile(file)) {
            return null;
        }
        try {
            JsonNode root = MAPPER.readTree(file.toFile());
            List<String> changed = new ArrayList<>();
            root.path("remoteChangedFiles").forEach(n -> changed.add(n.asText()));
            return new RepoState(root.path("repoId").asText(), Instant.parse(root.path("checkedAt").asText()),
                    root.path("remoteCommit").asText(null), changed);
        } catch (IOException | RuntimeException e) {
            logger.debug("Ignoring unreadable {}: {}", file, e.toString());
            return null;
        }
    }

    void write(Path file) {
        ObjectNode root = MAPPER.createObjectNode()
                .put("repoId", repoId)
                .put("checkedAt", checkedAt.toString())
                .put("remoteCommit", remoteCommit);
        ArrayNode changed = root.putArray("remoteChangedFiles");
        remoteChangedFiles.forEach(changed::add);
        CacheFiles.writeJson(file, root);
    }
}
