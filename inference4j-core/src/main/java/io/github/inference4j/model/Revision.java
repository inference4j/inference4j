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

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A repo's current state on the Hub, from {@code GET /api/models/{repo}/revision/main?blobs=true}: the commit
 * and every file's id, keyed by its path in the repo.
 */
record Revision(String commit, Map<String, RemoteFile> files) {

    /** A file on the Hub: its id is its SHA-256 when stored in LFS ({@code lfs}), its git blob id otherwise. */
    record RemoteFile(String id, boolean lfs) {
    }

    private static final ObjectMapper MAPPER = new ObjectMapper();

    static Revision parse(String json) throws IOException {
        JsonNode root = MAPPER.readTree(json);
        String commit = root.path("sha").asText(null);
        if (commit == null || commit.isEmpty()) {
            throw new IOException("Revision response has no sha");
        }
        Map<String, RemoteFile> files = new LinkedHashMap<>();
        for (JsonNode sibling : root.path("siblings")) {
            String path = sibling.path("rfilename").asText();
            JsonNode lfs = sibling.get("lfs");
            if (lfs != null && lfs.hasNonNull("sha256")) {
                files.put(path, new RemoteFile(lfs.get("sha256").asText(), true));
            } else if (sibling.hasNonNull("blobId")) {
                files.put(path, new RemoteFile(sibling.get("blobId").asText(), false));
            }
        }
        return new Revision(commit, Map.copyOf(files));
    }
}
