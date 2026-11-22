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

import io.github.inference4j.exception.ModelSourceException;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.List;
import java.util.UUID;
import java.util.regex.Pattern;

/**
 * Where a repo's files live in the cache:
 * <pre>
 * {cacheDir}/hub/.locks/models--{org}--{name}.lock     exclusive lock for sync, migration and eviction
 * {cacheDir}/hub/models--{org}--{name}/
 *     repo.json                 {@link RepoState}
 *     refs/main                 the current snapshot's name
 *     snapshots/{revision}/     the files at that commit, at their repo paths
 *     manifests/{revision}.json {@link SnapshotManifest}
 *     tmp/                      downloads in progress
 *     trash/                    snapshots being deleted
 * {cacheDir}/{org}/{name}/      the older flat layout, adopted into snapshots
 * </pre>
 */
final class RepoLayout {

    static final String HUB = "hub";
    static final String PREFIX = "models--";

    private static final Pattern REPO_ID = Pattern.compile("[\\w.-]+(/[\\w.-]+)?");
    private static final Pattern REVISION = Pattern.compile("[0-9A-Za-z]{1,64}");

    final String repoId;
    final Path root;
    final Path legacy;
    final Path lockFile;

    private RepoLayout(Path cacheDir, String repoId) {
        this.repoId = repoId;
        Path hub = cacheDir.resolve(HUB);
        String name = directoryName(repoId);
        this.root = hub.resolve(name);
        // A repo under an org named like the snapshot root has no flat layout to adopt
        this.legacy = repoId.split("/")[0].equals(HUB) ? null : cacheDir.resolve(repoId);
        this.lockFile = hub.resolve(".locks").resolve(name + ".lock");
    }

    static RepoLayout of(Path cacheDir, String repoId) {
        return new RepoLayout(cacheDir, validateRepoId(repoId));
    }

    static String directoryName(String repoId) {
        return PREFIX + repoId.replace("/", "--");
    }

    /** {@code repoId} if it names a Hub repo, such as {@code org/name}; it can never escape the cache. */
    static String validateRepoId(String repoId) {
        if (repoId == null || !REPO_ID.matcher(repoId).matches()) {
            throw new IllegalArgumentException("Invalid Hub repo id: '" + repoId + "'");
        }
        for (String part : repoId.split("/")) {
            if (part.equals(".") || part.equals("..")) {
                throw new IllegalArgumentException("Invalid Hub repo id: '" + repoId + "'");
            }
        }
        return repoId;
    }

    /** {@code revision} if it can name a snapshot directory. */
    static String validateRevision(String revision) {
        if (revision == null || !REVISION.matcher(revision).matches()) {
            throw new IllegalArgumentException("Invalid revision: '" + revision + "'");
        }
        return revision;
    }

    /** {@code path} if it is a relative path inside the repo, with {@code /} separators and no {@code ..}. */
    static String validatePath(String path) {
        if (path == null || path.isEmpty() || path.startsWith("/") || path.contains("\\") || path.contains(":")) {
            throw new IllegalArgumentException("Invalid file path in repo: '" + path + "'");
        }
        for (String part : path.split("/", -1)) {
            if (part.isEmpty() || part.equals(".") || part.equals("..")) {
                throw new IllegalArgumentException("Invalid file path in repo: '" + path + "'");
            }
        }
        return path;
    }

    /** Whether {@code path} is a valid repo path; for listings from the Hub, whose odd entries are skipped. */
    static boolean isValidPath(String path) {
        try {
            validatePath(path);
            return true;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    Path stateFile() {
        return root.resolve("repo.json");
    }

    Path refFile() {
        return root.resolve("refs").resolve("main");
    }

    Path snapshots() {
        return root.resolve("snapshots");
    }

    Path snapshot(String revision) {
        return snapshots().resolve(revision);
    }

    Path manifestFile(String revision) {
        return root.resolve("manifests").resolve(revision + ".json");
    }

    Path tmp() {
        return root.resolve("tmp");
    }

    Path trash() {
        return root.resolve("trash");
    }

    /** The current snapshot's name, or {@code null} if there is none. */
    String readRef() {
        Path ref = refFile();
        if (!Files.isRegularFile(ref)) {
            return null;
        }
        try {
            String name = Files.readString(ref, StandardCharsets.UTF_8).strip();
            return REVISION.matcher(name).matches() && Files.isDirectory(snapshot(name)) ? name : null;
        } catch (IOException e) {
            return null;
        }
    }

    /** Points {@code refs/main} at {@code revision}, atomically. */
    void writeRef(String revision) {
        Path ref = refFile();
        try {
            Files.createDirectories(ref.getParent());
            Path tmp = ref.resolveSibling("main." + UUID.randomUUID() + ".tmp");
            try {
                Files.writeString(tmp, revision + "\n", StandardCharsets.UTF_8);
                Files.move(tmp, ref, StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING);
            } finally {
                Files.deleteIfExists(tmp);
            }
        } catch (IOException e) {
            throw new ModelSourceException("Failed to write " + ref + ": " + e.getMessage(), e);
        }
    }

    void deleteRef() {
        try {
            Files.deleteIfExists(refFile());
        } catch (IOException e) {
            throw new ModelSourceException("Failed to delete " + refFile() + ": " + e.getMessage(), e);
        }
    }

    RepoState readState() {
        return RepoState.read(stateFile());
    }

    SnapshotManifest readManifest(String revision) {
        return SnapshotManifest.read(manifestFile(revision));
    }

    /** The snapshots on disk, by name. */
    List<String> snapshotNames() {
        return CacheFiles.directories(snapshots()).stream().filter(n -> REVISION.matcher(n).matches()).toList();
    }

    /** When {@code revision}'s snapshot was created: from its manifest, else the directory's modification time. */
    Instant createdAt(String revision) {
        SnapshotManifest manifest = readManifest(revision);
        if (manifest != null) {
            return manifest.createdAt();
        }
        try {
            return Files.getLastModifiedTime(snapshot(revision)).toInstant();
        } catch (IOException e) {
            return Instant.EPOCH;
        }
    }

    /** The repo id recorded for the repo directory {@code dir}, else decoded from its name. */
    static String repoIdOf(Path dir) {
        RepoState state = RepoState.read(dir.resolve("repo.json"));
        if (state != null && !state.repoId().isEmpty()) {
            return state.repoId();
        }
        String name = dir.getFileName().toString().substring(PREFIX.length());
        int separator = name.indexOf("--");
        return separator < 0 ? name : name.substring(0, separator) + "/" + name.substring(separator + 2);
    }
}
