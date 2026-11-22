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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.inference4j.exception.ModelSourceException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.DirectoryNotEmptyException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.Stream;

/** File operations of the snapshot cache. */
final class CacheFiles {

    private static final Logger logger = LoggerFactory.getLogger(CacheFiles.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private CacheFiles() {
    }

    /** Writes {@code json} to {@code file} atomically, so another process never reads half of it. */
    static void writeJson(Path file, ObjectNode json) {
        try {
            Files.createDirectories(file.getParent());
            Path tmp = file.resolveSibling(file.getFileName() + "." + UUID.randomUUID() + ".tmp");
            try {
                MAPPER.writerWithDefaultPrettyPrinter().writeValue(tmp.toFile(), json);
                Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } finally {
                Files.deleteIfExists(tmp);
            }
        } catch (IOException e) {
            throw new ModelSourceException("Failed to write " + file + ": " + e.getMessage(), e);
        }
    }

    /**
     * Places a complete copy of {@code source} at {@code target}: a hard link when the file system allows one,
     * otherwise (when {@code mayMove}) a move, otherwise a copy written to {@code tmpDir} and moved into place.
     * {@code target} appears complete or not at all.
     */
    static void place(Path source, Path target, Path tmpDir, boolean mayMove) throws IOException {
        Files.createDirectories(target.getParent());
        try {
            Files.createLink(target, source);
            return;
        } catch (UnsupportedOperationException | IOException e) {
            logger.debug("Cannot link {} to {}: {}", target, source, e.toString());
        }
        if (mayMove) {
            try {
                Files.move(source, target, StandardCopyOption.ATOMIC_MOVE);
                return;
            } catch (IOException e) {
                logger.debug("Cannot move {} to {}: {}", source, target, e.toString());
            }
        }
        Files.createDirectories(tmpDir);
        Path tmp = tmpDir.resolve(UUID.randomUUID() + ".part");
        try {
            Files.copy(source, tmp, StandardCopyOption.REPLACE_EXISTING);
            Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    /** The regular files under {@code dir}, as {@code /}-separated paths relative to it, sorted. */
    static List<String> list(Path dir) {
        if (!Files.isDirectory(dir)) {
            return List.of();
        }
        try (Stream<Path> files = Files.walk(dir)) {
            return files.filter(Files::isRegularFile)
                    .map(f -> dir.relativize(f).toString().replace('\\', '/'))
                    .sorted()
                    .toList();
        } catch (IOException | UncheckedIOException e) {
            if (!Files.isDirectory(dir)) {
                return List.of();   // deleted meanwhile, as by another process's eviction
            }
            throw new ModelSourceException("Failed to list " + dir + ": " + e.getMessage(), e);
        }
    }

    /** The regular files directly in {@code dir}, by name, sorted. */
    static List<String> listShallow(Path dir) {
        if (!Files.isDirectory(dir)) {
            return List.of();
        }
        try (Stream<Path> files = Files.list(dir)) {
            return files.filter(Files::isRegularFile).map(f -> f.getFileName().toString()).sorted().toList();
        } catch (IOException e) {
            return List.of();
        }
    }

    /** The subdirectories directly in {@code dir}, by name, sorted. */
    static List<String> directories(Path dir) {
        if (!Files.isDirectory(dir)) {
            return List.of();
        }
        try (Stream<Path> entries = Files.list(dir)) {
            return entries.filter(Files::isDirectory).map(f -> f.getFileName().toString()).sorted().toList();
        } catch (IOException e) {
            return List.of();
        }
    }

    /**
     * What identifies a file's content on disk, so that hard links to it count once: its file key (the inode on
     * Unix). Windows reports no file key; there hard links share their size and timestamps, which a separate
     * copy does not, since cached files are never modified after they are placed.
     */
    static Object key(Path file) {
        try {
            BasicFileAttributes attributes = Files.readAttributes(file, BasicFileAttributes.class);
            Object key = attributes.fileKey();
            return key != null ? key
                    : new LinkKey(attributes.size(), attributes.creationTime(), attributes.lastModifiedTime());
        } catch (IOException e) {
            return file.toAbsolutePath().normalize();
        }
    }

    /** Identifies a file by its size and timestamps, where the file system has no file key. */
    private record LinkKey(long size, FileTime created, FileTime modified) {
    }

    static long size(Path file) {
        try {
            return Files.size(file);
        } catch (IOException e) {
            return 0;
        }
    }

    /** Deletes {@code path} and everything under it, as far as possible; whether it is gone. */
    static boolean deleteRecursively(Path path) {
        if (!Files.exists(path)) {
            return true;
        }
        List<Path> all = new ArrayList<>();
        try (Stream<Path> walk = Files.walk(path)) {
            walk.sorted(Comparator.reverseOrder()).forEach(all::add);
        } catch (IOException | UncheckedIOException e) {
            logger.debug("Cannot list {}: {}", path, e.toString());
            return false;
        }
        boolean deleted = true;
        for (Path p : all) {
            try {
                Files.deleteIfExists(p);
            } catch (NoSuchFileException ignored) {
                // already gone
            } catch (IOException e) {
                logger.debug("Cannot delete {}: {}", p, e.toString());
                deleted = false;
            }
        }
        return deleted;
    }

    /** Deletes {@code dir} and its empty parents up to, not including, {@code stop}. */
    static void deleteEmptyDirectories(Path dir, Path stop) {
        Path current = dir;
        while (current != null && !current.equals(stop) && current.startsWith(stop)) {
            try {
                Files.deleteIfExists(current);
            } catch (DirectoryNotEmptyException e) {
                return;
            } catch (IOException e) {
                return;
            }
            current = current.getParent();
        }
    }
}
