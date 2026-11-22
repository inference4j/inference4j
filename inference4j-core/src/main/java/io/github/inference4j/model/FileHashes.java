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
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/** The ids the Hub gives files, computed from local copies. */
final class FileHashes {

    private FileHashes() {
    }

    /** The SHA-256 of the file, as the Hub reports it for files stored in LFS. */
    static String sha256(Path file) {
        return digest("SHA-256", file, null);
    }

    /** The git blob id of the file: SHA-1 of {@code "blob <size>\0"} followed by its bytes. */
    static String gitBlobId(Path file) {
        try {
            byte[] header = ("blob " + Files.size(file) + "\0").getBytes(StandardCharsets.US_ASCII);
            return digest("SHA-1", file, header);
        } catch (IOException e) {
            throw new ModelSourceException("Failed to read " + file + ": " + e.getMessage(), e);
        }
    }

    private static String digest(String algorithm, Path file, byte[] header) {
        try (InputStream in = Files.newInputStream(file)) {
            MessageDigest digest = MessageDigest.getInstance(algorithm);
            if (header != null) {
                digest.update(header);
            }
            byte[] buffer = new byte[1 << 16];
            for (int n = in.read(buffer); n >= 0; n = in.read(buffer)) {
                digest.update(buffer, 0, n);
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (IOException e) {
            throw new ModelSourceException("Failed to read " + file + ": " + e.getMessage(), e);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(algorithm + " is not available", e);
        }
    }
}
