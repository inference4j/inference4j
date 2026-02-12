package io.github.inference4j.core;

import io.github.inference4j.core.exception.ModelSourceException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class LocalModelSourceTest {

    @TempDir
    Path tempDir;

    @Test
    void resolve_returnsDirectoryPath() throws IOException {
        Path modelDir = tempDir.resolve("my-model");
        Files.createDirectory(modelDir);

        LocalModelSource source = new LocalModelSource(tempDir);
        Path resolved = source.resolve("my-model");

        assertEquals(modelDir, resolved);
        assertTrue(Files.isDirectory(resolved));
    }

    @Test
    void resolve_throwsWhenDirectoryNotFound() {
        LocalModelSource source = new LocalModelSource(tempDir);

        ModelSourceException ex = assertThrows(ModelSourceException.class, () ->
                source.resolve("nonexistent"));

        assertTrue(ex.getMessage().contains("nonexistent"));
    }

    @Test
    void resolve_throwsWhenPathIsFile() throws IOException {
        Files.writeString(tempDir.resolve("not-a-dir"), "content");

        LocalModelSource source = new LocalModelSource(tempDir);

        assertThrows(ModelSourceException.class, () ->
                source.resolve("not-a-dir"));
    }
}
