package io.github.inference4j.core;

import io.github.inference4j.core.exception.ModelLoadException;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class InferenceSessionTest {

    @Test
    void create_throwsModelLoadExceptionForNonexistentFile() {
        Path badPath = Path.of("/nonexistent/model.onnx");

        ModelLoadException ex = assertThrows(ModelLoadException.class, () ->
                InferenceSession.create(badPath));

        assertTrue(ex.getMessage().contains("/nonexistent/model.onnx"));
    }

    @Test
    void create_throwsModelLoadExceptionForInvalidFile() throws Exception {
        Path tempFile = java.nio.file.Files.createTempFile("not-a-model", ".onnx");
        try {
            java.nio.file.Files.writeString(tempFile, "this is not an onnx model");

            ModelLoadException ex = assertThrows(ModelLoadException.class, () ->
                    InferenceSession.create(tempFile));

            assertTrue(ex.getMessage().contains(tempFile.toString()));
        } finally {
            java.nio.file.Files.deleteIfExists(tempFile);
        }
    }
}
