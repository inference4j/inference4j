package io.github.inference4j;

import io.github.inference4j.exception.ModelSourceException;
import java.nio.file.Files;
import java.nio.file.Path;

public class LocalModelSource implements ModelSource {

    private final Path baseDir;

    public LocalModelSource(Path baseDir) {
        this.baseDir = baseDir;
    }

    @Override
    public Path resolve(String modelId) {
        Path resolved = baseDir.resolve(modelId);
        if (!Files.isDirectory(resolved)) {
            throw new ModelSourceException("Model directory not found: " + resolved);
        }
        return resolved;
    }
}
