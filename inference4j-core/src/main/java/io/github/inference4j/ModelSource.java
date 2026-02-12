package io.github.inference4j;

import java.nio.file.Path;

public interface ModelSource {
    Path resolve(String modelId);
}
