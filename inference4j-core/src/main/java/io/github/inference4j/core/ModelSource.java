package io.github.inference4j.core;

import java.nio.file.Path;

public interface ModelSource {
    Path resolve(String modelId);
}
