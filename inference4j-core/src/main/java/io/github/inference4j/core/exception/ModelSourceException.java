package io.github.inference4j.core.exception;

public class ModelSourceException extends InferenceException {

    public ModelSourceException(String message) {
        super(message);
    }

    public ModelSourceException(String message, Throwable cause) {
        super(message, cause);
    }
}
