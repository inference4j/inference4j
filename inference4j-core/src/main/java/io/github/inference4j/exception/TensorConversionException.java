package io.github.inference4j.exception;

public class TensorConversionException extends InferenceException {

    public TensorConversionException(String message) {
        super(message);
    }

    public TensorConversionException(String message, Throwable cause) {
        super(message, cause);
    }
}
