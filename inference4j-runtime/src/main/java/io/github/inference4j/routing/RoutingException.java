package io.github.inference4j.routing;

import io.github.inference4j.exception.InferenceException;

public class RoutingException extends InferenceException {

    public RoutingException(String message) {
        super(message);
    }

    public RoutingException(String message, Throwable cause) {
        super(message, cause);
    }
}
