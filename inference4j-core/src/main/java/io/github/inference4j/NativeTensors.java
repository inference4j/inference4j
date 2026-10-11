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

package io.github.inference4j;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;
import io.github.inference4j.exception.InferenceException;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * Tensors kept in ONNX Runtime memory between runs.
 *
 * <p>{@link InferenceSession#runNative(Map, NativeTensors, Set)} returns its outputs this way, and
 * accepts them back as inputs, so data that only flows from one run to the next, such as a language
 * model's key/value cache, never has to be copied into Java and back. Copy out only what Java needs
 * with {@link #copy(String)}.
 *
 * <p>{@link #select(Set)} and {@link #rename(UnaryOperator)} return views that share ownership with
 * this instance: closing any of them frees the memory of the run that produced them. Not thread-safe.
 */
public final class NativeTensors implements AutoCloseable {

    private static final NativeTensors EMPTY = new NativeTensors(new Owner(null), Map.of());

    private final Owner owner;
    private final Map<String, OnnxTensor> tensors;

    NativeTensors(Owner owner, Map<String, OnnxTensor> tensors) {
        this.owner = owner;
        this.tensors = Collections.unmodifiableMap(tensors);
    }

    /** No tensors; closing it does nothing. */
    public static NativeTensors empty() {
        return EMPTY;
    }

    /** The names of the tensors in this view. */
    public Set<String> names() {
        return tensors.keySet();
    }

    /**
     * Copies one tensor into Java memory.
     *
     * @throws InferenceException if there is no tensor with that name, or the memory was freed
     */
    public Tensor copy(String name) {
        OnnxTensor tensor = tensors().get(name);
        if (tensor == null) {
            throw new InferenceException("No native tensor named '" + name + "'");
        }
        return InferenceSession.fromOnnxTensor(tensor);
    }

    /** The named tensors of this view, sharing its ownership. Names not present are ignored. */
    public NativeTensors select(Set<String> names) {
        Map<String, OnnxTensor> selected = new LinkedHashMap<>();
        tensors().forEach((name, tensor) -> {
            if (names.contains(name)) {
                selected.put(name, tensor);
            }
        });
        return new NativeTensors(owner, selected);
    }

    /**
     * The same tensors under new names, sharing this view's ownership; nothing is copied. For example,
     * a model's {@code present.0.key} output becomes its {@code past_key_values.0.key} input.
     */
    public NativeTensors rename(UnaryOperator<String> naming) {
        Map<String, OnnxTensor> renamed = new LinkedHashMap<>();
        tensors().forEach((name, tensor) -> renamed.put(naming.apply(name), tensor));
        return new NativeTensors(owner, renamed);
    }

    /** Whether the memory behind these tensors has been freed. */
    public boolean isClosed() {
        return owner.closed;
    }

    /** Frees the ONNX Runtime memory of the run that produced these tensors, including other views of it. */
    @Override
    public void close() {
        owner.close();
    }

    /** The underlying tensors, for passing to {@link InferenceSession}. */
    Map<String, OnnxTensor> tensors() {
        if (owner.closed) {
            throw new InferenceException("Native tensors were already closed");
        }
        return tensors;
    }

    /** The run result that owns the native memory, shared by all views of it. */
    static final class Owner {
        private final OrtSession.Result result;
        private boolean closed;

        Owner(OrtSession.Result result) {
            this.result = result;
        }

        void close() {
            if (!closed && result != null) {
                closed = true;
                result.close();
            }
        }
    }
}
