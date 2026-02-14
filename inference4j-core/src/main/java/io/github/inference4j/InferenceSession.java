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

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.exception.TensorConversionException;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Manages an ONNX Runtime session for running model inference.
 *
 * <p>This is the primary entry point for loading an ONNX model and executing
 * inference. It wraps the ONNX Runtime C++ API, converting between
 * inference4j {@link Tensor} objects and native ONNX tensors.
 *
 * <p>Sessions are expensive to create (they load and optimize the model graph)
 * but cheap to call repeatedly. Create one session and reuse it across requests.
 *
 * <p>Example usage:
 * <pre>{@code
 * try (InferenceSession session = InferenceSession.create(Path.of("model.onnx"))) {
 *     Map<String, Tensor> inputs = Map.of("input", Tensor.fromFloats(data, shape));
 *     Map<String, Tensor> outputs = session.run(inputs);
 *     float[] result = outputs.get("output").toFloats();
 * }
 * }</pre>
 *
 * @see Tensor
 * @see SessionOptions
 */
public class InferenceSession implements AutoCloseable {

    private final OrtEnvironment environment;
    private final OrtSession session;

    private InferenceSession(OrtEnvironment environment, OrtSession session) {
        this.environment = environment;
        this.session = session;
    }

    /**
     * Creates a session from an ONNX model file with default options.
     *
     * @param modelPath path to the {@code .onnx} model file
     * @return a new session ready for inference
     * @throws io.github.inference4j.exception.ModelLoadException if the model cannot be loaded
     */
    public static InferenceSession create(Path modelPath) {
        return create(modelPath, SessionOptions.defaults());
    }

    /**
     * Creates a session from an ONNX model file with a configurer for custom options.
     *
     * <p>Applies default options (thread counts, optimization level) first, then
     * invokes the configurer to customize further (e.g., add GPU execution providers).
     *
     * @param modelPath  path to the {@code .onnx} model file
     * @param configurer callback to customize session options
     * @return a new session ready for inference
     * @throws io.github.inference4j.exception.ModelLoadException if the model cannot be loaded
     * @see SessionConfigurer
     */
    public static InferenceSession create(Path modelPath, SessionConfigurer configurer) {
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
                int cpus = Runtime.getRuntime().availableProcessors();
                opts.setIntraOpNumThreads(cpus);
                opts.setInterOpNumThreads(cpus);
                opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                configurer.configure(opts);
                OrtSession session = env.createSession(modelPath.toString(), opts);
                return new InferenceSession(env, session);
            }
        } catch (OrtException e) {
            throw new ModelLoadException(
                    "Failed to load model from " + modelPath + ": " + e.getMessage(), e);
        }
    }

    /**
     * Creates a session from an ONNX model file with custom options.
     *
     * @param modelPath path to the {@code .onnx} model file
     * @param options   session configuration (thread counts, optimization level)
     * @return a new session ready for inference
     * @throws io.github.inference4j.exception.ModelLoadException if the model cannot be loaded
     */
    public static InferenceSession create(Path modelPath, SessionOptions options) {
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            try (OrtSession.SessionOptions ortOptions = options.toOrtOptions()) {
                OrtSession session = env.createSession(modelPath.toString(), ortOptions);
                return new InferenceSession(env, session);
            }
        } catch (OrtException e) {
            throw new ModelLoadException(
                    "Failed to load model from " + modelPath + ": " + e.getMessage(), e);
        }
    }

    /**
     * Returns the names of all input tensors expected by the model.
     *
     * @return set of input tensor names
     */
    public Set<String> inputNames() {
        return session.getInputNames();
    }

    /**
     * Returns the shape of the named input tensor as defined in the model.
     *
     * <p>Dynamic dimensions are represented as {@code -1}.
     *
     * @param name the input tensor name
     * @return the tensor shape (e.g., {@code [1, 3, 224, 224]} for a batch-1 RGB image)
     * @throws InferenceException if the input name is not found
     */
    public long[] inputShape(String name) {
        try {
            NodeInfo info = session.getInputInfo().get(name);
            if (info == null) {
                throw new InferenceException("Unknown input: " + name);
            }
            return ((TensorInfo) info.getInfo()).getShape();
        } catch (OrtException e) {
            throw new InferenceException("Failed to get input shape: " + e.getMessage(), e);
        }
    }

    /**
     * Runs inference with the given input tensors and returns the model outputs.
     *
     * @param inputs map of input name to tensor (must match the model's expected inputs)
     * @return map of output name to tensor, in model-defined order
     * @throws InferenceException if inference fails
     * @throws io.github.inference4j.exception.TensorConversionException if a tensor type is unsupported
     */
    public Map<String, Tensor> run(Map<String, Tensor> inputs) {
        Map<String, OnnxTensor> onnxInputs = new HashMap<>();
        try {
            for (var entry : inputs.entrySet()) {
                onnxInputs.put(entry.getKey(), toOnnxTensor(entry.getValue()));
            }

            try (OrtSession.Result result = session.run(onnxInputs)) {
                Map<String, Tensor> outputs = new LinkedHashMap<>();
                for (Map.Entry<String, OnnxValue> entry : result) {
                    if (entry.getValue() instanceof OnnxTensor onnxTensor) {
                        outputs.put(entry.getKey(), fromOnnxTensor(onnxTensor));
                    }
                }
                return outputs;
            }
        } catch (OrtException e) {
            throw new InferenceException("Inference failed: " + e.getMessage(), e);
        } finally {
            onnxInputs.values().forEach(OnnxTensor::close);
        }
    }

    private OnnxTensor toOnnxTensor(Tensor tensor) throws OrtException {
        return switch (tensor.type()) {
            case FLOAT -> OnnxTensor.createTensor(environment,
                    FloatBuffer.wrap((float[]) tensor.rawData()), tensor.shape());
            case LONG -> OnnxTensor.createTensor(environment,
                    LongBuffer.wrap((long[]) tensor.rawData()), tensor.shape());
            default -> throw new TensorConversionException(
                    "Unsupported tensor type for ONNX conversion: " + tensor.type());
        };
    }

    private Tensor fromOnnxTensor(OnnxTensor onnxTensor) {
        TensorInfo info = onnxTensor.getInfo();
        long[] shape = info.getShape();

        return switch (info.type) {
            case FLOAT -> {
                FloatBuffer buffer = onnxTensor.getFloatBuffer();
                float[] data = new float[buffer.remaining()];
                buffer.get(data);
                yield Tensor.fromFloats(data, shape);
            }
            case INT64 -> {
                LongBuffer buffer = onnxTensor.getLongBuffer();
                long[] data = new long[buffer.remaining()];
                buffer.get(data);
                yield Tensor.fromLongs(data, shape);
            }
            default -> throw new TensorConversionException(
                    "Unsupported ONNX tensor type: " + info.type);
        };
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException e) {
            // Silently ignore close errors
        }
    }
}
