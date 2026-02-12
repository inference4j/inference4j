package io.github.inference4j.core;

import ai.onnxruntime.*;
import io.github.inference4j.core.exception.InferenceException;
import io.github.inference4j.core.exception.ModelLoadException;
import io.github.inference4j.core.exception.TensorConversionException;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class InferenceSession implements AutoCloseable {

    private final OrtEnvironment environment;
    private final OrtSession session;

    private InferenceSession(OrtEnvironment environment, OrtSession session) {
        this.environment = environment;
        this.session = session;
    }

    public static InferenceSession create(Path modelPath) {
        return create(modelPath, SessionOptions.defaults());
    }

    public static InferenceSession create(Path modelPath, SessionOptions options) {
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions ortOptions = options.toOrtOptions();
            OrtSession session = env.createSession(modelPath.toString(), ortOptions);
            return new InferenceSession(env, session);
        } catch (OrtException e) {
            throw new ModelLoadException(
                    "Failed to load model from " + modelPath + ": " + e.getMessage(), e);
        }
    }

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
