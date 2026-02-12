package io.github.inference4j.vision.classification;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.ModelSource;
import io.github.inference4j.OutputOperator;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.image.ImageTransformPipeline;
import io.github.inference4j.image.Labels;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Image classification model for EfficientNet architectures.
 *
 * <h2>Tested model</h2>
 * <p>This wrapper is tested against
 * <a href="https://huggingface.co/onnx/EfficientNet-Lite4">EfficientNet-Lite4</a>
 * (ONNX format, TensorFlow origin). That model has <strong>softmax built into the
 * graph</strong>, so this wrapper uses {@link OutputOperator#identity()} by default —
 * the output values are already probabilities.
 *
 * <p><strong>Other EfficientNet exports may differ.</strong> Some ONNX exports (e.g.,
 * from PyTorch's {@code timm} library) output raw logits without a final activation.
 * Using this wrapper with such a model would pass raw logits as confidence scores.
 * If your model outputs logits, override with
 * {@code .outputOperator(OutputOperator.softmax())} via the builder.
 *
 * <h2>Preprocessing</h2>
 * <ul>
 *   <li>Normalization: mean {@code [127/255, 127/255, 127/255]}, std {@code [128/255, 128/255, 128/255]}</li>
 *   <li>Input layout: auto-detected from model (typically NHWC for TF-origin models)</li>
 *   <li>Input size: auto-detected from model (B0=224, B1=240, B2=260, B3=300, B4=380, Lite4=280)</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (EfficientNet model = EfficientNet.fromPretrained("models/efficientnet-lite4")) {
 *     List<Classification> results = model.classify(Path.of("cat.jpg"));
 * }
 * }</pre>
 *
 * <h2>Larger variants via builder</h2>
 * <pre>{@code
 * try (EfficientNet model = EfficientNet.builder()
 *         .session(InferenceSession.create(modelPath))
 *         .pipeline(ImageTransformPipeline.imagenet(260))
 *         .labels(Labels.fromFile(Path.of("my-labels.txt")))
 *         .outputOperator(OutputOperator.softmax()) // if model outputs raw logits
 *         .defaultTopK(10)
 *         .build()) {
 *     List<Classification> results = model.classify(image, 3);
 * }
 * }</pre>
 *
 * @see OutputOperator
 */
public class EfficientNet extends AbstractImageClassificationModel {

    private EfficientNet(InferenceSession session, ImageTransformPipeline pipeline,
                         Labels labels, String inputName, int defaultTopK,
                         OutputOperator outputOperator) {
        super(session, pipeline, labels, inputName, defaultTopK, outputOperator);
    }

    public static EfficientNet fromPretrained(String modelPath) {
        Path dir = Path.of(modelPath);
        return fromModelDirectory(dir);
    }

    public static EfficientNet fromPretrained(String modelId, ModelSource source) {
        Path dir = source.resolve(modelId);
        return fromModelDirectory(dir);
    }

    public static Builder builder() {
        return new Builder();
    }

    private static final float[] EFFICIENTNET_MEAN = {127f / 255f, 127f / 255f, 127f / 255f};
    private static final float[] EFFICIENTNET_STD = {128f / 255f, 128f / 255f, 128f / 255f};

    private static EfficientNet fromModelDirectory(Path dir) {
        if (!Files.isDirectory(dir)) {
            throw new ModelSourceException("Model directory not found: " + dir);
        }

        Path modelPath = dir.resolve("model.onnx");
        if (!Files.exists(modelPath)) {
            throw new ModelSourceException("Model file not found: " + modelPath);
        }

        InferenceSession session = InferenceSession.create(modelPath);
        try {
            String inputName = session.inputNames().iterator().next();

            Path labelsPath = dir.resolve("labels.txt");
            Labels labels = Files.exists(labelsPath) ? Labels.fromFile(labelsPath) : Labels.imagenet();

            ImageTransformPipeline pipeline = detectPipeline(session, inputName,
                    EFFICIENTNET_MEAN, EFFICIENTNET_STD);

            return new EfficientNet(session, pipeline, labels, inputName, 5, OutputOperator.identity());
        } catch (Exception e) {
            session.close();
            throw e;
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {

        @Override
        protected Builder self() {
            return this;
        }

        public EfficientNet build() {
            validate();
            return new EfficientNet(session, pipeline, labels, inputName, defaultTopK, outputOperator);
        }
    }
}
