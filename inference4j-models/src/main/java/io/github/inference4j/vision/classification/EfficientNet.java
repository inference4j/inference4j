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
 * <p>Quick start with EfficientNet-B0 (224x224):
 * <pre>{@code
 * try (EfficientNet model = EfficientNet.fromPretrained("models/efficientnet-b0")) {
 *     List<Classification> results = model.classify(Path.of("cat.jpg"));
 * }
 * }</pre>
 *
 * <p>Larger variants (B1=240, B2=260, B3=300, etc.) via builder:
 * <pre>{@code
 * try (EfficientNet model = EfficientNet.builder()
 *         .session(InferenceSession.create(modelPath))
 *         .pipeline(ImageTransformPipeline.imagenet(260))
 *         .labels(Labels.fromFile(Path.of("my-labels.txt")))
 *         .defaultTopK(10)
 *         .build()) {
 *     List<Classification> results = model.classify(image, 3);
 * }
 * }</pre>
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
