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
 * Image classification model for ResNet architectures.
 *
 * <p>Quick start:
 * <pre>{@code
 * try (ResNet model = ResNet.fromPretrained("models/resnet50")) {
 *     List<Classification> results = model.classify(Path.of("cat.jpg"));
 * }
 * }</pre>
 *
 * <p>Custom configuration:
 * <pre>{@code
 * try (ResNet model = ResNet.builder()
 *         .session(InferenceSession.create(modelPath))
 *         .pipeline(ImageTransformPipeline.imagenet(224))
 *         .labels(Labels.fromFile(Path.of("my-labels.txt")))
 *         .defaultTopK(10)
 *         .build()) {
 *     List<Classification> results = model.classify(image, 3);
 * }
 * }</pre>
 */
public class ResNet extends AbstractImageClassificationModel {

    private ResNet(InferenceSession session, ImageTransformPipeline pipeline,
                   Labels labels, String inputName, int defaultTopK,
                   OutputOperator outputOperator) {
        super(session, pipeline, labels, inputName, defaultTopK, outputOperator);
    }

    public static ResNet fromPretrained(String modelPath) {
        Path dir = Path.of(modelPath);
        return fromModelDirectory(dir);
    }

    public static ResNet fromPretrained(String modelId, ModelSource source) {
        Path dir = source.resolve(modelId);
        return fromModelDirectory(dir);
    }

    public static Builder builder() {
        return new Builder();
    }

    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    private static ResNet fromModelDirectory(Path dir) {
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
                    IMAGENET_MEAN, IMAGENET_STD);

            return new ResNet(session, pipeline, labels, inputName, 5, OutputOperator.softmax());
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

        public ResNet build() {
            validate();
            return new ResNet(session, pipeline, labels, inputName, defaultTopK, outputOperator);
        }
    }
}
