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

import io.github.inference4j.processing.Postprocessor;
import io.github.inference4j.processing.Preprocessor;

import java.util.Map;

/**
 * Skeleton implementation of {@link InferenceTask} that enforces the
 * <strong>preprocess → infer → postprocess</strong> pipeline.
 *
 * <p>Every task is composed of a {@link Preprocessor} that converts domain input
 * into tensor inputs, an {@link InferenceSession} that runs the ONNX model, and
 * a {@link Postprocessor} that converts raw tensor outputs back into domain results.
 * The {@link #run(Object)} method is {@code final} — subclasses cannot bypass the
 * pipeline.
 *
 * <p>Subclasses that need parameterized overloads (e.g., {@code classify(image, topK)},
 * {@code detect(image, conf, iou)}) can access the {@code protected} fields directly
 * and compose the same building blocks with custom parameters.
 *
 * <h2>Example — adding a new task</h2>
 * <pre>{@code
 * public class MyClassifier
 *         extends AbstractInferenceTask<BufferedImage, List<Classification>>
 *         implements ImageClassifier {
 *
 *     public MyClassifier(InferenceSession session, ...) {
 *         super(session,
 *               image -> Map.of("input", preprocess(image)),
 *               ctx -> postprocess(ctx.outputs()));
 *     }
 *
 *     @Override
 *     public List<Classification> classify(BufferedImage image) {
 *         return run(image);
 *     }
 * }
 * }</pre>
 *
 * @param <I> the input type (e.g., {@code BufferedImage}, {@code String}, {@code Path})
 * @param <O> the output type (e.g., {@code List<Classification>}, {@code float[]})
 * @see InferenceContext
 * @see Preprocessor
 * @see Postprocessor
 */
public abstract class AbstractInferenceTask<I, O> implements InferenceTask<I, O> {

    protected final InferenceSession session;
    protected final Preprocessor<I, Map<String, Tensor>> preprocessor;
    protected final Postprocessor<InferenceContext<I>, O> postprocessor;

    protected AbstractInferenceTask(InferenceSession session,
                                    Preprocessor<I, Map<String, Tensor>> preprocessor,
                                    Postprocessor<InferenceContext<I>, O> postprocessor) {
        this.session = session;
        this.preprocessor = preprocessor;
        this.postprocessor = postprocessor;
    }

    @Override
    public final O run(I input) {
        Map<String, Tensor> inputs = preprocessor.process(input);
        Map<String, Tensor> outputs = session.run(inputs);
        return postprocessor.process(new InferenceContext<>(input, inputs, outputs));
    }

    @Override
    public void close() {
        session.close();
    }
}
