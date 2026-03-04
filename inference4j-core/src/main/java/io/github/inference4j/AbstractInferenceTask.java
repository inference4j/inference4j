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
 * <h2>Example — simple task (no metadata)</h2>
 * <pre>{@code
 * public class MyClassifier
 *         extends AbstractInferenceTask<BufferedImage, List<Classification>>
 *         implements ImageClassifier {
 *
 *     public MyClassifier(InferenceSession session, ...) {
 *         super(session,
 *               image -> PreprocessResult.of(Map.of("input", preprocess(image))),
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
 * <h2>Example — passing metadata from preprocessor to postprocessor</h2>
 * <pre>{@code
 * public class MyNerModel
 *         extends AbstractInferenceTask<String, List<Entity>> {
 *
 *     public MyNerModel(InferenceSession session, Tokenizer tokenizer) {
 *         super(session,
 *               text -> {
 *                   EncodedInput encoded = tokenizer.encode(text, 512);
 *                   Map<String, Tensor> tensors = Map.of(
 *                           "input_ids", Tensor.fromLongs(encoded.inputIds(), shape));
 *                   // Pass word IDs as metadata — not sent to ONNX session
 *                   return PreprocessResult.of(tensors, Map.of("wordIds", encoded.wordIds()));
 *               },
 *               ctx -> {
 *                   int[] wordIds = (int[]) ctx.metadata().get("wordIds");
 *                   // Use wordIds for subword-to-word alignment in postprocessing
 *                   return buildEntities(ctx.outputs(), wordIds, ctx.input());
 *               });
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
    protected final Preprocessor<I, PreprocessResult> preprocessor;
    protected final Postprocessor<InferenceContext<I>, O> postprocessor;

    protected AbstractInferenceTask(InferenceSession session,
                                    Preprocessor<I, PreprocessResult> preprocessor,
                                    Postprocessor<InferenceContext<I>, O> postprocessor) {
        this.session = session;
        this.preprocessor = preprocessor;
        this.postprocessor = postprocessor;
    }

    @Override
    public final O run(I input) {
        PreprocessResult result = preprocessor.process(input);
        Map<String, Tensor> outputs = session.run(result.tensors());
        return postprocessor.process(
                new InferenceContext<>(input, result.tensors(), outputs, result.metadata()));
    }

    @Override
    public void close() {
        session.close();
    }
}
