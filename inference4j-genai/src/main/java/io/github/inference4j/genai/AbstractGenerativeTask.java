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
package io.github.inference4j.genai;

import ai.onnxruntime.genai.Config;
import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.TokenizerStream;
import ai.onnxruntime.genai.GenAIException;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.generation.GenerativeTask;
import io.github.inference4j.generation.TokenStreamer;
import io.github.inference4j.model.ModelRevision;
import io.github.inference4j.model.ResolvedModel;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Base class for autoregressive generation tasks backed by onnxruntime-genai.
 *
 * <p>This is the generative counterpart to
 * {@link io.github.inference4j.AbstractInferenceTask AbstractInferenceTask}.
 * While AbstractInferenceTask enforces a preprocess &rarr; infer &rarr; postprocess pipeline
 * for single-pass models, this class owns the autoregressive generate loop:
 * prepare generator &rarr; generate tokens one by one &rarr; parse output.
 *
 * <p>Subclasses provide two hooks:
 * <ul>
 *   <li>{@link #prepareGenerator(Object, Generator)} &mdash; feed the generator with
 *       encoded input (text tokens, audio tensors, image tensors)</li>
 *   <li>{@link #parseOutput(String, Object, int, long)} &mdash; convert the generated
 *       text into the domain output type</li>
 * </ul>
 *
 * <p>The generate loop itself is {@code final} and cannot be overridden.
 *
 * @param <I> the domain input type
 * @param <O> the domain output type
 */
public abstract class AbstractGenerativeTask<I, O> implements GenerativeTask<I, O> {

    protected final Model model;
    private final ResolvedModel resolvedModel;
    private final ReentrantLock lock = new ReentrantLock();
    private boolean closed;

    protected AbstractGenerativeTask(Model model) {
        this(model, null);
    }

    /**
     * @param model         the loaded genai model
     * @param resolvedModel the handle on the directory the model was loaded from, which keeps a cached snapshot
     *                      on disk until {@link #close()}; may be {@code null}
     */
    protected AbstractGenerativeTask(Model model, ResolvedModel resolvedModel) {
        this.model = model;
        this.resolvedModel = resolvedModel;
    }

    /**
     * The cached revision the model was loaded from: present when it was loaded through a cache that manages
     * revisions, such as {@link io.github.inference4j.model.HuggingFaceModelSource}.
     */
    public Optional<ModelRevision> modelRevision() {
        return resolvedModel != null ? resolvedModel.revision() : Optional.empty();
    }

    @Override
    public final O generate(I input) {
        return generate(input, null);
    }

    /**
     * Generates output for {@code input}. Thread-safe: calls from several threads run one at a time,
     * because each generation preallocates a KV cache for the model's full context, so running them in
     * parallel would multiply memory. For parallel generation, create one task per thread. Calling
     * {@code generate} again from the token listener throws {@link IllegalStateException}, and so does
     * any call after {@link #close()}.
     */
    @Override
    public final O generate(I input, Consumer<String> tokenListener) {
        return exclusively(() -> {
            Generated generated = run(generator -> prepareGenerator(input, generator), tokenListener);
            return parseOutput(generated.text(), input, generated.promptTokens(), generated.generatedTokens(),
                    generated.durationMillis());
        });
    }

    /**
     * Runs the generate loop on a generator that {@code input} prepares, for subclasses with more than one kind of
     * input. Same locking, limits and stop sequences as {@link #generate(Object, Consumer)}.
     *
     * @param input         feeds the generator before the loop starts
     * @param tokenListener receives each decoded fragment; may be {@code null}
     * @return the generated text with prompt and generated token counts
     */
    protected final GenerationResult generateFrom(GeneratorInput input, Consumer<String> tokenListener) {
        return exclusively(() -> {
            Generated generated = run(input, tokenListener);
            return new GenerationResult(generated.text(), generated.promptTokens(), generated.generatedTokens(),
                    Duration.ofMillis(generated.durationMillis()));
        });
    }

    /** Feeds a generator before the generate loop starts. */
    @FunctionalInterface
    protected interface GeneratorInput {

        void prepare(Generator generator) throws GenAIException;

    }

    private record Generated(String text, int promptTokens, int generatedTokens, long durationMillis) {
    }

    private <R> R exclusively(Supplier<R> work) {
        if (lock.isHeldByCurrentThread()) {
            throw new IllegalStateException("generate() was called from its own token listener");
        }
        try {
            lock.lockInterruptibly();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new InferenceException("Interrupted while waiting for another generation to finish", e);
        }
        try {
            if (closed) {
                throw new IllegalStateException("The model was closed");
            }
            return work.get();
        } finally {
            lock.unlock();
        }
    }

    private Generated run(GeneratorInput input, Consumer<String> tokenListener) {
        try {
            try (GeneratorParams params = createParams();
                 Generator generator = new Generator(model, params)) {
                input.prepare(generator);

                long start = System.currentTimeMillis();
                int maxNewTokens = maxNewTokens();
                int tokenCount = 0;
                // Holds back text that could be the start of a stop sequence, so a stop sequence is never emitted
                TokenStreamer streamer = new TokenStreamer(stopSequences(),
                        tokenListener == null ? text -> { } : text -> {
                            if (!text.isEmpty()) {
                                tokenListener.accept(text);
                            }
                        });

                long sequenceLength = generator.tokenCount();
                int promptTokens = Math.toIntExact(sequenceLength);
                try (TokenizerStream stream = createStream()) {
                    while (!generator.isDone() && !streamer.isStopped()
                            && (maxNewTokens <= 0 || tokenCount < maxNewTokens)) {
                        generator.generateNextToken();
                        long newLength = generator.tokenCount();
                        if (newLength == sequenceLength) {
                            // End-of-text is not appended to the sequence, so there is no new token to decode
                            continue;
                        }
                        sequenceLength = newLength;
                        int token = generator.getLastTokenInSequence(0);
                        tokenCount++;
                        streamer.accept(stream.decode(token));
                    }
                }
                streamer.flush();

                return new Generated(streamer.getText(), promptTokens, tokenCount,
                        System.currentTimeMillis() - start);
            }
        } catch (GenAIException e) {
            throw new InferenceException(
                    "Generation failed: " + e.getMessage(), e);
        }
    }

    /**
     * Strings that end generation when they appear in the output. The stop sequence and anything after it are left
     * out of the result and are not passed to the token listener. The default is none.
     *
     * @return the stop sequences; empty for none
     */
    protected Set<String> stopSequences() {
        return Set.of();
    }

    /**
     * The maximum number of tokens to generate per call, not counting the prompt. Generation also stops at the
     * model's {@code max_length}, whichever comes first. The default, 0, sets no limit of its own.
     *
     * @return the limit, or 0 for none
     */
    protected int maxNewTokens() {
        return 0;
    }

    /**
     * Create a {@link TokenizerStream} for decoding generated tokens into text.
     *
     * <p>Subclasses backed by a {@code Tokenizer} (e.g. text generators) create the
     * stream from the tokenizer. Subclasses backed by a {@code MultiModalProcessor}
     * (e.g. Whisper audio transcription) create the stream from the processor.
     *
     * @return a new TokenizerStream for the current generation
     * @throws GenAIException if stream creation fails
     */
    protected abstract TokenizerStream createStream() throws GenAIException;

    /**
     * Feed the generator with encoded input before the generate loop starts.
     *
     * <p>Subclasses typically encode the input into token IDs and append them
     * to the generator's input sequence.
     *
     * @param input     the domain input
     * @param generator the generator to prepare
     */
    protected abstract void prepareGenerator(I input, Generator generator);

    /**
     * Convert the generated text and metadata into the domain output type.
     *
     * @param generatedText  the full generated text, without any stop sequence
     * @param input          the original domain input
     * @param promptTokens   number of tokens in the input sequence before generation
     * @param tokenCount     number of tokens generated
     * @param durationMillis wall-clock generation time in milliseconds
     * @return the domain output
     */
    protected abstract O parseOutput(String generatedText, I input, int promptTokens,
                                     int tokenCount, long durationMillis);

    /**
     * Create {@link GeneratorParams} for this generation run.
     *
     * <p>The default implementation returns bare params with no search options.
     * Subclasses should override this to configure model-specific options
     * (max_length, temperature, top_k, top_p, etc.).
     *
     * @return configured generator params
     * @throws GenAIException if param creation fails
     */
    protected GeneratorParams createParams() throws GenAIException {
        return new GeneratorParams(model);
    }

    /**
     * Loads a genai model from {@code modelDirectory}. With {@code intraOpThreads > 0}, every ONNX Runtime session
     * the model's {@code genai_config.json} declares (decoder, encoder, vision, embedding) is limited to that many
     * threads, through a config overlay; the files on disk are not changed.
     *
     * @param modelDirectory the directory holding {@code genai_config.json}
     * @param intraOpThreads the number of threads per session, or 0 for ONNX Runtime's default (all cores)
     * @return the loaded model
     * @throws GenAIException if the model cannot be loaded
     * @throws IOException    if {@code genai_config.json} cannot be read
     */
    protected static Model loadModel(Path modelDirectory, int intraOpThreads) throws GenAIException, IOException {
        if (intraOpThreads <= 0) {
            return new Model(modelDirectory.toString());
        }
        String overlay = threadsOverlay(modelDirectory.resolve("genai_config.json"), intraOpThreads);
        try (Config config = new Config(modelDirectory.toString())) {
            config.overlay(overlay);
            return new Model(config);
        }
    }

    private static String threadsOverlay(Path genaiConfig, int intraOpThreads) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode sections = mapper.readTree(genaiConfig.toFile()).path("model");
        ObjectNode overlay = mapper.createObjectNode();
        ObjectNode model = overlay.putObject("model");
        Iterator<Map.Entry<String, JsonNode>> fields = sections.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> section = fields.next();
            // A section with a model file is an ONNX Runtime session
            if (section.getValue().isObject() && section.getValue().has("filename")) {
                model.putObject(section.getKey())
                        .putObject("session_options")
                        .put("intra_op_num_threads", intraOpThreads);
            }
        }
        return mapper.writeValueAsString(overlay);
    }

    /**
     * Release subclass-owned resources (e.g. Tokenizer, MultiModalProcessor).
     *
     * <p>Called by {@link #close()} before closing the model. Subclasses should
     * override this to close any resources they own. The default implementation
     * does nothing.
     */
    protected void closeResources() {
    }

    @Override
    public final void close() {
        if (lock.isHeldByCurrentThread()) {
            throw new IllegalStateException("The model cannot be closed from its own token listener");
        }
        lock.lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            try {
                closeResources();
                model.close();
            } finally {
                if (resolvedModel != null) {
                    resolvedModel.close();
                }
            }
        } finally {
            lock.unlock();
        }
    }
}
