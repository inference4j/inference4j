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

/*
 * Portions of this file are ported from moonshine-ai/moonshine (core/moonshine-streaming-model.cpp
 * and core/transcriber.cpp), Copyright (c) 2025 Useful Sensors, Inc. (dba Moonshine AI),
 * licensed under the MIT License. See the NOTICE file for the full license text.
 */

package io.github.inference4j.audio;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.NativeTensors;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * A Moonshine v2 streaming speech-to-text model: six ONNX Runtime sessions and the arithmetic that
 * connects them.
 *
 * <p>Audio flows through the graphs in this order:
 * <ol>
 *   <li>{@code frontend.model} turns 80 ms chunks of 16 kHz audio into feature frames (50 per second),
 *       carrying its convolution buffers from one chunk to the next. Its weights come from
 *       {@code frontend.weights}, run once at load.</li>
 *   <li>{@code encoder} encodes features with sliding-window attention, so only new frames need work.</li>
 *   <li>{@code adapter} adds positions and projects encoded frames into the decoder's memory.</li>
 *   <li>{@code cross_kv} turns the memory into cross-attention keys and values.</li>
 *   <li>{@code decoder_kv} produces tokens from the cross-attention cache and its own self-attention
 *       cache.</li>
 * </ol>
 *
 * <p>Per-utterance state lives in a {@link Line}; the model itself is stateless after loading.
 */
final class MoonshineModel implements AutoCloseable {

    static final int SAMPLE_RATE = 16000;

    /** Samples per frontend call (80 ms). The published graphs need whole chunks. */
    static final int CHUNK_SAMPLES = 1280;

    /** Feature frames per second of audio. */
    static final int FRAMES_PER_SECOND = 50;

    static final List<String> FILES = List.of(
            "frontend.model.ort", "frontend.weights.ort", "encoder.ort", "adapter.ort",
            "cross_kv.ort", "decoder_kv.ort", "streaming_config.json");

    private static final Set<String> DECODER_OUTPUTS = Set.of("logits", "out_k_self", "out_v_self");

    private final Config config;
    private final InferenceSession frontend;
    private final InferenceSession encoder;
    private final InferenceSession adapter;
    private final InferenceSession crossKv;
    private final InferenceSession decoder;
    private final NativeTensors frontendWeights;

    private MoonshineModel(Config config, InferenceSession frontend, InferenceSession encoder,
                           InferenceSession adapter, InferenceSession crossKv, InferenceSession decoder,
                           NativeTensors frontendWeights) {
        this.config = config;
        this.frontend = frontend;
        this.encoder = encoder;
        this.adapter = adapter;
        this.crossKv = crossKv;
        this.decoder = decoder;
        this.frontendWeights = frontendWeights;
    }

    /**
     * Loads the model from a directory holding {@link #FILES}.
     *
     * @param dir         the model directory
     * @param openSession opens one ONNX Runtime session from a graph file
     */
    static MoonshineModel load(Path dir, Function<Path, InferenceSession> openSession) {
        Config config = Config.read(dir.resolve("streaming_config.json"));
        List<InferenceSession> opened = new ArrayList<>();
        try {
            InferenceSession weights = open(dir, "frontend.weights.ort", openSession, opened);
            NativeTensors frontendWeights = weights.runNative(Map.of(), null);
            return new MoonshineModel(config,
                    open(dir, "frontend.model.ort", openSession, opened),
                    open(dir, "encoder.ort", openSession, opened),
                    open(dir, "adapter.ort", openSession, opened),
                    open(dir, "cross_kv.ort", openSession, opened),
                    open(dir, "decoder_kv.ort", openSession, opened),
                    frontendWeights);
        } catch (RuntimeException e) {
            opened.forEach(InferenceSession::close);
            throw e;
        }
    }

    private static InferenceSession open(Path dir, String file, Function<Path, InferenceSession> openSession,
                                         List<InferenceSession> opened) {
        InferenceSession session = openSession.apply(dir.resolve(file));
        opened.add(session);
        return session;
    }

    Config config() {
        return config;
    }

    /** Starts a new utterance. */
    Line newLine() {
        return new Line();
    }

    @Override
    public void close() {
        frontendWeights.close();
        frontend.close();
        encoder.close();
        adapter.close();
        crossKv.close();
        decoder.close();
    }

    /**
     * The model shape from {@code streaming_config.json}.
     */
    record Config(int encoderDim, int decoderDim, int depth, int heads, int headDim, int vocabSize,
                  int bosId, int eosId, int lookahead, int frontendDim, int c1) {

        static Config read(Path file) {
            try {
                JsonNode json = new ObjectMapper().readTree(file.toFile());
                return new Config(
                        required(json, "encoder_dim"), required(json, "decoder_dim"), required(json, "depth"),
                        required(json, "nheads"), required(json, "head_dim"), required(json, "vocab_size"),
                        required(json, "bos_id"), required(json, "eos_id"), required(json, "total_lookahead"),
                        required(json, "d_model_frontend"), required(json, "c1"));
            } catch (IOException e) {
                throw new ModelSourceException("Failed to read " + file + ": " + e.getMessage(), e);
            }
        }

        private static int required(JsonNode json, String field) {
            JsonNode value = json.get(field);
            if (value == null || !value.isInt()) {
                throw new ModelSourceException("streaming_config.json is missing '" + field + "'");
            }
            return value.asInt();
        }
    }

    /**
     * The state of one utterance: frontend buffers, accumulated features, decoder memory and the
     * attention caches. Audio is added incrementally; {@link #encode(boolean)} and
     * {@link #decode(int, int[])} can run as often as needed. Not thread-safe.
     */
    final class Line implements AutoCloseable {

        // Frontend state, passed in and out of every frontend run
        private float[] sampleBuffer = new float[79];
        private long sampleLength;
        private float[] conv1Buffer = new float[config.frontendDim() * 4];
        private float[] conv2Buffer = new float[config.c1() * 4];
        private long frameCount;

        // Audio waiting for a whole chunk
        private final float[] pending = new float[CHUNK_SAMPLES];
        private int pendingLength;

        // Feature frames from the frontend, [frames, encoderDim]
        private float[] features = new float[0];
        private int featureFrames;

        // Encoder progress and decoder memory, [frames, decoderDim]
        private int encodedFrames;
        private float[] memory = new float[0];
        private int memoryFrames;

        // Attention caches, kept in ONNX Runtime memory
        private NativeTensors crossCache;
        private NativeTensors selfCache = NativeTensors.empty();

        private Line() {
        }

        /** Number of decoder memory frames encoded so far. */
        int memoryFrames() {
            return memoryFrames;
        }

        /**
         * Adds 16 kHz audio. Whole 80 ms chunks run through the frontend now; a remainder waits for the
         * next call.
         */
        void addAudio(float[] samples) {
            int offset = 0;
            if (pendingLength > 0) {
                int take = Math.min(CHUNK_SAMPLES - pendingLength, samples.length);
                System.arraycopy(samples, 0, pending, pendingLength, take);
                pendingLength += take;
                offset = take;
                if (pendingLength < CHUNK_SAMPLES) {
                    return;
                }
                runFrontend(pending, 0, CHUNK_SAMPLES);
                pendingLength = 0;
            }
            // Any whole number of chunks can go in one run; the frontend carries its state across them
            int whole = (samples.length - offset) / CHUNK_SAMPLES * CHUNK_SAMPLES;
            if (whole > 0) {
                runFrontend(samples, offset, whole);
                offset += whole;
            }
            pendingLength = samples.length - offset;
            System.arraycopy(samples, offset, pending, 0, pendingLength);
        }

        private void runFrontend(float[] samples, int offset, int length) {
            Map<String, Tensor> inputs = Map.of(
                    "audio_chunk", Tensor.fromFloats(Arrays.copyOfRange(samples, offset, offset + length),
                            new long[]{1, length}),
                    "sample_buffer", Tensor.fromFloats(sampleBuffer, new long[]{1, 79}),
                    "sample_len", Tensor.fromLongs(new long[]{sampleLength}, new long[]{1}),
                    "conv1_buffer", Tensor.fromFloats(conv1Buffer, new long[]{1, config.frontendDim(), 4}),
                    "conv2_buffer", Tensor.fromFloats(conv2Buffer, new long[]{1, config.c1(), 4}),
                    "frame_count", Tensor.fromLongs(new long[]{frameCount}, new long[]{1}));
            try (NativeTensors out = frontend.runNative(inputs, null, frontendWeights)) {
                Tensor newFeatures = out.copy("features");
                int frames = (int) newFeatures.shape()[1];
                if (frames > 0) {
                    features = append(features, featureFrames, newFeatures.toFloats(), frames, config.encoderDim());
                    featureFrames += frames;
                }
                sampleBuffer = out.copy("sample_buffer_out").toFloats();
                sampleLength = out.copy("sample_len_out").toLongs()[0];
                conv1Buffer = out.copy("conv1_buffer_out").toFloats();
                conv2Buffer = out.copy("conv2_buffer_out").toFloats();
                frameCount = out.copy("frame_count_out").toLongs()[0];
            }
        }

        /**
         * Encodes the frames added since the last call into decoder memory. Until the utterance is final,
         * the last {@code total_lookahead} frames are held back, because the encoder looks ahead at them.
         *
         * @param isFinal whether all audio of the utterance has been added
         */
        void encode(boolean isFinal) {
            int stable = isFinal ? featureFrames : Math.max(0, featureFrames - config.lookahead());
            int newFrames = stable - encodedFrames;
            if (newFrames <= 0) {
                return;
            }
            // Each encoder layer attends 16 frames back, so this much context reproduces a full pass
            int windowStart = Math.max(0, encodedFrames - 16 * config.depth());
            int windowFrames = featureFrames - windowStart;
            int dim = config.encoderDim();
            Tensor window = Tensor.fromFloats(
                    Arrays.copyOfRange(features, windowStart * dim, featureFrames * dim),
                    new long[]{1, windowFrames, dim});
            float[] encoded = encoder.run(Map.of("features", window)).get("encoded").toFloats();
            int start = encodedFrames - windowStart;
            Tensor fresh = Tensor.fromFloats(
                    Arrays.copyOfRange(encoded, start * dim, (start + newFrames) * dim),
                    new long[]{1, newFrames, dim});
            Tensor positions = Tensor.fromLongs(new long[]{encodedFrames}, new long[]{1});
            float[] newMemory = adapter.run(Map.of("encoded", fresh, "pos_offset", positions))
                    .get("memory").toFloats();
            memory = append(memory, memoryFrames, newMemory, newFrames, config.decoderDim());
            memoryFrames += newFrames;
            encodedFrames = stable;
            closeCrossCache();
        }

        /**
         * Decodes the utterance from the start with greedy search.
         *
         * <p>With a draft, typically the tokens of the previous decode of this utterance, all of it is
         * checked in one decoder pass and the longest prefix the model agrees with is kept; decoding
         * then continues from there. The result is the same as decoding without a draft.
         *
         * @param maxTokens the most tokens to produce
         * @param draft     tokens to verify, or an empty array
         * @return the tokens, without BOS and EOS
         */
        int[] decode(int maxTokens, int[] draft) {
            if (memoryFrames == 0) {
                return new int[0];
            }
            resetSelfCache();
            List<Integer> tokens = new ArrayList<>();
            int next;
            if (draft.length > 0) {
                float[] logits = runDecoder(prepend(config.bosId(), draft));
                int accepted = 0;
                while (accepted < draft.length && argmax(logits, accepted) == draft[accepted]) {
                    accepted++;
                }
                for (int i = 0; i < accepted; i++) {
                    tokens.add(draft[i]);
                }
                if (accepted == draft.length) {
                    next = argmax(logits, accepted);
                } else {
                    resetSelfCache();
                    float[] replay = runDecoder(prepend(config.bosId(), Arrays.copyOf(draft, accepted)));
                    next = argmax(replay, accepted);
                }
            } else {
                next = argmax(runDecoder(new int[]{config.bosId()}), 0);
            }
            while (next != config.eosId() && tokens.size() < maxTokens) {
                tokens.add(next);
                next = argmax(runDecoder(new int[]{next}), 0);
            }
            return tokens.stream().mapToInt(Integer::intValue).toArray();
        }

        /** Runs the decoder on some tokens, extending the self-attention cache; returns their logits. */
        private float[] runDecoder(int[] tokens) {
            if (crossCache == null) {
                Tensor mem = Tensor.fromFloats(Arrays.copyOf(memory, memoryFrames * config.decoderDim()),
                        new long[]{1, memoryFrames, config.decoderDim()});
                crossCache = crossKv.runNative(Map.of("memory", mem), null, NativeTensors.empty())
                        .rename(name -> "out_" + name);
            }
            long[] ids = Arrays.stream(tokens).asLongStream().toArray();
            Map<String, Tensor> inputs = Map.of("token", Tensor.fromLongs(ids, new long[]{1, ids.length}));
            NativeTensors out = selfCache == NativeTensors.empty()
                    ? decoder.runNative(withEmptySelfCache(inputs), DECODER_OUTPUTS, crossCache)
                    : decoder.runNative(inputs, DECODER_OUTPUTS, crossCache, selfCache);
            selfCache.close();
            float[] logits = out.copy("logits").toFloats();
            // The next step reads out_k_self/out_v_self as k_self/v_self
            selfCache = out.select(Set.of("out_k_self", "out_v_self")).rename(name -> name.substring(4));
            return logits;
        }

        private Map<String, Tensor> withEmptySelfCache(Map<String, Tensor> inputs) {
            long[] shape = {config.depth(), 1, config.heads(), 0, config.headDim()};
            return Map.of(
                    "token", inputs.get("token"),
                    "k_self", Tensor.fromFloats(new float[0], shape),
                    "v_self", Tensor.fromFloats(new float[0], shape));
        }

        private int argmax(float[] logits, int position) {
            int vocab = config.vocabSize();
            int base = position * vocab;
            int best = 0;
            for (int i = 1; i < vocab; i++) {
                if (logits[base + i] > logits[base + best]) {
                    best = i;
                }
            }
            return best;
        }

        private void resetSelfCache() {
            selfCache.close();
            selfCache = NativeTensors.empty();
        }

        private void closeCrossCache() {
            if (crossCache != null) {
                crossCache.close();
                crossCache = null;
            }
        }

        @Override
        public void close() {
            resetSelfCache();
            closeCrossCache();
        }
    }

    private static float[] append(float[] buffer, int frames, float[] values, int newFrames, int dim) {
        int needed = (frames + newFrames) * dim;
        float[] target = buffer.length >= needed ? buffer : Arrays.copyOf(buffer, Math.max(needed, buffer.length * 2));
        System.arraycopy(values, 0, target, frames * dim, newFrames * dim);
        return target;
    }

    private static int[] prepend(int first, int[] rest) {
        int[] all = new int[rest.length + 1];
        all[0] = first;
        System.arraycopy(rest, 0, all, 1, rest.length);
        return all;
    }
}
