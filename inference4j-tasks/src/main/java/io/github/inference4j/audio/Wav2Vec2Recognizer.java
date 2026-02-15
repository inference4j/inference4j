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

package io.github.inference4j.audio;

import io.github.inference4j.AbstractInferenceTask;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Wav2Vec2-CTC speech recognizer.
 *
 * <p>Wav2Vec2 is a non-autoregressive model that takes raw 16kHz mono audio
 * waveform and produces all tokens in a single forward pass via CTC decoding.
 * No FFT, mel spectrogram, or autoregressive loop is needed.
 *
 * <h2>Target model</h2>
 * <p>Designed for <a href="https://huggingface.co/facebook/wav2vec2-base-960h">
 * facebook/wav2vec2-base-960h</a> (~360MB ONNX). Pre-exported at
 * <a href="https://huggingface.co/Xenova/wav2vec2-base-960h">Xenova/wav2vec2-base-960h</a>.
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (Wav2Vec2Recognizer recognizer = Wav2Vec2Recognizer.builder().build()) {
 *     Transcription result = recognizer.transcribe(Path.of("audio.wav"));
 *     System.out.println(result.text());
 * }
 * }</pre>
 *
 * <h2>Custom configuration</h2>
 * <pre>{@code
 * try (Wav2Vec2Recognizer recognizer = Wav2Vec2Recognizer.builder()
 *         .modelId("my-org/my-wav2vec2")
 *         .modelSource(ModelSource.fromPath(localDir))
 *         .sessionOptions(opts -> opts.addCUDA(0))
 *         .vocabulary(Vocabulary.fromFile(vocabPath))
 *         .build()) {
 *     Transcription result = recognizer.transcribe(audioSamples, 16000);
 * }
 * }</pre>
 *
 * @see SpeechRecognizer
 * @see Transcription
 */
public class Wav2Vec2Recognizer
        extends AbstractInferenceTask<Path, Transcription>
        implements SpeechRecognizer {

    private static final String DEFAULT_MODEL_ID = "inference4j/wav2vec2-base-960h";
    private static final int DEFAULT_SAMPLE_RATE = 16000;
    private static final int DEFAULT_BLANK_INDEX = 0;
    private static final String DEFAULT_WORD_DELIMITER = "|";

    private final Vocabulary vocabulary;
    private final String inputName;
    private final int targetSampleRate;
    private final int blankIndex;
    private final String wordDelimiter;

    private Wav2Vec2Recognizer(InferenceSession session, Vocabulary vocabulary, String inputName,
                               int targetSampleRate, int blankIndex, String wordDelimiter) {
        super(session,
                createPreprocessor(inputName, targetSampleRate),
                ctx -> {
                    Tensor outputTensor = ctx.outputs().values().iterator().next();
                    long[] shape = outputTensor.shape();
                    int timeSteps = (int) shape[1];
                    int vocabSize = (int) shape[2];
                    return postProcess(outputTensor.toFloats(), timeSteps, vocabSize,
                            vocabulary, blankIndex, wordDelimiter);
                });
        this.vocabulary = vocabulary;
        this.inputName = inputName;
        this.targetSampleRate = targetSampleRate;
        this.blankIndex = blankIndex;
        this.wordDelimiter = wordDelimiter;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Transcription transcribe(Path audioPath) {
        return run(audioPath);
    }

    @Override
    public Transcription transcribe(float[] audioData, int sampleRate) {
        float[] resampled = AudioProcessor.resample(audioData, sampleRate, targetSampleRate);
        float[] normalized = AudioProcessor.normalize(resampled);
        Tensor inputTensor = Tensor.fromFloats(normalized, new long[]{1, normalized.length});
        Map<String, Tensor> inputs = Map.of(inputName, inputTensor);
        Map<String, Tensor> outputs = session.run(inputs);
        Tensor outputTensor = outputs.values().iterator().next();
        long[] shape = outputTensor.shape();
        int timeSteps = (int) shape[1];
        int vocabSize = (int) shape[2];
        return postProcess(outputTensor.toFloats(), timeSteps, vocabSize,
                vocabulary, blankIndex, wordDelimiter);
    }

    private static io.github.inference4j.processing.Preprocessor<Path, Map<String, Tensor>> createPreprocessor(
            String inputName, int targetSampleRate) {
        return audioPath -> {
            AudioData audio = AudioLoader.load(audioPath);
            float[] resampled = AudioProcessor.resample(audio.samples(), audio.sampleRate(), targetSampleRate);
            float[] normalized = AudioProcessor.normalize(resampled);
            Tensor inputTensor = Tensor.fromFloats(normalized, new long[]{1, normalized.length});
            return Map.of(inputName, inputTensor);
        };
    }

    static Transcription postProcess(float[] logits, int timeSteps, int vocabSize,
                                     Vocabulary vocabulary, int blankIndex, String wordDelimiter) {
        int[] tokenIds = MathOps.ctcGreedyDecode(logits, timeSteps, vocabSize, blankIndex);

        StringBuilder sb = new StringBuilder();
        for (int id : tokenIds) {
            String token = vocabulary.get(id);
            if (token.equals(wordDelimiter)) {
                sb.append(' ');
            } else {
                sb.append(token);
            }
        }

        return new Transcription(sb.toString().strip());
    }

    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private Vocabulary vocabulary;
        private String inputName;
        private int sampleRate = DEFAULT_SAMPLE_RATE;
        private int blankIndex = DEFAULT_BLANK_INDEX;
        private String wordDelimiter = DEFAULT_WORD_DELIMITER;

        Builder session(InferenceSession session) {
            this.session = session;
            return this;
        }

        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder vocabulary(Vocabulary vocabulary) {
            this.vocabulary = vocabulary;
            return this;
        }

        public Builder inputName(String inputName) {
            this.inputName = inputName;
            return this;
        }

        public Builder sampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }

        public Builder blankIndex(int blankIndex) {
            this.blankIndex = blankIndex;
            return this;
        }

        public Builder wordDelimiter(String wordDelimiter) {
            this.wordDelimiter = wordDelimiter;
            return this;
        }

        public Wav2Vec2Recognizer build() {
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id);
                loadFromDirectory(dir);
            }
            if (vocabulary == null) {
                throw new IllegalStateException("Vocabulary is required");
            }
            if (inputName == null) {
                inputName = session.inputNames().iterator().next();
            }
            return new Wav2Vec2Recognizer(session, vocabulary, inputName,
                    sampleRate, blankIndex, wordDelimiter);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }

            Path vocabPath = dir.resolve("vocab.json");
            if (!Files.exists(vocabPath)) {
                throw new ModelSourceException("Vocabulary file not found: " + vocabPath);
            }

            this.session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            try {
                if (this.vocabulary == null) {
                    this.vocabulary = Vocabulary.fromFile(vocabPath);
                }
            } catch (Exception e) {
                this.session.close();
                this.session = null;
                throw e;
            }
        }
    }
}
