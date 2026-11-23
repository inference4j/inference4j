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
import io.github.inference4j.PreprocessResult;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.preprocessing.audio.AudioData;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.Tensor;
import io.github.inference4j.model.AbstractModelBuilder;
import io.github.inference4j.preprocessing.audio.AudioTransformPipeline;
import io.github.inference4j.preprocessing.audio.Vocabulary;

import java.nio.file.Path;
import java.util.List;
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
 *         .modelSource(new LocalModelSource(modelsDir))   // resolves modelsDir/<modelId>
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

    private final io.github.inference4j.preprocessing.audio.Vocabulary vocabulary;
    private final String inputName;
    private final io.github.inference4j.preprocessing.audio.AudioTransformPipeline pipeline;
    private final int blankIndex;
    private final String wordDelimiter;

    private Wav2Vec2Recognizer(InferenceSession session, io.github.inference4j.preprocessing.audio.Vocabulary vocabulary, String inputName,
                               io.github.inference4j.preprocessing.audio.AudioTransformPipeline pipeline, int blankIndex, String wordDelimiter) {
        super(session,
                createPreprocessor(inputName, pipeline),
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
        this.pipeline = pipeline;
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
        AudioData audio = new AudioData(audioData, sampleRate);
        AudioData processed = pipeline.transform(audio);
        float[] samples = processed.samples();
        Tensor inputTensor = Tensor.fromFloats(samples, new long[]{1, samples.length});
        Map<String, Tensor> inputs = Map.of(inputName, inputTensor);
        Map<String, Tensor> outputs = session.run(inputs);
        Tensor outputTensor = outputs.values().iterator().next();
        long[] shape = outputTensor.shape();
        int timeSteps = (int) shape[1];
        int vocabSize = (int) shape[2];
        return postProcess(outputTensor.toFloats(), timeSteps, vocabSize,
                vocabulary, blankIndex, wordDelimiter);
    }

    private static io.github.inference4j.processing.Preprocessor<Path, PreprocessResult> createPreprocessor(
            String inputName, io.github.inference4j.preprocessing.audio.AudioTransformPipeline pipeline) {
        return audioPath -> {
            AudioData audio = io.github.inference4j.preprocessing.audio.AudioLoader.load(audioPath);
            AudioData processed = pipeline.transform(audio);
            float[] samples = processed.samples();
            Tensor inputTensor = Tensor.fromFloats(samples, new long[]{1, samples.length});
            return PreprocessResult.of(Map.of(inputName, inputTensor));
        };
    }

    static Transcription postProcess(float[] logits, int timeSteps, int vocabSize,
                                     io.github.inference4j.preprocessing.audio.Vocabulary vocabulary, int blankIndex, String wordDelimiter) {
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

    public static class Builder extends AbstractModelBuilder<Wav2Vec2Recognizer, Builder> {
        private io.github.inference4j.preprocessing.audio.Vocabulary vocabulary;
        private String inputName;
        private int sampleRate = DEFAULT_SAMPLE_RATE;
        private int blankIndex = DEFAULT_BLANK_INDEX;
        private String wordDelimiter = DEFAULT_WORD_DELIMITER;

        public Builder() {
            super(DEFAULT_MODEL_ID);
        }

        Builder session(InferenceSession session) {
            return useSession("model.onnx", session);
        }

        public Builder vocabulary(io.github.inference4j.preprocessing.audio.Vocabulary vocabulary) {
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

        @Override
        public Wav2Vec2Recognizer build() {
            AudioTransformPipeline pipeline = AudioTransformPipeline.builder()
                    .resample(sampleRate)
                    .normalize()
                    .build();
            List<String> required = vocabulary != null ? List.of("model.onnx") : List.of("model.onnx", "vocab.json");
            return load(required, files -> {
                InferenceSession session = files.session("model.onnx");
                Vocabulary resolvedVocabulary = vocabulary != null
                        ? vocabulary : Vocabulary.fromFile(files.file("vocab.json"));
                String input = inputName != null ? inputName : session.inputNames().iterator().next();
                return new Wav2Vec2Recognizer(session, resolvedVocabulary, input, pipeline, blankIndex,
                        wordDelimiter);
            });
        }
    }
}
