package io.github.inference4j.audio;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.MathOps;
import io.github.inference4j.ModelSource;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Wav2Vec2-CTC speech-to-text model wrapper.
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
 * try (Wav2Vec2 model = Wav2Vec2.fromPretrained("models/wav2vec2-base-960h")) {
 *     Transcription result = model.transcribe(Path.of("audio.wav"));
 *     System.out.println(result.text());
 * }
 * }</pre>
 *
 * <h2>Custom configuration</h2>
 * <pre>{@code
 * try (Wav2Vec2 model = Wav2Vec2.builder()
 *         .session(InferenceSession.create(modelPath))
 *         .vocabulary(Vocabulary.fromFile(vocabPath))
 *         .build()) {
 *     Transcription result = model.transcribe(audioSamples, 16000);
 * }
 * }</pre>
 *
 * @see SpeechToTextModel
 * @see Transcription
 */
public class Wav2Vec2 implements SpeechToTextModel {

    private static final int DEFAULT_SAMPLE_RATE = 16000;
    private static final int DEFAULT_BLANK_INDEX = 0;
    private static final String DEFAULT_WORD_DELIMITER = "|";

    private final InferenceSession session;
    private final io.github.inference4j.audio.Vocabulary vocabulary;
    private final String inputName;
    private final int targetSampleRate;
    private final int blankIndex;
    private final String wordDelimiter;

    private Wav2Vec2(InferenceSession session, Vocabulary vocabulary, String inputName,
                     int targetSampleRate, int blankIndex, String wordDelimiter) {
        this.session = session;
        this.vocabulary = vocabulary;
        this.inputName = inputName;
        this.targetSampleRate = targetSampleRate;
        this.blankIndex = blankIndex;
        this.wordDelimiter = wordDelimiter;
    }

    public static Wav2Vec2 fromPretrained(String modelPath) {
        Path dir = Path.of(modelPath);
        return fromModelDirectory(dir);
    }

    public static Wav2Vec2 fromPretrained(String modelId, ModelSource source) {
        Path dir = source.resolve(modelId);
        return fromModelDirectory(dir);
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Transcription transcribe(Path audioPath) {
        AudioData audio = AudioLoader.load(audioPath);
        return transcribe(audio.samples(), audio.sampleRate());
    }

    @Override
    public Transcription transcribe(float[] audioData, int sampleRate) {
        float[] resampled = AudioProcessor.resample(audioData, sampleRate, targetSampleRate);
        float[] normalized = AudioProcessor.normalize(resampled);

        Tensor inputTensor = Tensor.fromFloats(normalized, new long[]{1, normalized.length});

        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put(inputName, inputTensor);

        Map<String, Tensor> outputs = session.run(inputs);
        Tensor outputTensor = outputs.values().iterator().next();

        long[] shape = outputTensor.shape();
        int timeSteps = (int) shape[1];
        int vocabSize = (int) shape[2];

        return postProcess(outputTensor.toFloats(), timeSteps, vocabSize,
                vocabulary, blankIndex, wordDelimiter);
    }

    @Override
    public void close() {
        session.close();
    }

    /**
     * Post-processes CTC logits into a transcription.
     *
     * <p>Package-visible for unit testing without an ONNX session.
     *
     * @param logits        flat logit array of shape {@code [1, timeSteps, vocabSize]}
     * @param timeSteps     number of timesteps
     * @param vocabSize     vocabulary size
     * @param vocabulary    token vocabulary
     * @param blankIndex    index of the CTC blank token
     * @param wordDelimiter token that represents a word boundary (replaced with space)
     * @return the transcription result
     */
    static Transcription postProcess(float[] logits, int timeSteps, int vocabSize,
                                     Vocabulary vocabulary, int blankIndex, String wordDelimiter) {
        // Skip batch dimension: logits are [1, timeSteps, vocabSize], flat array starts at offset 0
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

    private static Wav2Vec2 fromModelDirectory(Path dir) {
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

        InferenceSession session = InferenceSession.create(modelPath);
        try {
            String inputName = session.inputNames().iterator().next();
            Vocabulary vocabulary = Vocabulary.fromFile(vocabPath);
            return new Wav2Vec2(session, vocabulary, inputName,
                    DEFAULT_SAMPLE_RATE, DEFAULT_BLANK_INDEX, DEFAULT_WORD_DELIMITER);
        } catch (Exception e) {
            session.close();
            throw e;
        }
    }

    public static class Builder {
        private InferenceSession session;
        private Vocabulary vocabulary;
        private String inputName;
        private int sampleRate = DEFAULT_SAMPLE_RATE;
        private int blankIndex = DEFAULT_BLANK_INDEX;
        private String wordDelimiter = DEFAULT_WORD_DELIMITER;

        public Builder session(InferenceSession session) {
            this.session = session;
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

        public Wav2Vec2 build() {
            if (session == null) {
                throw new IllegalStateException("InferenceSession is required");
            }
            if (vocabulary == null) {
                throw new IllegalStateException("Vocabulary is required");
            }
            if (inputName == null) {
                inputName = session.inputNames().iterator().next();
            }
            return new Wav2Vec2(session, vocabulary, inputName,
                    sampleRate, blankIndex, wordDelimiter);
        }
    }
}
