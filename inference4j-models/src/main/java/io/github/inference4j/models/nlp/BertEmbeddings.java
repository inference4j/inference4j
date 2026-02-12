package io.github.inference4j.models.nlp;

import io.github.inference4j.core.InferenceSession;
import io.github.inference4j.core.LocalModelSource;
import io.github.inference4j.core.ModelSource;
import io.github.inference4j.core.Tensor;
import io.github.inference4j.core.exception.ModelSourceException;
import io.github.inference4j.models.PoolingStrategy;
import io.github.inference4j.preprocessing.EncodedInput;
import io.github.inference4j.preprocessing.Tokenizer;
import io.github.inference4j.preprocessing.tokenizer.WordPieceTokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class BertEmbeddings implements AutoCloseable {

    private final InferenceSession session;
    private final Tokenizer tokenizer;
    private final PoolingStrategy poolingStrategy;
    private final int maxLength;

    private BertEmbeddings(InferenceSession session, Tokenizer tokenizer,
                           PoolingStrategy poolingStrategy, int maxLength) {
        this.session = session;
        this.tokenizer = tokenizer;
        this.poolingStrategy = poolingStrategy;
        this.maxLength = maxLength;
    }

    public static BertEmbeddings fromPretrained(String modelPath) {
        Path dir = Path.of(modelPath);
        return fromModelDirectory(dir);
    }

    public static BertEmbeddings fromPretrained(String modelId, ModelSource source) {
        Path dir = source.resolve(modelId);
        return fromModelDirectory(dir);
    }

    public static Builder builder() {
        return new Builder();
    }

    public float[] encode(String text) {
        EncodedInput encoded = tokenizer.encode(text, maxLength);

        long[] shape = {1, encoded.inputIds().length};

        Map<String, Tensor> inputs = Map.of(
                "input_ids", Tensor.fromLongs(encoded.inputIds(), shape),
                "attention_mask", Tensor.fromLongs(encoded.attentionMask(), shape),
                "token_type_ids", Tensor.fromLongs(encoded.tokenTypeIds(), shape)
        );

        Map<String, Tensor> outputs = session.run(inputs);

        Tensor outputTensor = outputs.values().iterator().next();
        return applyPooling(outputTensor.toFloats(), outputTensor.shape(),
                encoded.attentionMask(), poolingStrategy);
    }

    public List<float[]> encodeBatch(List<String> texts) {
        List<float[]> results = new ArrayList<>(texts.size());
        for (String text : texts) {
            results.add(encode(text));
        }
        return results;
    }

    static float[] applyPooling(float[] flatOutput, long[] shape,
                                long[] attentionMask, PoolingStrategy strategy) {
        int seqLen = (int) shape[1];
        int hiddenSize = (int) shape[2];

        return switch (strategy) {
            case CLS -> {
                float[] result = new float[hiddenSize];
                System.arraycopy(flatOutput, 0, result, 0, hiddenSize);
                yield result;
            }
            case MEAN -> {
                float[] result = new float[hiddenSize];
                int count = 0;
                for (int t = 0; t < seqLen; t++) {
                    if (attentionMask[t] == 1) {
                        for (int h = 0; h < hiddenSize; h++) {
                            result[h] += flatOutput[t * hiddenSize + h];
                        }
                        count++;
                    }
                }
                if (count > 0) {
                    for (int h = 0; h < hiddenSize; h++) {
                        result[h] /= count;
                    }
                }
                yield result;
            }
            case MAX -> {
                float[] result = new float[hiddenSize];
                Arrays.fill(result, -Float.MAX_VALUE);
                boolean anyValid = false;
                for (int t = 0; t < seqLen; t++) {
                    if (attentionMask[t] == 1) {
                        anyValid = true;
                        for (int h = 0; h < hiddenSize; h++) {
                            result[h] = Math.max(result[h], flatOutput[t * hiddenSize + h]);
                        }
                    }
                }
                if (!anyValid) {
                    Arrays.fill(result, 0f);
                }
                yield result;
            }
        };
    }

    @Override
    public void close() {
        session.close();
    }

    private static BertEmbeddings fromModelDirectory(Path dir) {
        if (!Files.isDirectory(dir)) {
            throw new ModelSourceException("Model directory not found: " + dir);
        }

        Path modelPath = dir.resolve("model.onnx");
        Path vocabPath = dir.resolve("vocab.txt");

        if (!Files.exists(modelPath)) {
            throw new ModelSourceException("Model file not found: " + modelPath);
        }
        if (!Files.exists(vocabPath)) {
            throw new ModelSourceException("Vocabulary file not found: " + vocabPath);
        }

        InferenceSession session = InferenceSession.create(modelPath);
        Tokenizer tokenizer = WordPieceTokenizer.fromVocabFile(vocabPath);

        return new BertEmbeddings(session, tokenizer, PoolingStrategy.MEAN, 512);
    }

    public static class Builder {
        private InferenceSession session;
        private Tokenizer tokenizer;
        private PoolingStrategy poolingStrategy = PoolingStrategy.MEAN;
        private int maxLength = 512;

        public Builder session(InferenceSession session) {
            this.session = session;
            return this;
        }

        public Builder tokenizer(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder poolingStrategy(PoolingStrategy poolingStrategy) {
            this.poolingStrategy = poolingStrategy;
            return this;
        }

        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }

        public BertEmbeddings build() {
            if (session == null) {
                throw new IllegalStateException("InferenceSession is required");
            }
            if (tokenizer == null) {
                throw new IllegalStateException("Tokenizer is required");
            }
            return new BertEmbeddings(session, tokenizer, poolingStrategy, maxLength);
        }
    }
}
