package io.github.inference4j;

import io.github.inference4j.processing.MathOps;
import io.github.inference4j.sampling.LogitsProcessor;
import io.github.inference4j.sampling.LogitsProcessors;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public class OnnxGenerativeSession implements GenerativeSession {

    private final InferenceSession session;
    private Map<String, Tensor> cache;
    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private int sequenceLength;

    public OnnxGenerativeSession(InferenceSession session) {
        this.session = session;
        this.cache = new LinkedHashMap<>();
        long[] cacheShape = this.session.inputShape("past_key_values.0.key");
        this.numHeads = (int) cacheShape[1];
        this.headDim = (int) cacheShape[3];
        this.numLayers = (int)session.inputNames().stream()
            .filter(n -> n.startsWith("past_key_values") && n.endsWith(".key"))
            .count();
    }

    @Override
    public ForwardResult prefill(long[] tokenIds) {
        this.sequenceLength = tokenIds.length;
        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(tokenIds, new long[]{1, tokenIds.length}));
        inputs.put("attention_mask", Tensor.fromLongs(ones(tokenIds.length), new long[]{1, tokenIds.length}));
        long[] positionIds = new long[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            positionIds[i] = i;
        }
        inputs.put("position_ids", Tensor.fromLongs(positionIds, new long[]{1, tokenIds.length}));
        preFillCache(inputs);
        Map<String, Tensor> outputs = session.run(inputs);
        var logitsOutput = outputs.get("logits").slice(0, 0).slice(0, -1).toFloats();
        for (int i = 0; i < this.numLayers; i++) {
            cache.put("past_key_values." + i + ".key",   outputs.get("present." + i + ".key"));
            cache.put("past_key_values." + i + ".value", outputs.get("present." + i + ".value"));
        }
        return new ForwardResult(logitsOutput);
    }

    @Override
    public ForwardResult decode(long tokenId) {
        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(new long[]{tokenId}, new long[]{1, 1}));
        inputs.put("attention_mask", Tensor.fromLongs(ones(sequenceLength + 1), new long[]{1, sequenceLength + 1}));
        inputs.put("position_ids", Tensor.fromLongs(new long[]{sequenceLength}, new long[]{1, 1}));
        inputs.putAll(this.cache);
        Map<String, Tensor> outputs = session.run(inputs);
        var logitsOutput = outputs.get("logits").slice(0, 0).slice(0, -1).toFloats();
        for (int i = 0; i < this.numLayers; i++) {
            cache.put("past_key_values." + i + ".key",   outputs.get("present." + i + ".key"));
            cache.put("past_key_values." + i + ".value", outputs.get("present." + i + ".value"));
        }
        this.sequenceLength++;
        return new ForwardResult(logitsOutput);
    }

    @Override
    public int cacheSequenceLength() {
        return 0;
    }

    @Override
    public void resetCache() {

    }

    @Override
    public void close() throws Exception {

    }

    private long[] ones(int length) {
        long[] ones = new long[length];
        Arrays.fill(ones, 1L);
        return ones;
    }

    private void preFillCache(Map<String, Tensor> inputs) {
        Tensor empty = Tensor.fromFloats(new float[0], new long[]{1, this.numLayers, 0, this.headDim});
        for (int i = 0; i < this.numLayers; i++) {
            inputs.put("past_key_values." + i + ".key", empty);
            inputs.put("past_key_values." + i + ".value", empty);
        }
    }

    public static void main(String[] args) {
        var session = InferenceSession.create(Path.of("/Users/vinicius/projects/python/model-debugger/models/gpt2/model.onnx"));
        var generativeSession = new OnnxGenerativeSession(session);
        ForwardResult result;
        LogitsProcessor pipeline = LogitsProcessors.temperature(0.7f).andThen(LogitsProcessors.topP(0.9f));
        var tokenIds = new long[]{15496, 995};
        result = generativeSession.prefill(tokenIds);
        float[] logits = result.logits().clone();
        float[] processed = pipeline.process(logits);
        float[] probs = MathOps.softmax(processed);
        printTop5(probs);
        printTop5(result.logits());
        int maxIndex = argMax(result.logits());
        for (int i = 0; i < 10; i++) {
            result = generativeSession.decode(maxIndex);
            System.out.println("=".repeat(40));
            printTop5(result.logits());
            maxIndex = argMax(result.logits());
        }

    }


    static void printTop5(float[] logits){
        int[] topIndices = {0, 0, 0, 0, 0};
        float[] topValues = {Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY,
            Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY};

        for (int i = 0; i < logits.length; i++) {
            for (int j = 0; j < 5; j++) {
                if (logits[i] > topValues[j]) {
                    // Shift everything down
                    for (int k = 4; k > j; k--) {
                        topIndices[k] = topIndices[k - 1];
                        topValues[k] = topValues[k - 1];
                    }
                    topIndices[j] = i;
                    topValues[j] = logits[i];
                    break;
                }
            }
        }

        for (int j = 0; j < 5; j++) {
            System.out.printf("  index=%d  logit=%.6f%n", topIndices[j], topValues[j]);
        }
    }

    static int argMax(float[] logits) {
        int maxIndex = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}
