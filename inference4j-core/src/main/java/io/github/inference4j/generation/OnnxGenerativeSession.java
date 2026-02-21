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

package io.github.inference4j.generation;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;

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
        this.numLayers = (int) session.inputNames().stream()
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
        return this.sequenceLength;
    }

    @Override
    public void resetCache() {
        this.cache.clear();
        this.sequenceLength = 0;
    }

    @Override
    public void close() throws Exception {
        this.session.close();
    }

    private long[] ones(int length) {
        long[] ones = new long[length];
        Arrays.fill(ones, 1L);
        return ones;
    }

    private void preFillCache(Map<String, Tensor> inputs) {
        Tensor empty = Tensor.fromFloats(new float[0], new long[]{1, this.numHeads, 0, this.headDim});
        for (int i = 0; i < this.numLayers; i++) {
            inputs.put("past_key_values." + i + ".key", empty);
            inputs.put("past_key_values." + i + ".value", empty);
        }
    }
}
