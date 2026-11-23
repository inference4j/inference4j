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

import io.github.inference4j.exception.InferenceException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * The contract of each test fixture in {@code src/test/resources/fixtures}, checked on the Java ONNX Runtime the
 * library ships with: input names, shapes and types, and the result of one canonical input.
 */
class FixtureModelsTest {

    @TempDir
    Path dir;

    @Test
    void tokensEchoReturnsEachUnmaskedTokenId() {
        try (InferenceSession session = open("tokens-echo")) {
            assertThat(session.inputNames()).containsExactly("input_ids", "attention_mask");
            assertTokenInputs(session, "input_ids", "attention_mask");

            Map<String, Tensor> outputs = session.run(tokens(new long[]{5, 6, 7}, new long[]{1, 1, 0}, null));

            assertThat(outputs).containsOnlyKeys("logits");
            assertThat(outputs.get("logits").shape()).containsExactly(1, 3);
            assertThat(outputs.get("logits").toFloats()).containsExactly(5f, 6f, 0f);
        }
    }

    @Test
    void tokensEchoTypesAddsAThousandPerTokenType() {
        try (InferenceSession session = open("tokens-echo-types")) {
            assertThat(session.inputNames()).containsExactly("input_ids", "attention_mask", "token_type_ids");
            assertTokenInputs(session, "input_ids", "attention_mask", "token_type_ids");

            Map<String, Tensor> outputs = session.run(
                    tokens(new long[]{5, 6, 7}, new long[]{1, 1, 1}, new long[]{0, 1, 1}));

            assertThat(outputs.get("logits").toFloats()).containsExactly(5f, 1006f, 1007f);
        }
    }

    @Test
    void tokensOneHotPutsTenAtEachTokenId() {
        try (InferenceSession session = open("tokens-onehot")) {
            assertThat(session.inputNames()).containsExactly("input_ids", "attention_mask");
            assertTokenInputs(session, "input_ids", "attention_mask");

            Map<String, Tensor> outputs = session.run(tokens(new long[]{0, 15, 3}, new long[]{1, 1, 0}, null));

            assertThat(outputs).containsOnlyKeys("last_hidden_state");
            Tensor hidden = outputs.get("last_hidden_state");
            assertThat(hidden.shape()).containsExactly(1, 3, 16);
            float[][] rows = hidden.toFloats3D()[0];
            assertThat(rows[0][0]).isEqualTo(10f);
            assertThat(rows[1][15]).isEqualTo(10f);
            assertThat(rows[2]).containsOnly(0f);
            assertThat(sum(rows[0]) + sum(rows[1])).isEqualTo(20f);
        }
    }

    @Test
    void tokensOneHotTypesAddsTheTokenTypeToEveryDimension() {
        try (InferenceSession session = open("tokens-onehot-types")) {
            assertThat(session.inputNames()).containsExactly("input_ids", "attention_mask", "token_type_ids");
            assertTokenInputs(session, "input_ids", "attention_mask", "token_type_ids");

            Tensor hidden = session.run(tokens(new long[]{0, 1}, new long[]{1, 1}, new long[]{0, 1}))
                    .get("last_hidden_state");

            float[][] rows = hidden.toFloats3D()[0];
            assertThat(rows[0][0]).isEqualTo(10f);
            assertThat(rows[0][1]).isZero();
            assertThat(rows[1][1]).isEqualTo(11f);
            assertThat(rows[1][0]).isEqualTo(1f);
        }
    }

    @Test
    void tokensPassageSumAddsTheSecondSegmentIds() {
        try (InferenceSession session = open("tokens-passage-sum")) {
            assertThat(session.inputNames()).containsExactly("input_ids", "attention_mask", "token_type_ids");
            assertTokenInputs(session, "input_ids", "attention_mask", "token_type_ids");

            Tensor logits = session.run(tokens(new long[]{9, 7, 9, 2, 1}, new long[]{1, 1, 1, 1, 1},
                    new long[]{0, 0, 0, 1, 1})).get("logits");

            assertThat(logits.shape()).containsExactly(1, 1);
            assertThat(logits.toFloats()).containsExactly(3f);
        }
    }

    @Test
    void depthChannelSumAddsTheChannels() {
        try (InferenceSession session = open("depth-channel-sum")) {
            assertThat(session.inputNames()).containsExactly("pixel_values");
            assertThat(session.inputShape("pixel_values")).containsExactly(-1, 3, -1, -1);

            float[] pixels = {1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400};
            Tensor depth = session.run(Map.of("pixel_values", Tensor.fromFloats(pixels, new long[]{1, 3, 2, 2})))
                    .get("predicted_depth");

            assertThat(depth.shape()).containsExactly(1, 2, 2);
            assertThat(depth.toFloats()).containsExactly(111f, 222f, 333f, 444f);
        }
    }

    @Test
    void sam2EncoderReturnsThreeChannelMeans() {
        try (InferenceSession session = open("sam2-encoder")) {
            assertThat(session.inputNames()).containsExactly("pixel_values");
            assertThat(session.inputShape("pixel_values")).containsExactly(1, 3, 1024, 1024);

            Map<String, Tensor> outputs = session.run(Map.of("pixel_values", sam2Pixels()));

            assertThat(outputs).containsOnlyKeys("image_embeddings.0", "image_embeddings.1", "image_embeddings.2");
            for (Tensor embedding : outputs.values()) {
                assertThat(embedding.shape()).containsExactly(1, 3, 1, 1);
                assertThat(embedding.toFloats()).containsExactly(1f, 2f, 3f);
            }
        }
    }

    @Test
    void sam2DecoderScoresByBoxAndRequiresEveryInput() {
        try (InferenceSession session = open("sam2-decoder")) {
            assertThat(session.inputNames()).containsExactlyInAnyOrder("image_embeddings.0", "image_embeddings.1",
                    "image_embeddings.2", "input_points", "input_labels", "input_boxes");
            assertThat(session.inputShape("input_points")).containsExactly(1, 1, -1, 2);
            assertThat(session.inputType("input_labels")).isEqualTo(TensorType.LONG);
            assertThat(session.inputShape("input_boxes")).containsExactly(1, -1, 4);

            Map<String, Tensor> inputs = sam2DecoderInputs();
            Map<String, Tensor> outputs = session.run(inputs);

            assertThat(outputs.get("iou_scores").shape()).containsExactly(1, 1, 3);
            assertThat(outputs.get("iou_scores").toFloats())
                    .containsExactly(new float[]{0.3f, 1.0f, 0.6f}, within(1e-6f));
            assertThat(outputs.get("pred_masks").shape()).containsExactly(1, 1, 3, 2, 2);
            assertThat(outputs.get("pred_masks").toFloats())
                    .containsExactly(-1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1);

            for (String name : inputs.keySet()) {
                Map<String, Tensor> missing = new HashMap<>(inputs);
                missing.remove(name);
                assertThatThrownBy(() -> session.run(missing)).as(name).isInstanceOf(InferenceException.class);
            }
        }
    }

    @Test
    void yoloV8OnePersonReturnsOneCandidate() {
        try (InferenceSession session = open("yolov8-one-person")) {
            assertThat(session.inputNames()).containsExactly("images");
            assertThat(session.inputShape("images")).containsExactly(1, 3, 32, 32);

            Tensor output = session.run(Map.of("images", zeros(1, 3, 32, 32))).get("output0");

            assertThat(output.shape()).containsExactly(1, 9, 1);
            assertThat(output.toFloats()).containsExactly(16f, 16f, 20f, 20f, 0.9f, 0.1f, 0f, 0f, 0f);
        }
    }

    @Test
    void yolo26OnePersonReturnsOneQuery() {
        try (InferenceSession session = open("yolo26-one-person")) {
            assertThat(session.inputNames()).containsExactly("images");
            assertThat(session.inputShape("images")).containsExactly(1, 3, 32, 32);

            Map<String, Tensor> outputs = session.run(Map.of("images", zeros(1, 3, 32, 32)));

            assertThat(outputs.get("logits").shape()).containsExactly(1, 1, 5);
            assertThat(outputs.get("logits").toFloats()).containsExactly(3f, -5f, -5f, -5f, -5f);
            assertThat(outputs.get("pred_boxes").shape()).containsExactly(1, 1, 4);
            assertThat(outputs.get("pred_boxes").toFloats()).containsExactly(0.5f, 0.5f, 0.25f, 0.25f);
        }
    }

    @Test
    void wav2Vec2AbcReturnsFiveSteps() {
        try (InferenceSession session = open("wav2vec2-abc")) {
            assertThat(session.inputNames()).containsExactly("input_values");
            assertThat(session.inputShape("input_values")).containsExactly(1, -1);

            Tensor logits = session.run(Map.of("input_values", zeros(1, 7))).get("logits");

            assertThat(logits.shape()).containsExactly(1, 5, 5);
            float[][] steps = logits.toFloats3D()[0];
            int[] argmax = new int[steps.length];
            for (int t = 0; t < steps.length; t++) {
                for (int k = 1; k < steps[t].length; k++) {
                    if (steps[t][k] > steps[t][argmax[t]]) {
                        argmax[t] = k;
                    }
                }
            }
            assertThat(argmax).containsExactly(1, 0, 2, 0, 3);
        }
    }

    @Test
    void sileroVadScoresTheLastSampleAndAdvancesTheState() {
        try (InferenceSession session = open("silero-vad")) {
            assertThat(session.inputNames()).containsExactly("input", "state", "sr");
            assertThat(session.inputShape("input")).containsExactly(1, -1);
            assertThat(session.inputShape("state")).containsExactly(2, 1, 128);
            assertThat(session.inputShape("sr")).isEmpty();
            assertThat(session.inputType("sr")).isEqualTo(TensorType.LONG);

            float[] input = new float[576];
            input[0] = 1f;
            input[575] = 0.3f;
            float[] state = new float[256];
            Arrays.fill(state, 2f);
            Map<String, Tensor> inputs = new LinkedHashMap<>();
            inputs.put("input", Tensor.fromFloats(input, new long[]{1, 576}));
            inputs.put("state", Tensor.fromFloats(state, new long[]{2, 1, 128}));
            inputs.put("sr", Tensor.fromLongs(new long[]{16000}, new long[]{}));

            Map<String, Tensor> outputs = session.run(inputs);

            assertThat(outputs.get("output").shape()).containsExactly(1, 1);
            assertThat(outputs.get("output").toFloats()[0]).isCloseTo(0.3f + 0.02f + 0.001f, within(1e-6f));
            assertThat(outputs.get("stateN").shape()).containsExactly(2, 1, 128);
            assertThat(outputs.get("stateN").toFloats()).containsOnly(3f);

            inputs.remove("sr");
            assertThatThrownBy(() -> session.run(inputs)).isInstanceOf(InferenceException.class);
        }
    }

    @Test
    void clipVisionAveragesEachChannel() {
        try (InferenceSession session = open("clip-vision")) {
            assertThat(session.inputNames()).containsExactly("pixel_values");
            assertThat(session.inputShape("pixel_values")).containsExactly(-1, 3, -1, -1);

            float[] pixels = {1, 3, 2, 6, 0, 0};
            Tensor embeds = session.run(Map.of("pixel_values", Tensor.fromFloats(pixels, new long[]{1, 3, 1, 2})))
                    .get("image_embeds");

            assertThat(embeds.shape()).containsExactly(1, 3);
            assertThat(embeds.toFloats()).containsExactly(2f, 4f, 0f);
        }
    }

    @Test
    void clipTextCountsUnmaskedIdsBelowThree() {
        try (InferenceSession session = open("clip-text")) {
            assertThat(session.inputNames()).containsExactly("input_ids", "attention_mask");
            assertTokenInputs(session, "input_ids", "attention_mask");

            Tensor embeds = session.run(tokens(new long[]{3, 0, 0, 0, 1, 1, 1, 1, 4, 2},
                    new long[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 0}, null)).get("text_embeds");

            assertThat(embeds.shape()).containsExactly(1, 3);
            assertThat(embeds.toFloats()).containsExactly(3f, 4f, 0f);
        }
    }

    @Test
    void classifierNhwcReturnsItsInput() {
        try (InferenceSession session = open("classifier-nhwc")) {
            assertThat(session.inputNames()).containsExactly("input");
            assertThat(session.inputShape("input")).containsExactly(1, 32, 32, 3);

            float[] pixels = new float[32 * 32 * 3];
            pixels[2] = 7f;
            Tensor scores = session.run(Map.of("input", Tensor.fromFloats(pixels, new long[]{1, 32, 32, 3})))
                    .get("scores");

            assertThat(scores.shape()).containsExactly(1, 32, 32, 3);
            assertThat(scores.toFloats()).isEqualTo(pixels);
        }
    }

    private InferenceSession open(String fixture) {
        return InferenceSession.create(TestModels.install(dir, fixture, fixture + ".onnx"));
    }

    private static void assertTokenInputs(InferenceSession session, String... names) {
        for (String name : names) {
            assertThat(session.inputShape(name)).as(name).containsExactly(-1, -1);
            assertThat(session.inputType(name)).as(name).isEqualTo(TensorType.LONG);
        }
    }

    private static Map<String, Tensor> tokens(long[] ids, long[] mask, long[] types) {
        long[] shape = {1, ids.length};
        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(ids, shape));
        inputs.put("attention_mask", Tensor.fromLongs(mask, shape));
        if (types != null) {
            inputs.put("token_type_ids", Tensor.fromLongs(types, shape));
        }
        return inputs;
    }

    private static Tensor sam2Pixels() {
        int plane = 1024 * 1024;
        float[] pixels = new float[3 * plane];
        Arrays.fill(pixels, 0, plane, 1f);
        Arrays.fill(pixels, plane, 2 * plane, 2f);
        Arrays.fill(pixels, 2 * plane, 3 * plane, 3f);
        return Tensor.fromFloats(pixels, new long[]{1, 3, 1024, 1024});
    }

    private static Map<String, Tensor> sam2DecoderInputs() {
        Map<String, Tensor> inputs = new HashMap<>();
        for (int i = 0; i < 3; i++) {
            inputs.put("image_embeddings." + i, Tensor.fromFloats(new float[]{1, 2, 3}, new long[]{1, 3, 1, 1}));
        }
        inputs.put("input_points", Tensor.fromFloats(new float[]{100, 200}, new long[]{1, 1, 1, 2}));
        inputs.put("input_labels", Tensor.fromLongs(new long[]{1}, new long[]{1, 1, 1}));
        inputs.put("input_boxes", Tensor.fromFloats(new float[]{0, 0, 512, 512}, new long[]{1, 1, 4}));
        return inputs;
    }

    private static Tensor zeros(long... shape) {
        long size = 1;
        for (long dim : shape) {
            size *= dim;
        }
        return Tensor.fromFloats(new float[(int) size], shape);
    }

    private static float sum(float[] values) {
        float total = 0;
        for (float value : values) {
            total += value;
        }
        return total;
    }
}
