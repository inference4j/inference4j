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
package io.github.inference4j.preprocessing.audio;

import io.github.inference4j.preprocessing.audio.AudioData;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AudioTransformPipelineTest {

    @Test
    void emptyPipeline_returnsInputUnchanged() {
        AudioData input = new AudioData(new float[]{0.1f, 0.2f, 0.3f}, 44100);

        AudioData result = io.github.inference4j.preprocessing.audio.AudioTransformPipeline.builder().build().transform(input);

        assertArrayEquals(input.samples(), result.samples());
        assertEquals(44100, result.sampleRate());
    }

    @Test
    void resample_changesSampleRate() {
        // 4 samples at 2 Hz = 2 seconds → resample to 4 Hz = 8 samples
        AudioData input = new AudioData(new float[]{0.0f, 0.5f, 1.0f, 0.5f}, 2);

        AudioData result = io.github.inference4j.preprocessing.audio.AudioTransformPipeline.builder()
                .resample(4)
                .build()
                .transform(input);

        assertEquals(4, result.sampleRate());
        assertEquals(8, result.samples().length);
    }

    @Test
    void resample_sameRate_returnsUnchanged() {
        float[] samples = {0.1f, 0.2f, 0.3f};
        AudioData input = new AudioData(samples, 16000);

        AudioData result = io.github.inference4j.preprocessing.audio.AudioTransformPipeline.builder()
                .resample(16000)
                .build()
                .transform(input);

        assertEquals(16000, result.sampleRate());
        assertSame(samples, result.samples());
    }

    @Test
    void normalize_producesZeroMeanUnitVariance() {
        float[] samples = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        AudioData input = new AudioData(samples, 16000);

        AudioData result = io.github.inference4j.preprocessing.audio.AudioTransformPipeline.builder()
                .normalize()
                .build()
                .transform(input);

        assertEquals(16000, result.sampleRate());

        // Check zero mean
        double mean = 0;
        for (float s : result.samples()) mean += s;
        mean /= result.samples().length;
        assertEquals(0.0, mean, 1e-5);

        // Check unit variance
        double variance = 0;
        for (float s : result.samples()) variance += s * s;
        variance /= result.samples().length;
        assertEquals(1.0, variance, 1e-5);
    }

    @Test
    void resampleThenNormalize_chainsCorrectly() {
        // 4 samples at 8000 Hz → resample to 16000 → normalize
        float[] samples = {0.5f, 1.0f, 0.5f, 0.0f};
        AudioData input = new AudioData(samples, 8000);

        AudioData result = io.github.inference4j.preprocessing.audio.AudioTransformPipeline.builder()
                .resample(16000)
                .normalize()
                .build()
                .transform(input);

        assertEquals(16000, result.sampleRate());
        assertTrue(result.samples().length > samples.length);

        // Verify normalized (zero mean)
        double mean = 0;
        for (float s : result.samples()) mean += s;
        mean /= result.samples().length;
        assertEquals(0.0, mean, 1e-5);
    }

    @Test
    void customTransform_isApplied() {
        AudioData input = new AudioData(new float[]{1.0f, 2.0f, 3.0f}, 16000);

        AudioData result = io.github.inference4j.preprocessing.audio.AudioTransformPipeline.builder()
                .addTransform(audio -> {
                    float[] scaled = new float[audio.samples().length];
                    for (int i = 0; i < scaled.length; i++) {
                        scaled[i] = audio.samples()[i] * 0.5f;
                    }
                    return new AudioData(scaled, audio.sampleRate());
                })
                .build()
                .transform(input);

        assertEquals(0.5f, result.samples()[0]);
        assertEquals(1.0f, result.samples()[1]);
        assertEquals(1.5f, result.samples()[2]);
    }

    @Test
    void processMethodDelegatesToTransform() {
        AudioData input = new AudioData(new float[]{1.0f, 2.0f}, 16000);

        io.github.inference4j.preprocessing.audio.AudioTransformPipeline pipeline = io.github.inference4j.preprocessing.audio.AudioTransformPipeline.builder()
                .normalize()
                .build();

        AudioData fromTransform = pipeline.transform(input);
        AudioData fromProcess = pipeline.process(
                new AudioData(new float[]{1.0f, 2.0f}, 16000));

        assertArrayEquals(fromTransform.samples(), fromProcess.samples());
    }
}
