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

/**
 * Resamples audio to a target sample rate.
 *
 * <p>Returns the input unchanged if already at the target rate.
 *
 * @see AudioProcessor#resample(float[], int, int)
 */
public class ResampleTransform implements AudioTransform {

	private final int targetSampleRate;

	public ResampleTransform(int targetSampleRate) {
		this.targetSampleRate = targetSampleRate;
	}

	@Override
	public AudioData apply(AudioData audio) {
		float[] resampled = AudioProcessor.resample(
				audio.samples(), audio.sampleRate(), targetSampleRate);
		return new AudioData(resampled, targetSampleRate);
	}
}
