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

import io.github.inference4j.processing.Preprocessor;

import java.util.ArrayList;
import java.util.List;

/**
 * Composes audio transforms into a preprocessing pipeline.
 *
 * <pre>{@code
 * AudioTransformPipeline pipeline = AudioTransformPipeline.builder()
 *     .resample(16000)
 *     .normalize()
 *     .build();
 * AudioData processed = pipeline.transform(audio);
 * }</pre>
 *
 * @see AudioTransform
 */
public class AudioTransformPipeline implements Preprocessor<AudioData, AudioData> {

	private final List<AudioTransform> transforms;

	private AudioTransformPipeline(List<AudioTransform> transforms) {
		this.transforms = List.copyOf(transforms);
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Applies all transforms sequentially and returns the processed audio.
	 *
	 * @param audio the input audio data
	 * @return the transformed audio data
	 */
	public AudioData transform(AudioData audio) {
		for (AudioTransform t : transforms) {
			audio = t.apply(audio);
		}
		return audio;
	}

	@Override
	public AudioData process(AudioData input) {
		return transform(input);
	}

	public static class Builder {

		private final List<AudioTransform> transforms = new ArrayList<>();

		/**
		 * Adds a resample step to the target sample rate.
		 *
		 * @param targetSampleRate the target sample rate in Hz (e.g., 16000)
		 */
		public Builder resample(int targetSampleRate) {
			transforms.add(new ResampleTransform(targetSampleRate));
			return this;
		}

		/**
		 * Adds a normalization step (zero mean, unit variance).
		 */
		public Builder normalize() {
			transforms.add(new NormalizeTransform());
			return this;
		}

		/**
		 * Adds a custom transform.
		 *
		 * @param transform the transform to add
		 */
		public Builder addTransform(AudioTransform transform) {
			transforms.add(transform);
			return this;
		}

		public AudioTransformPipeline build() {
			return new AudioTransformPipeline(transforms);
		}
	}
}
