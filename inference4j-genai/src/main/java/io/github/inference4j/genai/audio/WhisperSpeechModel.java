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
package io.github.inference4j.genai.audio;

import ai.onnxruntime.genai.Audios;
import ai.onnxruntime.genai.GenAIException;
import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.MultiModalProcessor;
import ai.onnxruntime.genai.NamedTensors;
import ai.onnxruntime.genai.TokenizerStream;
import io.github.inference4j.audio.Transcription;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.genai.AbstractGenerativeTask;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.preprocessing.audio.AudioData;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Whisper speech model backed by onnxruntime-genai.
 *
 * <p>Wraps OpenAI Whisper models (tiny, base, small) for automatic speech
 * recognition and translation. Audio preprocessing (mel spectrogram),
 * autoregressive decoding, KV cache, and beam search are all handled
 * natively by onnxruntime-genai's C++ layer.
 *
 * <h2>Transcription</h2>
 * <pre>{@code
 * try (var whisper = WhisperSpeechModel.builder()
 *         .modelId("inference4j/whisper-small-genai")
 *         .build()) {
 *     Transcription result = whisper.transcribe(Path.of("meeting.wav"));
 *     System.out.println(result.text());
 * }
 * }</pre>
 *
 * <h2>Translation to English</h2>
 * <pre>{@code
 * try (var whisper = WhisperSpeechModel.builder()
 *         .modelId("inference4j/whisper-small-genai")
 *         .language("fr")
 *         .task(WhisperTask.TRANSLATE)
 *         .build()) {
 *     Transcription result = whisper.transcribe(Path.of("french-audio.wav"));
 *     System.out.println(result.text());  // English text
 * }
 * }</pre>
 *
 * @see WhisperTask
 */
public class WhisperSpeechModel extends AbstractGenerativeTask<Path, Transcription> {

	private static final int CHUNK_DURATION_SECONDS = 30;

	private final MultiModalProcessor processor;
	private final String language;
	private final WhisperTask task;
	private final int maxLength;
	private final double temperature;
	private final int topK;
	private final double topP;

	WhisperSpeechModel(Model model, MultiModalProcessor processor,
					   String language, WhisperTask task,
					   int maxLength, double temperature, int topK, double topP) {
		super(model);
		this.processor = processor;
		this.language = language;
		this.task = task;
		this.maxLength = maxLength;
		this.temperature = temperature;
		this.topK = topK;
		this.topP = topP;
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Transcribes or translates speech from an audio file.
	 *
	 * <p>Audio longer than 30 seconds is automatically split into chunks,
	 * processed individually, and concatenated. The task mode (transcription
	 * or translation) is determined by the {@link WhisperTask} configured
	 * via the builder.
	 *
	 * @param audioPath path to a WAV audio file
	 * @return the transcription result
	 */
	public Transcription transcribe(Path audioPath) {
		AudioData audio = io.github.inference4j.preprocessing.audio.AudioLoader.load(audioPath);
		int chunkSamples = audio.sampleRate() * CHUNK_DURATION_SECONDS;

		if (audio.samples().length <= chunkSamples) {
			return generate(audioPath);
		}

		// Auto-chunking for audio > 30 seconds
		List<AudioData> chunks = io.github.inference4j.preprocessing.audio.AudioProcessor.chunk(audio, CHUNK_DURATION_SECONDS);
		StringBuilder fullText = new StringBuilder();
		for (AudioData chunk : chunks) {
			try {
				Path tempFile = Files.createTempFile("inference4j-whisper-chunk-", ".wav");
				try {
					io.github.inference4j.preprocessing.audio.AudioWriter.write(chunk, tempFile);
					Transcription partial = generate(tempFile);
					if (!partial.text().isEmpty()) {
						if (fullText.length() > 0) {
							fullText.append(' ');
						}
						fullText.append(partial.text());
					}
				} finally {
					deleteTempFile(tempFile);
				}
			} catch (IOException e) {
				throw new InferenceException(
						"Failed to write temp chunk file: " + e.getMessage(), e);
			}
		}
		return new Transcription(fullText.toString());
	}

	/**
	 * Transcribes or translates speech from raw audio samples.
	 *
	 * <p>Convenience method that writes the samples to a temporary WAV file
	 * and delegates to {@link #transcribe(Path)}.
	 *
	 * @param audioData  mono float32 PCM samples in {@code [-1.0, 1.0]}
	 * @param sampleRate sample rate in Hz (e.g., 16000)
	 * @return the transcription result
	 */
	public Transcription transcribe(float[] audioData, int sampleRate) {
		try {
			Path tempFile = Files.createTempFile("inference4j-whisper-", ".wav");
			try {
				io.github.inference4j.preprocessing.audio.AudioWriter.write(new AudioData(audioData, sampleRate), tempFile);
				return transcribe(tempFile);
			} finally {
				deleteTempFile(tempFile);
			}
		} catch (IOException e) {
			throw new InferenceException(
					"Failed to write temp audio file: " + e.getMessage(), e);
		}
	}

	@Override
	protected GeneratorParams createParams() throws GenAIException {
		GeneratorParams params = super.createParams();
		params.setSearchOption("max_length", maxLength);
		if (temperature > 0) {
			params.setSearchOption("temperature", temperature);
		}
		if (topK > 0) {
			params.setSearchOption("top_k", topK);
		}
		if (topP > 0) {
			params.setSearchOption("top_p", topP);
		}
		return params;
	}

	@Override
	protected void prepareGenerator(Path input, Generator generator) {
		try {
			try (Audios audios = new Audios(input.toString())) {
				String prompt = buildPrompt();
				try (NamedTensors inputs = processor.processAudios(prompt, audios)) {
					generator.setInputs(inputs);
				}
			}
		} catch (GenAIException e) {
			throw new InferenceException(
					"Failed to prepare Whisper generator: " + e.getMessage(), e);
		}
	}

	@Override
	protected TokenizerStream createStream() throws GenAIException {
		return processor.createStream();
	}

	@Override
	protected Transcription parseOutput(String generatedText, Path input,
										int tokenCount, long durationMillis) {
		return new Transcription(generatedText.strip());
	}

	@Override
	protected void closeResources() {
		processor.close();
	}

	/**
	 * Builds the Whisper decoder prompt string.
	 */
	String buildPrompt() {
		return "<|startoftranscript|><|" + language + "|>"
				+ task.token() + "<|notimestamps|>";
	}

	private static void deleteTempFile(Path file) {
		try {
			Files.deleteIfExists(file);
		} catch (IOException ignored) {
			// best-effort cleanup
		}
	}

	/**
	 * Builder for {@link WhisperSpeechModel}.
	 *
	 * <p>Minimal usage:
	 * <pre>{@code
	 * WhisperSpeechModel whisper = WhisperSpeechModel.builder()
	 *         .modelId("inference4j/whisper-small-genai")
	 *         .build();
	 * }</pre>
	 */
	public static class Builder {

		private ModelSource modelSource;
		private String modelId;
		private String language = "en";
		private WhisperTask task = WhisperTask.TRANSCRIBE;
		private int maxLength = 448;
		private double temperature = 0.0;
		private int topK = 0;
		private double topP = 0.0;

		// Package-private for testing
		Model model;
		MultiModalProcessor processor;

		public Builder modelId(String modelId) {
			this.modelId = modelId;
			return this;
		}

		public Builder modelSource(ModelSource modelSource) {
			this.modelSource = modelSource;
			return this;
		}

		public Builder language(String language) {
			this.language = language;
			return this;
		}

		public Builder task(WhisperTask task) {
			this.task = task;
			return this;
		}

		public Builder maxLength(int maxLength) {
			this.maxLength = maxLength;
			return this;
		}

		public Builder temperature(double temperature) {
			this.temperature = temperature;
			return this;
		}

		public Builder topK(int topK) {
			this.topK = topK;
			return this;
		}

		public Builder topP(double topP) {
			this.topP = topP;
			return this;
		}

		public WhisperSpeechModel build() {
			if (model == null) {
				if (modelId == null) {
					throw new IllegalStateException(
							"modelId is required — e.g., modelId(\"inference4j/whisper-small-genai\")");
				}
				ModelSource source = modelSource != null
						? modelSource : HuggingFaceModelSource.defaultInstance();
				Path modelDir = source.resolve(modelId);
				Model m = null;
				try {
					m = new Model(modelDir.toString());
					model = m;
					processor = new MultiModalProcessor(m);
				} catch (GenAIException e) {
					if (m != null) {
						m.close();
					}
					throw new ModelSourceException(
							"Failed to load Whisper model: " + e.getMessage(), e);
				}
			}
			return new WhisperSpeechModel(model, processor, language, task,
					maxLength, temperature, topK, topP);
		}
	}
}
