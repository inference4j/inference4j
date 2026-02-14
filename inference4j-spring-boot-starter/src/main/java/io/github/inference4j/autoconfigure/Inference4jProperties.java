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
package io.github.inference4j.autoconfigure;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Configuration properties for inference4j tasks.
 *
 * <p>Each task is opt-in: set {@code inference4j.<domain>.<task>.enabled=true}
 * in your application properties to create the corresponding bean.
 */
@ConfigurationProperties(prefix = "inference4j")
public class Inference4jProperties {

	private NlpProperties nlp = new NlpProperties();

	private VisionProperties vision = new VisionProperties();

	private AudioProperties audio = new AudioProperties();

	public NlpProperties getNlp() {
		return nlp;
	}

	public void setNlp(NlpProperties nlp) {
		this.nlp = nlp;
	}

	public VisionProperties getVision() {
		return vision;
	}

	public void setVision(VisionProperties vision) {
		this.vision = vision;
	}

	public AudioProperties getAudio() {
		return audio;
	}

	public void setAudio(AudioProperties audio) {
		this.audio = audio;
	}

	public static class NlpProperties {

		private TaskProperties textClassifier = new TaskProperties(
				"inference4j/distilbert-base-uncased-finetuned-sst-2-english");

		private TaskProperties textEmbedder = new TaskProperties();

		private TaskProperties searchReranker = new TaskProperties("inference4j/ms-marco-MiniLM-L-6-v2");

		public TaskProperties getTextClassifier() {
			return textClassifier;
		}

		public void setTextClassifier(TaskProperties textClassifier) {
			this.textClassifier = textClassifier;
		}

		public TaskProperties getTextEmbedder() {
			return textEmbedder;
		}

		public void setTextEmbedder(TaskProperties textEmbedder) {
			this.textEmbedder = textEmbedder;
		}

		public TaskProperties getSearchReranker() {
			return searchReranker;
		}

		public void setSearchReranker(TaskProperties searchReranker) {
			this.searchReranker = searchReranker;
		}

	}

	public static class VisionProperties {

		private TaskProperties imageClassifier = new TaskProperties("inference4j/resnet50-v1-7");

		private TaskProperties objectDetector = new TaskProperties("inference4j/yolov8n");

		private TaskProperties textDetector = new TaskProperties("inference4j/craft-mlt-25k");

		public TaskProperties getImageClassifier() {
			return imageClassifier;
		}

		public void setImageClassifier(TaskProperties imageClassifier) {
			this.imageClassifier = imageClassifier;
		}

		public TaskProperties getObjectDetector() {
			return objectDetector;
		}

		public void setObjectDetector(TaskProperties objectDetector) {
			this.objectDetector = objectDetector;
		}

		public TaskProperties getTextDetector() {
			return textDetector;
		}

		public void setTextDetector(TaskProperties textDetector) {
			this.textDetector = textDetector;
		}

	}

	public static class AudioProperties {

		private TaskProperties speechRecognizer = new TaskProperties("inference4j/wav2vec2-base-960h");

		private TaskProperties vad = new TaskProperties("inference4j/silero-vad");

		public TaskProperties getSpeechRecognizer() {
			return speechRecognizer;
		}

		public void setSpeechRecognizer(TaskProperties speechRecognizer) {
			this.speechRecognizer = speechRecognizer;
		}

		public TaskProperties getVad() {
			return vad;
		}

		public void setVad(TaskProperties vad) {
			this.vad = vad;
		}

	}

	public static class TaskProperties {

		private boolean enabled = false;

		private String modelId;

		public TaskProperties() {
		}

		public TaskProperties(String defaultModelId) {
			this.modelId = defaultModelId;
		}

		public boolean isEnabled() {
			return enabled;
		}

		public void setEnabled(boolean enabled) {
			this.enabled = enabled;
		}

		public String getModelId() {
			return modelId;
		}

		public void setModelId(String modelId) {
			this.modelId = modelId;
		}

	}

}
