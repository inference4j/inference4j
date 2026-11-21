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

	private MetricsProperties metrics = new MetricsProperties();

	private GenaiProperties genai = new GenaiProperties();

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

	public MetricsProperties getMetrics() {
		return metrics;
	}

	public void setMetrics(MetricsProperties metrics) {
		this.metrics = metrics;
	}

	public GenaiProperties getGenai() {
		return genai;
	}

	public void setGenai(GenaiProperties genai) {
		this.genai = genai;
	}

	public static class NlpProperties {

		private TaskProperties textClassifier = new TaskProperties(
				"inference4j/distilbert-base-uncased-finetuned-sst-2-english");

		private TaskProperties textEmbedder = new TaskProperties();

		private TaskProperties searchReranker = new TaskProperties("inference4j/ms-marco-MiniLM-L-6-v2");

		private TaskProperties namedEntityRecognizer = new TaskProperties("inference4j/distilbert-NER");

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

		public TaskProperties getNamedEntityRecognizer() {
			return namedEntityRecognizer;
		}

		public void setNamedEntityRecognizer(TaskProperties namedEntityRecognizer) {
			this.namedEntityRecognizer = namedEntityRecognizer;
		}

	}

	public static class VisionProperties {

		private TaskProperties imageClassifier = new TaskProperties("inference4j/resnet50-v1-7");

		private ObjectDetectorProperties objectDetector = new ObjectDetectorProperties();

		private TaskProperties textDetector = new TaskProperties("inference4j/craft-mlt-25k");

		public TaskProperties getImageClassifier() {
			return imageClassifier;
		}

		public void setImageClassifier(TaskProperties imageClassifier) {
			this.imageClassifier = imageClassifier;
		}

		public ObjectDetectorProperties getObjectDetector() {
			return objectDetector;
		}

		public void setObjectDetector(ObjectDetectorProperties objectDetector) {
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

	public static class MetricsProperties {

		private boolean enabled = true;

		public boolean isEnabled() {
			return enabled;
		}

		public void setEnabled(boolean enabled) {
			this.enabled = enabled;
		}

	}

	/**
	 * Generative models from {@code inference4j-genai}; used only when that module is on the classpath.
	 */
	public static class GenaiProperties {

		private GenerationProperties textGenerator = new GenerationProperties("phi-3-mini");

		private GenerationProperties visionLanguage = new GenerationProperties("phi-3.5-vision");

		private WhisperProperties whisper = new WhisperProperties();

		public GenerationProperties getTextGenerator() {
			return textGenerator;
		}

		public void setTextGenerator(GenerationProperties textGenerator) {
			this.textGenerator = textGenerator;
		}

		public GenerationProperties getVisionLanguage() {
			return visionLanguage;
		}

		public void setVisionLanguage(GenerationProperties visionLanguage) {
			this.visionLanguage = visionLanguage;
		}

		public WhisperProperties getWhisper() {
			return whisper;
		}

		public void setWhisper(WhisperProperties whisper) {
			this.whisper = whisper;
		}

	}

	/**
	 * A generative model chosen by preset name, with optional generation settings. Unset settings keep
	 * the builder's defaults.
	 */
	public static class GenerationProperties {

		private boolean enabled = false;

		private String model;

		private Integer maxLength;

		private Double temperature;

		private Integer topK;

		private Double topP;

		public GenerationProperties() {
		}

		public GenerationProperties(String defaultModel) {
			this.model = defaultModel;
		}

		public boolean isEnabled() {
			return enabled;
		}

		public void setEnabled(boolean enabled) {
			this.enabled = enabled;
		}

		public String getModel() {
			return model;
		}

		public void setModel(String model) {
			this.model = model;
		}

		public Integer getMaxLength() {
			return maxLength;
		}

		public void setMaxLength(Integer maxLength) {
			this.maxLength = maxLength;
		}

		public Double getTemperature() {
			return temperature;
		}

		public void setTemperature(Double temperature) {
			this.temperature = temperature;
		}

		public Integer getTopK() {
			return topK;
		}

		public void setTopK(Integer topK) {
			this.topK = topK;
		}

		public Double getTopP() {
			return topP;
		}

		public void setTopP(Double topP) {
			this.topP = topP;
		}

	}

	public static class WhisperProperties {

		private boolean enabled = false;

		private String modelId = "inference4j/whisper-small-genai";

		private String language = "en";

		private String task = "transcribe";

		private Integer maxLength;

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

		public String getLanguage() {
			return language;
		}

		public void setLanguage(String language) {
			this.language = language;
		}

		public String getTask() {
			return task;
		}

		public void setTask(String task) {
			this.task = task;
		}

		public Integer getMaxLength() {
			return maxLength;
		}

		public void setMaxLength(Integer maxLength) {
			this.maxLength = maxLength;
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

	/**
	 * The object detector: which wrapper builds the bean, and which model it loads.
	 *
	 * <p>When {@code model-id} is not set, the chosen architecture's own default model is used, so
	 * switching {@code architecture} to {@code yolov8} alone loads {@code inference4j/yolov8n} rather
	 * than the D-FINE weights.
	 */
	public static class ObjectDetectorProperties extends TaskProperties {

		private ObjectDetectorArchitecture architecture = ObjectDetectorArchitecture.DETR;

		public ObjectDetectorArchitecture getArchitecture() {
			return architecture;
		}

		public void setArchitecture(ObjectDetectorArchitecture architecture) {
			this.architecture = architecture;
		}

		/**
		 * Returns the configured model id, or the architecture's default model when none is set.
		 */
		@Override
		public String getModelId() {
			String modelId = super.getModelId();
			if (modelId != null && !modelId.isBlank()) {
				return modelId;
			}
			return architecture.getDefaultModelId();
		}

	}

	/**
	 * The detector families the starter can build an {@code ObjectDetector} from.
	 */
	public enum ObjectDetectorArchitecture {

		/** DETR-family models through {@code DetrDetector}, such as D-FINE and RT-DETRv2 (Apache-2.0). */
		DETR("inference4j/dfine-small-coco"),

		/** YOLOv8 models through {@code YoloV8Detector} (AGPL-3.0 weights). */
		YOLOV8("inference4j/yolov8n"),

		/** YOLO26 models through {@code Yolo26Detector} (AGPL-3.0 weights). */
		YOLO26("inference4j/yolo26n");

		private final String defaultModelId;

		ObjectDetectorArchitecture(String defaultModelId) {
			this.defaultModelId = defaultModelId;
		}

		/**
		 * Returns the model loaded when {@code model-id} is not set.
		 *
		 * @return the default HuggingFace model id for this architecture
		 */
		public String getDefaultModelId() {
			return defaultModelId;
		}

	}

}
