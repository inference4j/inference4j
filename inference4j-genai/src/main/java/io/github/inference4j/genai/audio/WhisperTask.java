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

/**
 * Represents the task mode for Whisper model inference.
 *
 * <p>Whisper supports two task modes that are specified as special tokens
 * in the decoder prompt:
 * <ul>
 *   <li>{@link #TRANSCRIBE} — transcribes speech in the source language</li>
 *   <li>{@link #TRANSLATE} — translates speech to English</li>
 * </ul>
 *
 * @see <a href="https://arxiv.org/abs/2212.04356">Whisper paper</a>
 */
public enum WhisperTask {

	TRANSCRIBE("<|transcribe|>"),

	TRANSLATE("<|translate|>");

	private final String token;

	WhisperTask(String token) {
		this.token = token;
	}

	/**
	 * Returns the special token string used in the Whisper decoder prompt
	 * to indicate this task mode.
	 * @return the task token, e.g. {@code "<|transcribe|>"}
	 */
	public String token() {
		return token;
	}

}
