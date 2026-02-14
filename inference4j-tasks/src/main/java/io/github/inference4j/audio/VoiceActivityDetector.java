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

package io.github.inference4j.audio;

import io.github.inference4j.Detector;

import java.nio.file.Path;
import java.util.List;

/**
 * Detects voice activity segments in audio.
 */
public interface VoiceActivityDetector extends Detector<Path, VoiceSegment> {

    @Override
    List<VoiceSegment> detect(Path audioPath);

    List<VoiceSegment> detect(float[] audioData, int sampleRate);

    @Override
    void close();
}
