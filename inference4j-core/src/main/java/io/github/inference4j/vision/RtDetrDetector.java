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

package io.github.inference4j.vision;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.preprocessing.image.Labels;
import io.github.inference4j.session.SessionConfigurer;

/**
 * The former name of {@link DetrDetector}, kept so existing code compiles.
 *
 * @deprecated since 0.15.0, for removal in 0.16.0. Use {@link DetrDetector}: same presets, same builder options,
 * same results.
 */
@Deprecated(since = "0.15.0", forRemoval = true)
public class RtDetrDetector extends DetrDetector {

    private RtDetrDetector(InferenceSession session, Labels labels, int inputSize, float defaultThreshold) {
        super(session, labels, inputSize, defaultThreshold);
    }

    /** @deprecated Use {@link DetrDetector#dfineSmall()}. */
    @Deprecated(since = "0.15.0", forRemoval = true)
    public static Builder dfineSmall() {
        return new Builder().modelId("inference4j/dfine-small-coco");
    }

    /** @deprecated Use {@link DetrDetector#dfineNano()}. */
    @Deprecated(since = "0.15.0", forRemoval = true)
    public static Builder dfineNano() {
        return new Builder().modelId("inference4j/dfine-nano-coco");
    }

    /** @deprecated Use {@link DetrDetector#rtDetrV2()}. */
    @Deprecated(since = "0.15.0", forRemoval = true)
    public static Builder rtDetrV2() {
        return new Builder().modelId("inference4j/rtdetr-v2-r18vd");
    }

    /** @deprecated Use {@link DetrDetector#builder()}. */
    @Deprecated(since = "0.15.0", forRemoval = true)
    public static Builder builder() {
        return dfineSmall();
    }

    /**
     * @deprecated Use {@link DetrDetector.Builder}.
     */
    @Deprecated(since = "0.15.0", forRemoval = true)
    public static class Builder extends DetrDetector.Builder {

        @Override
        public Builder modelSource(ModelSource modelSource) {
            super.modelSource(modelSource);
            return this;
        }

        @Override
        public Builder modelId(String modelId) {
            super.modelId(modelId);
            return this;
        }

        @Override
        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            super.sessionOptions(sessionConfigurer);
            return this;
        }

        @Override
        public Builder labels(Labels labels) {
            super.labels(labels);
            return this;
        }

        @Override
        public Builder inputSize(int inputSize) {
            super.inputSize(inputSize);
            return this;
        }

        @Override
        public Builder confidenceThreshold(float confidenceThreshold) {
            super.confidenceThreshold(confidenceThreshold);
            return this;
        }

        @Override
        public RtDetrDetector build() {
            return new RtDetrDetector(openSession(), labels, inputSize, confidenceThreshold);
        }
    }
}
