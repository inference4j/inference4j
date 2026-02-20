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

package io.github.inference4j.routing;

import io.github.inference4j.nlp.TextEmbedder;

import java.util.List;

/**
 * Routes embedding operations across multiple {@link TextEmbedder} instances
 * with weighted traffic splitting, shadow mode, and metrics support.
 *
 * <pre>{@code
 * TextEmbedder router = TextEmbedderRouter.builder()
 *     .name("embedding-ab-test")
 *     .route("fast", miniLm, 80)
 *     .route("accurate", mpnet, 20)
 *     .shadow("experimental", experimentalModel)
 *     .build();
 *
 * float[] embedding = router.encode("Hello world");
 * }</pre>
 */
public final class TextEmbedderRouter extends ModelRouter<TextEmbedder> implements TextEmbedder {

    private TextEmbedderRouter(Builder builder) {
        super(builder);
    }

    @Override
    public float[] encode(String text) {
        return execute(model -> model.encode(text));
    }

    @Override
    public List<float[]> encodeBatch(List<String> texts) {
        return execute(model -> model.encodeBatch(texts));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends BaseBuilder<TextEmbedder, Builder> {

        public TextEmbedderRouter build() {
            return new TextEmbedderRouter(this);
        }
    }
}
