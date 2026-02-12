package io.github.inference4j.embedding;

import io.github.inference4j.routing.ModelRouter;

import java.util.List;

/**
 * Routes embedding operations across multiple {@link EmbeddingModel} instances
 * with weighted traffic splitting, shadow mode, and metrics support.
 *
 * <pre>{@code
 * EmbeddingModel router = EmbeddingModelRouter.builder()
 *     .name("embedding-ab-test")
 *     .route("fast", miniLm, 80)
 *     .route("accurate", mpnet, 20)
 *     .shadow("experimental", experimentalModel)
 *     .build();
 *
 * float[] embedding = router.encode("Hello world");
 * }</pre>
 */
public final class EmbeddingModelRouter extends ModelRouter<EmbeddingModel> implements EmbeddingModel {

    private EmbeddingModelRouter(Builder builder) {
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

    public static final class Builder extends BaseBuilder<EmbeddingModel, Builder> {

        public EmbeddingModelRouter build() {
            return new EmbeddingModelRouter(this);
        }
    }
}
