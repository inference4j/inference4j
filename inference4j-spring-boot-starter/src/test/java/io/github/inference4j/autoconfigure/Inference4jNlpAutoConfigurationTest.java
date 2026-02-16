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

import java.util.List;

import io.github.inference4j.nlp.QueryDocumentPair;
import io.github.inference4j.nlp.SearchReranker;
import io.github.inference4j.nlp.TextClassification;
import io.github.inference4j.nlp.TextClassifier;
import io.github.inference4j.nlp.TextEmbedder;
import org.junit.jupiter.api.Test;

import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.assertThat;

class Inference4jNlpAutoConfigurationTest {

	private final ApplicationContextRunner runner = new ApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(Inference4jNlpAutoConfiguration.class));

	@Test
	void noBeansWhenDisabled() {
		runner.run(ctx -> {
			assertThat(ctx).doesNotHaveBean(TextClassifier.class);
			assertThat(ctx).doesNotHaveBean(TextEmbedder.class);
			assertThat(ctx).doesNotHaveBean(SearchReranker.class);
		});
	}

	@Test
	void noBeansWhenEnabledSetToFalse() {
		runner.withPropertyValues(
				"inference4j.nlp.text-classifier.enabled=false",
				"inference4j.nlp.text-embedder.enabled=false",
				"inference4j.nlp.search-reranker.enabled=false")
			.run(ctx -> {
				assertThat(ctx).doesNotHaveBean(TextClassifier.class);
				assertThat(ctx).doesNotHaveBean(TextEmbedder.class);
				assertThat(ctx).doesNotHaveBean(SearchReranker.class);
			});
	}

	@Test
	void beansAreLazyWhenEnabled() {
		runner.withPropertyValues(
				"inference4j.nlp.text-classifier.enabled=true",
				"inference4j.nlp.text-embedder.enabled=true",
				"inference4j.nlp.text-embedder.model-id=test/model",
				"inference4j.nlp.search-reranker.enabled=true")
			.run(ctx -> {
				assertThat(ctx.getBeanFactory().getBeanDefinition("textClassifier").isLazyInit()).isTrue();
				assertThat(ctx.getBeanFactory().getBeanDefinition("textEmbedder").isLazyInit()).isTrue();
				assertThat(ctx.getBeanFactory().getBeanDefinition("searchReranker").isLazyInit()).isTrue();
			});
	}

	@Test
	void userBeanOverridesAutoConfiguredTextClassifier() {
		runner.withPropertyValues("inference4j.nlp.text-classifier.enabled=true")
			.withUserConfiguration(CustomTextClassifierConfig.class)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(TextClassifier.class);
				assertThat(ctx).getBean(TextClassifier.class).isSameAs(CustomTextClassifierConfig.INSTANCE);
			});
	}

	@Test
	void userBeanOverridesAutoConfiguredTextEmbedder() {
		runner.withPropertyValues("inference4j.nlp.text-embedder.enabled=true")
			.withUserConfiguration(CustomTextEmbedderConfig.class)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(TextEmbedder.class);
				assertThat(ctx).getBean(TextEmbedder.class).isSameAs(CustomTextEmbedderConfig.INSTANCE);
			});
	}

	@Test
	void userBeanOverridesAutoConfiguredSearchReranker() {
		runner.withPropertyValues("inference4j.nlp.search-reranker.enabled=true")
			.withUserConfiguration(CustomSearchRerankerConfig.class)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(SearchReranker.class);
				assertThat(ctx).getBean(SearchReranker.class).isSameAs(CustomSearchRerankerConfig.INSTANCE);
			});
	}

	@Configuration(proxyBeanMethods = false)
	static class CustomTextClassifierConfig {

		static final TextClassifier INSTANCE = new TextClassifier() {
			@Override
			public List<TextClassification> classify(String text) {
				return List.of();
			}

			@Override
			public List<TextClassification> classify(String text, int topK) {
				return List.of();
			}

			@Override
			public void close() {
			}
		};

		@Bean
		TextClassifier textClassifier() {
			return INSTANCE;
		}

	}

	@Configuration(proxyBeanMethods = false)
	static class CustomTextEmbedderConfig {

		static final TextEmbedder INSTANCE = new TextEmbedder() {
			@Override
			public float[] encode(String text) {
				return new float[0];
			}

			@Override
			public List<float[]> encodeBatch(List<String> texts) {
				return List.of();
			}

			@Override
			public void close() {
			}
		};

		@Bean
		TextEmbedder textEmbedder() {
			return INSTANCE;
		}

	}

	@Configuration(proxyBeanMethods = false)
	static class CustomSearchRerankerConfig {

		static final SearchReranker INSTANCE = new SearchReranker() {
			@Override
			public Float run(QueryDocumentPair input) {
				return 0.0f;
			}

			@Override
			public void close() {
			}
		};

		@Bean
		SearchReranker searchReranker() {
			return INSTANCE;
		}

	}

}
