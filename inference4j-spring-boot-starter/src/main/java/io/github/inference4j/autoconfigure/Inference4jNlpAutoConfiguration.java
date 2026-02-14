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

import io.github.inference4j.nlp.DistilBertTextClassifier;
import io.github.inference4j.nlp.MiniLMSearchReranker;
import io.github.inference4j.nlp.SearchReranker;
import io.github.inference4j.nlp.SentenceTransformerEmbedder;
import io.github.inference4j.nlp.TextClassifier;
import io.github.inference4j.nlp.TextEmbedder;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Auto-configuration for inference4j NLP tasks.
 */
@AutoConfiguration
@ConditionalOnClass(DistilBertTextClassifier.class)
@EnableConfigurationProperties(Inference4jProperties.class)
public class Inference4jNlpAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean(TextClassifier.class)
	@ConditionalOnProperty(prefix = "inference4j.nlp.text-classifier", name = "enabled", havingValue = "true")
	public TextClassifier textClassifier(Inference4jProperties properties) {
		return DistilBertTextClassifier.builder()
			.modelId(properties.getNlp().getTextClassifier().getModelId())
			.build();
	}

	@Bean
	@ConditionalOnMissingBean(TextEmbedder.class)
	@ConditionalOnProperty(prefix = "inference4j.nlp.text-embedder", name = "enabled", havingValue = "true")
	public TextEmbedder textEmbedder(Inference4jProperties properties) {
		return SentenceTransformerEmbedder.builder()
			.modelId(properties.getNlp().getTextEmbedder().getModelId())
			.build();
	}

	@Bean
	@ConditionalOnMissingBean(SearchReranker.class)
	@ConditionalOnProperty(prefix = "inference4j.nlp.search-reranker", name = "enabled", havingValue = "true")
	public SearchReranker searchReranker(Inference4jProperties properties) {
		return MiniLMSearchReranker.builder()
			.modelId(properties.getNlp().getSearchReranker().getModelId())
			.build();
	}

}
