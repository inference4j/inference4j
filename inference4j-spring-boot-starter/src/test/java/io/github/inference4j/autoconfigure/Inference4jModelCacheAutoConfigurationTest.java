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

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import io.github.inference4j.model.CachedModel;
import io.github.inference4j.model.EvictionResult;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelCache;
import io.github.inference4j.model.UpdateCheck;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import org.springframework.boot.actuate.endpoint.annotation.Endpoint;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.FilteredClassLoader;
import org.springframework.boot.test.context.runner.WebApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.assertThat;

class Inference4jModelCacheAutoConfigurationTest {

	private final WebApplicationContextRunner runner = new WebApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(Inference4jModelCacheAutoConfiguration.class));

	@TempDir
	static Path cacheDir;

	@Test
	void modelCacheIsTheDefaultSourcesCache() {
		runner.run(ctx -> assertThat(ctx.getBean(ModelCache.class)).isSameAs(ModelCache.defaultCache()));
	}

	@Test
	void modelCacheIsTheCacheOfTheApplicationsSource() {
		runner.withUserConfiguration(SourceConfig.class).run(ctx -> {
			HuggingFaceModelSource source = ctx.getBean(HuggingFaceModelSource.class);
			assertThat(ctx.getBean(ModelCache.class)).isSameAs(source.cache());
			assertThat(ctx.getBean(ModelCache.class).directory()).isEqualTo(cacheDir);
		});
	}

	@Test
	void applicationsModelCacheReplacesTheAutoConfiguredOne() {
		runner.withUserConfiguration(CacheConfig.class).run(ctx -> {
			assertThat(ctx).hasSingleBean(ModelCache.class);
			assertThat(ctx.getBean(ModelCache.class)).isInstanceOf(EmptyCache.class);
		});
	}

	@Test
	void endpointIsCreatedWhenExposed() {
		runner.withPropertyValues("management.endpoints.web.exposure.include=models")
			.run(ctx -> assertThat(ctx).hasSingleBean(ModelsEndpoint.class));
	}

	@Test
	void endpointUsesTheApplicationsModelCache() {
		runner.withUserConfiguration(CacheConfig.class)
			.withPropertyValues("management.endpoints.web.exposure.include=models")
			.run(ctx -> assertThat(ctx.getBean(ModelsEndpoint.class).models().models()).isEmpty());
	}

	@Test
	void endpointIsNotCreatedWhenNotExposed() {
		runner.run(ctx -> {
			assertThat(ctx).doesNotHaveBean(ModelsEndpoint.class);
			assertThat(ctx).hasSingleBean(ModelCache.class);
		});
	}

	@Test
	void endpointIsNotCreatedWhenDisabled() {
		runner.withPropertyValues("management.endpoints.web.exposure.include=models",
				"inference4j.actuator.models.enabled=false")
			.run(ctx -> {
				assertThat(ctx).doesNotHaveBean(ModelsEndpoint.class);
				assertThat(ctx).hasSingleBean(ModelCache.class);
			});
	}

	@Test
	void endpointIsNotCreatedWhenItsAccessIsNone() {
		runner.withPropertyValues("management.endpoints.web.exposure.include=models",
				"management.endpoint.models.access=NONE")
			.run(ctx -> assertThat(ctx).doesNotHaveBean(ModelsEndpoint.class));
	}

	@Test
	void endpointIsNotCreatedWithoutActuator() {
		runner.withClassLoader(new FilteredClassLoader(Endpoint.class))
			.withPropertyValues("management.endpoints.web.exposure.include=models")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).doesNotHaveBean(ModelsEndpoint.class);
				assertThat(ctx).hasSingleBean(ModelCache.class);
			});
	}

	@Configuration(proxyBeanMethods = false)
	static class SourceConfig {

		@Bean
		HuggingFaceModelSource modelSource() {
			return HuggingFaceModelSource.builder().cacheDir(cacheDir).offline(true).build();
		}

	}

	@Configuration(proxyBeanMethods = false)
	static class CacheConfig {

		@Bean
		ModelCache myModelCache() {
			return new EmptyCache();
		}

	}

	/** A cache with nothing in it. */
	static class EmptyCache implements ModelCache {

		@Override
		public Path directory() {
			return cacheDir;
		}

		@Override
		public List<CachedModel> list() {
			return List.of();
		}

		@Override
		public Optional<CachedModel> find(String repoId) {
			return Optional.empty();
		}

		@Override
		public UpdateCheck check(String repoId) {
			throw new UnsupportedOperationException();
		}

		@Override
		public CachedModel update(String repoId) {
			throw new UnsupportedOperationException();
		}

		@Override
		public EvictionResult evict(String repoId) {
			return new EvictionResult(List.of(), List.of(), 0);
		}

		@Override
		public EvictionResult evict(String repoId, String revision) {
			return new EvictionResult(List.of(), List.of(), 0);
		}

		@Override
		public EvictionResult prune() {
			return new EvictionResult(List.of(), List.of(), 0);
		}

	}

}
