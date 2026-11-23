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

import io.github.inference4j.dashboard.ModelCacheService;
import io.github.inference4j.model.HuggingFaceModelSource;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.FilteredClassLoader;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.boot.test.context.runner.WebApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.function.RouterFunction;

import static org.assertj.core.api.Assertions.assertThat;

class Inference4jDashboardAutoConfigurationTest {

	private static final AutoConfigurations CONFIGURATIONS = AutoConfigurations
		.of(Inference4jModelCacheAutoConfiguration.class, Inference4jDashboardAutoConfiguration.class);

	@TempDir
	static Path cacheDir;

	private final WebApplicationContextRunner runner = new WebApplicationContextRunner()
		.withConfiguration(CONFIGURATIONS)
		.withUserConfiguration(SourceConfiguration.class);

	@Test
	void isOffByDefault() {
		runner.run(ctx -> {
			assertThat(ctx).doesNotHaveBean(RouterFunction.class);
			assertThat(ctx).hasSingleBean(ModelCacheService.class);
		});
	}

	@Test
	void servesWhenEnabled() {
		runner.withPropertyValues("inference4j.dashboard.enabled=true")
			.run(ctx -> assertThat(ctx).hasBean("inference4jDashboardRoutes"));
	}

	@Test
	void needsAServletWebApplication() {
		new ApplicationContextRunner().withConfiguration(CONFIGURATIONS)
			.withUserConfiguration(SourceConfiguration.class)
			.withPropertyValues("inference4j.dashboard.enabled=true")
			.run(ctx -> assertThat(ctx).doesNotHaveBean(RouterFunction.class));
	}

	@Test
	void needsSpringMvcFunctionalRoutes() {
		runner.withPropertyValues("inference4j.dashboard.enabled=true")
			.withClassLoader(new FilteredClassLoader(RouterFunction.class))
			.run(ctx -> assertThat(ctx).doesNotHaveBean("inference4jDashboardRoutes"));
	}

	@Test
	void rejectsAnEmptyPath() {
		runner.withPropertyValues("inference4j.dashboard.enabled=true", "inference4j.dashboard.path=/")
			.run(ctx -> assertThat(ctx).hasFailed()
				.getFailure()
				.rootCause()
				.hasMessageContaining("inference4j.dashboard.path"));
	}

	@Configuration(proxyBeanMethods = false)
	static class SourceConfiguration {

		@Bean
		HuggingFaceModelSource modelSource() {
			return HuggingFaceModelSource.builder().cacheDir(cacheDir).offline(true).build();
		}

	}

}
