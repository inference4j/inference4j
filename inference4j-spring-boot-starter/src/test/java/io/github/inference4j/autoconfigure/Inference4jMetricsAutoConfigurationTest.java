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

import io.github.inference4j.metrics.MicrometerRouterMetrics;
import io.github.inference4j.metrics.NoOpRouterMetrics;
import io.github.inference4j.metrics.RouterMetrics;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.Test;

import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.FilteredClassLoader;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.assertThat;

class Inference4jMetricsAutoConfigurationTest {

	private final ApplicationContextRunner runner = new ApplicationContextRunner().withConfiguration(
			AutoConfigurations.of(Inference4jMicrometerMetricsAutoConfiguration.class, Inference4jMetricsAutoConfiguration.class));

	@Test
	void noOpMetricsWhenNoMeterRegistry() {
		runner.run(ctx -> {
			assertThat(ctx).hasSingleBean(RouterMetrics.class);
			assertThat(ctx.getBean(RouterMetrics.class)).isSameAs(NoOpRouterMetrics.getInstance());
		});
	}

	@Test
	void noOpMetricsWhenMicrometerAbsentFromClasspath() {
		runner.withClassLoader(new FilteredClassLoader(MeterRegistry.class)).run(ctx -> {
			assertThat(ctx).hasSingleBean(RouterMetrics.class);
			assertThat(ctx.getBean(RouterMetrics.class)).isSameAs(NoOpRouterMetrics.getInstance());
		});
	}

	@Test
	void micrometerMetricsWhenMeterRegistryPresent() {
		runner.withUserConfiguration(MeterRegistryConfig.class)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(RouterMetrics.class);
				assertThat(ctx.getBean(RouterMetrics.class)).isInstanceOf(MicrometerRouterMetrics.class);
			});
	}

	@Test
	void routerMetricsDisabledViaProperty() {
		runner.withUserConfiguration(MeterRegistryConfig.class)
			.withPropertyValues("inference4j.metrics.enabled=false")
			.run(ctx -> assertThat(ctx).doesNotHaveBean(RouterMetrics.class));
	}

	@Test
	void userCanOverrideRouterMetrics() {
		runner.withUserConfiguration(MeterRegistryConfig.class, CustomMetricsConfig.class)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(RouterMetrics.class);
				assertThat(ctx.getBean(RouterMetrics.class)).isSameAs(CustomMetricsConfig.INSTANCE);
			});
	}

	@Configuration(proxyBeanMethods = false)
	static class MeterRegistryConfig {

		@Bean
		MeterRegistry meterRegistry() {
			return new SimpleMeterRegistry();
		}

	}

	@Configuration(proxyBeanMethods = false)
	static class CustomMetricsConfig {

		static final RouterMetrics INSTANCE = NoOpRouterMetrics.getInstance();

		@Bean
		RouterMetrics routerMetrics() {
			return INSTANCE;
		}

	}

}
