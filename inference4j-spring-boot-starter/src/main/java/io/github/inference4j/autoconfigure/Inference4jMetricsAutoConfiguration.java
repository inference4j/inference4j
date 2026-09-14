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

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Auto-configuration for a {@link RouterMetrics} bean.
 *
 * <p>{@link io.github.inference4j.routing.ModelRouter} builders in application code already accept
 * a {@code RouterMetrics} via {@code .metrics(...)}; this only removes the need to construct one
 * by hand. When a
 * {@link MeterRegistry} bean is present, routing gets a {@link MicrometerRouterMetrics} backed by
 * it; otherwise routing keeps working with a {@link NoOpRouterMetrics}, matching the default a
 * router builder already falls back to on its own.
 *
 * <p>Individual task wrappers (e.g. {@code ImageClassifier}, {@code TextClassifier}) have no
 * metrics hook of their own -- only {@code ModelRouter} does -- so this bean is only useful to
 * application code that builds and manages its own router.
 */
@AutoConfiguration
@ConditionalOnClass(MeterRegistry.class)
@EnableConfigurationProperties(Inference4jProperties.class)
public class Inference4jMetricsAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean(RouterMetrics.class)
	@ConditionalOnProperty(prefix = "inference4j.metrics", name = "enabled", matchIfMissing = true)
	public RouterMetrics routerMetrics(ObjectProvider<MeterRegistry> registry) {
		MeterRegistry meterRegistry = registry.getIfAvailable();
		if (meterRegistry == null) {
			return NoOpRouterMetrics.getInstance();
		}
		return new MicrometerRouterMetrics(meterRegistry);
	}

}
