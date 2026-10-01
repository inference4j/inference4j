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

import io.github.inference4j.metrics.NoOpRouterMetrics;
import io.github.inference4j.metrics.RouterMetrics;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Auto-configuration for a {@link RouterMetrics} bean, falling back to a {@link NoOpRouterMetrics}.
 * Unconditional on {@code inference4j.metrics.enabled} -- that property only gates the
 * Micrometer-backed bean, so disabling it still leaves a {@link RouterMetrics} bean available.
 */
@AutoConfiguration
@EnableConfigurationProperties(Inference4jProperties.class)
public class Inference4jMetricsAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean(RouterMetrics.class)
	public RouterMetrics routerMetrics() {
		return NoOpRouterMetrics.getInstance();
	}

}
