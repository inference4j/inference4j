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
import io.github.inference4j.metrics.RouterMetrics;

import io.micrometer.core.instrument.MeterRegistry;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;

/**
 * Auto-configuration for a Micrometer-backed {@link RouterMetrics} bean.
 */
@AutoConfiguration(before = Inference4jMetricsAutoConfiguration.class)
@ConditionalOnClass(MeterRegistry.class)
public class Inference4jMicrometerMetricsAutoConfiguration {

	@Bean
	@ConditionalOnBean(MeterRegistry.class)
	@ConditionalOnMissingBean(RouterMetrics.class)
	@ConditionalOnProperty(prefix = "inference4j.metrics", name = "enabled", matchIfMissing = true)
	public RouterMetrics routerMetrics(MeterRegistry registry) {
		return new MicrometerRouterMetrics(registry);
	}

}
