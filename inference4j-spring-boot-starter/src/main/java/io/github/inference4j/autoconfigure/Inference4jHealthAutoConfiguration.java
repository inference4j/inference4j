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

import io.github.inference4j.InferenceTask;

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;

/**
 * Auto-configuration for the inference4j health indicator.
 */
@AutoConfiguration
@ConditionalOnClass(HealthIndicator.class)
@ConditionalOnProperty(prefix = "inference4j.health", name = "enabled", matchIfMissing = true)
public class Inference4jHealthAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean(name = "inference4jHealthIndicator")
	public HealthIndicator inference4jHealthIndicator(ObjectProvider<InferenceTask<?, ?>> tasks) {
		return () -> {
			List<InferenceTask<?, ?>> activeTasks = tasks.orderedStream().toList();
			if (activeTasks.isEmpty()) {
				return Health.unknown().withDetail("reason", "No inference4j tasks configured").build();
			}
			return Health.up().withDetail("tasks", activeTasks.size()).build();
		};
	}

}
