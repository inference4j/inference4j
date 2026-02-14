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

import io.github.inference4j.InferenceTask;
import org.junit.jupiter.api.Test;

import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.actuate.health.Status;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.assertThat;

class Inference4jHealthAutoConfigurationTest {

	private final ApplicationContextRunner runner = new ApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(Inference4jHealthAutoConfiguration.class));

	@Test
	void healthIndicatorRegisteredByDefault() {
		runner.run(ctx -> assertThat(ctx).hasSingleBean(HealthIndicator.class));
	}

	@Test
	void healthIndicatorDisabledViaProperty() {
		runner.withPropertyValues("inference4j.health.enabled=false")
			.run(ctx -> assertThat(ctx).doesNotHaveBean(HealthIndicator.class));
	}

	@Test
	void healthUnknownWhenNoTasks() {
		runner.run(ctx -> {
			HealthIndicator indicator = ctx.getBean(HealthIndicator.class);
			Health health = indicator.health();
			assertThat(health.getStatus()).isEqualTo(Status.UNKNOWN);
			assertThat(health.getDetails()).containsEntry("reason", "No inference4j tasks configured");
		});
	}

	@Test
	void healthUpWhenTasksPresent() {
		runner.withUserConfiguration(TaskConfig.class).run(ctx -> {
			HealthIndicator indicator = ctx.getBean(HealthIndicator.class);
			Health health = indicator.health();
			assertThat(health.getStatus()).isEqualTo(Status.UP);
			assertThat(health.getDetails()).containsEntry("tasks", 1);
		});
	}

	@Test
	void userCanOverrideHealthIndicator() {
		runner.withUserConfiguration(CustomHealthConfig.class).run(ctx -> {
			assertThat(ctx).hasSingleBean(HealthIndicator.class);
			HealthIndicator indicator = ctx.getBean(HealthIndicator.class);
			assertThat(indicator.health().getStatus()).isEqualTo(Status.DOWN);
		});
	}

	@Configuration(proxyBeanMethods = false)
	static class TaskConfig {

		@Bean
		InferenceTask<String, String> dummyTask() {
			return new InferenceTask<>() {
				@Override
				public String run(String input) {
					return "result";
				}

				@Override
				public void close() {
				}
			};
		}

	}

	@Configuration(proxyBeanMethods = false)
	static class CustomHealthConfig {

		@Bean
		HealthIndicator inference4jHealthIndicator() {
			return () -> Health.down().build();
		}

	}

}
