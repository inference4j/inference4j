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
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import io.github.inference4j.InferenceTask;
import io.github.inference4j.generation.GenerativeTask;
import io.github.inference4j.model.CachedFile;
import io.github.inference4j.model.CachedModel;
import io.github.inference4j.model.CachedSnapshot;
import io.github.inference4j.model.EvictionResult;
import io.github.inference4j.model.ModelCache;
import io.github.inference4j.model.UpdateCheck;
import org.junit.jupiter.api.Test;

import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.health.contributor.Health;
import org.springframework.boot.health.contributor.HealthIndicator;
import org.springframework.boot.health.contributor.Status;
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
	void generativeTasksAreCountedAndDualTasksOnlyOnce() {
		runner.withUserConfiguration(TaskConfig.class, GenerativeTaskConfig.class).run(ctx -> {
			Health health = ctx.getBean(HealthIndicator.class).health();
			// dummyTask + generator + dual (both an InferenceTask and a GenerativeTask) = 3
			assertThat(health.getStatus()).isEqualTo(Status.UP);
			assertThat(health.getDetails()).containsEntry("tasks", 3);
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

	@Test
	void healthReportsStaleModelsAndUpdatesWithoutChangingTheStatus() {
		runner.withUserConfiguration(TaskConfig.class, CacheConfig.class).run(ctx -> {
			Health health = ctx.getBean(HealthIndicator.class).health();
			assertThat(health.getStatus()).isEqualTo(Status.UP);
			assertThat(health.getDetails()).containsEntry("tasks", 1);
			assertThat(health.getDetails()).containsEntry("staleModels",
					List.of(Map.of("repo", "org/stale", "loadedRevisions", List.of("old"), "cachedRevision", "new")));
			assertThat(health.getDetails()).containsEntry("updatesAvailable", List.of("org/outdated"));
		});
	}

	@Test
	void healthWithoutTasksKeepsItsStatusAndAddsCacheDetails() {
		runner.withUserConfiguration(CacheConfig.class).run(ctx -> {
			Health health = ctx.getBean(HealthIndicator.class).health();
			assertThat(health.getStatus()).isEqualTo(Status.UNKNOWN);
			assertThat(health.getDetails()).containsKeys("reason", "staleModels", "updatesAvailable");
		});
	}

	@Test
	void healthIgnoresACacheThatCannotBeListed() {
		runner.withUserConfiguration(TaskConfig.class, BrokenCacheConfig.class).run(ctx -> {
			Health health = ctx.getBean(HealthIndicator.class).health();
			assertThat(health.getStatus()).isEqualTo(Status.UP);
			assertThat(health.getDetails()).containsOnlyKeys("tasks");
		});
	}

	@Test
	void healthReusesTheCacheListingWithinTheTtl() {
		runner.withUserConfiguration(TaskConfig.class, CountingCacheConfig.class).run(ctx -> {
			HealthIndicator indicator = ctx.getBean(HealthIndicator.class);
			indicator.health();
			indicator.health();
			assertThat(ctx.getBean(CountingCacheConfig.class).listings.get()).isEqualTo(1);
		});
	}

	@Test
	void aZeroTtlListsTheCacheOnEveryHealthCall() {
		runner.withUserConfiguration(TaskConfig.class, CountingCacheConfig.class)
			.withPropertyValues("inference4j.health.model-cache-ttl=0s")
			.run(ctx -> {
				HealthIndicator indicator = ctx.getBean(HealthIndicator.class);
				indicator.health();
				indicator.health();
				assertThat(ctx.getBean(CountingCacheConfig.class).listings.get()).isEqualTo(2);
			});
	}

	@Configuration(proxyBeanMethods = false)
	static class CountingCacheConfig {

		final AtomicInteger listings = new AtomicInteger();

		@Bean
		ModelCache modelCache() {
			return new ListedCache(() -> {
				listings.incrementAndGet();
				return List.of();
			});
		}

	}

	@Configuration(proxyBeanMethods = false)
	static class CacheConfig {

		@Bean
		ModelCache modelCache() {
			CachedFile v1 = new CachedFile("model.onnx", "id-1", 10);
			CachedFile v2 = new CachedFile("model.onnx", "id-2", 10);
			CachedModel stale = new CachedModel("org/stale", "new", null, null, false,
					List.of(new CachedSnapshot("old", false, 1, List.of(v1), 10),
							new CachedSnapshot("new", true, 0, List.of(v2), 10)),
					20);
			CachedModel outdated = new CachedModel("org/outdated", "abc", Instant.EPOCH, "def", true,
					List.of(new CachedSnapshot("abc", true, 0, List.of(v1), 10)), 10);
			CachedModel fresh = new CachedModel("org/fresh", "abc", Instant.EPOCH, "abc", false,
					List.of(new CachedSnapshot("abc", true, 0, List.of(v1), 10)), 10);
			return new ListedCache(() -> List.of(fresh, outdated, stale));
		}

	}

	@Configuration(proxyBeanMethods = false)
	static class BrokenCacheConfig {

		@Bean
		ModelCache modelCache() {
			return new ListedCache(() -> {
				throw new IllegalStateException("unreadable");
			});
		}

	}

	/** A cache whose {@link #list()} answers from a supplier; nothing else is called. */
	static class ListedCache implements ModelCache {

		private final Supplier<List<CachedModel>> models;

		ListedCache(Supplier<List<CachedModel>> models) {
			this.models = models;
		}

		@Override
		public Path directory() {
			throw new UnsupportedOperationException();
		}

		@Override
		public List<CachedModel> list() {
			return models.get();
		}

		@Override
		public Optional<CachedModel> find(String repoId) {
			throw new UnsupportedOperationException();
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
			throw new UnsupportedOperationException();
		}

		@Override
		public EvictionResult evict(String repoId, String revision) {
			throw new UnsupportedOperationException();
		}

		@Override
		public EvictionResult prune() {
			throw new UnsupportedOperationException();
		}

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
	static class GenerativeTaskConfig {

		@Bean
		GenerativeTask<String, String> generator() {
			return new StubGenerator();
		}

		@Bean
		DualTask dual() {
			return new DualTask();
		}

	}

	static class StubGenerator implements GenerativeTask<String, String> {

		@Override
		public String generate(String input) {
			return input;
		}

		@Override
		public String generate(String input, java.util.function.Consumer<String> tokenListener) {
			return input;
		}

		@Override
		public void close() {
		}

	}

	/** Like WhisperSpeechModel: both kinds of task. */
	static class DualTask extends StubGenerator implements InferenceTask<String, String> {

		@Override
		public String run(String input) {
			return input;
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
