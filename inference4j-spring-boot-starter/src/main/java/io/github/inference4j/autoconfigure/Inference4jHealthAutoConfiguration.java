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

import java.time.Duration;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import io.github.inference4j.InferenceTask;
import io.github.inference4j.generation.GenerativeTask;
import io.github.inference4j.model.CachedModel;
import io.github.inference4j.model.CachedSnapshot;
import io.github.inference4j.model.ModelCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.health.contributor.Health;
import org.springframework.boot.health.contributor.HealthIndicator;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.context.annotation.Bean;
import org.springframework.core.env.Environment;

/**
 * Auto-configuration for the inference4j health indicator.
 *
 * <p>It reports the number of tasks and, when a {@link ModelCache} bean exists, the cached models a loaded model
 * serves an older revision of ({@code staleModels}) and those the last Hub check found changed
 * ({@code updatesAvailable}). Both come from local state, without calling the Hub, and never change the status.
 * The cache listing is reused for {@code inference4j.health.model-cache-ttl} (default 10 seconds), since health
 * probes can be frequent.
 */
@AutoConfiguration
@ConditionalOnClass(HealthIndicator.class)
@ConditionalOnProperty(prefix = "inference4j.health", name = "enabled", matchIfMissing = true)
public class Inference4jHealthAutoConfiguration {

	private static final Logger logger = LoggerFactory.getLogger(Inference4jHealthAutoConfiguration.class);

	@Bean
	@ConditionalOnMissingBean(name = "inference4jHealthIndicator")
	public HealthIndicator inference4jHealthIndicator(ObjectProvider<InferenceTask<?, ?>> tasks,
			ObjectProvider<GenerativeTask<?, ?>> generativeTasks, ObjectProvider<ModelCache> modelCaches,
			Environment environment) {
		Duration ttl = Binder.get(environment)
			.bind("inference4j.health.model-cache-ttl", Duration.class)
			.orElse(Duration.ofSeconds(10));
		CachedListing listing = new CachedListing(ttl);
		return () -> {
			// A bean can be both (WhisperSpeechModel is a SpeechRecognizer and a GenerativeTask): count it once
			Set<Object> activeTasks = Collections.newSetFromMap(new IdentityHashMap<>());
			tasks.orderedStream().forEach(activeTasks::add);
			generativeTasks.orderedStream().forEach(activeTasks::add);
			Health.Builder health = activeTasks.isEmpty()
					? Health.unknown().withDetail("reason", "No inference4j tasks configured")
					: Health.up().withDetail("tasks", activeTasks.size());
			ModelCache cache = modelCaches.getIfUnique();
			if (cache != null) {
				addCacheDetails(health, listing.get(cache));
			}
			return health.build();
		};
	}

	private static void addCacheDetails(Health.Builder health, List<CachedModel> models) {
		if (models == null) {
			return;
		}
		List<Map<String, Object>> stale = models.stream().filter(CachedModel::stale).map(model -> {
			Map<String, Object> entry = new LinkedHashMap<>();
			entry.put("repo", model.repoId());
			entry.put("loadedRevisions", model.snapshots().stream()
				.filter(snapshot -> !snapshot.current() && snapshot.leases() > 0)
				.map(CachedSnapshot::revision)
				.toList());
			entry.put("cachedRevision", model.revision());
			return entry;
		}).toList();
		List<String> updatesAvailable = models.stream()
			.filter(CachedModel::updateAvailable)
			.map(CachedModel::repoId)
			.toList();
		health.withDetail("staleModels", stale).withDetail("updatesAvailable", updatesAvailable);
	}

	/** The last cache listing, reused while it is younger than the TTL. Thread-safe. */
	private static final class CachedListing {

		private record Listing(ModelCache cache, long takenAt, List<CachedModel> models) {
		}

		private final AtomicReference<Listing> last = new AtomicReference<>();

		private final long ttlNanos;

		CachedListing(Duration ttl) {
			this.ttlNanos = ttl.toNanos();
		}

		/** The cache's models, or {@code null} when they cannot be listed. */
		List<CachedModel> get(ModelCache cache) {
			long now = System.nanoTime();
			Listing listing = last.get();
			if (listing != null && listing.cache() == cache
					&& now - listing.takenAt() < ttlNanos) {
				return listing.models();
			}
			try {
				List<CachedModel> models = cache.list();
				last.set(new Listing(cache, now, models));
				return models;
			}
			catch (RuntimeException ex) {
				logger.debug("Cannot list the model cache: {}", ex.toString());
				return null;
			}
		}

	}

}
