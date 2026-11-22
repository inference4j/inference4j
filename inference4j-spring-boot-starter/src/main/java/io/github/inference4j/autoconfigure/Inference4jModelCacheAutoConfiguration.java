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

import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelCache;

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.actuate.autoconfigure.endpoint.condition.ConditionalOnAvailableEndpoint;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Auto-configuration for the model cache: a {@link ModelCache} bean, and the {@code models} actuator endpoint over
 * it when Spring Boot Actuator is on the classpath.
 */
@AutoConfiguration
public class Inference4jModelCacheAutoConfiguration {

	/**
	 * The cache of the application's {@link HuggingFaceModelSource} bean when it defines one, otherwise the default
	 * source's cache, which the auto-configured models download into.
	 */
	@Bean
	@ConditionalOnMissingBean
	public ModelCache inference4jModelCache(ObjectProvider<HuggingFaceModelSource> sources) {
		HuggingFaceModelSource source = sources.getIfUnique();
		return source != null ? source.cache() : ModelCache.defaultCache();
	}

	@Configuration(proxyBeanMethods = false)
	@ConditionalOnClass(name = {
			"org.springframework.boot.actuate.endpoint.annotation.Endpoint",
			"org.springframework.boot.actuate.autoconfigure.endpoint.condition.ConditionalOnAvailableEndpoint" })
	@ConditionalOnProperty(prefix = "inference4j.actuator.models", name = "enabled", matchIfMissing = true)
	static class ModelsEndpointConfiguration {

		@Bean
		@ConditionalOnMissingBean
		@ConditionalOnAvailableEndpoint
		public ModelsEndpoint modelsEndpoint(ModelCache modelCache) {
			return new ModelsEndpoint(modelCache);
		}

	}

}
