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

import io.github.inference4j.dashboard.ModelCacheService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication;
import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication.Type;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.web.server.context.WebServerInitializedEvent;
import org.springframework.context.ApplicationListener;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.function.RouterFunction;
import org.springframework.web.servlet.function.ServerResponse;

/**
 * Auto-configuration for the inference4j dashboard: a local web page over the model cache, at
 * {@code inference4j.dashboard.path} ({@code /inference4j} by default). Off unless
 * {@code inference4j.dashboard.enabled=true}, and read-only unless {@code inference4j.dashboard.read-only=false}.
 * Requires a Spring MVC web application.
 */
@AutoConfiguration(after = Inference4jModelCacheAutoConfiguration.class)
@ConditionalOnWebApplication(type = Type.SERVLET)
@ConditionalOnClass(name = "org.springframework.web.servlet.function.RouterFunction")
@ConditionalOnProperty(prefix = "inference4j.dashboard", name = "enabled", havingValue = "true")
@EnableConfigurationProperties(Inference4jProperties.class)
public class Inference4jDashboardAutoConfiguration {

	/** The dashboard's page and JSON API. */
	@Bean
	public RouterFunction<ServerResponse> inference4jDashboardRoutes(ModelCacheService modelCacheService,
			Inference4jProperties properties) {
		Inference4jProperties.DashboardProperties dashboard = properties.getDashboard();
		return DashboardRoutes.create(modelCacheService, dashboard.getPath(), dashboard.isReadOnly());
	}

	@Configuration(proxyBeanMethods = false)
	@ConditionalOnClass(name = "org.springframework.boot.web.server.context.WebServerInitializedEvent")
	static class StartupLogConfiguration {

		@Bean
		DashboardStartupLog inference4jDashboardStartupLog(Inference4jProperties properties) {
			return new DashboardStartupLog(properties.getDashboard());
		}

	}

	/** Logs the dashboard's URL once the web server is up. */
	static final class DashboardStartupLog implements ApplicationListener<WebServerInitializedEvent> {

		private static final Logger logger = LoggerFactory.getLogger(Inference4jDashboardAutoConfiguration.class);

		private final Inference4jProperties.DashboardProperties dashboard;

		DashboardStartupLog(Inference4jProperties.DashboardProperties dashboard) {
			this.dashboard = dashboard;
		}

		@Override
		public void onApplicationEvent(WebServerInitializedEvent event) {
			if ("management".equals(event.getApplicationContext().getServerNamespace())) {
				return;
			}
			String contextPath = event.getApplicationContext()
				.getEnvironment()
				.getProperty("server.servlet.context-path", "")
				.replaceAll("/+$", "");
			logger.info("inference4j dashboard ({}): http://localhost:{}{}{}/",
					dashboard.isReadOnly() ? "read-only" : "writes enabled", event.getWebServer().getPort(),
					contextPath, DashboardRoutes.normalize(dashboard.getPath()));
		}

	}

}
