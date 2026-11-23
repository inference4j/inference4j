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

import java.net.URI;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;

import io.github.inference4j.dashboard.CacheError;
import io.github.inference4j.dashboard.ModelCacheService;
import io.github.inference4j.dashboard.UpdateStatus;

import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.http.CacheControl;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.servlet.function.RouterFunction;
import org.springframework.web.servlet.function.RouterFunctions;
import org.springframework.web.servlet.function.ServerRequest;
import org.springframework.web.servlet.function.ServerResponse;

/**
 * The dashboard's routes in a Spring MVC application: its static page from {@code inference4j-core}, and the JSON
 * API the page calls, over a {@link ModelCacheService}.
 *
 * <ul>
 *   <li>{@code GET <path>/api/info}: whether the dashboard is read-only.</li>
 *   <li>{@code GET <path>/api/cache}: the cache summary, with the disk usage of its volume.</li>
 *   <li>{@code GET <path>/api/cache/models}: every cached repo.</li>
 *   <li>{@code GET <path>/api/cache/model?repo=[&check=true]}: one cached repo, after a Hub check when asked.</li>
 *   <li>{@code POST <path>/api/cache/update?repo=}: starts an update in the background; 202.</li>
 *   <li>{@code DELETE <path>/api/cache/model?repo=[&revision=]}: evicts the repo, or one snapshot.</li>
 *   <li>{@code POST <path>/api/cache/prune}: prunes superseded snapshots.</li>
 * </ul>
 *
 * <p>When read-only, the write operations answer 403.
 */
final class DashboardRoutes {

	/** Where {@code inference4j-core} keeps the dashboard's page, styles and scripts. */
	static final String STATIC_LOCATION = "io/github/inference4j/dashboard/static/";

	private final ModelCacheService service;

	private final boolean readOnly;

	private DashboardRoutes(ModelCacheService service, boolean readOnly) {
		this.service = service;
		this.readOnly = readOnly;
	}

	/** The routes for a dashboard at {@code path}. */
	static RouterFunction<ServerResponse> create(ModelCacheService service, String path, boolean readOnly) {
		DashboardRoutes routes = new DashboardRoutes(service, readOnly);
		String base = normalize(path);
		String api = base + "/api";
		Resource index = new ClassPathResource(STATIC_LOCATION + "index.html");
		// Relative, so it also works under a servlet context path
		URI slash = URI.create(base.substring(base.lastIndexOf('/') + 1) + "/");
		return RouterFunctions.route()
			.GET(base, request -> ServerResponse.temporaryRedirect(slash).build())
			.GET(base + "/", request -> ServerResponse.ok()
				.contentType(MediaType.TEXT_HTML)
				.cacheControl(CacheControl.noCache())
				.body(index))
			.GET(api + "/info", request -> ServerResponse.ok().body(Map.of("readOnly", readOnly)))
			.GET(api + "/cache", request -> routes.read(routes.service::summary))
			.GET(api + "/cache/models", request -> routes.read(routes.service::models))
			.GET(api + "/cache/model", routes::model)
			.POST(api + "/cache/update", routes::update)
			.DELETE(api + "/cache/model", routes::evict)
			.POST(api + "/cache/prune", request -> routes.write(() -> ServerResponse.ok().body(routes.service.prune())))
			.add(RouterFunctions.resources(base + "/**", new ClassPathResource(STATIC_LOCATION),
					DashboardRoutes::noCache))
			.build();
	}

	/** {@code path} with a leading slash and no trailing one. */
	static String normalize(String path) {
		String trimmed = path == null ? "" : path.trim();
		while (trimmed.endsWith("/")) {
			trimmed = trimmed.substring(0, trimmed.length() - 1);
		}
		if (trimmed.isEmpty()) {
			throw new IllegalArgumentException("inference4j.dashboard.path must not be empty or '/'");
		}
		return trimmed.startsWith("/") ? trimmed : "/" + trimmed;
	}

	private ServerResponse model(ServerRequest request) {
		return handle(() -> {
			String repo = repo(request);
			boolean check = request.param("check").map(Boolean::parseBoolean).orElse(false);
			return found(check ? service.check(repo) : service.model(repo), repo);
		});
	}

	private ServerResponse update(ServerRequest request) {
		return write(() -> {
			String repo = repo(request);
			Optional<UpdateStatus> status = service.startUpdate(repo);
			if (status.isEmpty()) {
				return notCached(repo);
			}
			return ServerResponse.accepted()
				.body(Map.of("repoId", repo, "status", ModelsEndpoint.statusName(status.get())));
		});
	}

	private ServerResponse evict(ServerRequest request) {
		return write(() -> {
			String repo = repo(request);
			Optional<String> revision = request.param("revision");
			if (revision.isEmpty()) {
				return found(service.evict(repo), repo);
			}
			if (service.model(repo).isEmpty()) {
				return notCached(repo);
			}
			return service.evict(repo, revision.get())
				.map(result -> ServerResponse.ok().body(result))
				.orElseGet(() -> error(404, repo + " has no cached snapshot at revision " + revision.get()));
		});
	}

	private ServerResponse read(Supplier<Object> body) {
		return handle(() -> ServerResponse.ok().body(body.get()));
	}

	private ServerResponse write(Supplier<ServerResponse> action) {
		if (readOnly) {
			return error(403, "The dashboard is read-only: set inference4j.dashboard.read-only=false to allow changes");
		}
		return handle(action);
	}

	private ServerResponse handle(Supplier<ServerResponse> action) {
		try {
			return action.get();
		}
		catch (MissingParameterException ex) {
			return error(400, ex.getMessage());
		}
		catch (RuntimeException ex) {
			CacheError error = service.error(ex);
			return error(error.status(), error.message());
		}
	}

	private static ServerResponse found(Optional<?> result, String repo) {
		return result.<ServerResponse>map(body -> ServerResponse.ok().body(body)).orElseGet(() -> notCached(repo));
	}

	private static String repo(ServerRequest request) {
		return request.param("repo")
			.filter(repo -> !repo.isBlank())
			.orElseThrow(() -> new MissingParameterException("The 'repo' parameter is required"));
	}

	private static ServerResponse notCached(String repo) {
		return error(404, repo + " is not cached");
	}

	private static ServerResponse error(int status, String message) {
		return ServerResponse.status(status).body(Map.of("status", status, "message", message));
	}

	private static void noCache(Resource resource, HttpHeaders headers) {
		headers.setCacheControl(CacheControl.noCache());
	}

	/** A required request parameter is missing. */
	private static final class MissingParameterException extends RuntimeException {

		MissingParameterException(String message) {
			super(message);
		}

	}

}
