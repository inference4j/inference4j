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
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import io.github.inference4j.exception.ModelDownloadException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.CachedModel;
import io.github.inference4j.model.CachedSnapshot;
import io.github.inference4j.model.EvictionResult;
import io.github.inference4j.model.ModelCache;
import io.github.inference4j.model.ModelRevision;
import io.github.inference4j.model.UpdateCheck;
import org.jspecify.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.beans.factory.DisposableBean;
import org.springframework.boot.actuate.endpoint.Access;
import org.springframework.boot.actuate.endpoint.OperationResponseBody;
import org.springframework.boot.actuate.endpoint.annotation.DeleteOperation;
import org.springframework.boot.actuate.endpoint.annotation.ReadOperation;
import org.springframework.boot.actuate.endpoint.annotation.Selector;
import org.springframework.boot.actuate.endpoint.annotation.Selector.Match;
import org.springframework.boot.actuate.endpoint.annotation.WriteOperation;
import org.springframework.boot.actuate.endpoint.web.WebEndpointResponse;
import org.springframework.boot.actuate.endpoint.web.annotation.WebEndpoint;

/**
 * The {@code models} actuator endpoint: lists, checks, updates and evicts the models in a {@link ModelCache}.
 *
 * <ul>
 *   <li>{@code GET /actuator/models}: every cached repo, from local state.</li>
 *   <li>{@code GET /actuator/models/{org}/{name}}: one cached repo; {@code ?check=true} also asks the Hub whether
 *   it changed, reusing a check younger than a minute.</li>
 *   <li>{@code POST /actuator/models/{org}/{name}} with {@code {"action":"update"}}: starts an update in the
 *   background and answers 202.</li>
 *   <li>{@code DELETE /actuator/models/{org}/{name}[?revision=]}: evicts the repo, or one snapshot.</li>
 *   <li>{@code DELETE /actuator/models}: prunes superseded snapshots of every repo.</li>
 * </ul>
 *
 * <p>Read-only by default: the write and delete operations are mapped only with
 * {@code management.endpoint.models.access=unrestricted}. Responses never contain absolute paths.
 */
@WebEndpoint(id = "models", defaultAccess = Access.READ_ONLY)
public class ModelsEndpoint implements DisposableBean {

	private static final Logger logger = LoggerFactory.getLogger(ModelsEndpoint.class);

	/** How long a check's answer is reused, so repeated requests do not call the Hub each time. */
	private static final Duration CHECK_REUSE = Duration.ofMinutes(1);

	private static final String UPDATE = "update";

	private final ModelCache cache;

	private final ExecutorService updates;

	private final Map<String, UpdateCheck> checks = new ConcurrentHashMap<>();

	private final Set<String> pendingUpdates = ConcurrentHashMap.newKeySet();

	/**
	 * Creates the endpoint over {@code cache}. Updates run one at a time on a thread the endpoint owns, stopped
	 * when the endpoint is destroyed.
	 */
	public ModelsEndpoint(ModelCache cache) {
		this.cache = Objects.requireNonNull(cache, "cache");
		this.updates = Executors.newSingleThreadExecutor(runnable -> {
			Thread thread = new Thread(runnable, "inference4j-model-updates");
			thread.setDaemon(true);
			return thread;
		});
	}

	/** Every cached repo, read from local state only. */
	@ReadOperation
	public ModelsDescriptor models() {
		return new ModelsDescriptor(cache.list().stream().map(model -> describe(model, null)).toList());
	}

	/**
	 * One cached repo; 404 when it is not cached. With {@code check}, also asks the Hub whether it changed.
	 */
	@ReadOperation
	public WebEndpointResponse<Object> model(@Selector(match = Match.ALL_REMAINING) String[] repo,
			@Nullable Boolean check) {
		String repoId = String.join("/", repo);
		try {
			Optional<CachedModel> found = cache.find(repoId);
			if (found.isEmpty()) {
				return notCached(repoId);
			}
			CachedModel model = found.get();
			if (!Boolean.TRUE.equals(check)) {
				return new WebEndpointResponse<>(describe(model, null));
			}
			UpdateCheck result = check(model);
			// The check records its answer, so read the repo again
			model = cache.find(repoId).orElse(model);
			return new WebEndpointResponse<>(describe(model, result));
		}
		catch (RuntimeException ex) {
			return failure(repoId, "check", ex);
		}
	}

	/**
	 * Runs an action on a cached repo. {@code update} brings it up to date with the Hub in the background and
	 * answers 202; its progress shows as {@code updating} on the repo.
	 */
	@WriteOperation
	public WebEndpointResponse<Object> run(@Selector(match = Match.ALL_REMAINING) String[] repo, String action) {
		String repoId = String.join("/", repo);
		if (!UPDATE.equals(action)) {
			return error(WebEndpointResponse.STATUS_BAD_REQUEST,
					"Unknown action '" + action + "': expected '" + UPDATE + "'");
		}
		try {
			Optional<CachedModel> found = cache.find(repoId);
			if (found.isEmpty()) {
				return notCached(repoId);
			}
			String id = found.get().repoId();
			if (!pendingUpdates.add(id)) {
				return new WebEndpointResponse<>(new ActionDescriptor(id, UPDATE, "already-running"), 202);
			}
			try {
				updates.execute(() -> update(id));
			}
			catch (RejectedExecutionException ex) {
				pendingUpdates.remove(id);
				return error(WebEndpointResponse.STATUS_SERVICE_UNAVAILABLE, "The endpoint is shutting down");
			}
			return new WebEndpointResponse<>(new ActionDescriptor(id, UPDATE, "accepted"), 202);
		}
		catch (RuntimeException ex) {
			return failure(repoId, UPDATE, ex);
		}
	}

	/** Deletes a repo's snapshots that are not in use, or only {@code revision}; 404 when it is not cached. */
	@DeleteOperation
	public WebEndpointResponse<Object> evict(@Selector(match = Match.ALL_REMAINING) String[] repo,
			@Nullable String revision) {
		String repoId = String.join("/", repo);
		try {
			Optional<CachedModel> found = cache.find(repoId);
			if (found.isEmpty()) {
				return notCached(repoId);
			}
			if (revision == null) {
				return new WebEndpointResponse<>(EvictionDescriptor.of(cache.evict(repoId)));
			}
			boolean cached = found.get().snapshots().stream().anyMatch(s -> s.revision().equals(revision));
			// Rejects an invalid revision (400); a revision not cached removes nothing (404)
			EvictionResult result = cache.evict(repoId, revision);
			if (!cached) {
				return error(WebEndpointResponse.STATUS_NOT_FOUND,
						repoId + " has no cached snapshot at revision " + revision);
			}
			return new WebEndpointResponse<>(EvictionDescriptor.of(result));
		}
		catch (RuntimeException ex) {
			return failure(repoId, "evict", ex);
		}
	}

	/** Deletes every repo's superseded snapshots that are not in use. */
	@DeleteOperation
	public WebEndpointResponse<Object> prune() {
		try {
			return new WebEndpointResponse<>(EvictionDescriptor.of(cache.prune()));
		}
		catch (RuntimeException ex) {
			return failure(null, "prune", ex);
		}
	}

	@Override
	public void destroy() throws InterruptedException {
		updates.shutdown();
		if (!updates.awaitTermination(5, TimeUnit.SECONDS)) {
			updates.shutdownNow();
		}
	}

	private UpdateCheck check(CachedModel model) {
		UpdateCheck last = checks.get(model.repoId());
		if (last != null && Objects.equals(last.cachedRevision(), model.revision())
				&& last.checkedAt().isAfter(Instant.now().minus(CHECK_REUSE))) {
			return last;
		}
		UpdateCheck result = cache.check(model.repoId());
		checks.put(model.repoId(), result);
		return result;
	}

	private void update(String repoId) {
		try {
			CachedModel model = cache.update(repoId);
			logger.info("Updated {} to revision {}", repoId, model.revision());
		}
		catch (RuntimeException ex) {
			logger.warn("Update of {} failed: {}", repoId, safeMessage(ex));
		}
		finally {
			checks.remove(repoId);
			pendingUpdates.remove(repoId);
		}
	}

	private ModelDescriptor describe(CachedModel model, @Nullable UpdateCheck check) {
		return new ModelDescriptor(model.repoId(), model.revision(), model.lastChecked(), model.remoteRevision(),
				model.updateAvailable(), model.stale(), pendingUpdates.contains(model.repoId()), model.snapshots(),
				model.sizeBytes(), check != null ? CheckDescriptor.of(check) : null);
	}

	private static WebEndpointResponse<Object> notCached(String repoId) {
		return error(WebEndpointResponse.STATUS_NOT_FOUND, repoId + " is not cached");
	}

	/**
	 * Maps a failure to a status: 400 for an invalid repo id or revision, 502 when the Hub fails, 409 when the
	 * source cannot do it in its current state (offline, or the repo evicted meanwhile).
	 */
	private WebEndpointResponse<Object> failure(@Nullable String repoId, String operation, RuntimeException ex) {
		if (ex instanceof IllegalArgumentException) {
			return error(WebEndpointResponse.STATUS_BAD_REQUEST, safeMessage(ex));
		}
		if (ex instanceof ModelDownloadException) {
			logger.warn("Cannot {} {}: {}", operation, repoId, safeMessage(ex));
			return error(502, safeMessage(ex));
		}
		if (ex instanceof ModelSourceException) {
			return error(409, safeMessage(ex));
		}
		throw ex;
	}

	/** The exception's message, with the cache directory left out. */
	private String safeMessage(RuntimeException ex) {
		String message = Objects.requireNonNullElse(ex.getMessage(), ex.getClass().getSimpleName());
		String directory = cache.directory().toString();
		return directory.isEmpty() ? message : message.replace(directory, "<cache>");
	}

	private static WebEndpointResponse<Object> error(int status, String message) {
		return new WebEndpointResponse<>(new ErrorDescriptor(status, message), status);
	}

	/** The cached repos. */
	public record ModelsDescriptor(List<ModelDescriptor> models) implements OperationResponseBody {
	}

	/**
	 * A cached repo: {@link CachedModel}'s components, plus whether it is {@linkplain CachedModel#stale() stale},
	 * whether an update started here is still running, and the Hub check when one was asked for.
	 */
	public record ModelDescriptor(String repoId, @Nullable String revision, @Nullable Instant lastChecked,
			@Nullable String remoteRevision, boolean updateAvailable, boolean stale, boolean updating,
			List<CachedSnapshot> snapshots, long sizeBytes, @Nullable CheckDescriptor check)
			implements OperationResponseBody {
	}

	/** {@link UpdateCheck}'s components, plus whether an update would download anything. */
	public record CheckDescriptor(String cachedRevision, String remoteRevision, List<String> changedFiles,
			Instant checkedAt, boolean updateAvailable) {

		static CheckDescriptor of(UpdateCheck check) {
			return new CheckDescriptor(check.cachedRevision(), check.remoteRevision(), check.changedFiles(),
					check.checkedAt(), check.updateAvailable());
		}

	}

	/** {@link EvictionResult}'s components. */
	public record EvictionDescriptor(List<ModelRevision> removed, List<ModelRevision> inUse, long freedBytes)
			implements OperationResponseBody {

		static EvictionDescriptor of(EvictionResult result) {
			return new EvictionDescriptor(result.removed(), result.inUse(), result.freedBytes());
		}

	}

	/** The answer to an action started in the background. */
	public record ActionDescriptor(String repoId, String action, String status) implements OperationResponseBody {
	}

	/** Why a request failed. */
	public record ErrorDescriptor(int status, String message) implements OperationResponseBody {
	}

}
