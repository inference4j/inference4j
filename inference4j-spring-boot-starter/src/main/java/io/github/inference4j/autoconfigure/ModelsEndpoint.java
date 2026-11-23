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

import java.time.Instant;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;

import io.github.inference4j.dashboard.CacheError;
import io.github.inference4j.dashboard.ModelCacheService;
import io.github.inference4j.dashboard.ModelCheck;
import io.github.inference4j.dashboard.ModelView;
import io.github.inference4j.dashboard.UpdateStatus;
import io.github.inference4j.model.CachedSnapshot;
import io.github.inference4j.model.EvictionResult;
import io.github.inference4j.model.ModelRevision;
import org.jspecify.annotations.Nullable;

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
 * <p>The operations run on a {@link ModelCacheService}, shared with the dashboard. Read-only by default: the write and delete operations are mapped only with
 * {@code management.endpoint.models.access=unrestricted}. Responses never contain absolute paths.
 */
@WebEndpoint(id = "models", defaultAccess = Access.READ_ONLY)
public class ModelsEndpoint {

	private static final String UPDATE = "update";

	private final ModelCacheService service;

	/** Creates the endpoint over {@code service}, which runs the updates and owns their thread. */
	public ModelsEndpoint(ModelCacheService service) {
		this.service = Objects.requireNonNull(service, "service");
	}

	/** Every cached repo, read from local state only. */
	@ReadOperation
	public ModelsDescriptor models() {
		return new ModelsDescriptor(service.models().stream().map(ModelDescriptor::of).toList());
	}

	/**
	 * One cached repo; 404 when it is not cached. With {@code check}, also asks the Hub whether it changed.
	 */
	@ReadOperation
	public WebEndpointResponse<Object> model(@Selector(match = Match.ALL_REMAINING) String[] repo,
			@Nullable Boolean check) {
		String repoId = String.join("/", repo);
		try {
			Optional<ModelView> found = Boolean.TRUE.equals(check) ? service.check(repoId) : service.model(repoId);
			return found.<WebEndpointResponse<Object>>map(m -> new WebEndpointResponse<>(ModelDescriptor.of(m)))
				.orElseGet(() -> notCached(repoId));
		}
		catch (RuntimeException ex) {
			return failure(ex);
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
			Optional<UpdateStatus> status = service.startUpdate(repoId);
			if (status.isEmpty()) {
				return notCached(repoId);
			}
			String id = service.model(repoId).map(ModelView::repoId).orElse(repoId);
			return new WebEndpointResponse<>(new ActionDescriptor(id, UPDATE, statusName(status.get())), 202);
		}
		catch (RuntimeException ex) {
			return failure(ex);
		}
	}

	/** Deletes a repo's snapshots that are not in use, or only {@code revision}; 404 when it is not cached. */
	@DeleteOperation
	public WebEndpointResponse<Object> evict(@Selector(match = Match.ALL_REMAINING) String[] repo,
			@Nullable String revision) {
		String repoId = String.join("/", repo);
		try {
			if (revision == null) {
				return service.evict(repoId)
					.<WebEndpointResponse<Object>>map(r -> new WebEndpointResponse<>(EvictionDescriptor.of(r)))
					.orElseGet(() -> notCached(repoId));
			}
			if (service.model(repoId).isEmpty()) {
				return notCached(repoId);
			}
			return service.evict(repoId, revision)
				.<WebEndpointResponse<Object>>map(r -> new WebEndpointResponse<>(EvictionDescriptor.of(r)))
				.orElseGet(() -> error(WebEndpointResponse.STATUS_NOT_FOUND,
						repoId + " has no cached snapshot at revision " + revision));
		}
		catch (RuntimeException ex) {
			return failure(ex);
		}
	}

	/** Deletes every repo's superseded snapshots that are not in use. */
	@DeleteOperation
	public WebEndpointResponse<Object> prune() {
		try {
			return new WebEndpointResponse<>(EvictionDescriptor.of(service.prune()));
		}
		catch (RuntimeException ex) {
			return failure(ex);
		}
	}

	/** The JSON name of an update status: {@code accepted} or {@code already-running}. */
	static String statusName(UpdateStatus status) {
		return status.name().toLowerCase(Locale.ROOT).replace('_', '-');
	}

	private static WebEndpointResponse<Object> notCached(String repoId) {
		return error(WebEndpointResponse.STATUS_NOT_FOUND, repoId + " is not cached");
	}

	private WebEndpointResponse<Object> failure(RuntimeException ex) {
		CacheError error = service.error(ex);
		return error(error.status(), error.message());
	}

	private static WebEndpointResponse<Object> error(int status, String message) {
		return new WebEndpointResponse<>(new ErrorDescriptor(status, message), status);
	}

	/** The cached repos. */
	public record ModelsDescriptor(List<ModelDescriptor> models) implements OperationResponseBody {
	}

	/** A cached repo: {@link ModelView}'s components. */
	public record ModelDescriptor(String repoId, @Nullable String revision, @Nullable Instant lastChecked,
			@Nullable String remoteRevision, boolean updateAvailable, boolean stale, boolean updating,
			List<CachedSnapshot> snapshots, long sizeBytes, @Nullable ModelCheck check)
			implements OperationResponseBody {

		static ModelDescriptor of(ModelView model) {
			return new ModelDescriptor(model.repoId(), model.revision(), model.lastChecked(), model.remoteRevision(),
					model.updateAvailable(), model.stale(), model.updating(), model.snapshots(), model.sizeBytes(),
					model.check());
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
