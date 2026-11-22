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
import java.util.List;
import java.util.Map;

import io.github.inference4j.InferenceTask;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ResolvedModel;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.assertj.MockMvcTester;
import org.springframework.test.web.servlet.assertj.MvcTestResult;

import static org.assertj.core.api.Assertions.assertThat;

/** The models endpoint with its default, read-only access. */
@SpringBootTest(classes = ModelsEndpointReadOnlyTest.App.class, properties = {
		"management.endpoints.web.exposure.include=models,health",
		"management.endpoint.health.show-details=always", "management.endpoint.health.show-components=always",
		// Every health call lists the cache afresh, so each assertion sees the latest state
		"inference4j.health.model-cache-ttl=0s" })
@AutoConfigureMockMvc
class ModelsEndpointReadOnlyTest {

	private static final String MODEL = "inference4j/test-model";

	private static final String CHECKED = "inference4j/checked-model";

	private static final String HUB_DOWN = "inference4j/hub-down-model";

	private static final String STALE = "inference4j/stale-model";

	@TempDir
	static Path cacheDir;

	static TestModelCache models;

	@Autowired
	MockMvcTester mvc;

	@Autowired
	HuggingFaceModelSource source;

	@BeforeAll
	static void fillCache() throws Exception {
		models = new TestModelCache(cacheDir, MODEL, CHECKED, HUB_DOWN, STALE);
	}

	@AfterAll
	static void stopHub() {
		models.close();
	}

	@Test
	void listsTheCachedModels() throws Exception {
		MvcTestResult result = mvc.get().uri("/actuator/models").exchange();

		assertThat(result).hasStatusOk();
		assertThat(result).bodyJson().extractingPath("$.models[*].repoId").asArray().contains(MODEL, CHECKED);
		assertThat(result).bodyJson()
			.extractingPath("$.models[?(@.repoId == '" + MODEL + "')].snapshots[0].files[0].path")
			.asArray()
			.containsExactly("model.onnx");
		assertThat(result.getResponse().getContentAsString()).doesNotContain(cacheDir.toString());
	}

	@Test
	void findsAModelByItsIdWithASlash() throws Exception {
		MvcTestResult result = mvc.get().uri("/actuator/models/" + MODEL).exchange();

		assertThat(result).hasStatusOk();
		assertThat(result).bodyJson().extractingPath("$.repoId").isEqualTo(MODEL);
		assertThat(result).bodyJson().extractingPath("$.revision").isEqualTo(models.hub.commit(MODEL));
		assertThat(result).bodyJson().extractingPath("$.stale").isEqualTo(false);
		assertThat(result).bodyJson().extractingPath("$.updating").isEqualTo(false);
		assertThat(result).bodyJson().doesNotHavePath("$.check");
		assertThat(result.getResponse().getContentAsString()).doesNotContain(cacheDir.toString());
	}

	@Test
	void aModelNotCachedIsNotFound() {
		assertThat(mvc.get().uri("/actuator/models/inference4j/missing")).hasStatus(404)
			.bodyJson()
			.extractingPath("$.message")
			.isEqualTo("inference4j/missing is not cached");
	}

	@Test
	void anInvalidRepoIdIsABadRequest() {
		assertThat(mvc.get().uri("/actuator/models/inference4j/too/deep")).hasStatus(400)
			.bodyJson()
			.extractingPath("$.message")
			.isEqualTo("Invalid Hub repo id: 'inference4j/too/deep'");
	}

	@Test
	void checkAsksTheHubAndReusesARecentAnswer() {
		models.publishNewVersion(CHECKED, "weights v2");
		long before = models.hub.revisionChecks(CHECKED);

		assertThat(mvc.get().uri("/actuator/models/" + CHECKED + "?check=true")).hasStatusOk()
			.bodyJson()
			.satisfies(json -> {
				assertThat(json).extractingPath("$.check.remoteRevision").isEqualTo(models.hub.commit(CHECKED));
				assertThat(json).extractingPath("$.check.changedFiles").asArray().containsExactly("model.onnx");
				assertThat(json).extractingPath("$.check.updateAvailable").isEqualTo(true);
				assertThat(json).extractingPath("$.updateAvailable").isEqualTo(true);
			});
		assertThat(mvc.get().uri("/actuator/models/" + CHECKED + "?check=true")).hasStatusOk()
			.bodyJson()
			.extractingPath("$.check.updateAvailable")
			.isEqualTo(true);

		assertThat(models.hub.revisionChecks(CHECKED)).isEqualTo(before + 1);
	}

	@Test
	void checkOfAModelNotCachedNeverReachesTheHub() {
		assertThat(mvc.get().uri("/actuator/models/inference4j/not-cached?check=true")).hasStatus(404);

		assertThat(models.hub.revisionChecks("inference4j/not-cached")).isZero();
	}

	@Test
	void aHubFailureIsABadGateway() {
		models.hub.revisionStatus = 500;
		try {
			assertThat(mvc.get().uri("/actuator/models/" + HUB_DOWN + "?check=true")).hasStatus(502)
				.bodyJson()
				.extractingPath("$.message")
				.asString()
				.contains("the Hub could not be reached");
		}
		finally {
			models.hub.revisionStatus = null;
		}
	}

	@Test
	void writeOperationsAreNotMappedByDefault() {
		assertThat(mvc.post()
			.uri("/actuator/models/" + MODEL)
			.contentType(MediaType.APPLICATION_JSON)
			.content("{\"action\":\"update\"}")).hasStatus(405);
		assertThat(mvc.delete().uri("/actuator/models/" + MODEL)).hasStatus(405);
		assertThat(mvc.delete().uri("/actuator/models")).hasStatus(405);

		assertThat(source.cache().find(MODEL)).isPresent();
	}

	@Test
	void healthReportsAStaleModelAndStaysUp() {
		try (ResolvedModel loaded = source.open(STALE, TestModelCache.FILES)) {
			String loadedRevision = loaded.revision().orElseThrow().commit();
			models.publishNewVersion(STALE, "weights v2");
			source.cache().update(STALE);

			assertThat(mvc.get().uri("/actuator/health/inference4j")).hasStatusOk()
				.bodyJson()
				.satisfies(json -> {
					assertThat(json).extractingPath("$.status").isEqualTo("UP");
					assertThat(json).extractingPath("$.details.tasks").isEqualTo(1);
					assertThat(json).extractingPath("$.details.staleModels[?(@.repo == '" + STALE + "')]")
						.asArray()
						.singleElement()
						.isEqualTo(Map.of("repo", STALE, "loadedRevisions",
								List.of(loadedRevision), "cachedRevision", models.hub.commit(STALE)));
					assertThat(json).extractingPath("$.details.updatesAvailable").asArray().doesNotContain(STALE);
				});
		}
		assertThat(mvc.get().uri("/actuator/health/inference4j")).hasStatusOk()
			.bodyJson()
			.extractingPath("$.details.staleModels[*].repo")
			.asArray()
			.doesNotContain(STALE);
	}

	@SpringBootConfiguration
	@EnableAutoConfiguration
	static class App {

		@Bean
		HuggingFaceModelSource modelSource() {
			return models.source();
		}

		@Bean
		InferenceTask<String, String> echo() {
			return new InferenceTask<>() {
				@Override
				public String run(String input) {
					return input;
				}

				@Override
				public void close() {
				}
			};
		}

	}

}
