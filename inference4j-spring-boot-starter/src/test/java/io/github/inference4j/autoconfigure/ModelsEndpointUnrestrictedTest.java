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
import java.time.Duration;
import java.time.Instant;

import io.github.inference4j.model.CachedModel;
import io.github.inference4j.model.HuggingFaceModelSource;
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
import org.springframework.test.web.servlet.assertj.MockMvcTester.MockMvcRequestBuilder;

import static org.assertj.core.api.Assertions.assertThat;

/** The models endpoint with {@code management.endpoint.models.access=unrestricted}. */
@SpringBootTest(classes = ModelsEndpointUnrestrictedTest.App.class,
		properties = { "management.endpoints.web.exposure.include=models",
				"management.endpoint.models.access=unrestricted" })
@AutoConfigureMockMvc
class ModelsEndpointUnrestrictedTest {

	private static final String UPDATED = "inference4j/updated-model";

	private static final String EVICTED = "inference4j/evicted-model";

	private static final String PRUNED = "inference4j/pruned-model";

	private static final String KEPT = "inference4j/kept-model";

	@TempDir
	static Path cacheDir;

	static TestModelCache models;

	@Autowired
	MockMvcTester mvc;

	@Autowired
	HuggingFaceModelSource source;

	@BeforeAll
	static void fillCache() throws Exception {
		models = new TestModelCache(cacheDir, UPDATED, EVICTED, PRUNED, KEPT);
	}

	@AfterAll
	static void stopHub() {
		models.close();
	}

	@Test
	void updateRunsInTheBackgroundAndCompletes() throws Exception {
		models.publishNewVersion(UPDATED, "weights v2");
		String latest = models.hub.commit(UPDATED);

		assertThat(action(UPDATED, "update")).hasStatus(202)
			.bodyJson()
			.satisfies(json -> {
				assertThat(json).extractingPath("$.repoId").isEqualTo(UPDATED);
				assertThat(json).extractingPath("$.action").isEqualTo("update");
				assertThat(json).extractingPath("$.status").isIn("accepted", "already-running");
			});

		Instant deadline = Instant.now().plus(Duration.ofSeconds(10));
		while (!latest.equals(revision(UPDATED)) && Instant.now().isBefore(deadline)) {
			Thread.sleep(20);
		}
		assertThat(revision(UPDATED)).isEqualTo(latest);
		assertThat(models.hub.downloads(UPDATED, "model.onnx")).isEqualTo(2);
	}

	@Test
	void anUnknownActionIsABadRequest() {
		assertThat(action(KEPT, "refresh")).hasStatus(400)
			.bodyJson()
			.extractingPath("$.message")
			.isEqualTo("Unknown action 'refresh': expected 'update'");
	}

	@Test
	void aMissingActionIsABadRequest() {
		assertThat(mvc.post().uri("/actuator/models/" + KEPT).contentType(MediaType.APPLICATION_JSON).content("{}"))
			.hasStatus(400);
	}

	@Test
	void updatingAModelNotCachedIsNotFound() {
		assertThat(action("inference4j/missing", "update")).hasStatus(404);

		assertThat(models.hub.revisionChecks("inference4j/missing")).isZero();
	}

	@Test
	void anInvalidRepoIdIsABadRequest() {
		assertThat(action("inference4j/too/deep", "update")).hasStatus(400);
		assertThat(mvc.delete().uri("/actuator/models/inference4j/too/deep")).hasStatus(400);
	}

	@Test
	void evictDeletesTheModel() {
		String revision = revision(EVICTED);

		assertThat(mvc.delete().uri("/actuator/models/" + EVICTED)).hasStatusOk()
			.bodyJson()
			.satisfies(json -> {
				assertThat(json).extractingPath("$.removed[0].repoId").isEqualTo(EVICTED);
				assertThat(json).extractingPath("$.removed[0].commit").isEqualTo(revision);
				assertThat(json).extractingPath("$.inUse").asArray().isEmpty();
			});

		assertThat(source.cache().find(EVICTED)).isEmpty();
		assertThat(mvc.get().uri("/actuator/models/" + EVICTED)).hasStatus(404);
	}

	@Test
	void evictOfARevisionNotCachedIsNotFound() {
		assertThat(mvc.delete().uri("/actuator/models/" + KEPT + "?revision=0123abcd")).hasStatus(404);
		assertThat(mvc.delete().uri("/actuator/models/" + KEPT + "?revision=not..valid")).hasStatus(400);

		assertThat(source.cache().find(KEPT)).isPresent();
	}

	@Test
	void pruneDeletesSupersededSnapshots() {
		String old = revision(PRUNED);
		models.publishNewVersion(PRUNED, "weights v2");
		source.cache().update(PRUNED);
		assertThat(source.cache().find(PRUNED).orElseThrow().snapshots()).hasSize(2);

		assertThat(mvc.delete().uri("/actuator/models")).hasStatusOk()
			.bodyJson()
			.extractingPath("$.removed[?(@.repoId == '" + PRUNED + "')].commit")
			.asArray()
			.containsExactly(old);

		CachedModel pruned = source.cache().find(PRUNED).orElseThrow();
		assertThat(pruned.snapshots()).hasSize(1);
		assertThat(pruned.revision()).isEqualTo(models.hub.commit(PRUNED));
	}

	private MockMvcRequestBuilder action(String repo, String action) {
		return mvc.post()
			.uri("/actuator/models/" + repo)
			.contentType(MediaType.APPLICATION_JSON)
			.content("{\"action\":\"" + action + "\"}");
	}

	private String revision(String repo) {
		return source.cache().find(repo).map(CachedModel::revision).orElse(null);
	}

	@SpringBootConfiguration
	@EnableAutoConfiguration
	static class App {

		@Bean
		HuggingFaceModelSource modelSource() {
			return models.source();
		}

	}

}
