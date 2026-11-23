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
import org.springframework.test.web.servlet.assertj.MockMvcTester;
import org.springframework.test.web.servlet.assertj.MvcTestResult;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(classes = DashboardWebTest.App.class,
		properties = { "inference4j.dashboard.enabled=true", "inference4j.dashboard.read-only=false",
				"inference4j.dashboard.path=/console/" })
@AutoConfigureMockMvc
class DashboardWebTest {

	private static final String MODEL = "inference4j/dashboard-model";

	private static final String UPDATED = "inference4j/dashboard-updated";

	private static final String EVICTED = "inference4j/dashboard-evicted";

	private static final String PRUNED = "inference4j/dashboard-pruned";

	@TempDir
	static Path cacheDir;

	static TestModelCache models;

	@Autowired
	MockMvcTester mvc;

	@Autowired
	HuggingFaceModelSource source;

	@BeforeAll
	static void fillCache() throws Exception {
		models = new TestModelCache(cacheDir, MODEL, UPDATED, EVICTED, PRUNED);
	}

	@AfterAll
	static void stopHub() {
		models.close();
	}

	@Test
	void servesThePageAndItsAssets() throws Exception {
		assertThat(mvc.get().uri("/console")).hasStatus(307).hasHeader("Location", "console/");

		MvcTestResult page = mvc.get().uri("/console/").exchange();
		assertThat(page).hasStatusOk().hasContentTypeCompatibleWith("text/html");
		assertThat(page.getResponse().getContentAsString()).contains("x-data=\"dashboard\"");

		assertThat(mvc.get().uri("/console/dashboard.js")).hasStatusOk();
		assertThat(mvc.get().uri("/console/vendor/alpine-3.17.5.min.js")).hasStatusOk();
		assertThat(mvc.get().uri("/console/vendor/chartist-1.5.0.min.js")).hasStatusOk();
	}

	@Test
	void infoSaysWritesAreEnabled() {
		assertThat(mvc.get().uri("/console/api/info")).hasStatusOk()
			.bodyJson()
			.extractingPath("$.readOnly")
			.isEqualTo(false);
	}

	@Test
	void summaryHasTheCacheAndTheDisk() throws Exception {
		MvcTestResult result = mvc.get().uri("/console/api/cache").exchange();

		assertThat(result).hasStatusOk().bodyJson().satisfies(json -> {
			assertThat(json).extractingPath("$.models").asNumber().satisfies(n -> assertThat(n.intValue()).isPositive());
			assertThat(json).extractingPath("$.sizeBytes").asNumber().satisfies(n -> assertThat(n.longValue()).isPositive());
			assertThat(json).extractingPath("$.disk.totalBytes").asNumber()
				.satisfies(n -> assertThat(n.longValue()).isPositive());
		});
		assertThat(result.getResponse().getContentAsString()).doesNotContain(cacheDir.toString());
	}

	@Test
	void listsAndFindsModels() {
		assertThat(mvc.get().uri("/console/api/cache/models")).hasStatusOk()
			.bodyJson()
			.extractingPath("$[*].repoId")
			.asArray()
			.contains(MODEL);
		assertThat(mvc.get().uri("/console/api/cache/model?repo=" + MODEL)).hasStatusOk()
			.bodyJson()
			.extractingPath("$.revision")
			.isEqualTo(models.hub.commit(MODEL));
		assertThat(mvc.get().uri("/console/api/cache/model?repo=inference4j/missing")).hasStatus(404);
		assertThat(mvc.get().uri("/console/api/cache/model")).hasStatus(400);
		assertThat(mvc.get().uri("/console/api/cache/model?repo=inference4j/too/deep")).hasStatus(400);
	}

	@Test
	void checkAsksTheHub() {
		assertThat(mvc.get().uri("/console/api/cache/model?check=true&repo=" + MODEL)).hasStatusOk()
			.bodyJson()
			.satisfies(json -> {
				assertThat(json).extractingPath("$.check.remoteRevision").isEqualTo(models.hub.commit(MODEL));
				assertThat(json).extractingPath("$.check.updateAvailable").isEqualTo(false);
			});
	}

	@Test
	void updateRunsInTheBackground() throws Exception {
		models.publishNewVersion(UPDATED, "weights v2");
		String latest = models.hub.commit(UPDATED);

		assertThat(mvc.post().uri("/console/api/cache/update?repo=" + UPDATED)).hasStatus(202)
			.bodyJson()
			.extractingPath("$.status")
			.isIn("accepted", "already-running");

		Instant deadline = Instant.now().plus(Duration.ofSeconds(10));
		while (!latest.equals(revision(UPDATED)) && Instant.now().isBefore(deadline)) {
			Thread.sleep(20);
		}
		assertThat(revision(UPDATED)).isEqualTo(latest);
	}

	@Test
	void evictDeletesTheModel() {
		assertThat(mvc.delete().uri("/console/api/cache/model?repo=" + EVICTED + "&revision=0123abcd"))
			.hasStatus(404);
		assertThat(mvc.delete().uri("/console/api/cache/model?repo=" + EVICTED)).hasStatusOk()
			.bodyJson()
			.extractingPath("$.removed[0].repoId")
			.isEqualTo(EVICTED);

		assertThat(source.cache().find(EVICTED)).isEmpty();
	}

	@Test
	void pruneDeletesSupersededSnapshots() {
		String old = revision(PRUNED);
		models.publishNewVersion(PRUNED, "weights v2");
		source.cache().update(PRUNED);

		assertThat(mvc.post().uri("/console/api/cache/prune")).hasStatusOk()
			.bodyJson()
			.extractingPath("$.removed[?(@.repoId == '" + PRUNED + "')].commit")
			.asArray()
			.containsExactly(old);
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
