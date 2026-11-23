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

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(classes = DashboardReadOnlyTest.App.class, properties = "inference4j.dashboard.enabled=true")
@AutoConfigureMockMvc
class DashboardReadOnlyTest {

	private static final String MODEL = "inference4j/read-only-model";

	@TempDir
	static Path cacheDir;

	static TestModelCache models;

	@Autowired
	MockMvcTester mvc;

	@Autowired
	HuggingFaceModelSource source;

	@BeforeAll
	static void fillCache() throws Exception {
		models = new TestModelCache(cacheDir, MODEL);
	}

	@AfterAll
	static void stopHub() {
		models.close();
	}

	@Test
	void servesAtTheDefaultPath() {
		assertThat(mvc.get().uri("/inference4j/")).hasStatusOk();
		assertThat(mvc.get().uri("/inference4j/api/info")).hasStatusOk()
			.bodyJson()
			.extractingPath("$.readOnly")
			.isEqualTo(true);
		assertThat(mvc.get().uri("/inference4j/api/cache/models")).hasStatusOk();
	}

	@Test
	void writesAreForbidden() {
		assertThat(mvc.post().uri("/inference4j/api/cache/update?repo=" + MODEL)).hasStatus(403);
		assertThat(mvc.delete().uri("/inference4j/api/cache/model?repo=" + MODEL)).hasStatus(403)
			.bodyJson()
			.extractingPath("$.message")
			.asString()
			.contains("inference4j.dashboard.read-only=false");
		assertThat(mvc.post().uri("/inference4j/api/cache/prune")).hasStatus(403);

		assertThat(source.cache().find(MODEL)).isPresent();
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
