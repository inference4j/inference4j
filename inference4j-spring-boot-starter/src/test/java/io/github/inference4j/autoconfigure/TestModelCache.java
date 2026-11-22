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

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Set;

import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.PruneMode;
import io.github.inference4j.model.ResolvedModel;

/** A model cache in a temporary directory, filled from a {@link FakeHub}, for the endpoint's web tests. */
final class TestModelCache implements AutoCloseable {

	static final List<String> FILES = List.of("model.onnx");

	final Path cacheDir;

	final FakeHub hub;

	/** Serves {@code repos}, each with one model file, and caches them. */
	TestModelCache(Path cacheDir, String... repos) throws IOException {
		this.cacheDir = cacheDir;
		this.hub = new FakeHub(repos[0], Set.of("model.onnx"));
		for (int i = 1; i < repos.length; i++) {
			hub.addRepo(repos[i], Set.of("model.onnx"));
		}
		for (String repo : repos) {
			hub.put(repo, "model.onnx", "weights of " + repo + " v1");
		}
		HuggingFaceModelSource seeding = source();
		for (String repo : repos) {
			try (ResolvedModel model = seeding.open(repo, FILES)) {
				// downloaded
			}
		}
	}

	/** A source on the cache: online against the fake Hub, keeping superseded snapshots until pruned. */
	HuggingFaceModelSource source() {
		return HuggingFaceModelSource.builder()
			.cacheDir(cacheDir)
			.endpoint(hub.endpoint())
			.offline(false)
			.pruneMode(PruneMode.MANUAL)
			.minimumSnapshotAge(Duration.ZERO)
			.build();
	}

	/** Changes the repo's model file on the Hub; a new commit. */
	void publishNewVersion(String repo, String content) {
		hub.put(repo, "model.onnx", content);
	}

	@Override
	public void close() {
		hub.close();
	}

}
