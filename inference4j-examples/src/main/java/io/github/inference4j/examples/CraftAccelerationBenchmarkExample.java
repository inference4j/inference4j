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

package io.github.inference4j.examples;

import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.vision.CraftTextDetector;
import io.github.inference4j.vision.TextRegion;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Benchmarks CRAFT text detection across execution providers: CPU, CoreML (macOS), CUDA.
 *
 * <p>Run with:
 * <pre>
 * ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.CraftAccelerationBenchmarkExample
 * </pre>
 */
public class CraftAccelerationBenchmarkExample {

	private static final Path IMAGE_PATH;

	static {
		try {
			IMAGE_PATH = Path.of(CraftAccelerationBenchmarkExample.class.getResource("/fixtures/product_placement.jpg").toURI());
		}
		catch (java.net.URISyntaxException e) {
			throw new ExceptionInInitializerError(e);
		}
	}
	private static final int WARMUP_RUNS = 3;
	private static final int TIMED_RUNS = 10;

	public static void main(String[] args) throws IOException {
		System.out.println("=== CRAFT Text Detection — Execution Provider Benchmark ===");
		System.out.println("Image: " + IMAGE_PATH);
		System.out.printf("Warmup runs: %d, Timed runs: %d%n", WARMUP_RUNS, TIMED_RUNS);

		long cpuAvgMs = runProvider("CPU", null);

		long coremlAvgMs = runProvider("CoreML", opts -> opts.addCoreML());

		long cudaAvgMs = runProvider("CUDA", opts -> opts.addCUDA(0));

		// --- Summary ---
		System.out.println("\n=== Summary ===");
		printRow("CPU", cpuAvgMs);
		printRow("CoreML", coremlAvgMs);
		printRow("CUDA", cudaAvgMs);

		if (cpuAvgMs > 0) {
			if (coremlAvgMs > 0) {
				System.out.printf("  CoreML speedup: %.2fx%n", (double) cpuAvgMs / coremlAvgMs);
			}
			if (cudaAvgMs > 0) {
				System.out.printf("  CUDA speedup:   %.2fx%n", (double) cpuAvgMs / cudaAvgMs);
			}
		}
	}

	private static long runProvider(String name, SessionConfigurer configurer) throws IOException {
		System.out.printf("%n--- %s ---%n", name);
		try {
			CraftTextDetector.Builder builder = CraftTextDetector.builder();
			if (configurer != null) {
				builder.sessionOptions(configurer);
			}

			long loadStart = System.nanoTime();
			try (CraftTextDetector craft = builder.build()) {
				long loadMs = (System.nanoTime() - loadStart) / 1_000_000;
				System.out.printf("Model loaded in %d ms%n", loadMs);

				long avgMs = benchmark(craft);
				System.out.printf("Average inference: %d ms%n", avgMs);
				return avgMs;
			}
		}
		catch (Exception ex) {
			System.out.printf("%s not available: %s%n", name, ex.getMessage());
			return -1;
		}
	}

	private static long benchmark(CraftTextDetector craft) throws IOException {
		for (int i = 0; i < WARMUP_RUNS; i++) {
			List<TextRegion> regions = craft.detect(IMAGE_PATH, 0.4f, 0.3f);
			System.out.printf("  warmup %d: %d regions detected%n", i + 1, regions.size());
		}

		long totalNanos = 0;
		for (int i = 0; i < TIMED_RUNS; i++) {
			long start = System.nanoTime();
			List<TextRegion> regions = craft.detect(IMAGE_PATH, 0.4f, 0.3f);
			long elapsed = System.nanoTime() - start;
			totalNanos += elapsed;
			System.out.printf("  run %d: %d ms — %d regions%n", i + 1, elapsed / 1_000_000,
					regions.size());
		}

		return (totalNanos / TIMED_RUNS) / 1_000_000;
	}

	private static void printRow(String name, long avgMs) {
		if (avgMs > 0) {
			System.out.printf("  %-8s  Avg inference: %5d ms%n", name, avgMs);
		}
		else {
			System.out.printf("  %-8s  (not available)%n", name);
		}
	}

}
