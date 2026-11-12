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
/**
 * LangChain4j adapters for inference4j: run inference4j models in-process behind LangChain4j's
 * interfaces ({@code EmbeddingModel}, {@code ScoringModel}, {@code DocumentParser},
 * {@code DocumentTransformer}, {@code InputGuardrail}).
 *
 * <h2>Rules every adapter follows</h2>
 * <ul>
 *   <li><b>Adapters own their model.</b> Closing an adapter closes the inference4j model it wraps.
 *       Give each adapter its own model instance; do not wrap one instance in two adapters.</li>
 *   <li><b>Safe to share across threads.</b> Adapters delegate straight to inference4j models, which are
 *       thread-safe, so one adapter instance can serve a whole application. Generators and
 *       {@code MoonshineRecognizer} run one call at a time; other models run calls in parallel.</li>
 * </ul>
 *
 * <p>LangChain4j is a {@code compileOnly} dependency: applications bring their own LangChain4j
 * (built and tested against 1.23).
 */
package io.github.inference4j.langchain4j;
