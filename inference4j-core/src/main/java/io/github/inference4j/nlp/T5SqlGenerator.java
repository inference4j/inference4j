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

package io.github.inference4j.nlp;

import io.github.inference4j.generation.GenerationEngine;
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.tokenizer.TokenizerProvider;
import io.github.inference4j.tokenizer.UnigramTokenizer;

import java.util.function.BiFunction;
import java.util.function.Consumer;

/**
 * SQL generator backed by T5-based text-to-SQL encoder-decoder models.
 *
 * <p>Two presets are available out of the box:
 *
 * <ul>
 *   <li>{@link #t5SmallAwesome()} — T5-small (60M), fast and lightweight. Schema
 *       uses {@code CREATE TABLE} DDL statements.</li>
 *   <li>{@link #t5LargeSpider()} — T5-large (0.8B), highest accuracy. Schema uses
 *       the Spider format with {@code [SEP]} delimiters and quoted identifiers.</li>
 * </ul>
 *
 * <h2>Usage (T5-small)</h2>
 * <pre>{@code
 * try (var sqlGen = T5SqlGenerator.t5SmallAwesome().build()) {
 *     String sql = sqlGen.generateSql(
 *         "List the names of employees with salary above 50000",
 *         "CREATE TABLE employees (id INT, name VARCHAR, salary INT)");
 *     System.out.println(sql);
 * }
 * }</pre>
 *
 * <h2>Usage (T5-large Spider)</h2>
 * <pre>{@code
 * try (var sqlGen = T5SqlGenerator.t5LargeSpider().build()) {
 *     String sql = sqlGen.generateSql(
 *         "How many employees are in each department?",
 *         "\"employees\" \"id\" int, \"name\" varchar, \"dept_id\" int "
 *         + "[SEP] \"departments\" \"id\" int, \"name\" varchar");
 *     System.out.println(sql);
 * }
 * }</pre>
 *
 * @see TextGenerator
 * @see SqlGenerator
 * @see GenerationResult
 */
public class T5SqlGenerator implements TextGenerator, SqlGenerator {

	private final GenerationEngine engine;

	private final BiFunction<String, String, String> promptFormatter;

	T5SqlGenerator(GenerationEngine engine,
				   BiFunction<String, String, String> promptFormatter) {
		this.engine = engine;
		this.promptFormatter = promptFormatter;
	}

	/**
	 * T5-small-awesome-text-to-sql (60M parameters) preset.
	 *
	 * <p>Fast and lightweight text-to-SQL model. Expects schema as
	 * {@code CREATE TABLE} DDL statements. Downloads from
	 * {@code inference4j/t5-small-awesome-text-to-sql} on first use.
	 */
	public static Builder t5SmallAwesome() {
		return builder()
				.modelId("inference4j/t5-small-awesome-text-to-sql")
				.promptFormatter((query, schema) ->
						"tables: " + schema + " query for: " + query);
	}

	/**
	 * T5-LM-Large text2sql-spider (0.8B parameters) preset.
	 *
	 * <p>Highest accuracy text-to-SQL model, fine-tuned on Spider and Spider-Syn
	 * datasets. Expects schema in Spider format with quoted identifiers and
	 * {@code [SEP]} table delimiters. Downloads from
	 * {@code inference4j/T5-LM-Large-text2sql-spider} on first use.
	 * Requires external data files.
	 */
	public static Builder t5LargeSpider() {
		return builder()
				.modelId("inference4j/T5-LM-Large-text2sql-spider")
				.requiredFile("decoder_model.onnx_data")
				.requiredFile("encoder_model.onnx_data")
				.promptFormatter((query, schema) ->
						"Question: " + query + " Schema: " + schema);
	}

	/**
	 * Generic builder for custom T5-based text-to-SQL models.
	 *
	 * <p>Requires at minimum a {@link Builder#modelId(String) modelId} (or
	 * {@link Builder#modelSource(ModelSource) modelSource}) pointing to a directory
	 * with {@code encoder_model.onnx}, {@code decoder_model.onnx},
	 * {@code decoder_with_past_model.onnx}, and {@code config.json}.
	 *
	 * <p>Custom models must also set a {@link Builder#promptFormatter(BiFunction)
	 * promptFormatter} to specify how the query and schema are combined into a
	 * model prompt.
	 */
	public static Builder builder() {
		return new Builder();
	}

	// --- TextGenerator ---

	@Override
	public GenerationResult generate(String input) {
		return engine.generate(input);
	}

	@Override
	public GenerationResult generate(String input, Consumer<String> tokenListener) {
		return engine.generate(input, tokenListener);
	}

	// --- SqlGenerator ---

	@Override
	public GenerationResult generateSql(String query, String schema,
										  Consumer<String> tokenListener) {
		return engine.generate(promptFormatter.apply(query, schema), tokenListener);
	}

	@Override
	public void close() throws Exception {
		engine.close();
	}

	public static class Builder
			extends AbstractEncoderDecoderBuilder<T5SqlGenerator, Builder> {

		private BiFunction<String, String, String> promptFormatter;

		/**
		 * Sets the prompt formatter that combines the natural language query and
		 * schema into the model's expected input format.
		 *
		 * @param promptFormatter function taking (query, schema) and returning
		 *                        the formatted prompt string
		 * @return this builder
		 */
		public Builder promptFormatter(
				BiFunction<String, String, String> promptFormatter) {
			this.promptFormatter = promptFormatter;
			return this;
		}

		@Override
		protected TokenizerProvider defaultTokenizerProvider() {
			return UnigramTokenizer.provider();
		}

		@Override
		protected T5SqlGenerator createWrapper(GenerationEngine engine) {
			BiFunction<String, String, String> formatter = this.promptFormatter;
			if (formatter == null) {
				formatter = (query, schema) ->
						"tables: " + schema + " query for: " + query;
			}
			return new T5SqlGenerator(engine, formatter);
		}
	}
}
