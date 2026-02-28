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

import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.generation.GenerationEngine;
import io.github.inference4j.generation.GenerationResult;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.time.Duration;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class T5SqlGeneratorTest {

	private static final GenerationResult DUMMY_RESULT =
			new GenerationResult("SELECT count(*) FROM employees", 10, 5, Duration.ZERO);

	private GenerationEngine mockEngine() {
		GenerationEngine engine = mock(GenerationEngine.class);
		when(engine.generate(anyString(), any())).thenReturn(DUMMY_RESULT);
		when(engine.generate(anyString())).thenReturn(DUMMY_RESULT);
		return engine;
	}

	@Test
	void generateSql_smallFormat_prependsTablesAndQueryFor() {
		GenerationEngine engine = mockEngine();
		T5SqlGenerator generator = new T5SqlGenerator(engine,
				(query, schema) -> "tables: " + schema + " query for: " + query);

		generator.generateSql(
				"How many employees?",
				"CREATE TABLE employees (id INT, name VARCHAR)",
				token -> {});

		ArgumentCaptor<String> promptCaptor = ArgumentCaptor.forClass(String.class);
		verify(engine).generate(promptCaptor.capture(), any());
		assertEquals(
				"tables: CREATE TABLE employees (id INT, name VARCHAR) "
				+ "query for: How many employees?",
				promptCaptor.getValue());
	}

	@Test
	void generateSql_largeFormat_prependsQuestionAndSchema() {
		GenerationEngine engine = mockEngine();
		T5SqlGenerator generator = new T5SqlGenerator(engine,
				(query, schema) -> "Question: " + query + " Schema: " + schema);

		generator.generateSql(
				"How many employees?",
				"\"employees\" \"id\" int, \"name\" varchar",
				token -> {});

		ArgumentCaptor<String> promptCaptor = ArgumentCaptor.forClass(String.class);
		verify(engine).generate(promptCaptor.capture(), any());
		assertEquals(
				"Question: How many employees? Schema: \"employees\" \"id\" int, \"name\" varchar",
				promptCaptor.getValue());
	}

	@Test
	void generateSql_blocking_returnsText() {
		GenerationEngine engine = mockEngine();
		T5SqlGenerator generator = new T5SqlGenerator(engine,
				(query, schema) -> "tables: " + schema + " query for: " + query);

		String sql = generator.generateSql(
				"How many employees?",
				"CREATE TABLE employees (id INT)");

		assertEquals("SELECT count(*) FROM employees", sql);
	}

	@Test
	void generateSql_withStreaming_delegatesToEngine() {
		GenerationEngine engine = mockEngine();
		T5SqlGenerator generator = new T5SqlGenerator(engine,
				(query, schema) -> "tables: " + schema + " query for: " + query);
		Consumer<String> listener = token -> {};

		GenerationResult result = generator.generateSql(
				"List all departments",
				"CREATE TABLE departments (id INT, name VARCHAR)",
				listener);

		verify(engine).generate(anyString(), eq(listener));
		assertNotNull(result);
		assertEquals("SELECT count(*) FROM employees", result.text());
	}

	@Test
	void generate_delegatesToEngine() {
		GenerationEngine engine = mockEngine();
		T5SqlGenerator generator = new T5SqlGenerator(engine,
				(query, schema) -> query);

		GenerationResult result = generator.generate("raw prompt");

		verify(engine).generate("raw prompt");
		assertEquals("SELECT count(*) FROM employees", result.text());
	}

	@Test
	void t5SmallAwesome_preset_returnsBuilder() {
		T5SqlGenerator.Builder builder = T5SqlGenerator.t5SmallAwesome();
		assertNotNull(builder);
	}

	@Test
	void t5LargeSpider_preset_returnsBuilder() {
		T5SqlGenerator.Builder builder = T5SqlGenerator.t5LargeSpider();
		assertNotNull(builder);
	}

	@Test
	void builder_noModelIdOrSource_throws() {
		ModelLoadException ex = assertThrows(ModelLoadException.class, () ->
				T5SqlGenerator.builder().build());
		assertTrue(ex.getMessage().contains("modelId"));
		assertTrue(ex.getMessage().contains("modelSource"));
	}
}
