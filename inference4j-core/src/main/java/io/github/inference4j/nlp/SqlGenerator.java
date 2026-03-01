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

import io.github.inference4j.generation.GenerationResult;

import java.util.function.Consumer;

/**
 * Generates SQL queries from natural language questions and database schema descriptions.
 *
 * <p>Example usage:
 * <pre>{@code
 * try (var sqlGen = T5SqlGenerator.t5SmallAwesome().build()) {
 *     String sql = sqlGen.generateSql(
 *         "Who are the top 5 highest paid employees?",
 *         "CREATE TABLE employees (id INT, name VARCHAR, salary INT)");
 *     System.out.println(sql);
 * }
 * }</pre>
 *
 * @see T5SqlGenerator
 */
public interface SqlGenerator extends AutoCloseable {

    /**
     * Generates a SQL query from a natural language question and schema, blocking until complete.
     *
     * @param query  the natural language question
     * @param schema the database schema description
     * @return the generated SQL query
     */
    default String generateSql(String query, String schema) {
        return generateSql(query, schema, token -> {}).text();
    }

    /**
     * Generates a SQL query from a natural language question and schema, streaming tokens.
     *
     * @param query         the natural language question
     * @param schema        the database schema description
     * @param tokenListener receives each decoded text fragment as it is generated
     * @return the complete generation result
     */
    GenerationResult generateSql(String query, String schema, Consumer<String> tokenListener);
}
