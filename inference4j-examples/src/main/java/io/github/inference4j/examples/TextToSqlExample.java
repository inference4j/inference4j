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

import io.github.inference4j.nlp.T5SqlGenerator;

/**
 * Generates SQL queries from natural language using T5-small-awesome-text-to-sql.
 */
public class TextToSqlExample {

	private static final String SCHEMA =
			"CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT); "
			+ "CREATE TABLE departments (id INT, name VARCHAR, location VARCHAR)";

	public static void main(String[] args) throws Exception {
		try (var sqlGen = T5SqlGenerator.t5SmallAwesome().build()) {
			String[] questions = {
				"How many employees are in the engineering department?",
				"What is the average salary by department?",
				"List all employees in New York",
				"Which department has the most employees?"
			};

			System.out.println("Schema: " + SCHEMA);
			System.out.println();

			for (String question : questions) {
				System.out.println("Q: " + question);
				String sql = sqlGen.generateSql(question, SCHEMA);
				System.out.println("SQL: " + sql);
				System.out.println();
			}
		}
	}
}
