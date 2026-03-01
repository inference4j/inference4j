# Text-to-SQL

Generate SQL queries from natural language questions using T5-based text-to-SQL models.

## Quick example

```java
try (var sqlGen = T5SqlGenerator.t5SmallAwesome().build()) {
    String sql = sqlGen.generateSql(
            "How many employees are in the engineering department?",
            "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)");
    System.out.println(sql);
    // SELECT COUNT(*) FROM employees AS T1 JOIN departments AS T2
    //   ON T1.department = T2.id WHERE T2.name = 'Engineer'
}
```

## Full example

```java
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.nlp.T5SqlGenerator;

public class TextToSql {
    public static void main(String[] args) {
        String schema = "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT); " +
                         "CREATE TABLE departments (id INT, name VARCHAR, location VARCHAR)";

        try (var sqlGen = T5SqlGenerator.t5SmallAwesome()
                .maxNewTokens(200)
                .build()) {

            String[] questions = {
                "What is the average salary by department?",
                "List all employees in New York",
                "Which department has the most employees?"
            };

            for (String question : questions) {
                GenerationResult result = sqlGen.generateSql(question, schema,
                        token -> System.out.print(token));
                System.out.println();
                System.out.printf("  → %d tokens in %,d ms%n",
                        result.generatedTokens(), result.duration().toMillis());
            }
        }
    }
}
```

## Model presets

| Preset | Model | Parameters | Size | Schema format |
|--------|-------|-----------|------|---------------|
| `T5SqlGenerator.t5SmallAwesome()` | T5-small-awesome-text-to-sql | 60M | ~240 MB | `CREATE TABLE` DDL |
| `T5SqlGenerator.t5LargeSpider()` | T5-LM-Large-text2sql-spider | 0.8B | ~4.6 GB | Spider format with `[SEP]` |

### Choosing a preset

**T5-small-awesome** is recommended for most use cases. It's fast, lightweight, and handles
standard SQL patterns well. Schema is provided as familiar `CREATE TABLE` statements.

**T5-large-spider** produces higher accuracy on complex queries (JOINs, subqueries,
GROUP BY with HAVING). It uses a specialized schema format designed for the
[Spider benchmark](https://yale-lily.github.io/spider) with quoted identifiers and
`[SEP]` table delimiters — ideal for integration with JDBC metadata.

## Schema formats

### T5-small-awesome (CREATE TABLE DDL)

```java
String schema = "CREATE TABLE employees (id INT, name VARCHAR, salary INT); "
              + "CREATE TABLE departments (id INT, name VARCHAR)";
sqlGen.generateSql("What is the average salary?", schema);
```

### T5-large-spider (Spider format)

```java
String schema = "\"employees\" \"id\" int, \"name\" varchar, \"salary\" int, "
              + "foreign_key: primary key: \"id\" "
              + "[SEP] "
              + "\"departments\" \"id\" int, \"name\" varchar, "
              + "foreign_key: primary key: \"id\"";
sqlGen.generateSql("What is the average salary?", schema);
```

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | Preset-dependent | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.tokenizerProvider(TokenizerProvider)` | `TokenizerProvider` | `UnigramTokenizer` | Tokenizer construction strategy |
| `.promptFormatter(BiFunction)` | `BiFunction<String, String, String>` | Preset-dependent | Combines (query, schema) into model prompt |
| `.maxNewTokens(int)` | `int` | `256` | Maximum tokens to generate |
| `.temperature(float)` | `float` | `0.0` | Sampling temperature |
| `.topK(int)` | `int` | `0` (disabled) | Top-K sampling |
| `.topP(float)` | `float` | `0.0` (disabled) | Nucleus sampling |
| `.eosTokenId(int)` | `int` | Auto-detected | End-of-sequence token ID |

## Result type

`GenerationResult` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `text()` | `String` | The generated SQL query |
| `promptTokens()` | `int` | Number of tokens in the input |
| `generatedTokens()` | `int` | Number of tokens generated |
| `duration()` | `Duration` | Wall-clock generation time |

The convenience method `generateSql(query, schema)` returns the SQL as a plain `String`.

## Tips

- Use greedy decoding (default `temperature=0`) for SQL generation — deterministic output is what you want.
- Always validate and sanitize generated SQL before executing it against a real database.
- Include all relevant tables in the schema, even if the query only touches one — the model uses the full schema to resolve column references.
- For the T5-large-spider model, the schema format can be generated programmatically from JDBC `DatabaseMetaData`.
