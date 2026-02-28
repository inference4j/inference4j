# Text-to-SQL

Generate SQL queries from natural language questions using Flan-T5.

## Quick example

```java
try (var generator = FlanT5TextGenerator.flanT5Base().build()) {
    String sql = generator.generateSql(
            "How many employees are in the engineering department?",
            "employees(id, name, department, salary)");
    System.out.println(sql); // SELECT COUNT(*) FROM employees WHERE department = 'engineering'
}
```

## Full example

```java
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.nlp.FlanT5TextGenerator;

public class TextToSql {
    public static void main(String[] args) {
        String schema = "employees(id, name, department, salary), " +
                         "departments(id, name, location)";

        try (var generator = FlanT5TextGenerator.flanT5Base()
                .maxNewTokens(200)
                .build()) {

            String[] questions = {
                "What is the average salary by department?",
                "List all employees in New York",
                "Which department has the most employees?"
            };

            for (String question : questions) {
                GenerationResult result = generator.generateSql(question, schema,
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

| Preset | Model | Parameters | Size |
|--------|-------|-----------|------|
| `FlanT5TextGenerator.flanT5Small()` | Flan-T5 Small | 77M | ~300 MB |
| `FlanT5TextGenerator.flanT5Base()` | Flan-T5 Base | 250M | ~900 MB |
| `FlanT5TextGenerator.flanT5Large()` | Flan-T5 Large | 780M | ~3 GB |

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | Preset-dependent | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.tokenizerProvider(TokenizerProvider)` | `TokenizerProvider` | `SentencePieceBpeTokenizer` | Tokenizer construction strategy |
| `.maxNewTokens(int)` | `int` | `256` | Maximum tokens to generate |
| `.temperature(float)` | `float` | `0.0` | Sampling temperature |
| `.topK(int)` | `int` | `0` (disabled) | Top-K sampling |
| `.topP(float)` | `float` | `0.0` (disabled) | Nucleus sampling |
| `.eosTokenId(int)` | `int` | Auto-detected | End-of-sequence token ID |
| `.addedToken(String)` | `String` | — | Register a special token for atomic encoding |

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

- **Flan-T5 Base** offers the best balance of quality and size for text-to-SQL. Flan-T5 Small is faster but less accurate; Flan-T5 Large is more capable but requires ~3 GB.
- Include the table schema in the second argument — the model uses it to generate accurate column names and table references.
- The model automatically prepends the instruction: `"generate SQL given the question and schema. question: ... schema: ..."`.
- Use greedy decoding (default `temperature=0`) for SQL generation — deterministic output is what you want.
- Always validate and sanitize generated SQL before executing it against a real database.
