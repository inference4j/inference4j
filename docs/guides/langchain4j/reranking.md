# Reranking

`Inference4jScoringModel` is a LangChain4j `ScoringModel` backed by an inference4j cross-encoder
([`SearchReranker`](../../use-cases/semantic-search.md)), running in-process. A cross-encoder reads the query and
each retrieved segment together and scores their relevance. A typical pipeline retrieves a larger candidate set
from the embedding store, reranks it, and passes the highest-scoring segments to the LLM.

## Quick example

```java
var scoringModel = new Inference4jScoringModel(MiniLMSearchReranker.builder().build());

Response<Double> score = scoringModel.score("Java is a programming language.", "What is Java?");
```

The model (about 90 MB) is downloaded on first use.

## In a RAG pipeline

```java
ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
        .embeddingModel(embeddingModel)
        .embeddingStore(store)
        .maxResults(20)                       // candidates to rerank
        .build();

RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
        .contentRetriever(retriever)
        .contentAggregator(ReRankingContentAggregator.builder()
                .scoringModel(scoringModel)
                .minScore(0.5)                // relevance probability threshold
                .maxResults(4)                // segments passed to the LLM
                .build())
        .build();

Assistant assistant = AiServices.builder(Assistant.class)
        .chatModel(chatModel)
        .retrievalAugmentor(augmentor)
        .build();
```

## Options

The adapter has one constructor, `Inference4jScoringModel(SearchReranker)`. Options are set on the reranker; see its [builder options](../../use-cases/semantic-search.md#reranker-builder-options).

| Reranker option | Default | Description |
|-----------------|---------|----------------------|
| `.modelId(String)` | `inference4j/ms-marco-MiniLM-L-6-v2` | Another BERT-style cross-encoder (WordPiece tokenizer, one relevance logit) |
| `.maxLength(int)` | `512` | Tokens for query and segment together |
| `.stride(int)` | off | Score segments longer than `maxLength` passage by passage and keep the best score (MaxP), instead of truncating |
| `.sessionOptions(SessionConfigurer)` | default | Thread count or [hardware acceleration](../hardware-acceleration.md) |

## Notes

- **Scores are probabilities.** `MiniLMSearchReranker` returns a relevance probability between 0 and 1, passed through
  unchanged, so `minScore` on `ReRankingContentAggregator` is a probability threshold.
- **English.** `ms-marco-MiniLM-L-6-v2` was trained on English MS MARCO passages. A multilingual reranker is on the
  [roadmap](../../roadmap.md).
- **Segment length.** The model was trained on short passages. See
  [Reranking long documents](../../use-cases/semantic-search.md#reranking-long-documents-maxp).
- **One run per segment.** Segments are scored sequentially; each run uses all cores. See
  [Choosing an embedding or scoring model](index.md#choosing-an-embedding-or-scoring-model).
- **Argument order.** LangChain4j's `score(text, query)` takes the text first.
- **Token usage is not reported.**
- **Lifecycle.** `close()` closes the reranker. See
  [Lifecycle](index.md#lifecycle).
