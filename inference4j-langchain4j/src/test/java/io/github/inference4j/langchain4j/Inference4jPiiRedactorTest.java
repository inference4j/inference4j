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

package io.github.inference4j.langchain4j;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import io.github.inference4j.nlp.NamedEntity;
import io.github.inference4j.nlp.NamedEntityRecognizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static io.github.inference4j.langchain4j.Inference4jPiiRedactor.REDACTED_COUNT;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class Inference4jPiiRedactorTest {

    /** Finds every occurrence of the configured phrases, with their label and score. */
    private static final class PhraseRecognizer implements NamedEntityRecognizer {

        private record Phrase(String text, String label, float score) {
        }

        private final List<Phrase> phrases = new ArrayList<>();
        private final List<NamedEntity> extra = new ArrayList<>();
        boolean closed;

        PhraseRecognizer find(String text, String label) {
            return find(text, label, 0.9f);
        }

        PhraseRecognizer find(String text, String label, float score) {
            phrases.add(new Phrase(text, label, score));
            return this;
        }

        @Override
        public List<NamedEntity> recognize(String text) {
            List<NamedEntity> found = new ArrayList<>(extra);
            for (Phrase phrase : phrases) {
                for (int at = text.indexOf(phrase.text()); at >= 0; at = text.indexOf(phrase.text(), at + 1)) {
                    found.add(new NamedEntity(phrase.text(), phrase.label(), at, at + phrase.text().length(),
                            phrase.score()));
                }
            }
            return found;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static Inference4jPiiRedactor redactor(NamedEntityRecognizer recognizer) {
        return Inference4jPiiRedactor.builder().recognizer(recognizer).build();
    }

    // --- replacement ---

    @Test
    void replacesEachEntityWithItsLabelPlaceholder() {
        var recognizer = new PhraseRecognizer()
                .find("marie@example.com", "email")
                .find("555-0100", "phone number");

        String redacted = redactor(recognizer).redact("Mail marie@example.com or call 555-0100 today.");

        assertThat(redacted).isEqualTo("Mail [EMAIL] or call [PHONE_NUMBER] today.");
    }

    @Test
    void offsetsStayRightWhenPlaceholdersChangeLength() {
        var recognizer = new PhraseRecognizer()
                .find("Marie", "person")
                .find("4111 1111 1111 1111", "credit card number")
                .find("10.0.0.1", "ip address");

        String redacted = redactor(recognizer).redact("Marie paid with 4111 1111 1111 1111 from 10.0.0.1");

        assertThat(redacted).isEqualTo("[PERSON] paid with [CREDIT_CARD_NUMBER] from [IP_ADDRESS]");
    }

    @Test
    void textWithoutEntitiesIsUnchanged() {
        assertThat(redactor(new PhraseRecognizer()).redact("Nothing personal here.")).isEqualTo("Nothing personal here.");
    }

    @Test
    void entitiesOutsideTheTextAreIgnored() {
        var recognizer = new PhraseRecognizer();
        recognizer.extra.add(new NamedEntity("ghost", "person", 5, 50, 0.9f));
        recognizer.extra.add(new NamedEntity("empty", "person", 3, 3, 0.9f));

        assertThat(redactor(recognizer).redact("short text")).isEqualTo("short text");
    }

    // --- merging ---

    @Test
    void overlappingSpansBecomeOnePlaceholderWithTheBestLabel() {
        var recognizer = new PhraseRecognizer()
                .find("12 Main Street", "address", 0.8f)
                .find("Main Street, Springfield", "location", 0.6f);

        String redacted = redactor(recognizer).redact("Lives at 12 Main Street, Springfield now.");

        assertThat(redacted).isEqualTo("Lives at [ADDRESS] now.");
    }

    @Test
    void touchingSpansBecomeOnePlaceholder() {
        var recognizer = new PhraseRecognizer()
                .find("marie", "person", 0.6f)
                .find("@example.com", "email", 0.9f);

        assertThat(redactor(recognizer).redact("Write to marie@example.com.")).isEqualTo("Write to [EMAIL].");
    }

    @Test
    void sameLabelSpansSeparatedByWhitespaceBecomeOnePlaceholder() {
        var recognizer = new PhraseRecognizer().find("Marie", "person").find("Curie", "person");

        assertThat(redactor(recognizer).redact("Ask Marie  Curie about it.")).isEqualTo("Ask [PERSON] about it.");
    }

    @Test
    void sameLabelSpansSeparatedByWordsStaySeparate() {
        var recognizer = new PhraseRecognizer().find("Alice", "person").find("Bob", "person");

        assertThat(redactor(recognizer).redact("Alice and Bob met.")).isEqualTo("[PERSON] and [PERSON] met.");
    }

    @Test
    void differentLabelsSeparatedByWhitespaceStaySeparate() {
        var recognizer = new PhraseRecognizer().find("Marie", "person").find("555-0100", "phone number");

        assertThat(redactor(recognizer).redact("Marie 555-0100")).isEqualTo("[PERSON] [PHONE_NUMBER]");
    }

    // --- configuration ---

    @Test
    void labelsRestrictWhatIsRedactedIgnoringCase() {
        var recognizer = new PhraseRecognizer()
                .find("Marie", "person")
                .find("marie@example.com", "email");
        var redactor = Inference4jPiiRedactor.builder().recognizer(recognizer).labels(Set.of("EMAIL")).build();

        assertThat(redactor.redact("Marie: marie@example.com")).isEqualTo("Marie: [EMAIL]");
    }

    @Test
    void minScoreDropsLowConfidenceEntities() {
        var recognizer = new PhraseRecognizer()
                .find("Marie", "person", 0.4f)
                .find("marie@example.com", "email", 0.9f);
        var redactor = Inference4jPiiRedactor.builder().recognizer(recognizer).minScore(0.5f).build();

        assertThat(redactor.redact("Marie: marie@example.com")).isEqualTo("Marie: [EMAIL]");
    }

    @Test
    void placeholderIsConfigurable() {
        var recognizer = new PhraseRecognizer().find("Marie", "person");
        var redactor = Inference4jPiiRedactor.builder()
                .recognizer(recognizer)
                .placeholder(entity -> "<" + entity.label() + ">")
                .build();

        assertThat(redactor.redact("Hi Marie")).isEqualTo("Hi <person>");
    }

    // --- documents, segments and metadata ---

    @Test
    void transformKeepsMetadataAndCountsRedactionsPerLabel() {
        var recognizer = new PhraseRecognizer()
                .find("a@example.com", "email")
                .find("b@example.com", "email")
                .find("555-0100", "phone number");
        Document document = Document.from("a@example.com, b@example.com, 555-0100",
                Metadata.from(Document.FILE_NAME, "ticket.txt"));

        Document redacted = redactor(recognizer).transform(document);

        assertThat(redacted.text()).isEqualTo("[EMAIL], [EMAIL], [PHONE_NUMBER]");
        assertThat(redacted.metadata().getString(Document.FILE_NAME)).isEqualTo("ticket.txt");
        assertThat(redacted.metadata().getInteger(REDACTED_COUNT)).isEqualTo(3);
        assertThat(redacted.metadata().getInteger("pii_redacted_email")).isEqualTo(2);
        assertThat(redacted.metadata().getInteger("pii_redacted_phone_number")).isEqualTo(1);
        assertThat(document.metadata().containsKey(REDACTED_COUNT)).as("input metadata untouched").isFalse();
    }

    @Test
    void mergedSpansCountOnce() {
        var recognizer = new PhraseRecognizer().find("Marie", "person").find("Curie", "person");

        Document redacted = redactor(recognizer).transform(Document.from("Marie Curie"));

        assertThat(redacted.metadata().getInteger(REDACTED_COUNT)).isEqualTo(1);
        assertThat(redacted.metadata().getInteger("pii_redacted_person")).isEqualTo(1);
    }

    @Test
    void nothingFoundGivesAZeroCountAndNoPerLabelKeys() {
        Document redacted = redactor(new PhraseRecognizer()).transform(Document.from("Nothing personal."));

        assertThat(redacted.metadata().toMap()).containsOnly(java.util.Map.entry(REDACTED_COUNT, 0));
    }

    @Test
    void segmentTransformerRedactsSegmentsAndKeepsTheirMetadata() {
        var recognizer = new PhraseRecognizer().find("marie@example.com", "email");
        TextSegment segment = TextSegment.from("Mail marie@example.com", Metadata.from("index", "4"));

        List<TextSegment> redacted = redactor(recognizer).asTextSegmentTransformer().transformAll(List.of(segment));

        assertThat(redacted).singleElement().satisfies(s -> {
            assertThat(s.text()).isEqualTo("Mail [EMAIL]");
            assertThat(s.metadata().getString("index")).isEqualTo("4");
            assertThat(s.metadata().getInteger("pii_redacted_email")).isEqualTo(1);
        });
    }

    @Test
    void ingestorStoresOnlyRedactedText() {
        var recognizer = new PhraseRecognizer().find("marie@example.com", "email");
        var store = new InMemoryEmbeddingStore<TextSegment>();
        EmbeddingModel embeddingModel = new EmbeddingModel() {
            @Override
            public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
                return Response.from(segments.stream().map(s -> Embedding.from(new float[]{1f, 0f})).toList());
            }
        };

        EmbeddingStoreIngestor.builder()
                .documentTransformer(redactor(recognizer))
                .embeddingModel(embeddingModel)
                .embeddingStore(store)
                .build()
                .ingest(Document.from("Contact marie@example.com for access."));

        var matches = store.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(Embedding.from(new float[]{1f, 0f}))
                .build()).matches();
        assertThat(matches).singleElement()
                .satisfies(match -> assertThat(match.embedded().text()).isEqualTo("Contact [EMAIL] for access."));
    }

    // --- lifecycle ---

    @Test
    void aRecognizerIsRequired() {
        assertThatThrownBy(() -> Inference4jPiiRedactor.builder().build())
                .isInstanceOf(NullPointerException.class)
                .hasMessageContaining("recognizer");
    }

    @Test
    void closeClosesTheRecognizer() {
        var recognizer = new PhraseRecognizer();

        redactor(recognizer).close();

        assertThat(recognizer.closed).isTrue();
    }
}
