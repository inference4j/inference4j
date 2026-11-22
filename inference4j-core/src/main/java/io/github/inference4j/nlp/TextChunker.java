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

import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Cuts text into chunks a model takes whole, at paragraph, sentence and, for an over-long sentence, word
 * boundaries. Works on offsets into the original text, so the whitespace between chunks is kept exactly
 * and joining every chunk with its separator gives the text back.
 */
final class TextChunker {

    /** Whitespace that contains a line break: the boundary between paragraphs (or lines). */
    private static final Pattern PARAGRAPH_BREAK = Pattern.compile("\\s*\\R\\s*");

    /** How far one chunk may reach. */
    enum Packing {
        /** As many whole sentences as fit, within one paragraph. */
        WITHIN_PARAGRAPH,
        /** As many whole sentences as fit, across paragraphs. */
        ACROSS_PARAGRAPHS,
        /** One sentence per chunk. */
        ONE_SENTENCE
    }

    /**
     * One piece of text to process.
     *
     * @param text      the chunk, without surrounding whitespace
     * @param separator the original whitespace that follows it
     */
    record Chunk(String text, String separator) {
    }

    /**
     * @param leading the whitespace before the first chunk
     * @param chunks  the chunks in order; empty for blank text
     */
    record Chunks(String leading, List<Chunk> chunks) {
    }

    /** A sentence, or a word of an over-long sentence: the smallest piece packing works with. */
    private record Atom(int start, int contentEnd, int end, boolean sentenceEnd, boolean paragraphEnd) {
    }

    private TextChunker() {
    }

    /**
     * Splits {@code text} into chunks for which {@code fits} holds. A sentence that does not fit on its own
     * is split between words; a word that does not fit on its own becomes a chunk anyway.
     *
     * @param fits tells whether a candidate chunk can be processed whole
     */
    static Chunks split(String text, Locale locale, Predicate<String> fits, Packing packing) {
        int start = 0;
        while (start < text.length() && Character.isWhitespace(text.charAt(start))) {
            start++;
        }
        int end = text.length();
        while (end > start && Character.isWhitespace(text.charAt(end - 1))) {
            end--;
        }
        String leading = text.substring(0, start);
        if (start == end) {
            return new Chunks(leading, List.of());
        }
        List<Atom> atoms = atoms(text, start, end, locale, fits);
        return new Chunks(leading, pack(text, atoms, fits, packing));
    }

    private static List<Atom> atoms(String text, int start, int end, Locale locale, Predicate<String> fits) {
        List<Atom> atoms = new ArrayList<>();
        Matcher breaks = PARAGRAPH_BREAK.matcher(text).region(start, end);
        int paragraphStart = start;
        while (true) {
            boolean found = breaks.find();
            int paragraphEnd = found ? breaks.start() : end;
            // The paragraph's last atom takes the break, or the text's trailing whitespace, as its separator
            int separatorEnd = found ? breaks.end() : text.length();
            sentences(text, paragraphStart, paragraphEnd, separatorEnd, locale, fits, atoms);
            if (!found) {
                return atoms;
            }
            paragraphStart = breaks.end();
        }
    }

    private static void sentences(String text, int start, int end, int separatorEnd, Locale locale,
                                  Predicate<String> fits, List<Atom> atoms) {
        BreakIterator sentences = BreakIterator.getSentenceInstance(locale);
        sentences.setText(text.substring(start, end));
        int from = sentences.first();
        for (int to = sentences.next(); to != BreakIterator.DONE; from = to, to = sentences.next()) {
            int sentenceStart = start + from;
            int contentEnd = trimEnd(text, sentenceStart, start + to);
            boolean last = start + to == end;
            int sentenceEnd = last ? separatorEnd : start + to;
            if (contentEnd == sentenceStart) {
                continue;
            }
            if (fits.test(text.substring(sentenceStart, contentEnd))) {
                atoms.add(new Atom(sentenceStart, contentEnd, sentenceEnd, true, last));
            } else {
                words(text, sentenceStart, contentEnd, sentenceEnd, last, locale, atoms);
            }
        }
    }

    /** Splits an over-long sentence into words; punctuation stays a separate atom, glued back by packing. */
    private static void words(String text, int start, int contentEnd, int sentenceEnd, boolean paragraphEnd,
                              Locale locale, List<Atom> atoms) {
        BreakIterator words = BreakIterator.getWordInstance(locale);
        words.setText(text.substring(start, contentEnd));
        List<int[]> spans = new ArrayList<>();
        int from = words.first();
        for (int to = words.next(); to != BreakIterator.DONE; from = to, to = words.next()) {
            int wordStart = start + from;
            int wordEnd = start + to;
            if (isBlank(text, wordStart, wordEnd)) {
                if (!spans.isEmpty()) {
                    spans.get(spans.size() - 1)[2] = wordEnd;      // the space follows the previous word
                }
            } else {
                spans.add(new int[] {wordStart, wordEnd, wordEnd});
            }
        }
        for (int i = 0; i < spans.size(); i++) {
            int[] span = spans.get(i);
            boolean last = i == spans.size() - 1;
            atoms.add(new Atom(span[0], span[1], last ? sentenceEnd : span[2], last, last && paragraphEnd));
        }
    }

    private static List<Chunk> pack(String text, List<Atom> atoms, Predicate<String> fits, Packing packing) {
        List<Chunk> chunks = new ArrayList<>();
        int i = 0;
        while (i < atoms.size()) {
            Atom first = atoms.get(i);
            Atom last = first;
            int next = i + 1;
            while (next < atoms.size() && canExtend(last, packing)
                    && fits.test(text.substring(first.start(), atoms.get(next).contentEnd()))) {
                last = atoms.get(next);
                next++;
            }
            chunks.add(new Chunk(text.substring(first.start(), last.contentEnd()),
                    text.substring(last.contentEnd(), last.end())));
            i = next;
        }
        return chunks;
    }

    private static boolean canExtend(Atom last, Packing packing) {
        return switch (packing) {
            case WITHIN_PARAGRAPH -> !last.paragraphEnd();
            case ACROSS_PARAGRAPHS -> true;
            case ONE_SENTENCE -> !last.sentenceEnd();
        };
    }

    private static int trimEnd(String text, int start, int end) {
        while (end > start && Character.isWhitespace(text.charAt(end - 1))) {
            end--;
        }
        return end;
    }

    private static boolean isBlank(String text, int start, int end) {
        for (int i = start; i < end; i++) {
            if (!Character.isWhitespace(text.charAt(i))) {
                return false;
            }
        }
        return true;
    }
}
