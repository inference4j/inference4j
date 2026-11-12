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

package io.github.inference4j.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Holds UTF-8 bytes until they form complete characters. Backs the {@link TokenStream}s of the
 * tokenizers. Not thread-safe; each stream owns one.
 */
final class Utf8Accumulator {

    private final ByteArrayOutputStream pending = new ByteArrayOutputStream();

    void add(int b) {
        pending.write(b);
    }

    void add(byte[] bytes, int length) {
        pending.write(bytes, 0, length);
    }

    boolean isEmpty() {
        return pending.size() == 0;
    }

    /** Returns the complete characters held, keeping an unfinished character at the end. */
    String takeComplete() {
        byte[] bytes = pending.toByteArray();
        int cut = completeLength(bytes);
        pending.reset();
        pending.write(bytes, cut, bytes.length - cut);
        return new String(bytes, 0, cut, StandardCharsets.UTF_8);
    }

    /** Returns everything held; an unfinished character becomes U+FFFD. */
    String takeAll() {
        String text = pending.toString(StandardCharsets.UTF_8);
        pending.reset();
        return text;
    }

    /** Length of the prefix that does not end inside a multi-byte character. */
    static int completeLength(byte[] bytes) {
        int n = bytes.length;
        for (int i = n - 1; i >= Math.max(0, n - 4); i--) {
            int b = bytes[i] & 0xFF;
            if ((b & 0xC0) == 0x80) {
                continue;                       // continuation byte: keep looking for the lead byte
            }
            int length = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
            return i + length > n ? i : n;
        }
        return n;                               // stray continuation bytes: let the decoder replace them
    }
}
