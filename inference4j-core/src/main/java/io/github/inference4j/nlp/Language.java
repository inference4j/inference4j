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

/**
 * Supported languages for translation and other NLP tasks.
 *
 * <p>Constants use ISO 639-1 codes. {@link #isoCode()} returns the lowercase,
 * hyphenated form (e.g. {@code "pt-br"} for {@link #PT_BR}), and
 * {@link #displayName()} returns the human-readable name.
 */
public enum Language {

    // Western European
    EN("English"),
    FR("French"),
    DE("German"),
    ES("Spanish"),
    PT("Portuguese"),
    PT_BR("Brazilian Portuguese"),
    IT("Italian"),
    NL("Dutch"),
    CA("Catalan"),

    // Northern European
    SV("Swedish"),
    DA("Danish"),
    NO("Norwegian"),
    FI("Finnish"),

    // Eastern European
    PL("Polish"),
    CS("Czech"),
    HR("Croatian"),
    RO("Romanian"),

    // Other Latin-script
    TR("Turkish"),

    // Non-Latin
    JA("Japanese"),
    KO("Korean"),
    AR("Arabic"),
    ZH_CN("Chinese Simplified"),
    ZH_TW("Chinese Traditional"),
    HI("Hindi");

    private final String displayName;

    private final String isoCode;

    Language(String displayName) {
        this.displayName = displayName;
        this.isoCode = name().toLowerCase().replace('_', '-');
    }

    /**
     * Returns the human-readable name of this language (e.g. "Brazilian Portuguese").
     */
    public String displayName() {
        return this.displayName;
    }

    /**
     * Returns the ISO 639-1 code in lowercase with hyphens (e.g. "pt-br").
     */
    public String isoCode() {
        return this.isoCode;
    }

}
