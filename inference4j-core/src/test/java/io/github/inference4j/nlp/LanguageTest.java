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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class LanguageTest {

    @Test
    void isoCode_simpleConstant_returnsLowercase() {
        assertThat(Language.EN.isoCode()).isEqualTo("en");
        assertThat(Language.FR.isoCode()).isEqualTo("fr");
        assertThat(Language.DE.isoCode()).isEqualTo("de");
    }

    @Test
    void isoCode_underscoreConstant_returnsHyphenated() {
        assertThat(Language.PT_BR.isoCode()).isEqualTo("pt-br");
        assertThat(Language.ZH_CN.isoCode()).isEqualTo("zh-cn");
        assertThat(Language.ZH_TW.isoCode()).isEqualTo("zh-tw");
    }

    @Test
    void displayName_returnsReadableName() {
        assertThat(Language.EN.displayName()).isEqualTo("English");
        assertThat(Language.PT_BR.displayName()).isEqualTo("Brazilian Portuguese");
        assertThat(Language.ZH_CN.displayName()).isEqualTo("Chinese Simplified");
        assertThat(Language.JA.displayName()).isEqualTo("Japanese");
        assertThat(Language.HI.displayName()).isEqualTo("Hindi");
    }

    @Test
    void allLanguages_haveNonEmptyDisplayName() {
        for (Language language : Language.values()) {
            assertThat(language.displayName())
                    .as("displayName for %s", language.name())
                    .isNotEmpty();
        }
    }

    @Test
    void allLanguages_haveNonEmptyIsoCode() {
        for (Language language : Language.values()) {
            assertThat(language.isoCode())
                    .as("isoCode for %s", language.name())
                    .isNotEmpty();
        }
    }
}
