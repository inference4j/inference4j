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

import com.fasterxml.jackson.databind.JsonNode;
import io.github.inference4j.exception.ModelLoadException;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Reads the end-of-sequence token ids of a generative model from its {@code config.json}.
 */
final class EosTokenIds {

    private EosTokenIds() {
    }

    /**
     * The {@code eos_token_id} of a {@code config.json}: one id, or the integer ids of an array.
     *
     * @param config the parsed config.json
     * @return the ids, in their order in the file
     * @throws ModelLoadException if the config has no usable {@code eos_token_id}
     */
    static Set<Integer> fromConfig(JsonNode config) {
        JsonNode eosNode = config.get("eos_token_id");
        if (eosNode != null) {
            if (eosNode.isInt()) {
                return Set.of(eosNode.intValue());
            }
            if (eosNode.isArray()) {
                Set<Integer> ids = new LinkedHashSet<>();
                for (JsonNode element : eosNode) {
                    if (element.isInt()) {
                        ids.add(element.intValue());
                    }
                }
                if (!ids.isEmpty()) {
                    return ids;
                }
            }
        }
        throw new ModelLoadException("config.json missing eos_token_id — set it explicitly via eosTokenId()");
    }
}
