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

package io.github.inference4j.preprocessing.image;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class LabelsTest {

    @Test
    void imagenet_has1000Labels() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.imagenet();
        assertThat(labels.size()).isEqualTo(1000);
    }

    @Test
    void imagenet_firstLabelIsTench() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.imagenet();
        assertThat(labels.get(0)).isEqualTo("tench");
    }

    @Test
    void imagenet_lastLabelIsToiletTissue() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.imagenet();
        assertThat(labels.get(999)).isEqualTo("toilet tissue");
    }

    @Test
    void coco_has80Labels() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.coco();
        assertThat(labels.size()).isEqualTo(80);
    }

    @Test
    void coco_firstLabelIsPerson() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.coco();
        assertThat(labels.get(0)).isEqualTo("person");
    }

    @Test
    void fromFile_loadsLabelsFromDisk(@TempDir Path tempDir) throws IOException {
        Path labelsFile = tempDir.resolve("labels.txt");
        Files.write(labelsFile, List.of("alpha", "beta", "gamma"));

        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.fromFile(labelsFile);

        assertThat(labels.size()).isEqualTo(3);
        assertThat(labels.get(0)).isEqualTo("alpha");
        assertThat(labels.get(1)).isEqualTo("beta");
        assertThat(labels.get(2)).isEqualTo("gamma");
    }

    @Test
    void of_createsFromList() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.of(List.of("x", "y", "z"));

        assertThat(labels.size()).isEqualTo(3);
        assertThat(labels.get(0)).isEqualTo("x");
        assertThat(labels.get(2)).isEqualTo("z");
    }

    @Test
    void get_throwsOnInvalidIndex() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.of(List.of("a", "b"));

        assertThatThrownBy(() -> labels.get(2)).isInstanceOf(IndexOutOfBoundsException.class);
        assertThatThrownBy(() -> labels.get(-1)).isInstanceOf(IndexOutOfBoundsException.class);
    }
}
