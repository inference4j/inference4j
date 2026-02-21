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

import static org.junit.jupiter.api.Assertions.*;

class LabelsTest {

    @Test
    void imagenet_has1000Labels() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.imagenet();
        assertEquals(1000, labels.size());
    }

    @Test
    void imagenet_firstLabelIsTench() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.imagenet();
        assertEquals("tench", labels.get(0));
    }

    @Test
    void imagenet_lastLabelIsToiletTissue() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.imagenet();
        assertEquals("toilet tissue", labels.get(999));
    }

    @Test
    void coco_has80Labels() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.coco();
        assertEquals(80, labels.size());
    }

    @Test
    void coco_firstLabelIsPerson() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.coco();
        assertEquals("person", labels.get(0));
    }

    @Test
    void fromFile_loadsLabelsFromDisk(@TempDir Path tempDir) throws IOException {
        Path labelsFile = tempDir.resolve("labels.txt");
        Files.write(labelsFile, List.of("alpha", "beta", "gamma"));

        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.fromFile(labelsFile);

        assertEquals(3, labels.size());
        assertEquals("alpha", labels.get(0));
        assertEquals("beta", labels.get(1));
        assertEquals("gamma", labels.get(2));
    }

    @Test
    void of_createsFromList() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.of(List.of("x", "y", "z"));

        assertEquals(3, labels.size());
        assertEquals("x", labels.get(0));
        assertEquals("z", labels.get(2));
    }

    @Test
    void get_throwsOnInvalidIndex() {
        io.github.inference4j.preprocessing.image.Labels labels = io.github.inference4j.preprocessing.image.Labels.of(List.of("a", "b"));

        assertThrows(IndexOutOfBoundsException.class, () -> labels.get(2));
        assertThrows(IndexOutOfBoundsException.class, () -> labels.get(-1));
    }
}
