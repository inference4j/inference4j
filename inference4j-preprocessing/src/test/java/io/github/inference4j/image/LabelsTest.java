package io.github.inference4j.image;

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
        Labels labels = Labels.imagenet();
        assertEquals(1000, labels.size());
    }

    @Test
    void imagenet_firstLabelIsTench() {
        Labels labels = Labels.imagenet();
        assertEquals("tench", labels.get(0));
    }

    @Test
    void imagenet_lastLabelIsToiletTissue() {
        Labels labels = Labels.imagenet();
        assertEquals("toilet tissue", labels.get(999));
    }

    @Test
    void coco_has80Labels() {
        Labels labels = Labels.coco();
        assertEquals(80, labels.size());
    }

    @Test
    void coco_firstLabelIsPerson() {
        Labels labels = Labels.coco();
        assertEquals("person", labels.get(0));
    }

    @Test
    void fromFile_loadsLabelsFromDisk(@TempDir Path tempDir) throws IOException {
        Path labelsFile = tempDir.resolve("labels.txt");
        Files.write(labelsFile, List.of("alpha", "beta", "gamma"));

        Labels labels = Labels.fromFile(labelsFile);

        assertEquals(3, labels.size());
        assertEquals("alpha", labels.get(0));
        assertEquals("beta", labels.get(1));
        assertEquals("gamma", labels.get(2));
    }

    @Test
    void of_createsFromList() {
        Labels labels = Labels.of(List.of("x", "y", "z"));

        assertEquals(3, labels.size());
        assertEquals("x", labels.get(0));
        assertEquals("z", labels.get(2));
    }

    @Test
    void get_throwsOnInvalidIndex() {
        Labels labels = Labels.of(List.of("a", "b"));

        assertThrows(IndexOutOfBoundsException.class, () -> labels.get(2));
        assertThrows(IndexOutOfBoundsException.class, () -> labels.get(-1));
    }
}
