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

package io.github.inference4j.http;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.assertj.core.api.Assertions.assertThat;

class DownloadProgressBarTest {

    private final PrintStream originalOut = System.out;
    private ByteArrayOutputStream captured;

    @BeforeEach
    void setUp() {
        captured = new ByteArrayOutputStream();
        System.setOut(new PrintStream(captured));
    }

    @AfterEach
    void tearDown() {
        System.setOut(originalOut);
    }

    @Test
    void onProgress_completedDownload_printsProgressBarWithNewline() {
        var bar = new DownloadProgressBar("model.onnx", 10);

        bar.onProgress(1024, 1024);

        String output = captured.toString();
        assertThat(output).contains("model.onnx");
        assertThat(output).contains("100.0%");
        assertThat(output).contains("1.0 KiB / 1.0 KiB");
        assertThat(output).endsWith(System.lineSeparator());
    }

    @Test
    void onProgress_partialDownload_printsProgressBar() throws InterruptedException {
        var bar = new DownloadProgressBar("weights.bin", 10);

        bar.onProgress(512, 1024);
        // Wait past the 500ms throttle so the second call renders
        Thread.sleep(600);
        bar.onProgress(1024, 1024);

        String output = captured.toString();
        assertThat(output).contains("50.0%");
        assertThat(output).contains("100.0%");
    }

    @Test
    void onProgress_throttlesUpdatesWithin500ms() {
        var bar = new DownloadProgressBar("fast.bin", 10);

        bar.onProgress(100, 1000);
        bar.onProgress(200, 1000);
        bar.onProgress(300, 1000);

        // Only the first call should have rendered (subsequent ones within 500ms are skipped)
        String output = captured.toString();
        long progressCount = output.chars().filter(c -> c == '%').count();
        assertThat(progressCount).isEqualTo(1);
    }

    @Test
    void onProgress_unknownTotal_printsWithoutPercentage() {
        var bar = new DownloadProgressBar("stream.dat", 10);

        bar.onProgress(2048, 0);

        String output = captured.toString();
        assertThat(output).contains("stream.dat");
        assertThat(output).contains("2.0 KiB");
        assertThat(output).doesNotContain("%");
    }

    @Test
    void onProgress_longFileName_truncatesWithEllipsis() {
        var bar = new DownloadProgressBar("a-very-long-model-filename.onnx", 10);

        bar.onProgress(500, 500);

        String output = captured.toString();
        assertThat(output).contains("a-very-long-model...");
        assertThat(output).doesNotContain("a-very-long-model-filename.onnx");
    }

    @Test
    void onProgress_shortFileName_noTruncation() {
        var bar = new DownloadProgressBar("tiny.bin", 10);

        bar.onProgress(100, 100);

        String output = captured.toString();
        assertThat(output).contains("tiny.bin");
    }

    @Test
    void onProgress_largeFileSize_showsMiB() {
        var bar = new DownloadProgressBar("large.bin", 10);
        long totalBytes = 150 * 1024 * 1024L;

        bar.onProgress(totalBytes, totalBytes);

        String output = captured.toString();
        assertThat(output).contains("150.0 MiB");
    }

    @Test
    void onProgress_bytesLessThan1KiB_showsBytes() {
        var bar = new DownloadProgressBar("small.txt", 10);

        bar.onProgress(512, 512);

        String output = captured.toString();
        assertThat(output).contains("512 B");
    }
}
