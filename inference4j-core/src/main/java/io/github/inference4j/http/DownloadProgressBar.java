package io.github.inference4j.http;

/**
 * A {@link ProgressListener} that renders a console progress bar to {@code System.out}.
 *
 * <p>Displays file name, a visual bar, percentage, downloaded/total sizes, and current speed.
 * Updates are throttled to at most once every 500 ms to avoid excessive console output,
 * except for the final update when the download completes.
 *
 * <p>When the total size is unknown ({@code total <= 0}), only the downloaded size and speed
 * are shown, without a bar or percentage.
 *
 * <p>Example output:
 * <pre>
 * model.onnx           [████████████░░░░░░░░░░░░░░░░░░]  40.0%  48.0 MiB / 120.0 MiB @ 12.5 MiB/s
 * </pre>
 */
public class DownloadProgressBar implements ProgressListener {

    private final String fileName;
    private final int barWidth;
    private long lastUpdate = 0;
    private long lastBytes = 0;
    private long lastTime = System.currentTimeMillis();

    /**
     * Creates a progress bar with the default width of 30 characters.
     *
     * @param fileName the file name to display (truncated to 20 characters if longer)
     */
    public DownloadProgressBar(String fileName) {
        this(fileName, 30);
    }

    /**
     * Creates a progress bar with a custom width.
     *
     * @param fileName the file name to display (truncated to 20 characters if longer)
     * @param barWidth the number of characters for the visual bar
     */
    public DownloadProgressBar(String fileName, int barWidth) {
        this.fileName = fileName;
        this.barWidth = barWidth;
    }

    public void onProgress(long downloaded, long total) {
        long now = System.currentTimeMillis();
        if (downloaded < total && now - lastUpdate < 500) return;
        lastUpdate = now;

        long elapsed = now - lastTime;
        long deltaBytes = downloaded - lastBytes;
        double speed = elapsed > 0 ? (deltaBytes * 1000.0) / elapsed : 0;
        lastBytes = downloaded;
        lastTime = now;

        if (total > 0) {
            double fraction = (double) downloaded / total;
            int filled = (int) (fraction * barWidth);

            System.out.printf("\r%-20s [%s%s] %5.1f%% %s / %s @ %s/s",
                truncate(fileName, 20),
                "█".repeat(filled),
                "░".repeat(barWidth - filled),
                fraction * 100,
                humanSize(downloaded),
                humanSize(total),
                humanSize((long) speed));
        } else {
            System.out.printf("\r%-20s %s @ %s/s",
                truncate(fileName, 20),
                humanSize(downloaded),
                humanSize((long) speed));
        }

        if (downloaded == total) System.out.println();
    }

    private static String truncate(String name, int maxLen) {
        if (name.length() <= maxLen) return name;
        return name.substring(0, maxLen - 3) + "...";
    }

    private static String humanSize(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char unit = "KMGTPE".charAt(exp - 1);
        return String.format("%.1f %siB", bytes / Math.pow(1024, exp), unit);
    }
}
