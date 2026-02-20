package io.github.inference4j.sampling;

public class LogitsProcessors {

    public static LogitsProcessor temperature(float temperature) {
        return new TemperatureProcessor(temperature);
    }

    public static LogitsProcessor topK(int k) {
        return new TopKProcessor(k);
    }

    public static LogitsProcessor topP(float p) {
        return new TopPProcessor(p);
    }

}
