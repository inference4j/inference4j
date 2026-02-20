package io.github.inference4j.sampling;

public class TemperatureProcessor implements LogitsProcessor{

    private final float temperature;

    public TemperatureProcessor(float temperature) {
        this.temperature = temperature;
    }

    @Override
    public float[] process(float[] logits) {
        float[] result = logits.clone();
        for (int i = 0; i < result.length; i++) {
            result[i] /= temperature;
        }
        return result;
    }
}
