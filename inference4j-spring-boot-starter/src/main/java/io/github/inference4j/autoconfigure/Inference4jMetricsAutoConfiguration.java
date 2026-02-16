package io.github.inference4j.autoconfigure;

import io.github.inference4j.metrics.MicrometerRouterMetrics;
import io.github.inference4j.metrics.NoOpRouterMetrics;
import io.github.inference4j.metrics.RouterMetrics;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;

/**
 * Auto-configuration for inference4j metrics using Micrometer.
 */
@AutoConfiguration
@ConditionalOnClass({MeterRegistry.class, MicrometerRouterMetrics.class})
@ConditionalOnProperty(prefix = "inference4j.metrics", name = "enabled", matchIfMissing = true)
public class Inference4jMetricsAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnBean(MeterRegistry.class)
    public RouterMetrics routerMetrics(MeterRegistry registry) {
        return new MicrometerRouterMetrics(registry);
    }

    @Bean
    @ConditionalOnMissingBean
    public RouterMetrics noOpRouterMetrics() {
        return NoOpRouterMetrics.getInstance();
    }
}
