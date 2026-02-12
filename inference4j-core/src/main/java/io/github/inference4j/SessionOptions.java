package io.github.inference4j;

import ai.onnxruntime.OrtSession;

public class SessionOptions {

    private final int intraOpNumThreads;
    private final int interOpNumThreads;

    private SessionOptions(Builder builder) {
        this.intraOpNumThreads = builder.intraOpNumThreads;
        this.interOpNumThreads = builder.interOpNumThreads;
    }

    public static SessionOptions defaults() {
        return new Builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    OrtSession.SessionOptions toOrtOptions() throws ai.onnxruntime.OrtException {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(intraOpNumThreads);
        opts.setInterOpNumThreads(interOpNumThreads);
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        return opts;
    }

    public static class Builder {
        private int intraOpNumThreads = Runtime.getRuntime().availableProcessors();
        private int interOpNumThreads = Runtime.getRuntime().availableProcessors();

        public Builder intraOpNumThreads(int threads) {
            this.intraOpNumThreads = threads;
            return this;
        }

        public Builder interOpNumThreads(int threads) {
            this.interOpNumThreads = threads;
            return this;
        }

        public SessionOptions build() {
            return new SessionOptions(this);
        }
    }
}
