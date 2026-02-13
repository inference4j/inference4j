package io.github.inference4j.audio;

import java.nio.file.Path;

/**
 * Transcribes speech audio into text.
 */
public interface SpeechToTextModel extends AutoCloseable {

    /**
     * Transcribes audio from a WAV file.
     *
     * @param audioPath path to the audio file
     * @return the transcription result
     */
    Transcription transcribe(Path audioPath);

    /**
     * Transcribes raw audio samples.
     *
     * @param audioData  mono float32 PCM samples
     * @param sampleRate sample rate of the audio in Hz
     * @return the transcription result
     */
    Transcription transcribe(float[] audioData, int sampleRate);

    @Override
    void close();
}
