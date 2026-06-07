#pragma once

namespace chowdsp
{
/**
 *  A resampler using an internal resampling type
 *  as defined in the ResamplingTypes namespace.
 * 
 *  Note that the number of samples returned from the resampler
 *  may not be enough to fill the required output buffer, so it
 *  might be necessary to buffer some extra samples.
 */
template <Resampler ResamplerType, int MaxChannels = 8>
class ResamplingProcessor
{
public:

    /** Default constructor */
    ResamplingProcessor() = default;

    /** Move constructor */
    ResamplingProcessor (ResamplingProcessor&&) noexcept = default;

    /** Move assignment operator */
    ResamplingProcessor& operator= (ResamplingProcessor&&) noexcept = default;

    /** Prepares the resampler to process a new stream of data */
    void prepare (const juce::dsp::ProcessSpec& spec, double startRatio = 1.0)
    {
        jassert ((int) spec.numChannels <= MaxChannels);
        numActiveChannels = juce::jmin ((int) spec.numChannels, MaxChannels);

        for (int ch = 0; ch < numActiveChannels; ++ch)
            resamplers[(size_t) ch].prepare (spec.sampleRate, startRatio);

        outputBuffer.setMaxSize (numActiveChannels,
                                 (int) spec.maximumBlockSize * 20);
    }

    /** Resets the state of the resampler */
    void reset()
    {
        for (int ch = 0; ch < numActiveChannels; ++ch)
            resamplers[(size_t) ch].reset();
    }

    /** Sets the ratio of output sample rate over input sample rate
     * 
     *  @param ratio    The resampling ratio. Must be in [0.1, 10.0]
     */
    void setResampleRatio (float ratio)
    {
        auto ratioClamped = juce::jlimit (0.01f, 100.0f, ratio);
        for (int ch = 0; ch < numActiveChannels; ++ch)
            resamplers[(size_t) ch].setResampleRatio (ratioClamped);
    }

    /** Returns the ratio of output sample rate over input sample rate */
    [[nodiscard]] float getResampleRatio() const noexcept
    {
        return numActiveChannels > 0 ? resamplers[0].getResampleRatio() : 1.0f;
    }

    /** Processes an input block of samples
     * 
     *  @return the output block of generated samples at the new sample rate
     */
    BufferView<float> process (const BufferView<const float>& block) noexcept
    {
        const auto numChannels = juce::jmin ((int) block.getNumChannels(), numActiveChannels);
        const auto numSamples = block.getNumSamples();

        size_t outNumSamples = 0;
        #pragma clang loop vectorize(enable)
        for (int ch = 0; ch < numChannels; ++ch)
        {
            outNumSamples = resamplers[(size_t) ch].process (block.getReadPointer (ch),
                                                             outputBuffer.getWritePointer (ch),
                                                             (size_t) numSamples);
        }

        return { outputBuffer, 0, (int) outNumSamples };
    }

private:
    std::array<ResamplerType, (size_t) MaxChannels> resamplers;
    int numActiveChannels = 0;
    Buffer<float> outputBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ResamplingProcessor)
};

} // namespace chowdsp
