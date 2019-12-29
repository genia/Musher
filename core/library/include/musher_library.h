#ifndef __MUSHER_LIBRARY_H__
#define __MUSHER_LIBRARY_H__

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <algorithm>

#include "musher_core.h"
#include "utils.h"
#include "wavelib.h"
#include "minimp3_ex.h"

namespace musher
{
    void MUSHER_API CPrintFunctionalMessage(const char* message);

    std::vector<uint8_t> MUSHER_API CLoadAudioFile(const std::string& filePath);

    // bool MUSHER_API CAcceptDecode(const char* message, bool (*decodef)(const char*));

    /**
     * @brief Decodes a wav file, analysis stored into wav_decode_data
     * 
     * @tparam AudioBufferType type of the returned samples
     * @tparam Map unordered or ordered map
     * @tparam K string
     * @tparam V variant
     * @param wav_decoded_data map that will be used to store the music file analysis
     * @param file_path path to wav file
     * @return std::vector< std::vector< AudioBufferType > > CDecodeWav 2D vector of samples,
     * samples[0] holds channel 1
     * samples[1] holds channel 2 (Empty if mono file)
     */
    template <typename AudioBufferType, template <typename ...> class Map, typename K, typename V>
    std::vector< std::vector< AudioBufferType > > MUSHER_API CDecodeWav(Map<K, V>& wav_decoded_data, const std::vector<uint8_t>& file_data)
    {   
        std::vector< std::vector<AudioBufferType> > samples;

        if (!samples.empty()){
            std::string err_message = "Audio Buffer must be empty";
            throw std::runtime_error(err_message);
        }
        // -----------------------------------------------------------
        // HEADER CHUNK
        std::string header_chunk_id (file_data.begin(), file_data.begin() + 4);
        //int32_t fileSizeInBytes = fourBytesToInt (file_data, 4) + 8;
        std::string format (file_data.begin() + 8, file_data.begin() + 12);
        // -----------------------------------------------------------

        /* find data chunk in file_data */
        const std::string data_chunk_key = "data";
        int data_chunk_index = -1;
        auto data_chunk_it = std::search(file_data.begin(), file_data.end(), data_chunk_key.begin(), data_chunk_key.end());
        if(data_chunk_it != file_data.end())
        {
            data_chunk_index = std::distance(file_data.begin(), data_chunk_it);
        }

        /* find format chunk in file_data */
        std::string format_chunk_key = "fmt";
        int format_chunk_index = -1;
        auto formatChunkIt = std::search(file_data.begin(), file_data.end(), format_chunk_key.begin(), format_chunk_key.end());
        if(formatChunkIt != file_data.end())
        {
            format_chunk_index = std::distance(file_data.begin(), formatChunkIt);
        }

        // if we can't find the data or format chunks, or the IDs/formats don't seem to be as expected
        // then it is unlikely we'll able to read this file, so abort
        if (data_chunk_index == -1 || format_chunk_index == -1 || header_chunk_id != "RIFF" || format != "WAVE")
        {
            std::string err_message = "This doesn't seem to be a valid .WAV file";
            throw std::runtime_error(err_message);
        }

        // -----------------------------------------------------------
        // FORMAT CHUNK
        int f = format_chunk_index;
        std::string format_chunk_id (file_data.begin() + f, file_data.begin() + f + 4);
        //int32_t formatChunkSize = fourBytesToInt (file_data, f + 4);
        int16_t audio_format = twoBytesToInt (file_data, f + 8);
        int16_t num_channels = twoBytesToInt (file_data, f + 10);
        uint32_t sample_rate = (uint32_t) fourBytesToInt (file_data, f + 12);
        int32_t num_bytes_per_second = fourBytesToInt (file_data, f + 16);
        int16_t num_bytes_per_block = twoBytesToInt (file_data, f + 20);
        int bit_depth = (int) twoBytesToInt (file_data, f + 22);

        int num_bytes_per_sample = bit_depth / 8;
        
        // check that the audio format is PCM
        if (audio_format != 1)
        {
            std::string err_message = "This is a compressed .WAV file and this library does not support decoding them at present";
            throw std::runtime_error(err_message);
        }
        
        // check the number of channels is mono or stereo
        if (num_channels < 1 ||num_channels > 2)
        {
            std::string err_message = "This WAV file seems to be neither mono nor stereo (perhaps multi-track, or corrupted?";
            throw std::runtime_error(err_message);
        }
        
        // check header data is consistent
        if ((num_bytes_per_second != (num_channels * sample_rate * bit_depth) / 8) || (num_bytes_per_block != (num_channels * num_bytes_per_sample)))
        {
            std::string err_message = "The header data in this WAV file seems to be inconsistent";
            throw std::runtime_error(err_message);
        }
        
        // check bit depth is either 8, 16 or 24 bit
        if (bit_depth != 8 && bit_depth != 16 && bit_depth != 24)
        {
            std::string err_message = "This file has a bit depth that is not 8, 16 or 24 bits";
            throw std::runtime_error(err_message);
        }

        // -----------------------------------------------------------
        // DATA CHUNK
        int d = data_chunk_index;
        std::string data_chunk_id (file_data.begin() + d, file_data.begin() + d + 4);
        int32_t data_chunk_size = fourBytesToInt (file_data, d + 4);
        
        int num_samples = data_chunk_size / (num_channels * bit_depth / 8);
        int samples_start_index = data_chunk_index + 8;

        samples.resize (num_channels);

        for (int i = 0; i < num_samples; i++)
        {
            for (int channel = 0; channel < num_channels; channel++)
            {
                int sample_index = samples_start_index + (num_bytes_per_block * i) + channel * num_bytes_per_sample;
                
                if (bit_depth == 8)
                {   
                    /* normalize samples to between -1 and 1 */
                    AudioBufferType sample = normalizeInt8_t<AudioBufferType>(file_data[sample_index]);
                    samples[channel].push_back (sample);
                }
                else if (bit_depth == 16)
                {
                    int16_t sample_as_int = twoBytesToInt(file_data, sample_index);
                    /* normalize samples to between -1 and 1 */
                    AudioBufferType sample = normalizeInt16_t<AudioBufferType>(sample_as_int);
                    // samples[channel].push_back (sample);
                    samples[channel].push_back (sample_as_int);
                }
                else if (bit_depth == 24)
                {
                    int32_t sample_as_int = 0;
                    sample_as_int = (file_data[sample_index + 2] << 16) | (file_data[sample_index + 1] << 8) | file_data[sample_index];
                    
                    if (sample_as_int & 0x800000) // if the 24th bit is set, this is a negative number in 24-bit world
                        sample_as_int = sample_as_int | ~0xFFFFFF; // so make sure sign is extended to the 32 bit float

                    /* normalize samples to between -1 and 1 */
                    AudioBufferType sample = normalizeInt32_t<AudioBufferType>(sample_as_int);
                    // AudioBufferType sample = (AudioBufferType)sample_as_int / (AudioBufferType)8388608.;
                    // samples[channel].push_back (sample);
                    samples[channel].push_back (sample_as_int);
                }
                else
                {
                    std::string err_message = "This file has a bit depth that is not 8, 16 or 24 bits, not sure how you got past the first error check.";
                    throw std::runtime_error(err_message);
                }
            }
        }

        int num_channels_int = static_cast<int>(num_channels);
        bool mono = num_channels == 1;
        bool stereo = num_channels == 2;
        int num_samples_per_channel = 0;
        if (samples.size() > 0)
            num_samples_per_channel = static_cast<int>(samples[0].size());
        double lengthInSeconds = static_cast<double>(num_samples_per_channel) / static_cast<double>(sample_rate);
        std::string fileType = "wav";
        int avg_bitrate_kbps = (sample_rate * bit_depth * num_channels_int) / 1000;

        /* Add the decoded info into the unordered map */
        wav_decoded_data["sample_rate"] = sample_rate;
        wav_decoded_data["bit_depth"] = bit_depth;
        wav_decoded_data["channels"] = num_channels_int;
        wav_decoded_data["mono"] = mono;
        wav_decoded_data["stereo"] = stereo;
        wav_decoded_data["samples_per_channel"] = num_samples_per_channel;
        wav_decoded_data["length_in_seconds"] = lengthInSeconds;
        wav_decoded_data["filetype"] = fileType;
        wav_decoded_data["avg_bitrate_kbps"] = avg_bitrate_kbps;

        return samples;
    }

    /**
     * @brief Overloaded wrapper around CDecodeWav that accepts a file path to a wav file and returns interleaved samples
     * 
     * @tparam AudioBufferType type of the final samples
     * @tparam Map unordered or ordered map
     * @tparam K string
     * @tparam V variant
     * @param wav_decoded_data map that will be used to store the music file analysis
     * @param file_path path to wav file
     * @return std::vector< AudioBufferType > CDecodeWav interleaved samples
     */
    template <typename AudioBufferType, template <typename ...> class Map, typename K, typename V>
    std::vector< AudioBufferType > MUSHER_API CDecodeWav(Map<K, V>& wav_decoded_data, const std::string& file_path)
    {
        std::vector<uint8_t> file_data;
        file_data = CLoadAudioFile(file_path);
        std::vector< std::vector<double> > normalized_samples;
        normalized_samples = CDecodeWav<double>(wav_decoded_data, file_data);

        /* Return interleaved samples */
        return interleave2DVector(normalized_samples);
    }
    
    /**
     * @brief Decodes a mp3 file, analysis stored into wav_decode_data
     * 
     * @tparam AudioBufferType type of the final samples
     * @tparam Map unordered or ordered map
     * @tparam K string
     * @tparam V variant
     * @param wav_decoded_data map that will be used to store the music file analysis
     * @param file_path path to mp3 file
     * @return std::vector< AudioBufferType > CDecodeMp3 interleaved samples
     */
    template <typename AudioBufferType, template <typename ...> class Map, typename K, typename V>
    std::vector< AudioBufferType > MUSHER_API CDecodeMp3(Map<K, V>& mp3_decoded_data, const std::string file_path)
    {
        mp3dec_t mp3d;
        mp3dec_file_info_t info;
        if (mp3dec_load(&mp3d, file_path.c_str(), &info, NULL, NULL))
        {
            /* error */
            throw std::runtime_error("BAD MP3");
        }

        std::vector<int32_t> interleaved_samples(info.buffer, info.buffer + info.samples);
        int num_samples = static_cast<int>(info.samples);
        int num_samples_per_channel = static_cast<int>(info.samples) / 2;
        bool mono = info.channels == 1;
        bool stereo = info.channels == 2;
        uint32_t sample_rate = info.hz;
        double len_in_seconds = static_cast<double>(num_samples_per_channel) / static_cast<double>(sample_rate);
        std::string fileType = "mp3";

        /* Add the decoded info into the unordered map */
        mp3_decoded_data["sample_rate"] = sample_rate;
        mp3_decoded_data["channels"] = info.channels;
        mp3_decoded_data["mono"] = mono;
        mp3_decoded_data["stereo"] = stereo;
        mp3_decoded_data["samples_per_channel"] = num_samples;
        mp3_decoded_data["length_in_seconds"] = len_in_seconds;
        mp3_decoded_data["filetype"] = fileType;
        mp3_decoded_data["avg_bitrate_kbps"] = info.avg_bitrate_kbps;

        std::vector<double> normalized_samples(interleaved_samples.size());
        std::transform(
            interleaved_samples.begin(),
            interleaved_samples.end(),
            normalized_samples.begin(),
            []( const int32_t x ) { return normalizeInt32_t<AudioBufferType>(x); } );

        return normalized_samples;
    }

    template< typename vecType,
            typename = std::enable_if_t< std::is_floating_point<vecType>::value> >
    double bpmDetection(std::vector< vecType > &flattened_normalized_samples, uint32_t sample_rate)
    {
        wave_object obj;
        wt_object wt;
        int J = 1;

        const int total_levels = 4;
        const int max_decimation = pow(2, (total_levels - 1));

        double min_index = 60. / 220 * (sample_rate / max_decimation);
        double max_index = 60. / 40 * (sample_rate / max_decimation);

        const char *name = "db4";
        obj = wave_init(name);// Initialize the wavelet

        size_t cD_min_len;
        double decimated_signal_sum, decimated_signal_mean;
        std::vector<vecType> cD, cD_sum, cD_filtered, cD_decimated_signal, cD_mean_removed_signal, cD_mean_removed_signal_partial;
        std::vector<vecType> cA, cA_filtered, cA_mean_removed_signal_partial;
        for (int level = 0; level < total_levels; level++)
        {
            /* Discrete Wavelet Transform */
            if (level == 0) {
                wt = wt_init(obj, (char*) "dwt", flattened_normalized_samples.size(), J); // Initialize the wavelet transform object on input
                setDWTExtension(wt, (char*) "sym");
                setWTConv(wt, (char*) "direct");

                dwt(wt, flattened_normalized_samples.data()); // Perform DWT

                cD_min_len = wt->length[1] / max_decimation + 1;
                cD_sum.resize(cD_min_len, 0.0);
                cD_mean_removed_signal_partial.resize(cD_min_len);
            } else {
                wt = wt_init(obj,(char*) "dwt", cA.size(), J); // Initialize the wavelet transform object
                setDWTExtension(wt, (char*) "sym");
                setWTConv(wt, (char*) "direct");

                dwt(wt, cA.data()); // Perform remaining DWT's on cA
            }

            /* Fill cA */
            cA.clear();
            for (int i = 0; i < wt->length[0]; ++i) {
                    cA.push_back(wt->output[i]);
            }

            /* Fill cD */
            for (int i = wt->length[1]; i < wt->outlength; ++i) {
                    cD.push_back(wt->output[i]);
            }

            /* Perform One Pole filter on cD */
            cD_filtered = onePoleFilter(cD);

            /* Decimate */
            int dc = pow(2, (total_levels - level - 1));
            for (int ax = 0; ax < cD_filtered.size(); ax += dc)
            {
                cD_decimated_signal.push_back(std::abs(cD_filtered[ax]));
            }

            decimated_signal_sum = std::accumulate(cD_decimated_signal.begin(), cD_decimated_signal.end(), 0.0);
            decimated_signal_mean =  decimated_signal_sum / static_cast<double>(cD_decimated_signal.size());

            /* Remove the mean */
            cD_mean_removed_signal.resize(cD_decimated_signal.size());
            auto remove_mean = [decimated_signal_mean]( const vecType x ) { return x - decimated_signal_mean; };
            std::transform(
                cD_decimated_signal.begin(),
                cD_decimated_signal.end(),
                cD_mean_removed_signal.begin(),
                remove_mean );

            /* Reconstruct */
            std::copy_n ( cD_mean_removed_signal.begin(), cD_min_len, cD_mean_removed_signal_partial.begin() );
            /* Perform element-wise sum of 2 vectors and store into cD_sum */
            std::transform ( 
                        cD_sum.begin(),
                        cD_sum.end(),
                        cD_mean_removed_signal_partial.begin(),
                        cD_sum.begin(),
                        std::plus<vecType>() );

            /* Clear variables */
            wt_free(wt);
            cD.clear();
            cD_filtered.clear();
            cD_decimated_signal.clear();
            cD_mean_removed_signal.clear();
            cD_mean_removed_signal_partial.clear();
        }
        wave_free(obj);

        /* Check if cA has any useful data */
        bool zeros = std::all_of(cA.begin(), cA.end(), [](const vecType d) { return d == 0.0; });
        if (zeros)
            return 0.0;

        /* Filter cA */
        cA_filtered = onePoleFilter(cA);

        /* Make cA_filtered absolute */
        std::vector<vecType> cA_absolute(cA_filtered.size());
        auto absolute_val = []( const vecType x ) { return std::abs(x); };
        std::transform(
                cA_filtered.begin(),
                cA_filtered.end(),
                cA_absolute.begin(),
                absolute_val );
        
        /* Get mean */
        double cA_absolute_sum = std::accumulate(cA_absolute.begin(), cA_absolute.end(), 0.0);
        double cA_absolute_mean =  cA_absolute_sum / static_cast<double>(cA_absolute.size());

        std::vector<vecType> cA_mean_removed_signal(cA_absolute.size());
        auto remove_mean = [cA_absolute_mean]( const vecType x ) { return x - cA_absolute_mean; };
        std::transform(
                cA_absolute.begin(),
                cA_absolute.end(),
                cA_mean_removed_signal.begin(),
                remove_mean );

        cA_mean_removed_signal_partial.resize(cD_min_len);
        std::copy_n ( cA_mean_removed_signal.begin(), cD_min_len, cA_mean_removed_signal_partial.begin() );
        /* Add elements of cD_sum and cD_mean_removed_signal_partial together and store into cD_sum */
        std::transform ( 
                    cD_sum.begin(),
                    cD_sum.end(),
                    cA_mean_removed_signal_partial.begin(),
                    cD_sum.begin(),
                    std::plus<vecType>() );
        
        size_t data_len = cD_sum.size();
        std::vector<vecType> b(data_len * 2);

        /* Fill a section of b with cD_sum data */
        int k = 0;
        for (int i = data_len / 2; i < (data_len / 2) + data_len; ++i){
            b[i] = cD_sum[k];
            k += 1;
        }

        /* Reverse cD_sum */
        std::vector<vecType> reverse_cD(cD_sum);
        std::reverse(reverse_cD.begin(), reverse_cD.end());

        /* Perform an array flipped convolution, which is the same as a cross-correlation on the samples.  */
        std::vector<vecType> correl = fftConvolve<vecType>(b, reverse_cD);
        correl.pop_back(); // We don't need the last element
        size_t midpoint = correl.size() / 2;
        std::vector<vecType> correl_midpoint_tmp(correl.begin() + midpoint, correl.end());
        std::vector<vecType> sliced_correl_midpoint_tmp(correl_midpoint_tmp.begin() + std::floor(min_index), correl_midpoint_tmp.begin() + std::floor(max_index));

        /* Peak Detection */
        std::vector<vecType> sliced_correl_midpoint_tmp_abs(sliced_correl_midpoint_tmp.size());
        std::transform(
                sliced_correl_midpoint_tmp.begin(),
                sliced_correl_midpoint_tmp.end(),
                sliced_correl_midpoint_tmp_abs.begin(),
                []( const vecType x ) { return std::abs(x); } );
        std::vector< std::tuple< double, double > > peaks;
        double threshold = -1000.0;
        bool interpolate = true;
        std::string sort_by = "height";
        peaks = peakDetect(sliced_correl_midpoint_tmp_abs, threshold, interpolate, sort_by);

        /* Get the first item from peaks because we want the highest peak */
        const double peak_index = std::get<0>(peaks[0]);
 
        if (peak_index == 0.0)
            return 0.0;
        double peak_index_adjusted = peak_index + min_index;
        double bpm = 60. / peak_index_adjusted * (sample_rate / max_decimation);

        return bpm;
    }

    template< typename vecType,
            typename = std::enable_if_t< std::is_floating_point<vecType>::value> >
    double bpmsOverWindow(std::vector< vecType > &flattened_normalized_samples, size_t num_samples, uint32_t sample_rate, int windowSeconds)
    {
        int window_samples = windowSeconds * sample_rate;
        int sample_index = 0;
        int max_windows_index = num_samples / window_samples;
        std::vector<vecType> bpms(max_windows_index, 0.0);
        std::vector<vecType> seconds_mid(max_windows_index, 0.0);

        for (int window_index = 0; window_index < max_windows_index; window_index++)
        {
            typename std::vector<vecType>::iterator samp_it = flattened_normalized_samples.begin() + sample_index;
            std::vector<vecType> sliced_samples(samp_it, samp_it + window_samples);
            
            double bpm = bpmDetection(sliced_samples, sample_rate);
            bpms[window_index] = bpm;

            sample_index += window_samples;
        }

        return std::round(median(bpms));
    }

    // bool CDecodeAudio(const char* message, bool (*decodef)(const char*))
    // {
    //     // *decodef("hello")
    //     std::cout << "Hello from Accept Decode!" << std::endl;
    //     return decodef(message);
    // }

}

#endif /* __MUSHER_LIBRARY_H__ */