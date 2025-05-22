#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/core/framecutter.h"
#include "src/core/hpcp.h"
#include "src/core/key.h"
#include "src/core/spectral_peaks.h"
#include "src/core/spectrum.h"
#include "src/core/audio_decoders.h"
#include "src/core/windowing.h"
#include "src/core/mono_mixer.h"

using namespace musher::core;

/**
 * @brief Estimate Key C Major Classical
 *
 */
TEST(Key, EstimateKeyCMajorClassical) {
  const std::string file_path = TEST_DATA_DIR + std::string("audio_files/mozart_c_major_30sec.wav");
  // const std::string file_path = TEST_DATA_DIR + std::string("audio_files/1356281_Eb_major.wav");

  int pcp_size = 36;
  double sample_rate = 44100.;
  int num_harmonics = 4;

  WavDecoded wav_decoded = DecodeWav(file_path);
  std::vector<std::vector<double>> normalized_samples = wav_decoded.normalized_samples;
  std::vector<double> mixed_audio = MonoMixer(normalized_samples);

  Framecutter framecutter(mixed_audio, 4096, 512);

  int count = 0;
  std::vector<double> sums(static_cast<size_t>(pcp_size), 0.);

  for (const std::vector<double> &frame : framecutter) {
    std::vector<double> windowed_frame = Windowing(frame, BlackmanHarris62dB);
    std::vector<double> spectrum = ConvertToFrequencySpectrum(windowed_frame);
    std::vector<std::tuple<double, double>> spectral_peaks =
        SpectralPeaks(spectrum, -1000.0, "height", 100, sample_rate, 0, sample_rate / 2);
    std::vector<double> hpcp =
        HPCP(spectral_peaks, pcp_size, 440.0, num_harmonics - 1, true, 500.0, 40.0, 5000.0, "squared cosine", .5);

    for (int i = 0; i < static_cast<int>(hpcp.size()); i++) {
      sums[i] += hpcp[i];
    }
    count += 1;
  }
  std::vector<double> avgs(sums.size());
  std::transform(sums.begin(), sums.end(), avgs.begin(), [&count](auto const &sum) { return sum / count; });
  KeyOutput key_output = EstimateKey(avgs, true, true, 4, 0.6, "Temperley");

  EXPECT_EQ(key_output.key, "C");
  EXPECT_EQ(key_output.scale, "major");
  EXPECT_NEAR(key_output.strength, 0.760322, 0.000001);
  EXPECT_NEAR(key_output.first_to_second_relative_strength, 0.607807, 0.000001);
}

/**
 * @brief Detect Key C Major Classical Wav
 *
 */
TEST(Key, DetectKeyCMajorClassicalWav) {
  const std::string file_path = TEST_DATA_DIR + std::string("audio_files/mozart_c_major_30sec.wav");
  WavDecoded wav_decoded = DecodeWav(file_path);
  std::vector<std::vector<double>> normalized_samples = wav_decoded.normalized_samples;
  double sample_rate = wav_decoded.sample_rate;

  KeyOutput key_output = DetectKey(normalized_samples, sample_rate, "Temperley");
  EXPECT_EQ(key_output.key, "C");
  EXPECT_EQ(key_output.scale, "major");
  EXPECT_NEAR(key_output.strength, 0.760322, 0.000001);
  EXPECT_NEAR(key_output.first_to_second_relative_strength, 0.607807, 0.000001);
}

/**
 * @brief Detect Key C Major Classical Mp3
 *
 */
TEST(Key, DetectKeyCMajorClassicalMp3) {
  const std::string file_path = TEST_DATA_DIR + std::string("audio_files/mozart_c_major_30sec.mp3");
  Mp3Decoded wav_decoded = DecodeMp3(file_path);
  std::vector<std::vector<double>> normalized_samples = wav_decoded.normalized_samples;
  double sample_rate = wav_decoded.sample_rate;

  KeyOutput key_output = DetectKey(normalized_samples, sample_rate, "Temperley");
  EXPECT_EQ(key_output.key, "C");
  EXPECT_EQ(key_output.scale, "major");
  EXPECT_NEAR(key_output.strength, 0.760328, 0.000001);
  EXPECT_NEAR(key_output.first_to_second_relative_strength, 0.608866, 0.000001);
}

/**
 * @brief Estimate Key Eb Major EDM
 *
 */
TEST(Key, EstimateKeyEbMajorEDM) {
  const std::string filePath = TEST_DATA_DIR + std::string("audio_files/EDM_Eb_major_2min.wav");

  int pcp_size = 36;
  int num_harmonics = 4;

  WavDecoded wav_decoded = DecodeWav(filePath);
  double sample_rate = wav_decoded.sample_rate;
  std::vector<std::vector<double>> normalized_samples = wav_decoded.normalized_samples;
  std::vector<double> mixed_audio = MonoMixer(normalized_samples);

  Framecutter framecutter(mixed_audio, 4096, 512);

  int count = 0;
  std::vector<double> sums(static_cast<size_t>(pcp_size), 0.);

  for (const std::vector<double> &frame : framecutter) {
    // NOTE: windowing and ConvertToFrequencySpectrum are slowest functions here.
    std::vector<double> windowed_frame = Windowing(frame, BlackmanHarris62dB);
    std::vector<double> spectrum = ConvertToFrequencySpectrum(windowed_frame);
    std::vector<std::tuple<double, double>> spectral_peaks =
        SpectralPeaks(spectrum, -1000.0, "height", 100, sample_rate, 0, sample_rate / 2);
    std::vector<double> hpcp =
        HPCP(spectral_peaks, pcp_size, 440.0, num_harmonics - 1, true, 500.0, 40.0, 5000.0, "squared cosine", .5);

    for (int i = 0; i < static_cast<int>(hpcp.size()); i++) {
      sums[i] += hpcp[i];
    }
    count += 1;
  }
  std::vector<double> avgs(sums.size());
  std::transform(sums.begin(), sums.end(), avgs.begin(), [&count](auto const &sum) { return sum / count; });
  KeyOutput key_output = EstimateKey(avgs, true, true, 4, 0.6, "Edmm");

  EXPECT_EQ(key_output.key, "Eb");
  EXPECT_EQ(key_output.scale, "major");
  EXPECT_NEAR(key_output.strength, 0.613304, 0.000001);
  EXPECT_NEAR(key_output.first_to_second_relative_strength, 0.516593, 0.000001);
}

namespace fs = std::filesystem;

std::vector<std::string> get_all_files_in_directory(const std::string& directory_path) {
    std::vector<std::string> file_paths;
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (fs::is_regular_file(entry.status())) {
            file_paths.push_back(entry.path().string());
        }
    }
    return file_paths;
}
static bool ends_with(std::string_view str, std::string_view suffix)
{
    return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
}

/**
 * @brief Detect Key C Major Classical Mp3
 *
 */
TEST(Key, DetectGZ) {
    std::string directory_path = "/Users/zilberstein/Music/!Nickodemus - The Wonderwheel Spins 2013"; // Current directory
    std::vector<std::string> files = get_all_files_in_directory(directory_path);

    for (const auto& file_path : files) {//const std::string file_path = "/Users/zilberstein/Music/!Nickodemus - The Wonderwheel Spins 2013/Wonderwheel Recordings - The Wonderwheel Spins 2013 - 05 Gypsy Stepper.mp3";
        if (!ends_with(file_path, ".mp3"))
            continue;
        Mp3Decoded wav_decoded = DecodeMp3(file_path);
        std::vector<std::vector<double>> normalized_samples = wav_decoded.normalized_samples;
        double sample_rate = wav_decoded.sample_rate;
        
        KeyOutput key_output = DetectKey(normalized_samples, sample_rate, "Bgate",
                                         /* use_polphony */ true,
                                         /* use_three_chords */ true,
                                         /* num_harmonics */ 4,
                                         /* slope */ 0.6,
                                         /* use_maj_min */ false,
                                         /* pcp_size */ 36,
                                         /* frame_size */ 32768,
                                         /* hop_size */ 4096,
                                         /*const std::function<std::vector<double>(const std::vector<double>&)>& window_type_func */ BlackmanHarris62dB,
                                         /* max_num_peaks */ 100,
                                         /* window_size */ .5 );
        std::cout << file_path << " key " << key_output.key << key_output.scale << std::endl;
//        EXPECT_EQ(key_output.key, "F");
//        EXPECT_EQ(key_output.scale, "minor");
//        EXPECT_NEAR(key_output.strength, 0.614904, 0.000001);
//        EXPECT_NEAR(key_output.first_to_second_relative_strength, 0.192236, 0.000001);
    }
}
