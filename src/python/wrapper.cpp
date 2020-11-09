#include "src/python/wrapper.h"

#include "src/python/utils.h"

using namespace musher::core;

namespace musher {
namespace python {

int add(int i, int j) { return i + j; }

py::array_t<uint8_t> _LoadAudioFile(const std::string& file_path) {
  std::vector<uint8_t> fileData = LoadAudioFile(file_path);
  return ConvertSequenceToPyarray(fileData);
}

py::dict _DecodeWavFromData(std::vector<uint8_t>& file_data) {
  WavDecoded wav_decoded = DecodeWav(file_data);
  return ConvertWavDecodedToPyDict(wav_decoded);
}

py::dict _DecodeWavFromFile(const std::string file_path) {
  WavDecoded wav_decoded = DecodeWav(file_path);
  return ConvertWavDecodedToPyDict(wav_decoded);
}

py::array_t<double> _MonoMixer(const std::vector<std::vector<double>>& normalized_samples) {
  std::vector<double> mixed_audio = MonoMixer(normalized_samples);
  return ConvertSequenceToPyarray(mixed_audio);
}

py::array_t<double> _Windowing(const std::vector<double>& audio_frame,
                               const std::function<std::vector<double>(const std::vector<double>&)>& window_type_func,
                               unsigned int size,
                               unsigned int zero_padding_size,
                               bool zero_phase,
                               bool _normalize) {
  const std::vector<double> vec =
      Windowing(audio_frame, window_type_func, size, zero_padding_size, zero_phase, _normalize);
  return ConvertSequenceToPyarray(vec);
}

py::array_t<double> _BlackmanHarris(const std::vector<double>& window, double a0, double a1, double a2, double a3) {
  const std::vector<double> vec = BlackmanHarris(window, a0, a1, a2, a3);
  return ConvertSequenceToPyarray(vec);
}

py::array_t<double> _BlackmanHarris62dB(const std::vector<double>& window) {
  const std::vector<double> vec = BlackmanHarris62dB(window);
  return ConvertSequenceToPyarray(vec);
}

py::array_t<double> _BlackmanHarris92dB(const std::vector<double>& window) {
  const std::vector<double> vec = BlackmanHarris92dB(window);
  return ConvertSequenceToPyarray(vec);
}

py::array_t<double> _ConvertToFrequencySpectrum(const std::vector<double>& audio_frame) {
  const std::vector<double> vec = ConvertToFrequencySpectrum(audio_frame);
  return ConvertSequenceToPyarray(vec);
}

std::vector<std::tuple<double, double>> _SpectralPeaks(const std::vector<double>& input_spectrum,
                                                       double threshold,
                                                       std::string sort_by,
                                                       unsigned int max_num_peaks,
                                                       double sample_rate,
                                                       int min_pos,
                                                       int max_pos) {
  // Figure out how to pass vector of tuples back without copy.
  std::vector<std::tuple<double, double>> vec =
      SpectralPeaks(input_spectrum, threshold, sort_by, max_num_peaks, sample_rate, min_pos, max_pos);
  return vec;
  // py::array_t<std::tuple<double, double>> numpy_arr = py::cast(vec);
  // return numpy_arr;
}

py::array_t<double> _HPCP(const std::vector<std::tuple<double, double>>& peaks,
                          unsigned int size,
                          double reference_frequency,
                          unsigned int harmonics,
                          bool band_preset,
                          double band_split_frequency,
                          double min_frequency,
                          double max_frequency,
                          std::string _weight_type,
                          double window_size,
                          double sample_rate,
                          bool max_shifted,
                          bool non_linear,
                          std::string _normalized) {
  const std::vector<double> vec =
      HPCP(peaks, size, reference_frequency, harmonics, band_preset, band_split_frequency, min_frequency, max_frequency,
           _weight_type, window_size, sample_rate, max_shifted, non_linear, _normalized);
  return ConvertSequenceToPyarray(vec);
}

py::dict _DetectKey(const std::vector<double>& pcp,
                    const bool use_polphony,
                    const bool use_three_chords,
                    const unsigned int num_harmonics,
                    const double slope,
                    const std::string profile_type,
                    const bool use_maj_min) {
  KeyOutput key_output =
      DetectKey(pcp, use_polphony, use_three_chords, num_harmonics, slope, profile_type, use_maj_min);
  py::dict key_output_dict;
  key_output_dict["key"] = key_output.key;
  key_output_dict["scale"] = key_output.scale;
  key_output_dict["strength"] = key_output.strength;
  key_output_dict["first_to_second_relative_strength"] = key_output.first_to_second_relative_strength;
  return key_output_dict;
}

}  // namespace python
}  // namespace musher