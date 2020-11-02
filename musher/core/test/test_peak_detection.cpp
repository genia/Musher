#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "musher/core/test/gtest_extras.h"
#include "musher/core/utils.h"

using namespace musher::core;

/**
 * @brief Check that we can get a peak at the final position of the input.
 *
 */
TEST(PeakDetection, LastPositionPeak) {
  std::vector<double> inp{1.0, 1.0, 1.0, 1.0, 2.0};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, true);

  double actual_position;
  double actual_value;
  std::tie(actual_position, actual_value) = peaks[0];
  EXPECT_EQ(actual_position, 4.0);
  EXPECT_EQ(actual_value, 2.0);
}

/**
 * @brief Check that we get the middle value of a flat peak with interpolation on.
 * This was made to test a bug with interpolation that has since been fixed.
 *
 */
TEST(PeakDetection, FlatPeakMiddle1) {
  std::vector<double> inp{1.0, 2.0, 2.0, 2.0, 1.0};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, true);

  double actual_position;
  double actual_value;
  std::tie(actual_position, actual_value) = peaks[0];
  EXPECT_EQ(actual_position, 2.0);
  EXPECT_EQ(actual_value, 2.0);
}

/**
 * @brief Check that we get the middle value of a flat peak with interpolation on.
 *
 */
TEST(PeakDetection, FlatPeakMiddle2) {
  std::vector<double> inp{1.0, 2.0, 2.0, 2.0, 1.0, 0};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, true);

  double actual_position;
  double actual_value;
  std::tie(actual_position, actual_value) = peaks[0];
  EXPECT_EQ(actual_position, 2.0);
  EXPECT_EQ(actual_value, 2.0);
  // std::cout << "POSITIONS: " << std::endl;
  // for (auto & element : peaks) {
  //     std::cout << std::get<0>(element) << std::endl;
  // }

  // std::cout << "VALUES: " << std::endl;
  // for (auto & element : peaks) {
  //     std::cout << std::get<1>(element) << std::endl;
  // }
}

/**
 * @brief Check that we can interpolate a peak after some flatness.
 *
 */
TEST(PeakDetection, FlatToPeakInterpolation) {
  std::vector<double> inp{1, 2, 2, 2, 3, 0};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, true);

  double actual_position;
  double actual_value;
  std::tie(actual_position, actual_value) = peaks[0];
  EXPECT_NEAR(actual_position, 3.75, .01);
  EXPECT_NEAR(actual_value, 3.125, .001);
}

/**
 * @brief Check that we can find a peak after some flatness.
 * @details This test also shows how to convert peaks and values into seperate vectors
 *
 */
TEST(PeakDetection, FlatToPeakNoInterpolation) {
  std::vector<double> inp{1, 2, 2, 2, 3, 0};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, false);
  std::vector<double> actual_positions(peaks.size());
  std::vector<double> actual_height(peaks.size());

  std::transform(peaks.begin(), peaks.end(), actual_positions.begin(),
                 [](auto const& pair) { return std::get<0>(pair); });

  std::transform(peaks.begin(), peaks.end(), actual_height.begin(), [](auto const& pair) { return std::get<1>(pair); });

  std::vector<double> expected_positions = {4};
  EXPECT_VEC_EQ(actual_positions, expected_positions);

  std::vector<double> expected_height = {3};
  EXPECT_VEC_EQ(actual_height, expected_height);
}

/**
 * @brief Check that interpolation works with many peaks.
 *
 */
TEST(PeakDetection, ManyPeaksWithInterpolation) {
  std::vector<double> inp{0, 2, 1, 2, 1, 2, 0};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, true);

  double actual_position_0;
  double actual_height_0;
  std::tie(actual_position_0, actual_height_0) = peaks[0];
  EXPECT_NEAR(actual_position_0, 1.16667, 0.00001);
  EXPECT_NEAR(actual_height_0, 2.04167, 0.00001);

  double actual_position_1;
  double actual_height_1;
  std::tie(actual_position_1, actual_height_1) = peaks[1];
  EXPECT_DOUBLE_EQ(actual_position_1, 3);
  EXPECT_DOUBLE_EQ(actual_height_1, 2);

  double actual_position_2;
  double actual_height_2;
  std::tie(actual_position_2, actual_height_2) = peaks[2];
  EXPECT_NEAR(actual_position_2, 4.83333, 0.00001);
  EXPECT_NEAR(actual_height_2, 2.04167, 0.00001);
}

/**
 * @brief Check that the peaks are ordered by position.
 *
 */
TEST(PeakDetection, SortByPosition) {
  std::vector<double> inp{0, 2, 1, 4, 1, 6, 0};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, false, "position");
  std::vector<double> actual_positions(peaks.size());
  std::vector<double> actual_height(peaks.size());

  std::transform(peaks.begin(), peaks.end(), actual_positions.begin(),
                 [](auto const& pair) { return std::get<0>(pair); });

  std::transform(peaks.begin(), peaks.end(), actual_height.begin(), [](auto const& pair) { return std::get<1>(pair); });

  std::vector<double> expected_positions = {1, 3, 5};
  EXPECT_VEC_EQ(actual_positions, expected_positions);

  std::vector<double> expected_height = {2, 4, 6};
  EXPECT_VEC_EQ(actual_height, expected_height);
}

/**
 * @brief Check that the peaks are ordered by height.
 *
 */
TEST(PeakDetection, SortByHeight) {
  std::vector<double> inp{0, 2, 1, 4, 1, 6, 0};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, false, "height");
  std::vector<double> actual_positions(peaks.size());
  std::vector<double> actual_height(peaks.size());

  std::transform(peaks.begin(), peaks.end(), actual_positions.begin(),
                 [](auto const& pair) { return std::get<0>(pair); });

  std::transform(peaks.begin(), peaks.end(), actual_height.begin(), [](auto const& pair) { return std::get<1>(pair); });

  std::vector<double> expected_positions = {5, 3, 1};
  EXPECT_VEC_EQ(actual_positions, expected_positions);

  std::vector<double> expected_height = {6, 4, 2};
  EXPECT_VEC_EQ(actual_height, expected_height);
}

/**
 * @brief Check that we only get the first 2 peaks.
 *
 */
TEST(PeakDetection, MaxPeaks) {
  std::vector<double> inp{0, 2, 1, 4, 1, 6, 0};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, false, "position", 2);

  size_t actual_num_of_peaks = 2;
  EXPECT_EQ(actual_num_of_peaks, peaks.size());
}

/**
 * @brief Make sure we do not get peaks below min position.
 *
 */
TEST(PeakDetection, MinPosition) {
  std::vector<double> inp{5, 0, 1, 0, 0, 0, 0};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, false, "position", 0, 0.0, 2);

  double actual_position;
  double actual_height;
  std::tie(actual_position, actual_height) = peaks[0];
  EXPECT_EQ(actual_position, 2);
  EXPECT_EQ(actual_height, 1);
}

/**
 * @brief Make sure we do not get peaks above max position.
 *
 */
TEST(PeakDetection, MaxPosition) {
  std::vector<double> inp{0, 0, 0, 0, 1, 0, 2};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, false, "position", 0, 0.0, 0, 4);

  double actual_position;
  double actual_height;
  std::tie(actual_position, actual_height) = peaks[0];
  EXPECT_EQ(actual_position, 4);
  EXPECT_EQ(actual_height, 1);
}

/**
 * @brief Check that our input was scaled to fit the range of 3.
 *
 */
TEST(PeakDetection, Range) {
  std::vector<double> inp{5, 0, 1, 0, 2, 0, 1};
  std::vector<std::tuple<double, double>> peaks;
  peaks = PeakDetect(inp, -1000.0, false, "position", 0, 3.0);
  std::vector<double> actual_positions(peaks.size());
  std::vector<double> actual_height(peaks.size());

  std::transform(peaks.begin(), peaks.end(), actual_positions.begin(),
                 [](auto const& pair) { return std::get<0>(pair); });

  std::transform(peaks.begin(), peaks.end(), actual_height.begin(), [](auto const& pair) { return std::get<1>(pair); });

  std::vector<double> expected_positions = {0, 1, 2, 3};
  EXPECT_VEC_EQ(actual_positions, expected_positions);

  std::vector<double> expected_height = {5, 1, 2, 1};
  EXPECT_VEC_EQ(actual_height, expected_height);
}

/**
 * @brief Check that our input was scaled to fit the range of 3
 * and that we only get peaks within our min and max positions.
 *
 */
TEST(PeakDetection, RangeWithMinAndMaxPositions) {
  std::vector<double> inp{5, 0, 1, 0, 2, 0, 1};
  std::vector<std::tuple<double, double>> peaks;
  /* Min and Max positions should be within the range */
  peaks = PeakDetect(inp, -1000.0, false, "position", 0, 3.0, 2, 3);
  std::vector<double> actual_positions(peaks.size());
  std::vector<double> actual_height(peaks.size());

  std::transform(peaks.begin(), peaks.end(), actual_positions.begin(),
                 [](auto const& pair) { return std::get<0>(pair); });

  std::transform(peaks.begin(), peaks.end(), actual_height.begin(), [](auto const& pair) { return std::get<1>(pair); });

  std::vector<double> expected_positions = {2, 3};
  EXPECT_VEC_EQ(actual_positions, expected_positions);

  std::vector<double> expected_height = {2, 1};
  EXPECT_VEC_EQ(actual_height, expected_height);
}