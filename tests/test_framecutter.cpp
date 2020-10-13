#include "utils.h"
#include "test_utils.h"
#include "gtest_extras.h"

using namespace musher;

/**
 * @brief Test empty buffer.
 * 
 */
TEST(Framecutter, TestEmptyBuffer) {
    const std::vector<double> buffer({});
    std::vector<double> actual_frame;
    int frame_size = 100;
    int hop_size = 60;
    bool start_from_center = false;
    actual_frame = framecutter(buffer,
                               0,
                               frame_size,
                            //    hop_size,
                               start_from_center,
                               false,
                               0.);
    std::vector<double> expected_frame({});
    EXPECT_VEC_EQ(actual_frame, expected_frame);
}

/**
 * @brief Test empty centered.
 * 
 */
TEST(Framecutter, TestEmptyCentered) {
    const std::vector<double> buffer({});
    std::vector<double> actual_frame;
    int frame_size = 100;
    int hop_size = 60;
    bool start_from_center = true;
    actual_frame = framecutter(buffer,
                               0,
                               frame_size,
                            //    hop_size,
                               start_from_center,
                               false,
                               0.);
    std::vector<double> expected_frame({});
    EXPECT_VEC_EQ(actual_frame, expected_frame);
}

/**
 * @brief Test One.
 * 
 */
TEST(Framecutter, TestOne) {
    const std::vector<double> buffer({23});
    std::vector<double> actual_frame;
    int frame_size = 100;
    int hop_size = 60;
    bool start_from_center = false;
    actual_frame = framecutter(buffer,
                               0,
                               frame_size,
                            //    hop_size,
                               start_from_center,
                               false,
                               0.);
    std::vector<double> expected_frame(100);
    expected_frame[0] = 23;
    EXPECT_VEC_EQ(actual_frame, expected_frame);
}

/**
 * @brief Test Centered.
 * 
 */
TEST(Framecutter, TestOneCentered) {
    const std::vector<double> buffer({23});
    std::vector<double> actual_frame;
    int frame_size = 100;
    int hop_size = 60;
    bool start_from_center = true;
    actual_frame = framecutter(buffer,
                               0,
                               frame_size,
                            //    hop_size,
                               start_from_center,
                               false,
                               0.);
    std::vector<double> expected_frame(100);
    expected_frame[50] = 23;
    EXPECT_VEC_EQ(actual_frame, expected_frame);
}

/**
 * @brief Test Last Frame.
 * 
 */
TEST(Framecutter, TestLastFrame) {
    std::vector<double> buffer(100);
    /* Fill buffer with increasing numbers of 0 to 100 */
    std::iota(std::begin(buffer), std::end(buffer), (double)0.);
    std::vector<double> actual_frame;
    int frame_size = 100;
    int hop_size = 60;
    bool start_from_center = false;
    actual_frame = framecutter(buffer,
                               0,
                               frame_size,
                            //    hop_size,
                               start_from_center,
                               false,
                               0.);
    std::vector<double> expected_frame(100);
    std::iota(std::begin(expected_frame), std::end(expected_frame), (double)0.);
    EXPECT_VEC_EQ(actual_frame, expected_frame);
}

/**
 * @brief Test Last Frame.
 * 
 */
TEST(Framecutter, TestLastFrame2) {
    std::vector<double> buffer(100);
    /* Fill buffer with increasing numbers of 0 to 100 */
    std::iota(std::begin(buffer), std::end(buffer), (double)0.);
    std::vector<double> actual_frame;
    int frame_size = 101;
    int hop_size = 60;
    bool start_from_center = false;
    actual_frame = framecutter(buffer,
                               0,
                               frame_size,
                            //    hop_size,
                               start_from_center,
                               false,
                               0.);
    std::vector<double> expected_frame(101);
    std::iota(std::begin(expected_frame), std::end(expected_frame)-1, (double)0.);
    EXPECT_VEC_EQ(actual_frame, expected_frame);
}

/**
 * @brief Test Last Frame Centered.
 * 
 */
TEST(Framecutter, TestLastFrameCentered) {
    std::vector<double> buffer(100);
    /* Fill buffer with increasing numbers of 0 to 100 */
    std::iota(std::begin(buffer), std::end(buffer), (double)0.);
    std::vector<std::vector<double>> actual_frames;
    std::vector<double> actual_frame;
    int frame_size = 100;
    int hop_size = 60;
    bool start_from_center = true;
    actual_frames = allCutFrames(buffer,
                                 hop_size,
                                 frame_size,
                                 start_from_center,
                                 false,
                                 0.);

    std::vector<double> expected_frame1(frame_size);
    std::iota(std::begin(expected_frame1)+50, std::end(expected_frame1), (double)0.);

    std::vector<double> expected_frame2(frame_size);
    std::iota(std::begin(expected_frame2), std::end(expected_frame2)-10, (double)10.);

    std::vector<double> expected_frame3(frame_size);
    std::iota(std::begin(expected_frame3), std::end(expected_frame3)-70, (double)70.);

    std::vector<std::vector<double>> expected_frames({
        expected_frame1,
        expected_frame2,
        expected_frame3
    });

    EXPECT_MATRIX_EQ(actual_frames, expected_frames);
}

/**
 * @brief Test Last Frame Centered2.
 * 
 */
TEST(Framecutter, TestLastFrameCentered2) {
    std::vector<double> buffer(100);
    /* Fill buffer with increasing numbers of 0 to 100 */
    std::iota(std::begin(buffer), std::end(buffer), (double)0.);
    std::vector<std::vector<double>> actual_frames;
    std::vector<double> actual_frame;
    int frame_size = 102;
    int hop_size = 60;
    bool start_from_center = true;
    actual_frames = allCutFrames(buffer,
                                 hop_size,
                                 frame_size,
                                 start_from_center,
                                 false,
                                 0.);

    std::vector<double> expected_frame1(frame_size);
    std::iota(std::begin(expected_frame1)+51, std::end(expected_frame1), (double)0.);

    std::vector<double> expected_frame2(frame_size);
    std::iota(std::begin(expected_frame2), std::end(expected_frame2)-11, (double)9.);

    std::vector<double> expected_frame3(frame_size);
    std::iota(std::begin(expected_frame3), std::end(expected_frame3)-71, (double)69.);

    std::vector<std::vector<double>> expected_frames({
        expected_frame1,
        expected_frame2,
        expected_frame3
    });

    EXPECT_MATRIX_EQ(actual_frames, expected_frames);
}

/**
 * @brief Test Last Frame Centered3.
 * 
 */
TEST(Framecutter, TestLastFrameCentered3) {
    std::vector<double> buffer(100);
    /* Fill buffer with increasing numbers of 0 to 100 */
    std::iota(std::begin(buffer), std::end(buffer), (double)0.);
    std::vector<std::vector<double>> actual_frames;
    int frame_size = 101;
    int hop_size = 60;
    bool start_from_center = true;
    actual_frames = allCutFrames(buffer,
                                 hop_size,
                                 frame_size,
                                 start_from_center,
                                 false,
                                 0.);

    std::vector<double> expected_frame1(frame_size);
    std::iota(std::begin(expected_frame1)+51, std::end(expected_frame1), (double)0.);

    std::vector<double> expected_frame2(frame_size);
    std::iota(std::begin(expected_frame2), std::end(expected_frame2)-10, (double)9.);

    std::vector<double> expected_frame3(frame_size);
    std::iota(std::begin(expected_frame3), std::end(expected_frame3)-70, (double)69.);

    std::vector<std::vector<double>> expected_frames({
        expected_frame1,
        expected_frame2,
        expected_frame3
    });

    EXPECT_MATRIX_EQ(actual_frames, expected_frames);
}

/**
 * @brief Test big hop size.
 * 
 */
TEST(Framecutter, TestBigHopSize) {
    std::vector<double> buffer(100);
    /* Fill buffer with increasing numbers of 0 to 100 */
    std::iota(std::begin(buffer), std::end(buffer), (double)0.);
    std::vector<double> actual_frame;
    int frame_size = 20;
    int hop_size = 40;
    std::vector<std::vector<double>> actual_frames;
    actual_frames = allCutFrames(buffer,
                                 hop_size,
                                 frame_size,
                                 false,
                                 false,
                                 0.);

    // printVector("actual_frame", actual_frame);
    printMatrix(actual_frames);
    // std::vector<double> expected_frame(101);
    // std::iota(std::begin(expected_frame), std::end(expected_frame)-1, (double)0.);
    // EXPECT_VEC_EQ(actual_frame, expected_frame);
}
