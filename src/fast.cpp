#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include "Fast.cuh"

std::vector<cv::Point> circle_offsets = {
    { 0, -3}, { 1, -3}, { 2, -2}, { 3, -1},
    { 3,  0}, { 3,  1}, { 2,  2}, { 1,  3},
    { 0,  3}, {-1,  3}, {-2,  2}, {-3,  1},
    {-3,  0}, {-3, -1}, {-2, -2}, {-1, -3}
};

void FAST_detect(
    const cv::Mat& image,
    std::vector<Keypoint>& keypoints,
    int threshold,
    int n,
    int nms_window,
    int nfeatures
) 
{
    CV_Assert(image.type() == CV_8UC1);
    CV_Assert(circle_offsets.size() == 16);

    cv::Mat scores = cv::Mat::zeros(image.size(), CV_32F);

    const int rows = image.rows;
    const int cols = image.cols;

    for (int y = 3; y < rows - 3; y++) {
        for (int x = 3; x < cols - 3; x++) {

            int Ip = static_cast<int>(image.at<uchar>(y, x));

            // FAST early rejection: pixels 1,5,9,13
            const int check_idx[4] = {0, 4, 8, 12};
            int brighter = 0;
            int darker   = 0;

            for (int k = 0; k < 4; k++) {
                int idx = check_idx[k];
                const cv::Point& off = circle_offsets[idx];
                int cp = static_cast<int>(
                    image.at<uchar>(y + off.y, x + off.x)
                );

                if (cp >= Ip + threshold)
                    brighter++;
                else if (cp <= Ip - threshold)
                    darker++;
            }

            if (std::max(brighter, darker) < 3)
                continue;

            // Load full circle (duplicate for wrap-around)
            int circle_vals[32];
            for (int i = 0; i < 16; i++) {
                const cv::Point& off = circle_offsets[i];
                int v = static_cast<int>(
                    image.at<uchar>(y + off.y, x + off.x)
                );
                circle_vals[i]      = v;
                circle_vals[i + 16] = v;
            }

            // Check for n contiguous pixels
            bool is_keypoint = false;
            for (int i = 0; i < 16; i++) {

                bool all_brighter = true;
                bool all_darker   = true;

                for (int j = 0; j < n; j++) {
                    int v = circle_vals[i + j];
                    if (v < Ip + threshold)
                        all_brighter = false;
                    if (v > Ip - threshold)
                        all_darker = false;
                }

                if (all_brighter || all_darker) {
                    is_keypoint = true;

                    // Score for NMS (same as Python)
                    float score = 0.0f;
                    for (const auto& off : circle_offsets) {
                        int v = static_cast<int>(
                            image.at<uchar>(y + off.y, x + off.x)
                        );
                        score += std::abs(Ip - v);
                    }

                    scores.at<float>(y, x) = score;
                    break;
                }
            }
        }
    }

    int nms_radius = nms_window / 2;

    // NMS
    for (int y = 3; y < rows - 3; y++) {
        for (int x = 3; x < cols - 3; x++) {
            if (scores.at<float>(y, x) <= 0.0f || keypoints.size() >= nfeatures) continue;
            
            if (nms_radius != 0) {
                // Extract window (same slicing semantics as NumPy)
                cv::Rect roi(
                    x - nms_radius,
                    y - nms_radius,
                    2 * nms_radius + 1,
                    2 * nms_radius + 1
                );

                cv::Mat window = scores(roi);

                double minVal, maxVal;
                cv::minMaxLoc(window, &minVal, &maxVal);

                if (std::abs(scores.at<float>(y, x) - maxVal) < 1e-6f) {
                    keypoints.push_back({x, y});
                }
            }
            else {
                keypoints.push_back({x, y});
            }
        }
    }
}

int main() {
    cv::Mat image = cv::imread("/home/marvin/Visual-Odometry-GPU/000000.png", cv::IMREAD_GRAYSCALE);
    std::vector<Keypoint> keypoints_cpu;
    std::vector<Keypoint> keypoints_gpu;

    int threshold = 50;
    int n = 9;
    int nms_window = 3;
    int nfeatures = 3000;

    FAST_detect(image, keypoints_cpu, threshold, n, nms_window, nfeatures);
    Fast(image, keypoints_gpu, threshold, n, nms_window, nfeatures);

    // cv::Mat diff;
    // cv::compare(scores_cpu, scores_gpu, diff, cv::CmpTypes::CMP_NE);
    // int same = cv::countNonZero(diff.reshape(1)) == 0;
    // std::cout << (same ? "Same" : "Different") << std::endl;

    std::cout << (keypoints_cpu.size() == keypoints_gpu.size() ? "Same" : "Different") << std::endl;
    std::cout << (keypoints_cpu.size()) << std::endl;
    std::cout << (keypoints_gpu.size()) << std::endl;

}