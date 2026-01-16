#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include "orb.hpp"
#include "orb_cpu.hpp"
#include "orb_pattern.hpp"

std::vector<cv::Point> circle_offsets = {
    { 0, -3}, { 1, -3}, { 2, -2}, { 3, -1},
    { 3,  0}, { 3,  1}, { 2,  2}, { 1,  3},
    { 0,  3}, {-1,  3}, {-2,  2}, {-3,  1},
    {-3,  0}, {-3, -1}, {-2, -2}, {-1, -3}
};

OrientedFASTCPU::OrientedFASTCPU(int nfeatures, int threshold, int n, int nms_window, int patch_size) {
    this->nfeatures = nfeatures;
    this->threshold = threshold;
    this->n = n;
    this->nms_window = nms_window;
    this->patch_size = patch_size;
}

std::vector<Keypoint> OrientedFASTCPU::detect(const cv::Mat& image) {
    std::vector<Keypoint> keypoints;

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

    return keypoints;
}

std::vector<float> OrientedFASTCPU::compute_orientations(const cv::Mat& image, const std::vector<Keypoint>& keypoints) {
    std::vector<float> orientations;
    
    int patch_radius = patch_size / 2;

    // Ensure grayscale float or uchar image
    CV_Assert(image.channels() == 1);

    for (const auto& kp : keypoints) {
        int x = kp.x;
        int y = kp.y;

        // Skip keypoints too close to border
        if (x - patch_radius < 0 || x + patch_radius >= image.cols ||
            y - patch_radius < 0 || y + patch_radius >= image.rows) {
            orientations.push_back(0.0f);
            continue;
        }

        float m_10 = 0.0f;
        float m_01 = 0.0f;
        float m_00 = 0.0f;

        for (int r = -patch_radius; r <= patch_radius; ++r) {
            for (int c = -patch_radius; c <= patch_radius; ++c) {
                float intensity;

                if (image.type() == CV_8U) {
                    intensity = static_cast<float>(image.at<uchar>(y + r, x + c));
                } else {
                    intensity = image.at<float>(y + r, x + c);
                }

                m_10 += c * intensity;
                m_01 += r * intensity;
                m_00 += intensity;
            }
        }

        float angle = std::atan2(m_01, m_10);
        orientations.push_back(angle);
    }    
    
    return orientations;
}

RotatedBRIEFCPU::RotatedBRIEFCPU() {
    this->n_bits = 256;
    this->patch_size = 31;
}

int RotatedBRIEFCPU::sum5x5(const cv::Mat &integral, int x, int y, int width)
{
    int x0 = x - 2;
    int y0 = y - 2;
    int x1 = x + 3;
    int y1 = y + 3;

    return integral.at<int>(y1, x1)
         + integral.at<int>(y0, x0)
         - integral.at<int>(y0, x1)
         - integral.at<int>(y1, x0);
}

std::vector<ORBDescriptor> RotatedBRIEFCPU::compute(const cv::Mat& image, const std::vector<Keypoint>& keypoints, const std::vector<float>& orientations) {
    std::vector<ORBDescriptor> descriptors;
    descriptors.resize(keypoints.size());

    cv::Mat integral;
    cv::integral(image, integral);

    const int height = integral.rows;
    const int width = integral.cols;

    for (int idx=0; idx<keypoints.size(); idx++) {
        const Keypoint& kp = keypoints[idx];
        float angle = orientations[idx];

        float c = std::cos(angle);
        float s = std::sin(angle);

        ORBDescriptor desc{};

        for (int i = 0; i < 256; i++) {
            int x1 = bit_pattern_31_[i * 4];
            int y1 = bit_pattern_31_[i * 4 + 1];
            int x2 = bit_pattern_31_[i * 4 + 2];
            int y2 = bit_pattern_31_[i * 4 + 3];

            int dx1 = static_cast<int>(std::lround(c * x1 - s * y1));
            int dy1 = static_cast<int>(std::lround(s * x1 + c * y1));

            int dx2 = static_cast<int>(std::lround(c * x2 - s * y2));
            int dy2 = static_cast<int>(std::lround(s * x2 + c * y2));

            int cx1 = kp.x + dx1;
            int cy1 = kp.y + dy1;
            int cx2 = kp.x + dx2;
            int cy2 = kp.y + dy2;

            // Boundary check for 5x5 window
            const int subwindow_radius = 5 / 2;
            if (cx1 < subwindow_radius || cy1 < subwindow_radius ||
                cx1 > width - subwindow_radius || cy1 > height - subwindow_radius ||
                cx2 < subwindow_radius || cy2 < subwindow_radius ||
                cx2 > width - subwindow_radius || cy2 > height - subwindow_radius)
                continue;

            int s1 = sum5x5(integral, cx1, cy1, width);
            int s2 = sum5x5(integral, cx2, cy2, width);

            if (s1 < s2) {
                desc.data[i >> 3] |= (1u << (i & 7));
            }
        }
        descriptors[idx] = desc;
    }

    return descriptors;
}

ORBCPU::ORBCPU(int nfeatures, float scaleFactor, int nlevels) {
    this->nfeatures = nfeatures;
    this->scaleFactor = scaleFactor;
    this->nlevels = nlevels;

    pyramid.resize(nlevels);

    std::cout << "scaleFactor: " << scaleFactor << std::endl;
    std::cout << "nlevels: " << nlevels << std::endl;
}

void ORBCPU::detectAndCompute(const cv::Mat& image, std::vector<Keypoint>& keypoints, std::vector<float>& orientations, std::vector<ORBDescriptor>& descriptors) {
    keypoints = fast.detect(image);
    std::cout << keypoints.size() << std::endl;
    orientations = fast.compute_orientations(image, keypoints);
    descriptors = brief.compute(image, keypoints, orientations);
}

void ORBCPU::buildPyramid(const cv::Mat& image) {
    pyramid[0] = image;
    int H = image.rows;
    int W = image.cols;

    for (int i=1; i<nlevels; i++) {
        float scale = pow(scaleFactor, i);
        cv::Size newSize(round(W / scale), round(H / scale));

        cv::Mat resized;
        cv::resize(pyramid[0], resized, newSize, 0, 0, cv::INTER_LINEAR);        
        cv::GaussianBlur(resized, pyramid[i], cv::Size(5, 5), 0);    
    }

    // visualize
    for (int i=0; i<pyramid.size(); i++) {
        cv::imshow("Octave " + std::to_string(i), pyramid[i]);
    }
    cv::waitKey(0);
}