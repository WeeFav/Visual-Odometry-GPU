#pragma once
#include <opencv2/opencv.hpp>
#include "orb.hpp"

void Brief(const cv::Mat& image, const std::vector<Keypoint>& keypoints, const std::vector<float>& orientations, std::vector<ORBDescriptor> descriptors, int n_bits, int patch_size);