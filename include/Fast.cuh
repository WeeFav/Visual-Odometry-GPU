#pragma once
#include <opencv2/opencv.hpp>
#include "orb.hpp"

int Fast(const cv::Mat& image, std::vector<Keypoint>& keypoints, int threshold, int n, int nms_window, int nfeatures);
void Orientations(const cv::Mat& image, const std::vector<Keypoint>& keypoints, std::vector<float>& orientations, int patch_size);