#pragma once
#include <opencv2/opencv.hpp>
#include "orb.hpp"

void HarrisScore(const cv::Mat& image, std::vector<Keypoint>& keypoints, std::vector<float>& harris_scores, int corner_window, int k);
