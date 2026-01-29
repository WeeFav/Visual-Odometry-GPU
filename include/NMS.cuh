#pragma once
#include <opencv2/opencv.hpp>
#include "orb.hpp"

void NMS(const cv::Mat& input, std::vector<Keypoint>& keypoints, int nms_window, int nfeatures, float threshold);
