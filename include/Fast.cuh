#pragma once
struct Keypoint { int x, y; };

void Fast(const cv::Mat& image, std::vector<Keypoint>& keypoints, int threshold, int n, int nms_window, int nfeatures);