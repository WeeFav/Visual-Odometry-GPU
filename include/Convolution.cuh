#pragma once
#include <opencv2/opencv.hpp>
#include "orb.hpp"

int conv2d(const cv::Mat& image, cv::Mat& dst, float* kernel, int kernel_size);