#ifndef SOBEL_H
#define SOBEL_H

#include <opencv2/opencv.hpp>

void SobelCUDA(const cv::Mat& image, cv::Mat& dst, int dir);

#endif // SOBEL_H