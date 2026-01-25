#ifndef GAUSSIANBLUR_H
#define GAUSSIANBLUR_H

#include <opencv2/opencv.hpp>

void GaussianBlurCUDA(const cv::Mat& image, cv::Mat& dst, int kernel_size);

#endif // GAUSSIANBLUR_H