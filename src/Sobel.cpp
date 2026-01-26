#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Convolution.cuh"

static float SOBEL_X[9] = {
    -1.f,  0.f,  1.f,
    -2.f,  0.f,  2.f,
    -1.f,  0.f,  1.f
};

static float SOBEL_Y[9] = {
    -1.f, -2.f, -1.f,
     0.f,  0.f,  0.f,
     1.f,  2.f,  1.f
};

void SobelCUDA(const cv::Mat& image, cv::Mat& dst, int dir) {
    float* kernel;
    if (dir == 0) {
        kernel = SOBEL_X; 
    }
    else {
        kernel = SOBEL_Y; 
    }
    
    int kernel_radius = 1;
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, kernel_radius, kernel_radius, kernel_radius, kernel_radius, cv::BORDER_REFLECT_101);

    conv2d(padded, dst, kernel, 3);
}