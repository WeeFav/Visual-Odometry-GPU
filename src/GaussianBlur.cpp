#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Convolution.cuh"

// Function to create a 2D Gaussian kernel stored as a 1D float array
float* createGaussianKernel(int kernelSize, float sigma = -1.0f) {
    if (kernelSize % 2 == 0) {
        std::cerr << "Kernel size must be odd." << std::endl;
        return nullptr;
    }

    if (sigma <= 0.0f) {
        // Heuristic if sigma not provided
        sigma = 0.3f * ((kernelSize - 1) * 0.5f) + 0.8f;
    }

    int halfSize = kernelSize / 2;
    float* kernel = new float[kernelSize * kernelSize];
    float sum = 0.0f;

    // Compute Gaussian values
    for (int y = -halfSize; y <= halfSize; ++y) {
        for (int x = -halfSize; x <= halfSize; ++x) {
            float value = std::exp(-(x*x + y*y) / (2 * sigma * sigma));
            kernel[(y + halfSize) * kernelSize + (x + halfSize)] = value;
            sum += value;
        }
    }

    // Normalize
    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        kernel[i] /= sum;
    }

    return kernel;
}

void GaussianBlurCUDA(const cv::Mat& image, cv::Mat& dst, int kernel_size) {
    float* kernel = createGaussianKernel(kernel_size);
    
    conv2d(image, dst, kernel, kernel_size);

    delete[] kernel; // Free memory
}