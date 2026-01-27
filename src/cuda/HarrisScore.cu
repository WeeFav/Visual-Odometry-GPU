#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "GaussianBlur.hpp"
#include "Sobel.hpp"
#include "orb.hpp"

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const int block_size = 256;

__global__ void d_HarrisScore(const float* Sx2, const float* Sy2, const float* Sxy, Keypoint* keypoints, float* output, int kp_count, int width, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < kp_count) {
        int x = keypoints[idx].x;
        int y = keypoints[idx].y;

        int i = y * width + x;
        float a = Sx2[i];
        float b = Sxy[i];
        float c = Sy2[i];

        float det = a * c - b * b;
        float trace = a + c;

        output[idx] = det - k * trace * trace;        
    }
}

void HarrisScore(const cv::Mat& image, std::vector<Keypoint>& keypoints, std::vector<float>& harris_scores, int corner_window, int k) {
    int height = image.rows;
    int width = image.cols;
    
    cv::Mat Ix, Iy;
    SobelCUDA(image, Ix, 0);
    SobelCUDA(image, Iy, 1);

    cv::Mat Ix2 = Ix.mul(Ix);
    cv::Mat Iy2 = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);
    
    cv::Mat Sx2, Sy2, Sxy;
    GaussianBlurCUDA(Ix2, Sx2, corner_window);
    GaussianBlurCUDA(Iy2, Sy2, corner_window);
    GaussianBlurCUDA(Iy2, Sxy, corner_window);

    float* h_Sx2 = Sx2.ptr<float>();
    float* h_Sy2 = Sy2.ptr<float>();
    float* h_Sxy = Sxy.ptr<float>();
    Keypoint* h_keypoints = keypoints.data();

    // Allocate device memory and copy input data over to GPU
    float *d_Sx2, *d_Sy2, *d_Sxy, *d_output;
    Keypoint *d_keypoints;
    cudaMalloc(&d_Sx2, height*width*sizeof(float));
    cudaMalloc(&d_Sy2, height*width*sizeof(float));
    cudaMalloc(&d_Sxy, height*width*sizeof(float));
    cudaMalloc(&d_keypoints, keypoints.size()*sizeof(Keypoint));
    cudaMalloc(&d_output, keypoints.size()*sizeof(Keypoint));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_Sx2, h_Sx2, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sy2, h_Sy2, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sxy, h_Sxy, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keypoints, h_keypoints, keypoints.size()*sizeof(Keypoint), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, keypoints.size()*sizeof(Keypoint));
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Run GPU kernel
    d_HarrisScore<<<(keypoints.size()+block_size-1)/block_size, block_size>>>(d_Sx2, d_Sy2, d_Sxy, d_keypoints, d_output, keypoints.size(), width, k);
    cudaCheckErrors("kernel launch failure");

    // Copy output from device to host
    harris_scores.resize(keypoints.size());
    cudaMemcpy(harris_scores.data(), d_output, keypoints.size()*sizeof(Keypoint), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
}