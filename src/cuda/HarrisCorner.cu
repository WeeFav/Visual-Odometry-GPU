#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "GaussianBlur.hpp"
#include "Sobel.hpp"

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

const int block_size = 16; // all threads in the block participates in loading input into shared memory but not in computing output

__global__ void d_HarrisCorner(const float* Sx2, const float* Sy2, const float* Sxy, float* output, int height, int width, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        int i = idy * width + idx;
        float a = Sx2[i];
        float b = Sxy[i];
        float c = Sy2[i];

        float det = a * c - b * b;
        float trace = a + c;

        output[i] = det - k * trace * trace;        
    }
}

void HarrisCorner(const cv::Mat& image, cv::Mat& dst, int corner_window, int k) {
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

    cv::Mat harris(image.size(), CV_32F);

    float* h_Sx2 = Sx2.ptr<float>();
    float* h_Sy2 = Sy2.ptr<float>();
    float* h_Sxy = Sxy.ptr<float>();
    float *h_output = harris.ptr<float>();

    // Allocate device memory and copy input data over to GPU
    float *d_Sx2, *d_Sy2, *d_Sxy, *d_output;
    cudaMalloc(&d_Sx2, height*width*sizeof(float));
    cudaMalloc(&d_Sy2, height*width*sizeof(float));
    cudaMalloc(&d_Sxy, height*width*sizeof(float));
    cudaMalloc(&d_output, height*width*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_Sx2, h_Sx2, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sy2, h_Sy2, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sxy, h_Sxy, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, height*width*sizeof(float));
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Run GPU kernel
    dim3 block(block_size, block_size);
    dim3 grid((width+block_size-1)/block_size, (height+block_size-1)/block_size);
    d_HarrisCorner<<<grid, block>>>(d_Sx2, d_Sy2, d_Sxy, d_output, height, width, k);
    cudaCheckErrors("kernel launch failure");

    // Copy output from device to host
    cudaMemcpy(h_output, d_output, height*width*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
}