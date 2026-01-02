#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
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


const int block_size = 256; // all threads in the block participates in loading input into shared memory and compute ouput. some threads need to additionally load halo cells

__global__ void d_Orientations(
    const float* image,
    const Keypoint* keypoints,
    float* orientations,
    int width,
    int height,
    int patch_size,
    int kp_count
) 
{
    int PATCH_RADIUS = patch_size / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= kp_count) return;

    Keypoint kp = keypoints[idx];
    int x = kp.x;
    int y = kp.y;

    // Must have full patch
    if (x < PATCH_RADIUS || x >= width - PATCH_RADIUS ||
        y < PATCH_RADIUS || y >= height - PATCH_RADIUS) {
        orientations[idx] = 0.0f;
        return;
    }

    float m10 = 0.0f;
    float m01 = 0.0f;

    // Compute intensity centroid
    for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
        int row = (y + dy) * width;
        for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
            float I = image[row + (x + dx)];
            m10 += dx * I;
            m01 += dy * I;
        }
    }

    orientations[idx] = atan2f(m01, m10);
}

void Orientations(const cv::Mat& image, std::vector<Keypoint>& keypoints, std::vector<float>& orientations, int patch_size) {
    int width = image.cols;
    int height = image.rows;

    cv::Mat img_float;
    image.convertTo(img_float, CV_32F);
    
    float h_image[height * width];
    std::memcpy(h_image, img_float.ptr<float>(), height * width * sizeof(float));

    Keypoint* h_keypoints = keypoints.data();

    // Allocate device memory and copy input data over to GPU
    float *d_image, *d_orientations;
    Keypoint *d_keypoints;
    cudaMalloc(&d_image, height*width*sizeof(float));
    cudaMalloc(&d_keypoints, keypoints.size()*sizeof(Keypoint));
    cudaMalloc(&d_orientations, keypoints.size()*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_image, h_image, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keypoints, h_keypoints, keypoints.size()*sizeof(Keypoint), cudaMemcpyHostToDevice);
    cudaMemset(d_orientations, 0, keypoints.size()*sizeof(float));
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Run GPU kernel
    d_Orientations<<<(keypoints.size()+block_size-1)/block_size, block_size>>>(d_image, d_keypoints, d_orientations, width, height, patch_size, keypoints.size());
    cudaCheckErrors("kernel launch failure");

    // Copy output from device to host
    orientations.resize(keypoints.size());
    cudaMemcpy(orientations.data(), d_orientations, keypoints.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
}