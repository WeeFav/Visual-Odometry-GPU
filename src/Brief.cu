#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "orb.hpp"
#include "orb_pattern.hpp"

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
const int subwindow_radius = 5 / 2;

__constant__ int d_briefPattern[256 * 4];

__device__ __forceinline__
int sum5x5(const int* ii, int x, int y, int width)
{
    int x0 = x - 2;
    int y0 = y - 2;
    int x1 = x + 3;
    int y1 = y + 3;

    return ii[y1 * width + x1]
         + ii[y0 * width + x0]
         - ii[y0 * width + x1]
         - ii[y1 * width + x0];
}

__global__ void d_Brief(
    const int* integral,
    const Keypoint* keypoints,
    const float* orientations,
    ORBDescriptor* descriptors,
    int width,
    int height,
    int kp_count
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kp_count) return;

    Keypoint kp = keypoints[idx];
    float angle = orientations[idx];

    float cos = cosf(angle);
    float sin = sinf(angle);

    ORBDescriptor desc;

    #pragma unroll
    for (int i = 0; i < 256; i++) {
        int x1 = d_briefPattern[i * 4];
        int y1 = d_briefPattern[i * 4 + 1];
        int x2 = d_briefPattern[i * 4 + 2];
        int y2 = d_briefPattern[i * 4 + 3];

        // Rotate offsets
        int dx1 = __float2int_rn(cos * x1 - sin * y1);
        int dy1 = __float2int_rn(sin * x1 + cos * y1);

        int dx2 = __float2int_rn(cos * x2 - sin * y2);
        int dy2 = __float2int_rn(sin * x2 + cos * y2);

        int cx1 = kp.x + dx1;
        int cy1 = kp.y + dy1;
        int cx2 = kp.x + dx2;
        int cy2 = kp.y + dy2;

        // Boundary check for 5x5 window
        if (cx1 < subwindow_radius || cy1 < subwindow_radius || cx1 > width-subwindow_radius || cy1 > height-subwindow_radius ||
            cx2 < subwindow_radius || cy2 < subwindow_radius || cx2 > width-subwindow_radius || cy2 > height-subwindow_radius)
            continue;

        int s1 = sum5x5(integral, cx1, cy1, width);
        int s2 = sum5x5(integral, cx2, cy2, width);

        if (s1 < s2) {
            // i >> 3   ==   i / 8 (selects which byte the bit belongs to)
            // i & 7   ==   i % 8 (selects which bit inside the byte)
            desc.data[i >> 3] |= (1 << (i & 7));
        }
    }
}

void Brief(const cv::Mat& image, const std::vector<Keypoint>& keypoints, const std::vector<float>& orientations, std::vector<ORBDescriptor> descriptors, int n_bits, int patch_size) {
    int width = image.cols + 1; // integral image will have extra padding
    int height = image.rows + 1;

    cv::Mat integral;
    cv::integral(image, integral);
    
    int h_integral[height * width];
    std::memcpy(h_integral, integral.ptr<int>(), height * width * sizeof(int));
    
    const Keypoint* h_keypoints = keypoints.data();
    const float* h_orientations = orientations.data();
    
    // Allocate device memory and copy input data over to GPU
    int *d_integral;
    float *d_orientations;
    Keypoint *d_keypoints;
    ORBDescriptor* d_descriptors;

    cudaMalloc(&d_integral, height*width*sizeof(int));
    cudaMalloc(&d_keypoints, keypoints.size()*sizeof(Keypoint));
    cudaMalloc(&d_orientations, keypoints.size()*sizeof(float));
    cudaMalloc(&d_descriptors, keypoints.size()*sizeof(ORBDescriptor));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_integral, h_integral, height*width*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keypoints, h_keypoints, keypoints.size()*sizeof(Keypoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_orientations, h_orientations, keypoints.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_descriptors, 0, keypoints.size()*sizeof(ORBDescriptor));
    cudaMemcpyToSymbol(d_briefPattern, bit_pattern_31_, 256*4*sizeof(int));
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Run GPU kernel
    d_Brief<<<(keypoints.size()+block_size-1)/block_size, block_size>>>(d_integral, d_keypoints, d_orientations, d_descriptors, width, height, keypoints.size());
    cudaCheckErrors("kernel launch failure");

    // Copy output from device to host
    descriptors.resize(keypoints.size());
    cudaMemcpy(descriptors.data(), d_descriptors, keypoints.size()*sizeof(ORBDescriptor), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
}