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

const int block_size = 16; // all threads in the block participates in loading input into shared memory and compute ouput. some threads need to additionally load halo cells

__global__ void d_NMS(
    const float* scores,
    Keypoint* keypoints,
    int* kp_count,
    int width,
    int height,
    int NMS_RADIUS,
    int nfeatures,
    float threshold
)
{
    // Dynamic shared memory (1D)
    extern __shared__ float score_tile[];
    int shared_width = block_size + 2 * NMS_RADIUS; 

    // global pixel coords
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // shared memory coords
    int sx = threadIdx.x + NMS_RADIUS;
    int sy = threadIdx.y + NMS_RADIUS;       

    // load center
    float val = 0.0f;
    if (x < width && y < height)
        val = scores[y * width + x];

    score_tile[sy * shared_width + sx] = val;

    int gx, gy;

    // load halo
    if (threadIdx.x < NMS_RADIUS) {
        // left
        gx = x - NMS_RADIUS;
        score_tile[sy * shared_width + (sx - NMS_RADIUS)] = (gx >= 0 && y < height) ? scores[y * width + gx] : 0.0f;
        
        // right
        gx = x + block_size;
        score_tile[sy * shared_width + (sx + block_size)] = (gx < width && y < height) ? scores[y * width + gx] : 0.0f;
    }

    if (threadIdx.y < NMS_RADIUS) {
        // top
        gy = y - NMS_RADIUS;
        score_tile[(sy - NMS_RADIUS) * shared_width + sx] = (gy >= 0 && x < width) ? scores[gy * width + x] : 0.0f;

        // bottom
        gy = y + block_size;
        score_tile[(sy + block_size) * shared_width + sx] = (gy < height && x < width) ? scores[gy * width + x] : 0.0f;
    }

    // corners
    if (threadIdx.x < NMS_RADIUS && threadIdx.y < NMS_RADIUS) {
        // top-left
        gx = x - NMS_RADIUS;
        gy = y - NMS_RADIUS;
        score_tile[(sy - NMS_RADIUS) * shared_width + (sx - NMS_RADIUS)] = (gx >= 0 && gy >= 0) ? scores[gy * width + gx] : 0.0f;

        // top-right
        gx = x + block_size;
        gy = y - NMS_RADIUS;
        score_tile[(sy - NMS_RADIUS) * shared_width + (sx + block_size)] = (gx < width && gy >= 0) ? scores[gy * width + gx] : 0.0f;

        // bottom-left
        gx = x - NMS_RADIUS;
        gy = y + block_size;
        score_tile[(sy + block_size) * shared_width + (sx - NMS_RADIUS)] = (gx >= 0 && gy < height) ? scores[gy * width + gx] : 0.0f;

        // bottom-right
        gx = x + block_size;
        gy = y + block_size;
        score_tile[(sy + block_size) * shared_width + (sx + block_size)] = (gx < width && gy < height) ? scores[gy * width + gx] : 0.0f;
    }

    __syncthreads();

    // boundary check
    if (x < NMS_RADIUS || y < NMS_RADIUS || x >= width - NMS_RADIUS || y >= height - NMS_RADIUS) return;       

    // corner check
    if (val <= threshold) return; 

    // NMS test
    bool is_max = true;
    #pragma unroll // expand loop at compile time
    for (int dy = -NMS_RADIUS; dy <= NMS_RADIUS; ++dy) {
        #pragma unroll
        for (int dx = -NMS_RADIUS; dx <= NMS_RADIUS; ++dx) {
            if (dx == 0 && dy == 0) continue; // center pixel
            if (score_tile[(sy + dy) * shared_width + (sx + dx)] > val) {
                is_max = false;
                break;
            }
        }
        if (!is_max) break;
    }

    if (!is_max) return;

    // append keypoint
    int idx = atomicAdd(kp_count, 1);
    if (idx < nfeatures) {
        keypoints[idx].x = x;
        keypoints[idx].y = y;
    }
}

void NMS(const cv::Mat& input, std::vector<Keypoint>& keypoints, int nms_window, int nfeatures, float threshold) {
    int height = input.rows;
    int width = input.cols;
    
    const float* h_input = input.ptr<float>();

    // Allocate device memory and copy input data over to GPU
    float *d_input;
    Keypoint* d_keypoints;
    int *d_kp_count;
    cudaMalloc(&d_input, height*width*sizeof(float));
    cudaMalloc(&d_keypoints, nfeatures*sizeof(Keypoint));
    cudaMalloc(&d_kp_count, sizeof(int));

    cudaMemcpy(d_input, h_input, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_keypoints, 0, nfeatures*sizeof(Keypoint));
    cudaMemset(d_kp_count, 0, sizeof(int));

    // Run GPU kernel
    int NMS_RADIUS = nms_window / 2;
    dim3 block(block_size, block_size);
    dim3 grid((width+block_size-1)/block_size, (height+block_size-1)/block_size);
    size_t shared_mem_bytes = (block_size + 2 * NMS_RADIUS) * (block_size + 2 * NMS_RADIUS) * sizeof(float);
    d_NMS<<<grid, block, shared_mem_bytes>>>(d_input, d_keypoints, d_kp_count, width, height, NMS_RADIUS, nfeatures, threshold);
    cudaCheckErrors("kernel launch failure");

    // Copy output from device to host
    int h_kp_count = 0;
    cudaMemcpy(&h_kp_count, d_kp_count, sizeof(int), cudaMemcpyDeviceToHost);
    keypoints.resize(h_kp_count);
    cudaMemcpy(keypoints.data(), d_keypoints, h_kp_count*sizeof(Keypoint), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
}