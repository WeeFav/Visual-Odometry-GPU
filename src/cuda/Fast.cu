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
const int RADIUS = 3;

__constant__ int d_circle_offsets[16][2] = {
    {0, -3}, {1, -3}, {2, -2}, {3, -1},
    {3,  0}, {3,  1}, {2,  2}, {1,  3},
    {0,  3}, {-1, 3}, {-2, 2}, {-3, 1},
    {-3, 0}, {-3,-1}, {-2,-2}, {-1,-3}
};

__global__ void d_Fast(
    const float *input, 
    float *scores,
    int height,
    int width,
    int threshold, 
    int n) 
{
    __shared__ float input_tile[block_size + 2 * RADIUS][block_size + 2 * RADIUS];

    // location in input
    int idx = blockIdx.x * block_size + threadIdx.x;
    int idy = blockIdx.y * block_size + threadIdx.y;

    // location in shared memory
    int shm_x = threadIdx.x + RADIUS;
    int shm_y = threadIdx.y + RADIUS;

    // load non-halo region
    if ((idx < width) && (idy < height)) {
        input_tile[shm_y][shm_x] = input[idy * width + idx];
    }

    // LEFT halo
    if (threadIdx.x < RADIUS) {
        int gx = idx - RADIUS;
        int gy = idy;
        int sx = shm_x - RADIUS;
        int sy = shm_y;

        if (gx >= 0 && gy < height)
            input_tile[sy][sx] = input[gy * width + gx];
        else
            input_tile[sy][sx] = 0;    // or clamp, mirror, etc.
    }

    // RIGHT halo
    if (threadIdx.x >= block_size - RADIUS) {
        int gx = idx + RADIUS;
        int gy = idy;
        int sx = shm_x + RADIUS;
        int sy = shm_y;

        if (gx < width && gy < height)
            input_tile[sy][sx] = input[gy * width + gx];
        else
            input_tile[sy][sx] = 0;
    }   

    // TOP halo
    if (threadIdx.y < RADIUS) {
        int gx = idx;
        int gy = idy - RADIUS;
        int sx = shm_x;
        int sy = shm_y - RADIUS;

        if (gy >= 0 && gx < width)
            input_tile[sy][sx] = input[gy * width + gx];
        else
            input_tile[sy][sx] = 0;
    }

    // BOTTOM halo
    if (threadIdx.y >= block_size - RADIUS) {
        int gx = idx;
        int gy = idy + RADIUS;
        int sx = shm_x;
        int sy = shm_y + RADIUS;

        if (gy < height && gx < width)
            input_tile[sy][sx] = input[gy * width + gx];
        else
            input_tile[sy][sx] = 0;
    }

    // TOP-LEFT
    if (threadIdx.x < RADIUS && threadIdx.y < RADIUS) {
        int gx = idx - RADIUS;
        int gy = idy - RADIUS;
        int sx = shm_x - RADIUS;
        int sy = shm_y - RADIUS;

        if (gx >= 0 && gy >= 0)
            input_tile[sy][sx] = input[gy * width + gx];
        else
            input_tile[sy][sx] = 0;
    }

    // TOP-RIGHT
    if (threadIdx.x >= block_size - RADIUS && threadIdx.y < RADIUS) {
        int gx = idx + RADIUS;
        int gy = idy - RADIUS;
        int sx = shm_x + RADIUS;
        int sy = shm_y - RADIUS;

        if (gx < width && gy >= 0)
            input_tile[sy][sx] = input[gy * width + gx];
        else
            input_tile[sy][sx] = 0;
    }

    // BOTTOM-LEFT
    if (threadIdx.x < RADIUS && threadIdx.y >= block_size - RADIUS) {
        int gx = idx - RADIUS;
        int gy = idy + RADIUS;
        int sx = shm_x - RADIUS;
        int sy = shm_y + RADIUS;

        if (gx >= 0 && gy < height)
            input_tile[sy][sx] = input[gy * width + gx];
        else
            input_tile[sy][sx] = 0;
    }

    // BOTTOM-RIGHT
    if (threadIdx.x >= block_size - RADIUS && threadIdx.y >= block_size - RADIUS) {
        int gx = idx + RADIUS;
        int gy = idy + RADIUS;
        int sx = shm_x + RADIUS;
        int sy = shm_y + RADIUS;

        if (gx < width && gy < height)
            input_tile[sy][sx] = input[gy * width + gx];
        else
            input_tile[sy][sx] = 0;
    }

    __syncthreads();

    // Only consider pixels that have a full 3-pixel neighborhood
    if (idx < RADIUS || idx >= (width - RADIUS) || idy < RADIUS || idy >= (height - RADIUS)) return;

    float Ip = input_tile[shm_y][shm_x];

    // FAST check: examine pixels 1, 5, 9, 13 first (indices 0, 4, 8, 12)
    const int check_idx[4] = {0, 4, 8, 12};
    int brighter = 0, darker = 0;
    for (int k = 0; k < 4; ++k) {
        int offx = d_circle_offsets[check_idx[k]][0];
        int offy = d_circle_offsets[check_idx[k]][1];
        float cp = input_tile[shm_y + offy][shm_x + offx]; // safe: within halo
        if (cp >= Ip + threshold) brighter++;
        else if (cp <= Ip - threshold) darker++;
    }

    // If none are sufficiently bright/dark, skip
    if (max(brighter, darker) < 3) return;

    // Full test: check for n contiguous pixels
    float circ[16];
    for (int i = 0; i < 16; ++i) {
        int offx = d_circle_offsets[i][0];
        int offy = d_circle_offsets[i][1];
        circ[i] = input_tile[shm_y + offy][shm_x + offx]; // safe: within halo
    } 
       
    // Double the array logically via modulo indexing to handle wrap-around
    bool is_corner = false;
    for (int start = 0; start < 16 && !is_corner; ++start) {
        // check contiguous segment of length n
        bool all_br = true; // brighter
        bool all_dr = true; // darker
        for (int k = 0; k < n; ++k) {
            float v = circ[(start + k) & 15]; // modulo 16 (faster with &15)
            if (!(v >= Ip + threshold)) all_br = false;
            if (!(v <= Ip - threshold)) all_dr = false;
            if (!all_br && !all_dr) break;
        }
        if (all_br || all_dr) is_corner = true;
    }

    if (!is_corner) return;

    // --- compute score (sum abs diffs) ---
    float score = 0.0f;
    for (int i = 0; i < 16; ++i) score += fabsf(Ip - circ[i]);

    // write score to score image
    scores[idy * width + idx] = score;
}

__global__ void d_NMS(
    const float* scores,
    Keypoint* keypoints,
    int* kp_count,
    int width,
    int height,
    int NMS_RADIUS,
    int nfeatures
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
    if (val <= 0.0f) return; 

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

int Fast(const cv::Mat& image, std::vector<Keypoint>& keypoints, int threshold, int n, int nms_window, int nfeatures) {
    int width = image.cols;
    int height = image.rows;

    cv::Mat img_float;
    image.convertTo(img_float, CV_32F);
    
    float h_image[height * width];
    std::memcpy(h_image, img_float.ptr<float>(), height * width * sizeof(float));

    // Allocate device memory and copy input data over to GPU
    float *d_image, *d_scores;
    cudaMalloc(&d_image, height*width*sizeof(float));
    cudaMalloc(&d_scores, height*width*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_image, h_image, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_scores, 0, height*width*sizeof(float));
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Run GPU kernel
    dim3 block(block_size, block_size);
    dim3 grid((width+block_size-1)/block_size, (height+block_size-1)/block_size);
    d_Fast<<<grid, block>>>(d_image, d_scores, height, width, threshold, n);
    cudaCheckErrors("kernel launch failure");

    // Copy output from device to host
    float h_scores[height * width];
    cudaMemcpy(h_scores, d_scores, height*width*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    // cv::Mat score_img(height, width, CV_32F, h_scores);
    // score_img = score_img.clone();

    // NMS
    Keypoint* d_keypoints;
    int *d_kp_count;

    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_keypoints, nfeatures*sizeof(Keypoint));
    cudaMalloc(&d_kp_count, sizeof(int));

    cudaMemset(d_keypoints, 0, nfeatures*sizeof(Keypoint));
    cudaMemset(d_kp_count, 0, sizeof(int));

    // Run GPU kernel
    int NMS_RADIUS = nms_window / 2;
    size_t shared_mem_bytes = (block_size + 2 * NMS_RADIUS) * (block_size + 2 * NMS_RADIUS) * sizeof(float);
    d_NMS<<<grid, block, shared_mem_bytes>>>(d_scores, d_keypoints, d_kp_count, width, height, NMS_RADIUS, nfeatures);
    cudaCheckErrors("kernel launch failure");

    // Copy output from device to host
    int h_kp_count = 0;
    keypoints.resize(nfeatures);
    cudaMemcpy(keypoints.data(), d_keypoints, nfeatures*sizeof(Keypoint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_kp_count, d_kp_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    return h_kp_count;
}