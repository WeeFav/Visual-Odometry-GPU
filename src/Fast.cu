#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

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

struct Keypoint { int x, y; };

const int block_size = 16; // all threads in the block participates in loading input into shared memory and compute ouput. some threads need to additionally load halo cells
const RADIUS = 3;

__constant__ int d_circle_offsets[16][2] = {
    {0, -3}, {1, -3}, {2, -2}, {3, -1},
    {3,  0}, {3,  1}, {2,  2}, {1,  3},
    {0,  3}, {-1, 3}, {-2, 2}, {-3, 1},
    {-3, 0}, {-3,-1}, {-2,-2}, {-1,-3}
};

__global__ void d_Fast(
    const float *input, 
    Keypoint* keypoints,
    float *score_img,
    int* kp_count,
    int height,
    int width,
    int nfeatures
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
    score_img[idy * width + idx] = score;
    
    // append keypoint atomically (check capacity)
    int index = atomicAdd(kp_count, 1);
    if (index < nfeatures) {
        keypoints[index].x = idx;
        keypoints[index].y = idy;
    }
}

void Fast(const cv::Mat& image, std::vector<Keypoint>& keypoints, int nfeatures, int threshold, int n) {
    int width = image.cols;
    int height = image.rows;

    cv::Mat img_float;
    image.convertTo(img_float, CV_32F);
    
    float h_input[height * width];
    std::memcpy(h_input, img_float.ptr<float>(), height * width * sizeof(float));

    // Allocate device memory and copy input data over to GPU
    float *d_input, *d_output;
    Keypoint* d_keypoints;
    cudaMalloc(&d_input, height*width*sizeof(float));
    cudaMalloc(&d_output, height*width*sizeof(float));
    cudaMalloc(&d_keypoints, nfeatures*sizeof(Keypoint));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_input, h_input, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Run GPU kernel
    dim3 block(block_size, block_size);
    dim3 grid((width+block_size-1)/block_size, (height+block_size-1)/block_size);
    d_Fast<<<grid, block>>>(d_input, d_keypoints, d_output, height, width, nfeatures, threshold, n);

}