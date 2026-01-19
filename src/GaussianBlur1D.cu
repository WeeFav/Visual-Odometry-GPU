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

const int kernel_size = 5;
const int kernel_radius = kernel_size / 2;
const float h_kernel[kernel_size] = {1, 4, 6, 4, 1};

const int block_x = 32;
const int block_y = 8;

__constant__ float d_kernel[kernel_size];

__device__ __forceinline__
int reflect101(int p, int len) {
    if (p < 0)       return -p;
    if (p >= len)    return 2 * len - p - 2;
    return p;
}

__global__ void d_GaussianHorizontal(const float *input, float *output, const int height, const int width) {
    __shared__ float shm[block_y][block_x + 2 * kernel_radius];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int sx = threadIdx.x + kernel_radius;

    // load center cells
    if ((idy < height) && (idx < width)) {
        shm[threadIdx.y][sx] = input[idy * width + idx];
    }
    else {
        shm[threadIdx.y][sx] = 0.0f;
    }

    // load halo cells
    if ((idy < height) && (threadIdx.x < kernel_radius)) {
        int left  = idx - kernel_radius;
        int right = idx + block_x;

        shm[threadIdx.y][sx - kernel_radius] = input[idy * width + reflect101(left, width)];
        shm[threadIdx.y][sx + block_x] = input[idy * width + reflect101(right, width)];
    }

    __syncthreads();

    if ((idy < height) && (idx < width)) {
        float sum = 0.0f;
        #pragma unroll
        for (int k=0; k<kernel_size; k++) {
            sum += d_kernel[k] * shm[threadIdx.y][sx - kernel_radius + k];
        }
        output[idy * width + idx] = sum / 16.0f;        
    }
}

__global__ void d_GaussianVertical(const float *input, float *output, const int height, const int width) {
    __shared__ float shm[block_y + 2 * kernel_radius][block_x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int sy = threadIdx.y + kernel_radius;

    // load center cells
    if ((idy < height) && (idx < width)) {
        shm[sy][threadIdx.x] = input[idy * width + idx];
    }
    else {
        shm[sy][threadIdx.x] = 0.0f;
    }

    // load halo cells
    if ((idy < height) && (idx < width) && (threadIdx.y < kernel_radius)) {
        int top  = idy - kernel_radius;
        int bottom = idy + block_y;

        shm[sy - kernel_radius][threadIdx.x] = input[reflect101(top, height) * width + idx];
        shm[sy + block_y][threadIdx.x] = input[reflect101(bottom, height) * width + idx];
    }

    __syncthreads();

    if ((idy < height) && (idx < width)) {
        float sum = 0.0f;
        #pragma unroll
        for (int k=0; k<kernel_size; k++) {
            sum += d_kernel[k] * shm[sy - kernel_radius + k][threadIdx.x];
        }
        output[idy * width + idx] = sum / 16.0f;        
    }
}

void GaussianBlur1D(const cv::Mat& image, cv::Mat& dst) {
    int width = image.cols;
    int height = image.rows;

    cv::Mat img_float;
    image.convertTo(img_float, CV_32F);

    float h_input[height * width];
    std::memcpy(h_input, img_float.ptr<float>(), height * width * sizeof(float));

    // Allocate device memory and copy input data over to GPU
    float *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, height*width*sizeof(float));
    cudaMalloc(&d_output, height*width*sizeof(float));
    cudaMalloc(&d_temp, height*width*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_input, h_input, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_size*sizeof(float));
    cudaMemset(d_temp, 0, height*width*sizeof(float));
    cudaMemset(d_output, 0, height*width*sizeof(float));
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Run GPU kernel
    dim3 block(block_x, block_y);
    dim3 grid((width+block_x-1)/block_x, (height+block_y-1)/block_y);
    d_GaussianHorizontal<<<grid, block>>>(d_input, d_temp, height, width);
    cudaCheckErrors("kernel launch failure");

    d_GaussianVertical<<<grid, block>>>(d_temp, d_output, height, width);
    cudaCheckErrors("kernel launch failure");

    // Copy output from device to host
    float h_output[height * width];
    cudaMemcpy(h_output, d_output, height*width*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    cv::Mat output_mat(height, width, CV_32F);
    std::memcpy(output_mat.ptr<float>(), h_output, height * width * sizeof(float));    
    output_mat.convertTo(dst, CV_8U);
}