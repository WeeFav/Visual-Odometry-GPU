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

const int block_size = 16; // all threads in the block participates in loading input into shared memory but not in computing output

__global__ void d_conv2d(const float *input, const float *kernel, float *output, int height, int width, int height_out, int width_out, int kernel_size, int kernel_radius, int output_tile_size) {
    __shared__ float tile[block_size][block_size];
    
    // location in input
    int idx = blockIdx.x * output_tile_size + threadIdx.x;
    int idy = blockIdx.y * output_tile_size + threadIdx.y;

    if ((idx < width) && (idy < height)) {
        tile[threadIdx.y][threadIdx.x] = input[idy * width + idx];
    }
    else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // location in input tile
    int tile_idx = threadIdx.x - kernel_radius;
    int tile_idy = threadIdx.y - kernel_radius;

    // only non-halo cells compute output 
    if (tile_idx >= 0 && tile_idx < output_tile_size && tile_idy >= 0 && tile_idy < output_tile_size) {
        float sum = 0;
        for (int i=0; i<kernel_size; i++) {
            for (int j=0; j<kernel_size; j++) {
                sum += tile[i + tile_idy][j + tile_idx] * kernel[i * kernel_size + j];
            }
        }

        int out_x = idx - kernel_radius;
        int out_y = idy - kernel_radius;
        
        if(out_x >= 0 && out_x < width_out && out_y >= 0 && out_y < height_out) {
            output[out_y * width_out + out_x] = sum;
        }
    }
}

void conv2d(const cv::Mat& image, cv::Mat& dst, float* kernel, int kernel_size) {
    int height = image.rows;
    int width = image.cols;

    cv::Mat img_float, output_float;
    image.convertTo(img_float, CV_32F);

    // For a kernel of size F x F, padding P, and stride S
    // H_out = (H + 2P - F) / S + 1
    int padding = 0; // input image should be pre-padded
    int stride = 1;
    int height_out = (height - 2 * padding - kernel_size) / stride + 1;
    int width_out = (width - 2 * padding - kernel_size) / stride + 1; 
    
    output_float.create(height_out, width_out, CV_32F);
    
    float* h_input = img_float.ptr<float>();
    float *h_output = output_float.ptr<float>();

    // Allocate device memory and copy input data over to GPU
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, height*width*sizeof(float));
    cudaMalloc(&d_kernel, kernel_size*kernel_size*sizeof(float));
    cudaMalloc(&d_output, height_out*width_out*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_input, h_input, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, height_out*width_out*sizeof(float));
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Run GPU kernel
    int kernel_radius = kernel_size / 2;
    int output_tile_size = block_size - kernel_radius * 2;
    dim3 block(block_size, block_size);
    dim3 grid((width+output_tile_size-1)/output_tile_size, (height+output_tile_size-1)/output_tile_size);
    d_conv2d<<<grid, block>>>(d_input, d_kernel, d_output, height, width, height_out, width_out, kernel_size, kernel_radius, output_tile_size);
    cudaCheckErrors("kernel launch failure");

    // Copy output from device to host
    cudaMemcpy(h_output, d_output, height_out*width_out*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    output_float.convertTo(dst, CV_8U);
}