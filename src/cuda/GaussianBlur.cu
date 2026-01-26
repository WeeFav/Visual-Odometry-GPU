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
const float h_kernel [kernel_size * kernel_size] = {
    1, 4, 7, 4, 1,
    4, 16, 26, 16, 4,
    7, 26, 41, 26, 7,
    4, 16, 26, 16, 4,
    1, 4, 7, 4, 1
};


const int block_size = 16; // all threads in the block participates in loading input into shared memory but not in computing output
const int output_tile_size = block_size - kernel_radius * 2;

__constant__ float kFilter_d[kernel_size][kernel_size];

__global__ void d_GaussianBlur(const float *input, float *output, const int height, const int width, const int height_out, const int width_out) {
    __shared__ float input_tile[block_size][block_size];
    
    // location in input
    int idx = blockIdx.x * output_tile_size + threadIdx.x;
    int idy = blockIdx.y * output_tile_size + threadIdx.y;

    if ((idx < width) && (idy < height)) {
        input_tile[threadIdx.y][threadIdx.x] = input[idy * width + idx];
    }
    else {
        input_tile[threadIdx.y][threadIdx.x] = 0.0f;
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
                sum += input_tile[i + tile_idy][j + tile_idx] * kFilter_d[i][j];
            }
        }

        int out_x = idx - kernel_radius;
        int out_y = idy - kernel_radius;

        if(out_x >= 0 && out_x < width_out && out_y >= 0 && out_y < height_out) {
            output[out_y * width_out + out_x] = sum / 273.0f;
        }

    }
}

void GaussianBlur(const cv::Mat& image, cv::Mat& dst) {
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, kernel_radius, kernel_radius, kernel_radius, kernel_radius, cv::BORDER_REFLECT_101);
    
    int width = padded.cols;
    int height = padded.rows;

    cv::Mat img_float;
    padded.convertTo(img_float, CV_32F);

    float h_input[height * width];
    std::memcpy(h_input, img_float.ptr<float>(), height * width * sizeof(float));

    int height_out = (height - kernel_size + 2 * 0) / 1 + 1;
    int width_out = (width - kernel_size + 2 * 0) / 1 + 1;    
    
    cudaFree(0); // warm up

    /* create and start timer */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Allocate device memory and copy input data over to GPU
    float *d_input, *d_output;
    cudaMalloc(&d_input, height*width*sizeof(float));
    cudaMalloc(&d_output, height_out*width_out*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    
    cudaMemcpy(d_input, h_input, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kFilter_d, h_kernel, kernel_size*kernel_size*sizeof(float));
    cudaCheckErrors("cudaMemcpy H2D failure");
    
    // Run GPU kernel
    // Normally all threads in a block load to shm and contribute to the output. When halo is needed, shm will be larger than block size and some threads will load halo. Since every thread in the block contributes to output, stride for grid is equals to block size
    // Here, block size includes threads that only load halo and do no computing. We can imagine block size equals to the expanded shm from above, and output_tile_size is the block size from above. Thus,  stride for grid is equals to output_tile_size since we can only skip over threads that actually contribute to output.
    dim3 block(block_size, block_size);
    dim3 grid((width+output_tile_size-1)/output_tile_size, (height+output_tile_size-1)/output_tile_size);
    d_GaussianBlur<<<grid, block>>>(d_input, d_output, height, width, height_out, width_out);
    cudaCheckErrors("kernel launch failure");
    
    // Copy output from device to host
    float h_output[height_out * width_out];
    cudaMemcpy(h_output, d_output, height_out*width_out*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    /* stop CPU timer */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );

    cv::Mat output_mat(height_out, width_out, CV_32F);
    std::memcpy(output_mat.ptr<float>(), h_output, height_out * width_out * sizeof(float));    
    output_mat.convertTo(dst, CV_8U);
}