#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "GaussianBlur.cuh"

int main() {
    cv::Mat image = cv::imread("/home/d300/VO/data/kitti/data_odometry_gray/dataset/sequences/00/image_0/000000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat output;
    GaussianBlur(image, output);

    std::cout << output.rows << std::endl;
    std::cout << output.cols << std::endl;

    cv::imshow("output", output);
    cv::waitKey(0);
    return 0;
}