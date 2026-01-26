#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <map>
#include <tuple>
#include <string>
#include "orb.hpp"
#include "orb_cpu.hpp"
// #include "GaussianBlur.cuh"
#include "GaussianBlur.hpp"
#include "Sobel.hpp"

int main() {
    cv::Mat image = cv::imread("/home/marvin/Visual-Odometry-GPU/000000.png", cv::IMREAD_GRAYSCALE);
    
    cv::Mat a, b;
    GaussianBlurCUDA(image, a, 5);
    SobelCUDA(image, b, 0);
    // GaussianBlur1D(image, b);

    std::cout << image.rows << image.cols << std::endl;
    std::cout << a.rows << a.cols << std::endl;
    std::cout << b.rows << b.cols << std::endl;

    // cv::Mat diff; 
    // cv::compare(a, b, diff, cv::CMP_NE);
    // std::cout << diff.cols * diff.rows << std::endl;
    // std::cout << cv::countNonZero(diff) << std::endl;

    // cv::cvtColor(image, image_cpu, cv::COLOR_GRAY2BGR);
    // cv::imshow("orig", image);
    cv::imshow("blur", a);
    cv::imshow("sobel", b);
    // cv::imshow("1d", b);
    cv::waitKey(0);
    cv::destroyAllWindows();


    // ORBCPU orb_cpu;
    ORB orb_gpu(100);

    // std::vector<Keypoint> keypoints_cpu;
    // std::vector<Keypoint> keypoints_gpu;
    // std::vector<float> orientations_cpu;
    // std::vector<float> orientations_gpu;
    // std::vector<ORBDescriptor> descriptors_cpu;
    // std::vector<ORBDescriptor> descriptors_gpu;

    // // CPU
    // orb_cpu.detectAndCompute(image, keypoints_cpu, orientations_cpu, descriptors_cpu);

    // cv::Mat image_cpu;
    // cv::cvtColor(image, image_cpu, cv::COLOR_GRAY2BGR);
    // for (int i=0; i<keypoints_cpu.size(); i++) {
    //     int x = keypoints_cpu[i].x;
    //     int y = keypoints_cpu[i].y;
    //     float dx = 10 * std::cos(orientations_cpu[i]);
    //     float dy = 10 * std::sin(orientations_cpu[i]);
    //     cv::circle(image_cpu, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), 1);
    //     cv::arrowedLine(image_cpu, cv::Point(x, y), cv::Point(static_cast<int>(x + dx), static_cast<int>(y + dy)), (0, 0, 255), 1, 8, 0, 0.3);
    // }

    // cv::imshow("CPU", image_cpu);

    // // GPU
    // orb_gpu.detectAndCompute(image, keypoints_gpu, orientations_gpu, descriptors_gpu);

    // cv::Mat image_gpu;
    // cv::cvtColor(image, image_gpu, cv::COLOR_GRAY2BGR);
    // for (int i=0; i<keypoints_gpu.size(); i++) {
    //     int x = keypoints_gpu[i].x;
    //     int y = keypoints_gpu[i].y;
    //     float dx = 10 * std::cos(orientations_gpu[i]);
    //     float dy = 10 * std::sin(orientations_gpu[i]);
    //     cv::circle(image_gpu, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), 1);
    //     cv::arrowedLine(image_gpu, cv::Point(x, y), cv::Point(static_cast<int>(x + dx), static_cast<int>(y + dy)), (0, 0, 255), 1, 8, 0, 0.3);
    // }

    // cv::imshow("GPU", image_gpu);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // std::map<std::tuple<int, int>, int> m;
    // for (int i=0; i<keypoints_gpu.size(); i++) {
    //     m[{keypoints_gpu[i].x, keypoints_gpu[i].y}] = i;
    // }

    // std::cout << descriptors_cpu.size() << std::endl;
    // std::cout << descriptors_gpu.size() << std::endl;

    // for (int i=0; i<keypoints_cpu.size(); i++) {
    //     auto it = m.find({keypoints_cpu[i].x, keypoints_cpu[i].y});
    //     if (it != m.end()) {
    //         int j = it->second;
    //         ORBDescriptor desc_cpu = descriptors_cpu[i];
    //         ORBDescriptor desc_gpu = descriptors_gpu[j];

    //         int dist = 0;
    //         for (int k=0; k<32; k++) {
    //             dist += __builtin_popcount(desc_cpu.data[k] ^ desc_gpu.data[k]);
    //         }

    //         std::cout << "Hamming Distance: " << dist << std::endl;
    //     }
    //     else {
    //         std::cout << "Keypoint not found in GPU" << std::endl;
    //     }
    // }

}