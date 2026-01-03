#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "GaussianBlur.cuh"
#include "Fast.cuh"
#include "Brief.cuh"

class OrientedFAST {
public:
    OrientedFAST(int nfeatures=3000, int threshold=50, int n=9, int nms_window=3, int patch_size=9) {
        this->nfeatures = nfeatures;
        this->threshold = threshold;
        this->n = n;
        this->nms_window = nms_window;
        this->patch_size = patch_size;
    }

    std::vector<Keypoint> detect(const cv::Mat& image) {
        std::vector<Keypoint> keypoints;
        int kp_count = Fast(image, keypoints, threshold, n, nms_window, nfeatures);
        keypoints.resize(kp_count);
        return keypoints;
    }

    std::vector<float> compute_orientations(const cv::Mat& image, const std::vector<Keypoint>& keypoints) {
        std::vector<float> orientations;
        Orientations(image, keypoints, orientations, patch_size);
        return orientations;
    }

private:
    int nfeatures;
    int threshold;
    int n;
    int nms_window;
    int patch_size;
};

class RotatedBRIEF {
public:
    RotatedBRIEF() {
        this->n_bits = 256;
        this->patch_size = 31;
    }

    std::vector<ORBDescriptor> compute(const cv::Mat& image, const std::vector<Keypoint>& keypoints, const std::vector<float>& orientations) {
        std::vector<ORBDescriptor> descriptors;
        Brief(image, keypoints, orientations, descriptors, n_bits, patch_size);
        return descriptors;
    }
private:
    int n_bits;
    int patch_size;
};

class ORB {
public:
    ORB(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8) {
        this->nfeatures = nfeatures;
        this->scaleFactor = scaleFactor;
        this->nlevels = nlevels;

        pyramid.resize(nlevels);

        std::cout << "scaleFactor: " << scaleFactor << std::endl;
        std::cout << "nlevels: " << nlevels << std::endl;
    }

    void detectAndCompute(const cv::Mat& image, std::vector<Keypoint> keypoints, std::vector<ORBDescriptor> descriptors) {
        buildPyramid(image);
    }

private:
    int nfeatures;
    float scaleFactor;
    int nlevels;
    std::vector<cv::Mat> pyramid;

    void buildPyramid(const cv::Mat& image) {
        pyramid[0] = image;
        int H = image.rows;
        int W = image.cols;

        for (int i=1; i<nlevels; i++) {
            float scale = pow(scaleFactor, i);
            cv::Size newSize(round(W / scale), round(H / scale));

            cv::Mat resized;
            cv::resize(pyramid[0], resized, newSize, 0, 0, cv::INTER_LINEAR);        
            GaussianBlur(resized, pyramid[i]);    
        }

        // visualize
        for (int i=0; i<pyramid.size(); i++) {
            cv::imshow("Octave " + std::to_string(i), pyramid[i]);
        }
        cv::waitKey(0);

    }
};

int main() {
    cv::Mat image = cv::imread("/home/marvin/Visual-Odometry-GPU/000000.png", cv::IMREAD_GRAYSCALE);
    // ORB orb;
    // orb.detectAndCompute(image);

    OrientedFAST fast;
    std::vector<Keypoint> keypoints = fast.detect(image);
    std::vector<float> orientations = fast.compute_orientations(image, keypoints);
    
    cv::Mat image_color;
    cv::cvtColor(image, image_color, cv::COLOR_GRAY2BGR);
    for (int i=0; i<keypoints.size(); i++) {
        int x = keypoints[i].x;
        int y = keypoints[i].y;
        float dx = 10 * std::cos(orientations[i]);
        float dy = 10 * std::sin(orientations[i]);
        cv::circle(image_color, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), 1);
        cv::arrowedLine(image_color, cv::Point(x, y), cv::Point(static_cast<int>(x + dx), static_cast<int>(y + dy)), (0, 0, 255), 1, 8, 0, 0.3);
    }
    cv::imshow("Fast", image_color);
    cv::waitKey(0);

    RotatedBRIEF brief;
    std::vector<ORBDescriptor> descriptors = brief.compute(image, keypoints, orientations);

    return 0;
}