#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "orb.hpp"
#include "GaussianBlur.cuh"
#include "Fast.cuh"
#include "Brief.cuh"

OrientedFAST::OrientedFAST(int nfeatures, int threshold, int n, int nms_window, int patch_size) {
    this->nfeatures = nfeatures;
    this->threshold = threshold;
    this->n = n;
    this->nms_window = nms_window;
    this->patch_size = patch_size;
}

std::vector<Keypoint> OrientedFAST::detect(const cv::Mat& image) {
    std::vector<Keypoint> keypoints;
    int kp_count = Fast(image, keypoints, threshold, n, nms_window, nfeatures);
    keypoints.resize(kp_count);
    return keypoints;
}

std::vector<float> OrientedFAST::compute_orientations(const cv::Mat& image, const std::vector<Keypoint>& keypoints) {
    std::vector<float> orientations;
    Orientations(image, keypoints, orientations, patch_size);
    return orientations;
}

RotatedBRIEF::RotatedBRIEF() {
    this->n_bits = 256;
    this->patch_size = 31;
}

std::vector<ORBDescriptor> RotatedBRIEF::compute(const cv::Mat& image, const std::vector<Keypoint>& keypoints, const std::vector<float>& orientations) {
    std::vector<ORBDescriptor> descriptors;
    Brief(image, keypoints, orientations, descriptors, n_bits, patch_size);
    return descriptors;
}

ORB::ORB(int nfeatures, float scaleFactor, int nlevels) {
    this->nfeatures = nfeatures;
    this->scaleFactor = scaleFactor;
    this->nlevels = nlevels;

    pyramid.resize(nlevels);

    std::cout << "scaleFactor: " << scaleFactor << std::endl;
    std::cout << "nlevels: " << nlevels << std::endl;
}

void ORB::detectAndCompute(const cv::Mat& image, std::vector<Keypoint>& keypoints, std::vector<float>& orientations, std::vector<ORBDescriptor>& descriptors) {
    keypoints = fast.detect(image);
    std::cout << keypoints.size() << std::endl;
    orientations = fast.compute_orientations(image, keypoints);
    descriptors = brief.compute(image, keypoints, orientations);
}

void ORB::buildPyramid(const cv::Mat& image) {
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