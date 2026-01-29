#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "orb.hpp"
#include "GaussianBlur.cuh"
#include "Fast.cuh"
#include "Brief.cuh"
#include "HarrisScore.cuh"

OrientedFAST::OrientedFAST(int threshold, int n, int nms_window, int patch_size) :
    threshold(threshold),
    n(n),
    nms_window(nms_window),
    patch_size(patch_size)
{
    std::cout << "[FAST PARAM] threshold: " << threshold << std::endl; // pixel must be > or < center pixel + threshold to be consider brighter or darker
    std::cout << "[FAST PARAM] n: " << n << std::endl; // number of contiguous pixel that needs to be > or < center pixel 
    std::cout << "[FAST PARAM] nms_window: " << nms_window << std::endl; // nms size for fast feature detection
    std::cout << "[FAST PARAM] patch_size: " << patch_size << std::endl; // patch size for orientation calculation for a given keypoint
}

std::vector<Keypoint> OrientedFAST::detect(const cv::Mat& image, int nfeatures) {
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
    std::cout << "[BRIEF PARAM] n_bits: " << n_bits << std::endl; // number of bits for each descriptor
    std::cout << "[BRIEF PARAM] patch_size: " << patch_size << std::endl; // size of the brief comparison region
}

std::vector<ORBDescriptor> RotatedBRIEF::compute(const cv::Mat& image, const std::vector<Keypoint>& keypoints, const std::vector<float>& orientations) {
    std::vector<ORBDescriptor> descriptors;
    Brief(image, keypoints, orientations, descriptors, n_bits, patch_size);
    return descriptors;
}

ORB::ORB(int nfeatures, float scaleFactor, int nlevels) :
    nfeatures(nfeatures),
    scaleFactor(scaleFactor),
    nlevels(nlevels)
{
    std::cout << "[ORB PARAM] nfeatures: " << nfeatures << std::endl; // maximum features to detec
    std::cout << "[ORB PARAM] scaleFactor: " << scaleFactor << std::endl; // pyramid scale factor
    std::cout << "[ORB PARAM] nlevels: " << nlevels << std::endl; // levels of pyramid

    pyramid.resize(nlevels);
}

void ORB::detectAndCompute(const cv::Mat& image, std::vector<Keypoint>& keypoints, std::vector<float>& orientations, std::vector<ORBDescriptor>& descriptors) {
    buildPyramid(image);

    for (int l=0; l<nlevels; l++) {
        int nfeatures_l = nfeatures * ((1 - 1/scaleFactor) / (1 - std::pow(1/scaleFactor, nlevels))) * std::pow(1/scaleFactor, l);
        std::vector<Keypoint> keypoints_l = fast.detect(pyramid[l], nfeatures_l * 2);
        std::vector<float> harris_scores;
        HarrisScore(pyramid[l], keypoints_l, harris_scores, 7, 0.04);

        std::vector<std::pair<int, float>> kp_scores;
        for (int i=0; i<keypoints_l.size(); i++) {
            kp_scores.push_back({i, harris_scores[i]});
        }

        // Semi sort array
        std::nth_element(
            kp_scores.begin(), 
            kp_scores.begin() + nfeatures_l, 
            kp_scores.end(),
            [](const std::pair<int, float> a, const std::pair<Keypoint, float> b) {
                return a.second > b.second;
            }
        );

        // Keep top N harris score
        std::vector<Keypoint> keypoints_filtered;
        for (auto& kp_score : kp_scores) {
            keypoints_filtered.push_back(keypoints_l[kp_score.first]);
        }

        std::cout << keypoints_filtered.size() << std::endl;

        std::vector<float> orientations_l = fast.compute_orientations(pyramid[l], keypoints_filtered);
        std::vector<ORBDescriptor> descriptors_l = brief.compute(image, keypoints_filtered, orientations_l);

        // Map coordinate back to orignial input space
        for (auto& kp : keypoints_filtered) {
            float scale = std::pow(scaleFactor, l);
            kp.x *= scale;
            kp.y *= scale;
        }

        keypoints.insert(keypoints.end(), keypoints_filtered.begin(), keypoints_filtered.end());
        orientations.insert(orientations.end(), orientations_l.begin(), orientations_l.end());
        descriptors.insert(descriptors.end(), descriptors_l.begin(), descriptors_l.end());
    }
    
    // keypoints = fast.detect(image);
    // std::cout << keypoints.size() << std::endl;
    // orientations = fast.compute_orientations(image, keypoints);
    // descriptors = brief.compute(image, keypoints, orientations);
}

void ORB::buildPyramid(const cv::Mat& image) {
    pyramid[0] = image;
    int H = image.rows;
    int W = image.cols;

    for (int i=1; i<nlevels; i++) {
        float scale = pow(scaleFactor, i);
        cv::Size newSize(round(W / scale), round(H / scale));
        cv::resize(pyramid[0], pyramid[i], newSize, 0, 0, cv::INTER_LINEAR); // INTER_LINEAR acts as gaussian blur 
    }

    // // visualize
    // for (int i=0; i<pyramid.size(); i++) {
    //     cv::imshow("Octave " + std::to_string(i), pyramid[i]);
    // }
    // cv::waitKey(0);
}