#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "GaussianBlur.cuh"
#include "Fast.cuh"

struct Keypoint { int x, y; };

class OrientedFAST {
public:
    OrientedFAST(int nfeatures=500, int threshold=50, int n=9, int nms_window=1, int patch_size=9) {
        this->nfeatures = nfeatures;
        this->threshold = threshold;
        this->n = n;
        this->nms_window = nms_window;
        this->patch_size = patch_size;
    }

    void detect(const cv::Mat& image) {
        std::vector<Keypoint> keypoints;

        Fast(image, keypoints, nfeatures, threshold, n);
    }

private:
    int nfeatures;
    int threshold;
    int n;
    int nms_window;
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

    void detectAndCompute(const cv::Mat& image) {
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
    cv::Mat image = cv::imread("/home/d300/VO/data/kitti/data_odometry_gray/dataset/sequences/00/image_0/000000.png", cv::IMREAD_GRAYSCALE);
    ORB orb;
    orb.detectAndCompute(image);
    return 0;
}