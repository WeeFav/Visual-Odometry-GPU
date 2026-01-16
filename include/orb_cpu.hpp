#ifndef ORB_CPU_H
#define ORB_CPU_H

class OrientedFASTCPU {
public:
    OrientedFASTCPU(int nfeatures=3000, int threshold=50, int n=9, int nms_window=3, int patch_size=9);
    std::vector<Keypoint> detect(const cv::Mat& image);
    std::vector<float> compute_orientations(const cv::Mat& image, const std::vector<Keypoint>& keypoints);

private:
    int nfeatures;
    int threshold;
    int n;
    int nms_window;
    int patch_size;
};

class RotatedBRIEFCPU {
public:
    RotatedBRIEFCPU();
    int sum5x5(const cv::Mat &integral, int x, int y, int width);
    std::vector<ORBDescriptor> compute(const cv::Mat& image, const std::vector<Keypoint>& keypoints, const std::vector<float>& orientations);
private:
    int n_bits;
    int patch_size;
};

class ORBCPU {
public:
    ORBCPU(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8);
    void detectAndCompute(const cv::Mat& image, std::vector<Keypoint>& keypoints, std::vector<float>& orientations, std::vector<ORBDescriptor>& descriptors);

private:
    int nfeatures;
    float scaleFactor;
    int nlevels;
    std::vector<cv::Mat> pyramid;

    OrientedFASTCPU fast;
    RotatedBRIEFCPU brief;

    void buildPyramid(const cv::Mat& image);
};

#endif // ORB_CPU_H