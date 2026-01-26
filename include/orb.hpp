#ifndef ORB_H
#define ORB_H

struct Keypoint { int x, y; };

struct ORBDescriptor {
    uint8_t data[32];
};

class OrientedFAST {
public:
    OrientedFAST(int nfeatures=3000, int threshold=50, int n=9, int nms_window=3, int patch_size=9);
    std::vector<Keypoint> detect(const cv::Mat& image);
    std::vector<float> compute_orientations(const cv::Mat& image, const std::vector<Keypoint>& keypoints);

private:
    int nfeatures;
    int threshold;
    int n;
    int nms_window;
    int patch_size;
};

class RotatedBRIEF {
public:
    RotatedBRIEF();
    std::vector<ORBDescriptor> compute(const cv::Mat& image, const std::vector<Keypoint>& keypoints, const std::vector<float>& orientations);
    
private:
    int n_bits;
    int patch_size;
};

class ORB {
public:
    ORB(int nfeatures=3000, float scaleFactor=1.2f, int nlevels=8);
    void detectAndCompute(const cv::Mat& image, std::vector<Keypoint>& keypoints, std::vector<float>& orientations, std::vector<ORBDescriptor>& descriptors);

private:
    int nfeatures;
    float scaleFactor;
    int nlevels;
    std::vector<cv::Mat> pyramid;

    OrientedFAST fast;
    RotatedBRIEF brief;

    void buildPyramid(const cv::Mat& image);
};

#endif // ORB_H