#ifndef ORB_H
#define ORB_H

struct Keypoint { int x, y; };

struct ORBDescriptor {
    uint8_t data[32];
};

class OrientedFAST {
public:
    OrientedFAST(int threshold=20, int n=9, int nms_window=3, int patch_size=31);
    std::vector<Keypoint> detect(const cv::Mat& image, int nfeatures);
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
    int n_bits = 256;
    int patch_size = 31;
};

class ORB {
public:
    ORB(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8);
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