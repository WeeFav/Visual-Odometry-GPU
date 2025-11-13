#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <iterator>

namespace fs = std::filesystem;

class VisualOdom {
public:
    VisualOdom(const std::string& KITTI_DIR, const std::string& seq) {
        std::string images_dir = KITTI_DIR + "/data_odometry_gray/dataset/sequences/" + seq + "/image_0";
        for (auto& p : fs::directory_iterator(images_dir)) {
            images.push_back(p.path().string());
        }
        std::sort(images.begin(), images.end());

        orb = cv::ORB::create(3000);
        
        cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
        cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
        flann = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);

        // Load ground truth poses
        readPoses(KITTI_DIR, seq);
        // Load calibration
        readCalib(KITTI_DIR, seq);
    }

    void run() {
        std::vector<cv::Point2d> gt_path;
        std::vector<cv::Point2d> est_path;
        cv::Mat cur_pose;

        for (size_t i = 0; i < 1000 && i < images.size(); i++) {
            cv::Mat gt_pose = gt_poses[i].clone();

            if (i == 0) {
                cv::Mat img1 = cv::imread(images[0], cv::IMREAD_GRAYSCALE);
                orb->detectAndCompute(img1, cv::noArray(), kp1, des1);
                cur_pose = gt_pose;
            } else {
                cv::Mat img2 = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
                std::vector<cv::Point2f> pts1, pts2;
                get_matches(img2, pts1, pts2);

                cv::Mat T = get_pose(pts1, pts2);

                double true_scale = cv::norm(gt_poses[i](cv::Range(0, 3), cv::Range(3, 4)) -
                                             gt_poses[i - 1](cv::Range(0, 3), cv::Range(3, 4)));

                T(cv::Range(0, 3), cv::Range(3, 4)) *= true_scale;

                cur_pose = cur_pose * T.inv();

                // Shift the cache: current becomes previous
                kp1 = kp2;
                des1 = des2.clone();
            }

            gt_path.push_back(cv::Point2d(gt_pose.at<double>(0, 3), gt_pose.at<double>(2, 3)));
            est_path.push_back(cv::Point2d(cur_pose.at<double>(0, 3), cur_pose.at<double>(2, 3)));
                
            // Draw paths
            drawPaths(gt_path, est_path);
        }
        cv::waitKey(0);
    }

private:
    std::vector<std::string> images;
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::FlannBasedMatcher> flann;
    std::vector<cv::Mat> gt_poses;
    cv::Mat K;
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    int w = 1000, h = 1000;
    cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC3);

    void readPoses(const std::string& KITTI_DIR, const std::string& seq) {
        std::ifstream pose_file(KITTI_DIR + "/data_odometry_poses/dataset/poses/" + seq + ".txt");
        std::string line;
        while (std::getline(pose_file, line)) {
            std::istringstream ss(line);
            std::vector<double> data((std::istream_iterator<double>(ss)), std::istream_iterator<double>());
            cv::Mat T = cv::Mat::zeros(4, 4, CV_64F);
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 4; c++) {
                    T.at<double>(r, c) = data[r * 4 + c];
                }
            }
            T.at<double>(3, 3) = 1.0;
            gt_poses.push_back(T);
        }        
    }

    void readCalib(const std::string& KITTI_DIR, const std::string& seq) {
        std::ifstream calib_file(KITTI_DIR + "/data_odometry_gray/dataset/sequences/" + seq + "/calib.txt");
        std::string line;
        std::getline(calib_file, line);
        std::string data_str = line.substr(4);
        std::istringstream ss(data_str);
        std::vector<double> calib_params((std::istream_iterator<double>(ss)), std::istream_iterator<double>());
        cv::Mat P = cv::Mat(calib_params).reshape(1, 3);
        K = P(cv::Range(0, 3), cv::Range(0, 3)).clone();
    }

    void get_matches(const cv::Mat &img2, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2) {
        // Find the keypoints and descriptors
        orb->detectAndCompute(img2, cv::noArray(), kp2, des2);

        // Match frame 1-2
        std::vector<std::vector<cv::DMatch>> matches;
        flann->knnMatch(des1, des2, matches, 2);

        pts1.clear();
        pts2.clear();

        // Extract corresponding keypoints
        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i].size() < 2) continue;
            cv::DMatch m = matches[i][0]; // first match
            cv::DMatch n = matches[i][1]; // second match
            if (m.distance < 0.8 * n.distance) {
                pts1.push_back(kp1[m.queryIdx].pt);
                pts2.push_back(kp2[m.trainIdx].pt);
            }
        }
    }

    cv::Mat get_pose(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2) {
        // find essential matrix using RANSAC 5-point algorithm
        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);

        // // keep inliers from RANSAC
        // std::vector<cv::Point2f> inliers1, inliers2;
        // for (size_t i = 0; i < mask.rows; i++) {
        //     if (mask.at<uchar>(i)) {
        //         inliers1.push_back(pts1[i]);
        //         inliers2.push_back(pts2[i]);
        //     }
        // }

        // decompose essential matrix to get best R and best unscaled t
        cv::Mat R, t;
        cv::recoverPose(E, pts1, pts2, K, R, t, mask);

        // construct unscaled relative transform betwen frames
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        R.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
        t.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));

        return T;
    }

    void drawPaths(const std::vector<cv::Point2d> &gt_path, const std::vector<cv::Point2d> &est_path) {
        auto draw_path = [&](const std::vector<cv::Point2d> &path, const cv::Scalar &color) {
            cv::line(canvas,
                    cv::Point(int(path[path.size() - 2].x*1 + w/2), int(path[path.size() - 2].y*1 + h/2)),
                    cv::Point(int(path[path.size() - 1].x*1 + w/2), int(path[path.size() - 1].y*1 + h/2)),
                    color, 2);
        };

        draw_path(gt_path, cv::Scalar(0, 255, 0));
        draw_path(est_path, cv::Scalar(0, 0, 255));

        cv::imshow("VO Path", canvas);
        cv::waitKey(1);
    }
};

int main() {
    std::string KITTI_DIR = "/home/d300/VO/data/kitti";
    std::string seq = "05";

    VisualOdom vo(KITTI_DIR, seq);
    vo.run();

    return 0;
}