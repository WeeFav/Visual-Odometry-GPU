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

        sift = cv::SIFT::create();
        
        cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
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
        std::vector<double> gt_scale;
        std::vector<double> est_scale;
        cv::Mat cur_pose;

        for (size_t i = 0; i < 1000 && i < images.size(); i++) {
            cv::Mat gt_pose = gt_poses[i].clone();

            if (i == 0) {
                img1 = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
                sift->detect(img1, kp1);
                cv::KeyPoint::convert(kp1, tracked_pts1);
                cur_pose = gt_pose;
            } else {
                img2 = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
                // Optical flow tracking
                track_optical_flow(img2); 

                // Not enough points? Reinitialize
                if (tracked_pts2.size() < 150) {                   // GPT OF:
                    sift->detect(img2, kp1);                        // GPT OF:
                    cv::KeyPoint::convert(kp1, tracked_pts2);      // GPT OF:
                    tracked_pts1 = tracked_pts2;                   // GPT OF:
                    prev_img = img.clone();                        // GPT OF:
                    continue;                                      // GPT OF:
                }

                cv::Mat R, t;
                get_pose(pts1, pts2, R, t);

                double scale = get_scale(R, t, pts1, pts2);

                double true_scale = cv::norm(gt_poses[i](cv::Range(0, 3), cv::Range(3, 4)) -
                                             gt_poses[i - 1](cv::Range(0, 3), cv::Range(3, 4)));

                
                // construct unscaled relative transform betwen frames
                cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
                R.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
                t.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));
                T(cv::Range(0, 3), cv::Range(3, 4)) *= scale;

                cur_pose = cur_pose * T.inv();

                // Shift the cache: current becomes previous
                kp1 = kp2;
                des1 = des2.clone();
                prev_points_3d = points_3d;

                gt_scale.push_back(true_scale);
                est_scale.push_back(scale);
            }

            gt_path.push_back(cv::Point2d(gt_pose.at<double>(0, 3), gt_pose.at<double>(2, 3)));
            est_path.push_back(cv::Point2d(cur_pose.at<double>(0, 3), cur_pose.at<double>(2, 3)));
            
            // Draw paths
            drawPaths(i, gt_path, est_path);
        }

        cv::waitKey(0);

        savePaths("./gt_path.txt", "./est_path.txt", "./scale.txt", gt_path, est_path, gt_scale, est_scale);
    }

private:
    std::vector<std::string> images;
    cv::Ptr<cv::SIFT> sift;
    cv::Ptr<cv::FlannBasedMatcher> flann;
    std::vector<cv::Mat> gt_poses;
    cv::Mat K;
    std::vector<cv::KeyPoint> kp1, kp2;
    std::vector<cv::Point2f> tracked_pts1, tracked_pts2;
    cv::Mat img1;
    int w = 1000, h = 1000;
    cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC3);
    std::vector<cv::Point3f> prev_points_3d;
    std::vector<cv::Point3f> points_3d;

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

    void track_optical_flow(const cv::Mat &img2)
    {
        if (img1.empty() || pts1.empty()) return;

        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(
            img1, img2,
            pts1, pts2,
            status, err,
            cv::Size(21,21), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01)
        );

        // Remove lost tracks
        std::vector<cv::Point2f> pts1_filt, pts2_filt;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                pts1_filt.push_back(pts1[i]);
                pts2_filt.push_back(pts2[i]);
            }
        }

        pts1 = pts1_filt;
        pts2 = pts2_filt;
    }

    void get_pose(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2, cv::Mat &R, cv::Mat &t) {
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
        cv::recoverPose(E, pts1, pts2, K, R, t, mask);
    }

    double get_scale(
        const cv::Mat &R, 
        const cv::Mat &t,
        const std::vector<cv::Point2f> &pts1, 
        const std::vector<cv::Point2f> &pts2) 
    {
        // triangulation to get 3D points
        cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F); // first camera as origin
        P1 = K * P1;

        cv::Mat Rt;
        cv::hconcat(R, t, Rt);
        cv::Mat P2 = K * Rt; // second camera relative to first

        // Convert points to cv::Mat (2xN)
        cv::Mat pts1_T(2, pts1.size(), CV_64F);
        cv::Mat pts2_T(2, pts2.size(), CV_64F);
        for (size_t i = 0; i < pts1.size(); i++) {
            pts1_T.at<double>(0, i) = pts1[i].x;
            pts1_T.at<double>(1, i) = pts1[i].y;
            pts2_T.at<double>(0, i) = pts2[i].x;
            pts2_T.at<double>(1, i) = pts2[i].y;
        }

        cv::Mat points_4d_h;
        cv::triangulatePoints(P1, P2, pts1_T, pts2_T, points_4d_h); // 4xN

        // Convert from homogeneous to 3D
        points_3d.clear();
        for (int i = 0; i < points_4d_h.cols; ++i) {
            double w = points_4d_h.at<double>(3, i);
            points_3d.push_back(cv::Point3f(
                points_4d_h.at<double>(0, i) / w,
                points_4d_h.at<double>(1, i) / w,
                points_4d_h.at<double>(2, i) / w
            ));
        }

        // Estimate scale between previous and current 3D points
        if (prev_points_3d.empty() || points_3d.empty())
            return 1.0;

        size_t min_idx = std::min(prev_points_3d.size(), points_3d.size());

        std::vector<cv::Point3f> prev_pts(prev_points_3d.begin(), prev_points_3d.begin() + min_idx);
        std::vector<cv::Point3f> cur_pts(points_3d.begin(), points_3d.begin() + min_idx);

        std::vector<double> prev_dist, cur_dist;

        for (size_t i = 1; i < min_idx; ++i) {
            cv::Point3f dp = prev_pts[i] - prev_pts[i - 1];
            prev_dist.push_back(std::sqrt(dp.x * dp.x + dp.y * dp.y + dp.z * dp.z));

            cv::Point3f dc = cur_pts[i] - cur_pts[i - 1];
            cur_dist.push_back(std::sqrt(dc.x * dc.x + dc.y * dc.y + dc.z * dc.z));
        }

        std::vector<double> ratios;
        for (size_t i = 0; i < prev_dist.size(); ++i)
            ratios.push_back(prev_dist[i] / (cur_dist[i] + 1e-6));

        std::nth_element(ratios.begin(), ratios.begin() + ratios.size()/2, ratios.end());
        double scale = ratios[ratios.size()/2];  // median

        scale = std::max(0.1, std::min(5.0, scale));
        return scale;                
    }

    void drawPaths(int i, const std::vector<cv::Point2d> &gt_path, const std::vector<cv::Point2d> &est_path) {
        auto draw_path = [&](const std::vector<cv::Point2d> &path, const cv::Scalar &color) {
            cv::line(canvas,
                    cv::Point(int(w/2 + path[path.size() - 2].x*1), int(h/2 - path[path.size() - 2].y*1)),
                    cv::Point(int(w/2 + path[path.size() - 1].x*1), int(h/2 - path[path.size() - 1].y*1)),
                    color, 2);
        };

        draw_path(gt_path, cv::Scalar(0, 255, 0));
        draw_path(est_path, cv::Scalar(0, 0, 255));
        
        cv::Mat display;
        canvas.copyTo(display);  // copy current canvas
        cv::putText(display, std::to_string(i), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255));
        cv::imshow("VO Path", display);
        cv::waitKey(1);    
    }


    void savePaths(
        const std::string &gt_file, 
        const std::string &est_file,
        const std::string &scale_file,
        const std::vector<cv::Point2d> &gt_path,
        const std::vector<cv::Point2d> &est_path,
        const std::vector<double> &gt_scale,
        const std::vector<double> &est_scale
    )
    {
        std::ofstream gt_out(gt_file);
        std::ofstream est_out(est_file);
        std::ofstream scale_out(scale_file);

        for (size_t i = 0; i < gt_path.size(); i++) {
            gt_out << gt_path[i].x << " " << gt_path[i].y << "\n";
        }
        for (size_t i = 0; i < est_path.size(); i++) {
            est_out << est_path[i].x << " " << est_path[i].y << "\n";
        }
        for (size_t i = 0; i < gt_scale.size(); i++) {
            scale_out << gt_scale[i] << " " << est_scale[i] << "\n";
        }

        gt_out.close();
        est_out.close();
        scale_out.close();
    }
};

int main() {
    std::string KITTI_DIR = "/home/d300/VO/data/kitti";
    std::string seq = "05";

    VisualOdom vo(KITTI_DIR, seq);
    vo.run();

    return 0;
}