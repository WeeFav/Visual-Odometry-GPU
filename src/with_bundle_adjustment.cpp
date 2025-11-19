#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <iterator>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace fs = std::filesystem;

struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y, const cv::Mat& K)
        : x_(observed_x), y_(observed_y)
    {
        fx_ = K.at<double>(0,0);
        fy_ = K.at<double>(1,1);
        cx_ = K.at<double>(0,2);
        cy_ = K.at<double>(1,2);
    }

    template <typename T>
    bool operator()(const T* const pose,    // 6 DOF: angle-axis + translation
                    const T* const point3d, // landmark X,Y,Z
                    T* residuals) const 
    {
        // Pose format: pose[0:3] = angle-axis, pose[3:6] = t
        T p[3];
        ceres::AngleAxisRotatePoint(pose, point3d, p);
        p[0] += pose[3];
        p[1] += pose[4];
        p[2] += pose[5];

        // Project into pixel coordinates
        T u = T(fx_) * p[0] / p[2] + T(cx_);
        T v = T(fy_) * p[1] / p[2] + T(cy_);

        residuals[0] = u - T(x_);
        residuals[1] = v - T(y_);
        return true;
    }

    static ceres::CostFunction* Create(double observed_x,
                                       double observed_y,
                                       const cv::Mat& K)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(observed_x, observed_y, K)));
    }

    double x_, y_;
    double fx_, fy_, cx_, cy_;
};

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
                cv::Mat img1 = cv::imread(images[0], cv::IMREAD_GRAYSCALE);
                sift->detectAndCompute(img1, cv::noArray(), kp1, des1);
                cur_pose = gt_pose;
            } else {
                cv::Mat img2 = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
                
                std::vector<cv::Point2f> pts1, pts2;
                get_matches(img2, pts1, pts2);

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
                
                // *** Bundle Adjustment *** //
                pose_window.push_back(cur_pose.clone()); // Add current pose
                landmark_window.push_back(points_3d); // Add landmarks of this frame
                observations_window.push_back(pts2); // Add 2D observations
                
                // Keep window size fixed
                if (pose_window.size() > WINDOW_SIZE) {
                    pose_window.pop_front();
                    landmark_window.pop_front();
                    observations_window.pop_front();
                }

                // Run BA every 10 frames
                if (i % 10 == 0 && pose_window.size() >= 3) {
                    run_bundle_adjustment(pose_window, landmark_window, observations_window, K);
                }
                
                cur_pose = pose_window.back();

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
    cv::Mat des1, des2;
    std::vector<cv::Point3f> prev_points_3d;
    std::vector<cv::Point3f> points_3d;
    
    // Path Visualization
    int w = 1000, h = 1000;
    cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC3);

    // Bundle Adjustment
    int WINDOW_SIZE = 5;
    std::deque<cv::Mat> pose_window; // last N poses
    std::deque<std::vector<cv::Point3f>> landmark_window; // 3D points per frame
    std::deque<std::vector<cv::Point2f>> observations_window; // 2D matches per frame

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
        sift->detectAndCompute(img2, cv::noArray(), kp2, des2);

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

    void run_bundle_adjustment(
        std::deque<cv::Mat>& pose_window,
        std::deque<std::vector<cv::Point3f>>& landmark_window,
        std::deque<std::vector<cv::Point2f>>& observations_window,
        const cv::Mat& K)
    {
        if (pose_window.size() < 3) return; // not enough yet

        ceres::Problem problem;

        const int N = pose_window.size();

        // Convert poses to double arrays (angle-axis + t)
        std::vector<std::array<double,6>> pose_params(N);
        for (int i = 0; i < N; i++) {
            cv::Mat R = pose_window[i](cv::Range(0,3), cv::Range(0,3));
            cv::Mat t = pose_window[i](cv::Range(0,3), cv::Range(3,4));

            cv::Mat rvec;
            cv::Rodrigues(R, rvec);

            pose_params[i][0] = rvec.at<double>(0);
            pose_params[i][1] = rvec.at<double>(1);
            pose_params[i][2] = rvec.at<double>(2);
            pose_params[i][3] = t.at<double>(0);
            pose_params[i][4] = t.at<double>(1);
            pose_params[i][5] = t.at<double>(2);

            problem.AddParameterBlock(pose_params[i].data(), 6);
            if (i == 0)
                problem.SetParameterBlockConstant(pose_params[i].data()); // fix first pose
        }

        // Landmarks
        std::vector<std::array<double,3>> points3d_params;
        for (auto& lm_set : landmark_window)
            for (auto& p : lm_set) {
                points3d_params.push_back({p.x, p.y, p.z});
            }

        int point_index = 0;
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < landmark_window[k].size(); j++) {
                problem.AddParameterBlock(points3d_params[point_index].data(), 3); // add landmark as parameter for each landmark in a frame for N frames 

                ceres::CostFunction* cost =
                    ReprojectionError::Create(
                        observations_window[k][j].x,
                        observations_window[k][j].y,
                        K
                    );

                problem.AddResidualBlock(
                    cost, 
                    new ceres::HuberLoss(1.0),
                    pose_params[k].data(),
                    points3d_params[point_index].data()
                );
                point_index++;
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.max_num_iterations = 25;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 4;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Put optimized poses back
        for (int i = 0; i < N; i++) {
            cv::Mat rvec = (cv::Mat_<double>(3,1) << 
                pose_params[i][0], pose_params[i][1], pose_params[i][2]);

            cv::Mat R;
            cv::Rodrigues(rvec, R);

            pose_window[i](cv::Range(0,3), cv::Range(0,3)) = R.clone();
            pose_window[i].at<double>(0,3) = pose_params[i][3];
            pose_window[i].at<double>(1,3) = pose_params[i][4];
            pose_window[i].at<double>(2,3) = pose_params[i][5];
        }
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