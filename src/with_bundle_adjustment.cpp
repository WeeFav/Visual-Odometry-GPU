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
#include <chrono>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace fs = std::filesystem;

struct Landmark {
    int id;
    cv::Point3d pos;
    // observations: pair(frame_index_in_window, 2D point)
    std::vector<std::pair<int, cv::Point2d>> observations;
};

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

// --- Helper functions ---

// Extract R (3x3) and t (3x1) from a 4x4 pose matrix
inline void extract_R_t_from_pose(const cv::Mat& pose4x4, cv::Mat& R, cv::Mat& t) {
    CV_Assert(pose4x4.rows == 4 && pose4x4.cols == 4 && pose4x4.type() == CV_64F);
    R = pose4x4(cv::Range(0,3), cv::Range(0,3)).clone();
    t = pose4x4(cv::Range(0,3), cv::Range(3,4)).clone();
}

// Compose a 4x4 pose from R and t (R 3x3, t 3x1)
inline cv::Mat compose_pose_from_R_t(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat pose = cv::Mat::eye(4,4,CV_64F);
    R.copyTo(pose(cv::Range(0,3), cv::Range(0,3)));
    t.copyTo(pose(cv::Range(0,3), cv::Range(3,4)));
    return pose;
}

// Build projection matrix P = K * [R|t]
inline cv::Mat buildProjection(const cv::Mat& K, const cv::Mat& R, const cv::Mat& t) {
    cv::Mat Rt(3,4,CV_64F);
    R.copyTo(Rt(cv::Range(0,3), cv::Range(0,3)));
    t.copyTo(Rt(cv::Range(0,3), cv::Range(3,4)));
    cv::Mat P = K * Rt;
    return P;
}

// Triangulate points using OpenCV (expects matching vector of points in frame0 & frame1)
inline std::vector<cv::Point3d> triangulatePointsLinear(
    const std::vector<cv::Point2f>& pts0,
    const std::vector<cv::Point2f>& pts1,
    const cv::Mat& P0,
    const cv::Mat& P1)
{
    std::vector<cv::Point3d> points_3d;
    if (pts0.empty()) return points_3d;

    // Convert points to cv::Mat (2xN)
    cv::Mat pts0_T(2, pts0.size(), CV_64F);
    cv::Mat pts1_T(2, pts1.size(), CV_64F);
    for (size_t i = 0; i < pts0.size(); i++) {
        pts0_T.at<double>(0, i) = pts0[i].x;
        pts0_T.at<double>(1, i) = pts0[i].y;
        pts1_T.at<double>(0, i) = pts1[i].x;
        pts1_T.at<double>(1, i) = pts1[i].y;
    }

    cv::Mat points_4d_h;
    cv::triangulatePoints(P0, P1, pts0_T, pts1_T, points_4d_h); // 4 x N
    
    points_3d.resize(points_4d_h.cols);
    // Convert from homogeneous to 3D
    for (int i = 0; i < points_4d_h.cols; ++i) {
        double w = points_4d_h.at<double>(3, i);
        double X = points_4d_h.at<double>(0, i) / w;
        double Y = points_4d_h.at<double>(1, i) / w;
        double Z = points_4d_h.at<double>(2, i) / w; 
        // std::cout << "w: " << w << std::endl;    
        points_3d[i] = cv::Point3d(X, Y, Z);
    }

    return points_3d;
}

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
        std::vector<cv::Point2d> opt_path;
        std::vector<double> gt_scale;
        std::vector<double> est_scale;

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < 1000 && i < images.size(); i++) {
            cv::Mat gt_pose = gt_poses[i].clone();

            if (i == 0) {
                img1 = cv::imread(images[0], cv::IMREAD_GRAYSCALE);
                std::vector<cv::KeyPoint> kp1;
                sift->detect(img1, kp1);
                cv::KeyPoint::convert(kp1, pts1);
                cur_pose = gt_pose;
                // *** Bundle Adjustment *** //
                pose_window.push_back(cur_pose.clone()); // Add current pose
                observations_window.push_back(pts1); // Add 2D observations
                imgs_window.push_back(img1.clone()); // Add images
            } else {
                cv::Mat img2 = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
                
                std::vector<cv::Point2f> pts2;
                track_optical_flow(img2, pts2);

                // Not enough points? Use feature matching for this frame
                if (pts2.size() < 150) {             
                    get_matches(img2, pts2);
                }

                cv::Mat R, t;
                get_pose(pts2, R, t);

                std::vector<cv::Point3f> points_3d;
                double scale = get_scale(R, t, pts2, points_3d);

                double true_scale = cv::norm(gt_poses[i](cv::Range(0, 3), cv::Range(3, 4)) -
                                             gt_poses[i - 1](cv::Range(0, 3), cv::Range(3, 4)));
                
                
                // construct unscaled relative transform betwen frames
                cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
                R.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
                t.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));
                T(cv::Range(0, 3), cv::Range(3, 4)) *= scale;

                cur_pose = prev_pose * T.inv();

                // *** Bundle Adjustment *** //
                pose_window.push_back(cur_pose.clone()); // Add current pose
                observations_window.push_back(pts2); // Add 2D observations
                imgs_window.push_back(img2.clone()); // Add images

                // Shift the cache: current becomes previous
                img1 = img2.clone();
                pts1 = pts2;
                prev_points_3d = points_3d;

                gt_scale.push_back(true_scale);
                est_scale.push_back(scale);
            }

            // *** Bundle Adjustment *** //
            // Keep window size fixed
            if (pose_window.size() > WINDOW_SIZE) {
                pose_window.pop_front();
                observations_window.pop_front();
                imgs_window.pop_front();
            }

            // Run BA every 10 frames
            if (i % 10 == 0 && pose_window.size() == WINDOW_SIZE) {
                run_bundle_adjustment();
            }
            
            gt_path.push_back(cv::Point2d(gt_pose.at<double>(0, 3), gt_pose.at<double>(2, 3)));
            
            cur_pose = pose_window.back();
            est_path.push_back(cv::Point2d(cur_pose.at<double>(0, 3), cur_pose.at<double>(2, 3)));

            if (i % 10 == 0) {
                int N = est_path.size();
                if (N >= WINDOW_SIZE) {
                    for (int k = 0; k < WINDOW_SIZE; k++) {
                        cv::Mat pose = pose_window[k];
                        double x = pose.at<double>(0, 3);
                        double z = pose.at<double>(2, 3);
                        est_path[N - WINDOW_SIZE + k] = cv::Point2d(x, z);
                    }
                }
            }

            prev_pose = cur_pose.clone();
            
            // Draw paths
            drawPaths(i, gt_path, est_path);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

        cv::waitKey(0);

        savePaths("./gt_path.txt", "./est_path.txt", "./scale.txt", gt_path, est_path, gt_scale, est_scale);
    }

private:
    std::vector<std::string> images;
    cv::Ptr<cv::SIFT> sift;
    cv::Ptr<cv::FlannBasedMatcher> flann;
    std::vector<cv::Mat> gt_poses;
    cv::Mat K;

    // Cache
    cv::Mat img1;
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point3f> prev_points_3d;
    cv::Mat prev_pose, cur_pose; // camera pose C, camera to world

    // Path Visualization
    int w = 1000, h = 1000;
    cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC3);

    // Bundle Adjustment
    int WINDOW_SIZE = 5;
    std::deque<cv::Mat> pose_window; // last N poses
    std::deque<std::vector<cv::Point2f>> observations_window; // 2D matches per frame
    std::deque<cv::Mat> imgs_window;

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

    void track_optical_flow(const cv::Mat &img2,
                            std::vector<cv::Point2f> &pts2)
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
        std::vector<cv::Point2f> pts1_valid, pts2_valid;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                pts1_valid.push_back(pts1[i]);
                pts2_valid.push_back(pts2[i]);
            }
        }

        pts1 = pts1_valid;
        pts2 = pts2_valid;
    }

    void get_matches(const cv::Mat &img2, std::vector<cv::Point2f> &pts2) 
    {
        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat des1, des2;
        
        // Find the keypoints and descriptors
        sift->detectAndCompute(img1, cv::noArray(), kp1, des1);
        sift->detectAndCompute(img2, cv::noArray(), kp2, des2);

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

    void get_pose(
        const std::vector<cv::Point2f> &pts2, 
        cv::Mat &R, 
        cv::Mat &t) 
    {
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
        const std::vector<cv::Point2f> &pts2,
        std::vector<cv::Point3f> &points_3d) 
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

    // --- Optical-flow tracking across window ---
    // Track initial points (pts0) from frame 0 through all frames in `imgs`.
    // Returns tracks: vector per point, each entry is pair(frame_index, point2f) where it was observed
    std::vector<std::vector<std::pair<int, cv::Point2f>>> trackPointsAcrossWindow(const std::vector<cv::Point2f>& keypoints0)
    {
        std::vector<std::vector<std::pair<int, cv::Point2f>>> tracks;
        tracks.resize(keypoints0.size());

        // track each initial point from frame 0 to each frame
        // point is tracked through frame by frame propagation
        for (size_t i = 0; i < keypoints0.size(); ++i) {
            tracks[i].clear();
            tracks[i].emplace_back(0, keypoints0[i]);
            cv::Point2f prevPt = keypoints0[i];
            bool alive = true;
            for (int fi = 1; fi < (int)imgs_window.size(); ++fi) {
                std::vector<cv::Point2f> in = { prevPt };
                std::vector<cv::Point2f> out;
                std::vector<unsigned char> status;
                std::vector<float> err;
                cv::calcOpticalFlowPyrLK(
                    imgs_window[fi-1], imgs_window[fi],
                    in, out,
                    status, err,
                    cv::Size(21,21), 3,
                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01)
                );
                if (status[0]) {
                    tracks[i].emplace_back(fi, out[0]);
                    prevPt = out[0];
                } else {
                    alive = false;
                    break;
                }
            }
        }

        return tracks;
    }

    // --- Build landmarks by triangulating from frames 0 and 1 and attaching observations from tracks ---
    bool buildLandmarksFromFirstTwoFramesAndTracks(std::vector<cv::Point2f> keypoints0, std::vector<Landmark> &landmarks)
    {
        std::vector<std::vector<std::pair<int, cv::Point2f>>> tracks = trackPointsAcrossWindow(keypoints0);
        
        // Build projection matrices for frame 0 and 1

        cv::Mat R0, t0, R1, t1;
        // pose_window is cam to world, but projection matrix need world to cam
        extract_R_t_from_pose(pose_window[0].inv(), R0, t0);
        extract_R_t_from_pose(pose_window[1].inv(), R1, t1);
        cv::Mat P0 = buildProjection(K, R0, t0);
        cv::Mat P1 = buildProjection(K, R1, t1);

        double baseline = cv::norm(t0 - t1);
        if (baseline < 0.1 || baseline > 100) return false;

        // Prepare vectors for triangulation: only for points that have a valid observation in frame 1
        std::vector<cv::Point2f> pts0_for_tri, pts1_for_tri;
        std::vector<int> original_idx_for_tri;
        for (size_t i = 0; i < tracks.size(); ++i) {
            // require that track has an observation at frame 1
            bool hasFrame1 = false;
            cv::Point2f p0 = keypoints0[i];
            cv::Point2f p1;
            for (auto &obs : tracks[i]) {
                if (obs.first == 1) { 
                    hasFrame1 = true; 
                    p1 = obs.second; 
                    break; 
                }
            }
            if (hasFrame1) {
                pts0_for_tri.push_back(p0);
                pts1_for_tri.push_back(p1);
                original_idx_for_tri.push_back((int)i);
            }
        }
        
        // std::cout << "P0: " << P0 << std::endl;
        // std::cout << "P1: " << P1 << std::endl;

        // for (int j = 0; j < pts0_for_tri.size(); j++) {
        //     std::cout << "Pair " << j << ": "
        //             << pts0_for_tri[j] << " -> "
        //             << pts1_for_tri[j] << std::endl;
        // }

        // points_3d is in world frame
        std::vector<cv::Point3d> points_3d = triangulatePointsLinear(pts0_for_tri, pts1_for_tri, P0, P1);
        assert(points_3d.size() == original_idx_for_tri.size());

        // Build landmarks using 3d points from frame 0, 1 and images points across frames
        landmarks.reserve(points_3d.size());
        for (size_t j = 0; j < points_3d.size(); ++j) {
            cv::Point3d X = points_3d[j];

            // Simple depth check
            if (X.z <= 0) continue;

            Landmark lm;
            lm.id = j;
            lm.pos = X;

            // add observations from tracks for frames where track succeeded
            int orig_idx = original_idx_for_tri[j];
            for (auto &obs : tracks[orig_idx]) {
                // obs: (frame_idx, pt)
                lm.observations.emplace_back(obs.first, cv::Point2d(obs.second.x, obs.second.y));
            }
            landmarks.push_back(lm);
        }

        return true;
    }

    void run_bundle_adjustment()
    {
        // Check everything has WINDOW_SIZE frames
        if (pose_window.size() != WINDOW_SIZE || observations_window.size() != WINDOW_SIZE || imgs_window.size() != WINDOW_SIZE) {
            std::cerr << "Need same size.\n";
            return;
        }

        // Initial keypoints: use features from first image.
        std::vector<cv::Point2f> keypoints0;
        // If caller provided observations for frame 0, use them; else detect features
        if (!observations_window[0].empty()) {
            for (auto &p : observations_window[0]) keypoints0.emplace_back((float)p.x, (float)p.y);
        } else {
            // detect good features in frame 0
            cv::goodFeaturesToTrack(imgs_window[0], keypoints0, 2000, 0.01, 8);
        }

        if (keypoints0.empty()) {
            std::cerr << "No initial keypoints to track.\n";
            return;
        }

        // Build landmarks by tracking & triangulation
        std::vector<Landmark> landmarks;
        if (!buildLandmarksFromFirstTwoFramesAndTracks(keypoints0, landmarks))
        {
            std::cerr << "buildLandmarksFromFirstTwoFramesAndTracks failed.\n";
            return;
        }
        if (landmarks.empty()) {
            std::cerr << "No landmarks after triangulation.\n";
            return;
        }

        // --- Setup Ceres problem ---
        ceres::Problem problem;

        // Parameter blocks for poses: vector<double> per frame of size 6 (angle-axis 3 + translation 3)
        std::vector<std::array<double,6>> pose_params(WINDOW_SIZE);
        for (int i = 0; i < WINDOW_SIZE; ++i) {
            cv::Mat R, t;
            // we need world to cam because we want to project 3d world points to camera 
            extract_R_t_from_pose(pose_window[i].inv(), R, t);
            cv::Mat rvec;
            cv::Rodrigues(R, rvec); // rvec is 3x1 rotation vector (angle-axis)
            pose_params[i][0] = rvec.at<double>(0,0);
            pose_params[i][1] = rvec.at<double>(1,0);
            pose_params[i][2] = rvec.at<double>(2,0);
            pose_params[i][3] = t.at<double>(0,0);
            pose_params[i][4] = t.at<double>(1,0);
            pose_params[i][5] = t.at<double>(2,0);
            problem.AddParameterBlock(pose_params[i].data(), 6);
        }

        // Parameter blocks for points
        std::vector<std::array<double,3>> point_params;
        point_params.reserve(landmarks.size());
        for (size_t i = 0; i < landmarks.size(); ++i) {
            std::array<double,3> p;
            p[0] = landmarks[i].pos.x;
            p[1] = landmarks[i].pos.y;
            p[2] = landmarks[i].pos.z;
            point_params.push_back(p);
            problem.AddParameterBlock(point_params.back().data(), 3);
            
            // for (int k = 0; k < 3; k++) {
            //     std::cout << "point_params[" << i << "][" << k << "] = "
            //             << point_params[i][k] << std::endl;
            // }        

        }

        // Add residuals: for each landmark and each observation in the window
        for (size_t i = 0; i < landmarks.size(); ++i) {
            const Landmark& lm = landmarks[i];
            for (auto &obs : lm.observations) {
                int frame_idx = obs.first;
                if (frame_idx < 0 || frame_idx >= WINDOW_SIZE) continue;
                
                // Build reprojection residual
                ceres::CostFunction* cost = ReprojectionError::Create(obs.second.x, obs.second.y, K);
                problem.AddResidualBlock(
                    cost, 
                    new ceres::HuberLoss(1.0), 
                    pose_params[frame_idx].data(), // frame's pose
                    point_params[i].data() // landmark's 3D point
                );
            }
        }

        // fix the first pose
        problem.SetParameterBlockConstant(pose_params[0].data());

        // Solve
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        // options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 200;
        options.num_threads = 4;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // std::cout << summary.FullReport() << std::endl;

        if (summary.termination_type == ceres::CONVERGENCE)
        {
            // Write optimized poses back into pose_window
            for (int i = 0; i < WINDOW_SIZE; ++i) {
                cv::Mat rvec = (cv::Mat_<double>(3,1) << pose_params[i][0], pose_params[i][1], pose_params[i][2]);
                cv::Mat R;
                cv::Rodrigues(rvec, R);
                cv::Mat t = (cv::Mat_<double>(3,1) << pose_params[i][3], pose_params[i][4], pose_params[i][5]);
                cv::Mat newPose = compose_pose_from_R_t(R, t);

                cv::Mat oldPose = pose_window[i].inv(); // world->cam

                // --- compute difference ---
                cv::Mat R_old = oldPose(cv::Range(0,3), cv::Range(0,3));
                cv::Mat t_old = oldPose(cv::Range(0,3), cv::Range(3,4));

                cv::Mat R_diff = R * R_old.t(); // rotation difference
                cv::Mat rvec_diff;
                cv::Rodrigues(R_diff, rvec_diff);
                double angle_diff = cv::norm(rvec_diff); // in radians

                cv::Mat t_diff = t - t_old;
                double trans_diff = cv::norm(t_diff);

                // set thresholds
                const double MAX_ROT_DIFF = 0.5;   // radians ~ 30 degrees
                const double MAX_TRANS_DIFF = 50.0; // meters (or units of your system)

                if (angle_diff < MAX_ROT_DIFF && trans_diff < MAX_TRANS_DIFF) {
                    // pose_params is world to cam, so we need to invert to get cam to world                
                    pose_window[i] = newPose.inv();
                } else {
                    std::cout << "Pose " << i << " differs too much. Not updating." << std::endl;
                    std::cout << "Pose " << i << " angle_diff " << angle_diff << " trans_diff " << trans_diff << std::endl;
                }

            }
        }

    }

    void drawPaths(
        int i, 
        const std::vector<cv::Point2d> &gt_path, 
        const std::vector<cv::Point2d> &est_path) 
    {
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