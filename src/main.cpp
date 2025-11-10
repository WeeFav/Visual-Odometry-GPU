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

namespace fs = std::filesystem;

// ---------- Helpers ----------

bool readPoses(const std::string &path, std::vector<cv::Mat> &poses) {
    std::ifstream fin(path);
    if (!fin.is_open()) return false;
    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        if (vals.size() != 12) continue;

        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                T.at<double>(r, c) = vals[r * 4 + c];
        poses.push_back(T);
    }
    return true;
}

bool readCalibP0(const std::string &path, cv::Mat &P, cv::Mat &K) {
    std::ifstream fin(path);
    if (!fin.is_open()) return false;
    std::string line;
    // Look for line starting with "P0:"
    while (std::getline(fin, line)) {
        if (line.rfind("P0:", 0) == 0 || line.rfind("P0 ", 0) == 0) {
            std::string payload = line.substr(4);
            std::istringstream iss(payload);
            std::vector<double> vals;
            double v;
            while (iss >> v) vals.push_back(v);
            if (vals.size() >= 12) {
                P = cv::Mat(3, 4, CV_64F);
                for (int i = 0; i < 12; ++i)
                    P.at<double>(i / 4, i % 4) = vals[i];
                K = P(cv::Range(0,3), cv::Range(0,3)).clone();
                return true;
            }
        }
    }
    // fallback: try first line trimmed after "P0:"
    return false;
}

// Convert vector<cv::KeyPoint> + DMatches to arrays of points
void matchedPoints(const std::vector<cv::KeyPoint>& k1,
                   const std::vector<cv::KeyPoint>& k2,
                   const std::vector<cv::DMatch>& matches,
                   std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2) {
    q1.clear(); q2.clear();
    for (auto &m : matches) {
        q1.push_back(k1[m.queryIdx].pt);
        q2.push_back(k2[m.trainIdx].pt);
    }
}

// triangulate and count positive z and compute relative scale estimate
// returns pair(sum_positive_z, relative_scale)
std::pair<int, double> sum_z_cal_relative_scale(
    const cv::Mat &R, const cv::Mat &t,
    const cv::Mat &P_left, 
    const std::vector<cv::Point2f> &q1,
    const std::vector<cv::Point2f> &q2,
    const cv::Mat &K) 
{
    // Build projection matrices for cv::triangulatePoints (3x4 doubles)
    cv::Mat T = cv::Mat::eye(4,4,CV_64F);
    R.copyTo(T(cv::Range(0,3), cv::Range(0,3)));
    t.copyTo(T(cv::Range(0,3), cv::Range(3,4)));

    // P_left is 3x4 projection from calib (global P)
    cv::Mat P1 = P_left.clone(); // 3x4
    cv::Mat P2 = K * (cv::Mat_<double>(3,4) << 
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2)
    );

    // Convert points to homogeneous for triangulation
    if (q1.empty() || q2.empty()) return {0, 1.0};

    cv::Mat pts1(2, (int)q1.size(), CV_64F), pts2(2, (int)q2.size(), CV_64F);
    for (size_t i = 0; i < q1.size(); ++i) {
        pts1.at<double>(0,i) = q1[i].x;
        pts1.at<double>(1,i) = q1[i].y;
        pts2.at<double>(0,i) = q2[i].x;
        pts2.at<double>(1,i) = q2[i].y;
    }

    cv::Mat homQ1; // 4xN
    cv::triangulatePoints(P1, P2, pts1, pts2, homQ1);

    // project into cam2: hom_Q2 = T * homQ1
    cv::Mat homQ2 = T * homQ1;

    // Convert to Euclidean
    int N = homQ1.cols;
    std::vector<cv::Point3d> eu1, eu2;
    eu1.reserve(N); eu2.reserve(N);
    for (int i=0;i<N;i++) {
        double w1 = homQ1.at<double>(3,i);
        double w2 = homQ2.at<double>(3,i);
        if (fabs(w1) < 1e-8 || fabs(w2) < 1e-8) {
            eu1.emplace_back(0,0, -1e6);
            eu2.emplace_back(0,0, -1e6);
        } else {
            eu1.emplace_back(
                homQ1.at<double>(0,i)/w1,
                homQ1.at<double>(1,i)/w1,
                homQ1.at<double>(2,i)/w1
            );
            eu2.emplace_back(
                homQ2.at<double>(0,i)/w2,
                homQ2.at<double>(1,i)/w2,
                homQ2.at<double>(2,i)/w2
            );
        }
    }

    int sum_pos = 0;
    for (int i=0;i<N;i++) {
        if (eu1[i].z > 0 && eu2[i].z > 0) sum_pos++;
    }

    // relative scale: mean of ratios of consecutive point distances (like your python)
    double relative_scale = 1.0;
    if (N > 1) {
        std::vector<double> ratios;
        for (int i=0;i<N-1;i++) {
            cv::Point3d a1 = eu1[i], b1 = eu1[i+1];
            cv::Point3d a2 = eu2[i], b2 = eu2[i+1];
            double d1 = cv::norm(cv::Vec3d(a1.x - b1.x, a1.y - b1.y, a1.z - b1.z));
            double d2 = cv::norm(cv::Vec3d(a2.x - b2.x, a2.y - b2.y, a2.z - b2.z));
            if (d2 > 1e-8 && d1 > 1e-8) ratios.push_back(d1 / d2);
        }
        if (!ratios.empty()) {
            double sum=0;
            for (double r: ratios) sum += r;
            relative_scale = sum / ratios.size();
            if (relative_scale <= 0) relative_scale = 1.0;
        }
    }
    return {sum_pos, relative_scale};
}

cv::Mat decomp_essential_mat_and_select(
    const cv::Mat &E,
    const std::vector<cv::Point2f> &q1,
    const std::vector<cv::Point2f> &q2,
    const cv::Mat &P_calib,
    const cv::Mat &K)
{
    // decomposeEssentialMat -> returns R1,R2,t (t is up-to-scale)
    cv::Mat R1, R2, tvec;
    cv::decomposeEssentialMat(E, R1, R2, tvec);
    // tvec is 3x1. Build candidate pairs
    std::vector<std::pair<cv::Mat, cv::Mat>> candidates;
    candidates.push_back({R1,  tvec});
    candidates.push_back({R1, -tvec});
    candidates.push_back({R2,  tvec});
    candidates.push_back({R2, -tvec});

    int best_idx = 0;
    int best_count = -1;
    double best_scale = 1.0;

    for (size_t i=0;i<candidates.size();++i) {
        auto &pr = candidates[i];
        auto [count, relscale] = sum_z_cal_relative_scale(pr.first, pr.second, P_calib, q1, q2, K);
        if (count > best_count) {
            best_count = count;
            best_idx = (int)i;
            best_scale = relscale;
        }
    }

    cv::Mat R = candidates[best_idx].first.clone();
    cv::Mat t = candidates[best_idx].second.clone();
    // scale t
    t = t * best_scale;

    // build 4x4 transform
    cv::Mat T = cv::Mat::eye(4,4,CV_64F);
    R.copyTo(T(cv::Range(0,3), cv::Range(0,3)));
    t.copyTo(T(cv::Range(0,3), cv::Range(3,4)));
    return T;
}

// ---------- Main ----------

int main(int argc, char** argv) {
    std::string KITTI_DIR = "/media/d300/T9/KITTI";

    std::string seq = "02";
    std::string images_dir = KITTI_DIR + "/data_odometry_gray/dataset/sequences/" + seq + "/image_0";

    // enumerate images
    std::vector<std::string> images;
    for (auto &entry : fs::directory_iterator(images_dir)) {
        if (entry.is_regular_file()) {
            images.push_back(entry.path().string());
        }
    }
    std::sort(images.begin(), images.end());

    // feature detector / descriptor
    cv::Ptr<cv::ORB> orb = cv::ORB::create(3000);

    // FLANN LSH parameters for ORB (binary descriptors)
    cv::Ptr<cv::FlannBasedMatcher> flann;
    // LSH index parameters are provided through cv::flann::LshIndexParams
    cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
    flann = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);

    // read ground truth poses
    std::vector<cv::Mat> gt_poses;
    std::string pose_path = KITTI_DIR + "/data_odometry_poses/dataset/poses/" + seq + ".txt";
    if (!readPoses(pose_path, gt_poses)) {
        std::cerr << "Failed to read poses from " << pose_path << "\n";
        return -1;
    }

    // read calibration (P0)
    cv::Mat P_calib, K;
    std::string calib_path = KITTI_DIR + "/data_odometry_gray/dataset/sequences/01/calib.txt";
    if (!readCalibP0(calib_path, P_calib, K)) {
        std::cerr << "Failed to read calib from " << calib_path << "\n";
        return -1;
    }

    std::cout << "K:\n" << K << "\n";
    std::cout << "P_calib:\n" << P_calib << "\n";

    // visualization canvas (simple)
    int canvas_w = 800, canvas_h = 800;
    cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));

    std::vector<cv::Point2d> gt_path_points;
    std::vector<cv::Point2d> est_path_points;

    cv::Mat cur_pose = gt_poses[0].clone();

    // windows
    cv::namedWindow("matches", cv::WINDOW_NORMAL);
    cv::namedWindow("trajectory", cv::WINDOW_NORMAL);

    int max_frames = std::min((int)images.size()-1, 100); // match code uses i and i-1
    for (int i = 1; i < max_frames; ++i) {
        cv::Mat img1 = cv::imread(images[i-1], cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(images[i],   cv::IMREAD_GRAYSCALE);
        if (img1.empty() || img2.empty()) {
            std::cerr << "Cannot open images: " << images[i-1] << " or " << images[i] << "\n";
            continue;
        }

        // detect and compute
        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat des1, des2;
        orb->detectAndCompute(img1, cv::noArray(), kp1, des1);
        orb->detectAndCompute(img2, cv::noArray(), kp2, des2);

        if (des1.empty() || des2.empty()) {
            std::cerr << "Empty descriptors at frame " << i << "\n";
            continue;
        }

        // Flann + ratio test
        std::vector<std::vector<cv::DMatch>> knn_matches;
        try {
            flann->knnMatch(des1, des2, knn_matches, 2);
        } catch (const cv::Exception &e) {
            std::cerr << "Flann knnMatch exception: " << e.what() << "\n";
            continue;
        }

        std::vector<cv::DMatch> good;
        for (auto &km : knn_matches) {
            if (km.size() >= 2) {
                if (km[0].distance < 0.8f * km[1].distance) good.push_back(km[0]);
            }
        }

        // draw matches
        cv::Mat outImg;
        cv::drawMatches(img1, kp1, img2, kp2, good, outImg);
        cv::imshow("matches", outImg);
        cv::waitKey(1);

        // get matched points
        std::vector<cv::Point2f> q1, q2;
        matchedPoints(kp1, kp2, good, q1, q2);
        if (q1.size() < 8) {
            std::cerr << "Not enough matches: " << q1.size() << "\n";
            // still update gt/est paths
            cv::Mat gt = gt_poses[i];
            gt_path_points.emplace_back(gt.at<double>(0,3), gt.at<double>(2,3));
            est_path_points.emplace_back(cur_pose.at<double>(0,3), cur_pose.at<double>(2,3));
            // draw
            canvas.setTo(cv::Scalar(255,255,255));
            for (size_t k=1;k<gt_path_points.size();++k) {
                cv::line(canvas, cv::Point((int)(gt_path_points[k-1].x)+400, (int)(gt_path_points[k-1].y)+400),
                         cv::Point((int)(gt_path_points[k].x)+400, (int)(gt_path_points[k].y)+400),
                         cv::Scalar(0,200,0), 2);
            }
            for (size_t k=1;k<est_path_points.size();++k) {
                cv::line(canvas, cv::Point((int)(est_path_points[k-1].x)+400, (int)(est_path_points[k-1].y)+400),
                         cv::Point((int)(est_path_points[k].x)+400, (int)(est_path_points[k].y)+400),
                         cv::Scalar(0,0,200), 2);
            }
            cv::imshow("trajectory", canvas);
            if (cv::waitKey(1) == 27) break;
            continue;
        }

        // find essential
        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(q1, q2, K, cv::RANSAC, 0.999, 0.4, mask);

        if (E.empty()) {
            std::cerr << "Empty essential matrix\n";
            continue;
        }

        // Decompose & select correct solution (with triangulation check)
        cv::Mat T = decomp_essential_mat_and_select(E, q1, q2, P_calib, K);

        // compose: cur_pose = cur_pose * inv(T)  (same as python cur_pose @ inv(T))
        cv::Mat T_inv = T.inv();
        cur_pose = cur_pose * T_inv;

        // store paths (x,z)
        cv::Mat gt = gt_poses[i];

        gt_path_points.emplace_back(gt.at<double>(0,3), gt.at<double>(2,3));
        est_path_points.emplace_back(cur_pose.at<double>(0,3), cur_pose.at<double>(2,3));

        // draw paths on canvas (centered)
        canvas.setTo(cv::Scalar(255,255,255));
        for (size_t k=1;k<gt_path_points.size();++k) {
            cv::Point p1((int)(gt_path_points[k-1].x)+400, (int)(gt_path_points[k-1].y)+400);
            cv::Point p2((int)(gt_path_points[k].x)+400, (int)(gt_path_points[k].y)+400);
            cv::line(canvas, p1, p2, cv::Scalar(0,200,0), 2);
        }
        for (size_t k=1;k<est_path_points.size();++k) {
            cv::Point p1((int)(est_path_points[k-1].x)+400, (int)(est_path_points[k-1].y)+400);
            cv::Point p2((int)(est_path_points[k].x)+400, (int)(est_path_points[k].y)+400);
            cv::line(canvas, p1, p2, cv::Scalar(0,0,200), 2);
        }

        cv::imshow("trajectory", canvas);
        if (cv::waitKey(1) == 27) break; // ESC to exit
    }

    cv::waitKey(0);
    return 0;
}
