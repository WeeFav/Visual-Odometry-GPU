// bundle_adjustment_with_lk.cpp
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <vector>
#include <deque>
#include <iostream>
#include <map>
#include <set>
#include <memory>

// --- YOUR ReprojectionError (unchanged) ---
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

// --- Data structs ---
struct Landmark {
    int id;
    cv::Point3d pos;
    // observations: pair(frame_index_in_window, 2D point)
    std::vector<std::pair<int, cv::Point2d>> observations;
};

// --- Helper functions ---

// Extract R (3x3) and t (3x1) from a 4x4 pose matrix (world->camera): X_cam = R*X_world + t
inline void extract_R_t_from_pose(const cv::Mat& pose4x4, cv::Mat& R, cv::Mat& t) {
    CV_Assert(pose4x4.rows == 4 && pose4x4.cols == 4 && pose4x4.type() == CV_64F);
    R = pose4x4(cv::Range(0,3), cv::Range(0,3)).clone();
    t = pose4x4(cv::Range(0,3), cv::Range(3,4)).clone();
}

// Compose a 4x4 pose from R and t (R 3x3, t 3x1) world->camera
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
    std::vector<cv::Point3d> out;
    if (pts0.empty()) return out;

    cv::Mat pts4;
    cv::triangulatePoints(P0, P1, pts0, pts1, pts4); // 4 x N
    out.resize(pts4.cols);
    for (int i = 0; i < pts4.cols; ++i) {
        double w = pts4.at<double>(3,i);
        if (std::abs(w) < 1e-8) { out[i] = cv::Point3d(NAN,NAN,NAN); continue; }
        double X = pts4.at<double>(0,i) / w;
        double Y = pts4.at<double>(1,i) / w;
        double Z = pts4.at<double>(2,i) / w;
        out[i] = cv::Point3d(X,Y,Z);
    }
    return out;
}

// Compute reprojection of 3D point to pixel using K and R,t (world->camera)
inline cv::Point2d projectPoint(const cv::Point3d& X, const cv::Mat& K, const cv::Mat& R, const cv::Mat& t) {
    cv::Mat Xw = (cv::Mat_<double>(3,1) << X.x, X.y, X.z);
    cv::Mat Xc = R * Xw + t;
    double z = Xc.at<double>(2,0);
    double x = Xc.at<double>(0,0) / z;
    double y = Xc.at<double>(1,0) / z;
    double u = K.at<double>(0,0) * x + K.at<double>(0,2);
    double v = K.at<double>(1,1) * y + K.at<double>(1,2);
    return cv::Point2d(u,v);
}

// --- Optical-flow tracking across window ---
// Track initial points (pts0) from frame 0 through all frames in `imgs`.
// Returns tracks: vector per point, each entry is pair(frame_index, point2f) where it was observed
static std::vector<std::vector<std::pair<int, cv::Point2f>>> trackPointsAcrossWindow(
    const cv::Mat& img0,
    const std::vector<cv::Mat>& imgs, // imgs[0] corresponds to frame 0
    const std::vector<cv::Point2f>& pts0_in,
    const cv::Size& winSize = cv::Size(21,21))
{
    int N = (int)imgs.size();
    std::vector<std::vector<std::pair<int, cv::Point2f>>> tracks;
    tracks.resize(pts0_in.size());

    // initialize tracks with first frame observation
    for (size_t i = 0; i < pts0_in.size(); ++i) {
        tracks[i].emplace_back(0, pts0_in[i]);
    }

    std::vector<cv::Point2f> prevPts = pts0_in;
    cv::Mat prevImg = imgs[0];

    // Re-implement: track each initial point from frame 0 to each frame independently (robust & simple)
    for (size_t i = 0; i < pts0_in.size(); ++i) {
        tracks[i].clear();
        tracks[i].emplace_back(0, pts0_in[i]);
        cv::Point2f prevPt = pts0_in[i];
        bool alive = true;
        for (int fi = 1; fi < (int)imgs.size(); ++fi) {
            std::vector<cv::Point2f> in = { prevPt };
            std::vector<cv::Point2f> out;
            std::vector<unsigned char> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(imgs[fi-1], imgs[fi], in, out, status, err,
                                     winSize, 3, cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                                     cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 1e-4);
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
// imgs: vector of gray images for the window (size = window_size)
// pts0_in: keypoints in frame 0 (in pixel coordinates)
// pose_window: deque<cv::Mat> of 4x4 poses (world->camera) length == imgs.size()
static std::vector<Landmark> buildLandmarksFromFirstTwoFramesAndTracks(
    const std::vector<cv::Mat>& imgs,
    const std::vector<cv::Point2f>& pts0_in,
    const std::deque<cv::Mat>& pose_window,
    const cv::Mat& K)
{
    int W = (int)imgs.size();
    CV_Assert(W == (int)pose_window.size());
    // Track points from frame0 to all frames
    auto tracks = trackPointsAcrossWindow(imgs[0], imgs, pts0_in);

    // Build projection matrices for frame 0 and 1
    cv::Mat R0, t0, R1, t1;
    extract_R_t_from_pose(pose_window[0], R0, t0);
    extract_R_t_from_pose(pose_window[1], R1, t1);
    cv::Mat P0 = buildProjection(K, R0, t0);
    cv::Mat P1 = buildProjection(K, R1, t1);

    // Prepare vectors for triangulation: only for points that have a valid observation in frame1
    std::vector<cv::Point2f> pts0_for_tri, pts1_for_tri;
    std::vector<int> original_idx_for_tri;
    for (size_t i = 0; i < tracks.size(); ++i) {
        // require that track has an observation at frame 1
        bool hasFrame1 = false;
        cv::Point2f p0 = pts0_in[i];
        cv::Point2f p1;
        for (auto &obs : tracks[i]) {
            if (obs.first == 1) { hasFrame1 = true; p1 = obs.second; break; }
        }
        if (hasFrame1) {
            pts0_for_tri.push_back(p0);
            pts1_for_tri.push_back(p1);
            original_idx_for_tri.push_back((int)i);
        }
    }

    std::vector<cv::Point3d> tri3d = triangulatePointsLinear(pts0_for_tri, pts1_for_tri, P0, P1);

    std::vector<Landmark> landmarks;
    landmarks.reserve(tri3d.size());
    int next_id = 0;
    for (size_t j = 0; j < tri3d.size(); ++j) {
        cv::Point3d X = tri3d[j];
        if (!cv::checkRange(X)) continue;
        // Simple depth check: re-project into camera coordinates to inspect z
        cv::Mat Xw = (cv::Mat_<double>(3,1) << X.x, X.y, X.z);
        cv::Mat Xc0 = R0 * Xw + t0;
        double z0 = Xc0.at<double>(2,0);
        cv::Mat Xc1 = R1 * Xw + t1;
        double z1 = Xc1.at<double>(2,0);
        if (z0 <= 0 || z1 <= 0) continue;

        // Optionally check reprojection residual small in frames 0 & 1
        // (here we accept all)
        Landmark lm;
        lm.id = next_id++;
        lm.pos = X;

        // add observations from tracks for frames where track succeeded
        int orig_idx = original_idx_for_tri[j];
        for (auto &obs : tracks[orig_idx]) {
            // obs: (frame_idx, pt)
            lm.observations.emplace_back(obs.first, cv::Point2d(obs.second.x, obs.second.y));
        }
        landmarks.push_back(lm);
    }

    return landmarks;
}

// --- The main bundle adjustment function ---
// Inputs:
//   pose_window: deque of 4x4 CV_64F poses (world->camera)
//   landmark_window: (unused as we create landmarks from tracking) - kept for API compatibility
//   observations_window: (unused directly)
//   K: camera intrinsics CV_64F 3x3
void run_bundle_adjustment(std::deque<cv::Mat>& pose_window,
                           std::deque<std::vector<cv::Point3d>>& landmark_window,
                           std::deque<std::vector<cv::Point2f>>& observations_window,
                           const cv::Mat& K)
{
    // Convert to images for optical flow (assume you have access to the images corresponding to the poses).
    // For demo purpose we'll assume observations_window has the 2D keypoints for each frame? If not available,
    // user must supply a vector<cv::Mat> imgs (grayscale) for the window. Here we try to construct dummy imgs
    // from observations_window if possible. Better to pass images directly in a real system.
    int W = (int)pose_window.size();
    if (W < 2) {
        std::cerr << "Need at least 2 frames for BA\n";
        return;
    }

    // ==== Build imgs vector (caller should ideally pass actual cv::Mat frames). ====
    // Here we cannot reconstruct images; assume caller has a global vector<cv::Mat> last_window_imgs.
    // For the code to be self-contained, we'll require a global `std::vector<cv::Mat> last_window_imgs` variable.
    extern std::vector<cv::Mat> last_window_imgs; // must be set by caller (grayscale CV_8U)
    if ((int)last_window_imgs.size() != W) {
        std::cerr << "Error: last_window_imgs must be set to the same length as pose_window (grayscale images)\n";
        return;
    }

    // Initial keypoints: use features from first image.
    std::vector<cv::Point2f> keypoints0;
    // If caller provided observations for frame 0, use them; else detect features
    if (!observations_window.empty() && !observations_window[0].empty()) {
        for (auto &p : observations_window[0]) keypoints0.emplace_back((float)p.x, (float)p.y);
    } else {
        // detect good features in frame 0
        cv::goodFeaturesToTrack(last_window_imgs[0], keypoints0, 2000, 0.01, 8);
    }

    if (keypoints0.empty()) {
        std::cerr << "No initial keypoints to track.\n";
        return;
    }

    // Build landmarks by tracking & triangulation
    std::vector<cv::Mat> imgsVec;
    imgsVec.reserve(W);
    for (int i = 0; i < W; ++i) imgsVec.push_back(last_window_imgs[i]);

    std::vector<Landmark> landmarks = buildLandmarksFromFirstTwoFramesAndTracks(imgsVec, keypoints0, pose_window, K);

    if (landmarks.empty()) {
        std::cerr << "No landmarks after triangulation.\n";
        return;
    }

    // --- Setup Ceres problem ---
    ceres::Problem problem;

    // Parameter blocks for poses: vector<double> per frame of size 6 (angle-axis 3 + translation 3)
    std::vector<std::array<double,6>> pose_params(W);
    for (int i = 0; i < W; ++i) {
        cv::Mat R, t;
        extract_R_t_from_pose(pose_window[i], R, t);
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
    }

    // Add residuals: for each landmark and each observation in the window
    for (size_t i = 0; i < landmarks.size(); ++i) {
        const Landmark& lm = landmarks[i];
        for (auto &obs : lm.observations) {
            int frame_idx = obs.first;
            if (frame_idx < 0 || frame_idx >= W) continue;
            // Check depth of point wrt that camera and skip if behind camera (optional)
            // Build reprojection residual
            ceres::CostFunction* cost = ReprojectionError::Create(obs.second.x, obs.second.y, K);
            problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), pose_params[frame_idx].data(), point_params[i].data());
        }
    }

    // Optional: fix the first pose to remove gauge freedom
    problem.SetParameterBlockConstant(pose_params[0].data());

    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // Write optimized poses back into pose_window
    for (int i = 0; i < W; ++i) {
        cv::Mat rvec = (cv::Mat_<double>(3,1) << pose_params[i][0], pose_params[i][1], pose_params[i][2]);
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        cv::Mat t = (cv::Mat_<double>(3,1) << pose_params[i][3], pose_params[i][4], pose_params[i][5]);
        cv::Mat newPose = compose_pose_from_R_t(R, t);
        pose_window[i] = newPose.clone();
    }

    // Write optimized points back into landmark_window[0] (or a separate structure)
    // We'll pack optimized points into landmark_window[0] for convenience (caller can change)
    std::vector<cv::Point3d> optimizedPoints;
    optimizedPoints.reserve(point_params.size());
    for (size_t i = 0; i < point_params.size(); ++i) {
        optimizedPoints.emplace_back(point_params[i][0], point_params[i][1], point_params[i][2]);
    }
    // Replace first element of landmark_window (or push)
    if (!landmark_window.empty()) {
        landmark_window[0] = optimizedPoints;
    } else {
        landmark_window.push_back(optimizedPoints);
    }
}

// --------------------------------------------------
// Example usage notes (not compiled here):
// - Must provide `last_window_imgs` (grayscale CV_8U images) externally and set it to the frames
//   that correspond (in the same order) to the poses in `pose_window`.
// - pose_window must contain CV_64F 4x4 poses world->camera.
// - observations_window may be left empty; function will detect features in frame 0.
// --------------------------------------------------

// A small global variable required by the function (you must set this in your main loop)
std::vector<cv::Mat> last_window_imgs; // set to grayscale images for the current BA window

