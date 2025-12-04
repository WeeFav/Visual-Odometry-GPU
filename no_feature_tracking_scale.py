import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import time

class VisualOdom:
    def __init__(self, KITTI_DIR, seq, feature):
        images_dir = f"{KITTI_DIR}/data_odometry_gray/dataset/sequences/{seq}/image_0"
        self.images = [os.path.join(images_dir, p) for p in sorted(os.listdir(images_dir))]
        
        if feature == "orb":
            self.feature = cv2.ORB_create(3000)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        elif feature == "sift":
            self.feature = cv2.SIFT_create()
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.gt_poses = []
        with open(f"{KITTI_DIR}/data_odometry_poses/dataset/poses/{seq}.txt", 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                self.gt_poses.append(T)
                
        with open(f"{KITTI_DIR}/data_odometry_gray/dataset/sequences/{seq}/calib.txt", 'r') as f:
            params = np.fromstring(f.readline()[4:], dtype=np.float64, sep=' ')
            self.P = np.reshape(params, (3, 4))
            self.K = self.P[0:3, 0:3]

        self.prev_points_3d = []
    
    def get_matches(self, img2):
        # Find the keypoints and descriptors with ORB
        self.kp2, self.des2 = self.feature.detectAndCompute(img2, None) # train

        # Match frame 1-2
        matches = self.flann.knnMatch(self.des1, self.des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
                
        # Extract corresponding keypoints
        pts1 = np.float32([self.kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([self.kp2[m.trainIdx].pt for m in good])
        
        return pts1, pts2

    def draw_matches(self, i, pts1, pts2, pts3):
        img1 = cv2.imread(self.images[i-2])
        img2 = cv2.imread(self.images[i-1])
        img3 = cv2.imread(self.images[i])

        h, w = img1.shape[:2]
        canvas = np.zeros((h, w*3, 3), dtype=np.uint8)
        canvas[:h, :w] = img1
        canvas[:h, w:w*2] = img2
        canvas[:h, w*2:w*3] = img3

        for pt1, pt2, pt3 in zip(pts1, pts2, pts3):
            # Points in each image
            pt1 = tuple(np.round(pt1).astype(int))
            pt2 = tuple(np.round(pt2).astype(int))
            pt3 = tuple(np.round(pt3).astype(int))
            
            # Shift points according to their position in the canvas
            pt2_shifted = (pt2[0] + w, pt2[1])
            pt3_shifted = (pt3[0] + w*2, pt3[1])
            
            color = tuple(np.random.randint(0, 255, 3).tolist())  # random color per match
            
            # Draw lines connecting the keypoints across images
            # cv2.line(canvas, pt1, pt2_shifted, color, 1)
            # cv2.line(canvas, pt2_shifted, pt3_shifted, color, 1)
            
            # Optional: Draw keypoints
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.circle(canvas, pt2_shifted, 3, color, -1)
            cv2.circle(canvas, pt3_shifted, 3, color, -1)

        cv2.imshow("3-frame matches", canvas)

    def get_pose(self, pts2, pts3):
        # find essential matrix using RANSAC 5-point algorithm
        E, mask = cv2.findEssentialMat(pts2, pts3, self.K, cv2.RANSAC, prob=0.999, threshold=1.0, mask=None)
        # keep inliers from RANSAC
        pts2 = pts2[mask.ravel()==1]
        pts3 = pts3[mask.ravel()==1]
           
        # decompose essential matrix to get best R and best unscaled t
        retval, R, t, mask = cv2.recoverPose(E, pts2, pts3, self.K)
        
        return R, t
    
    def triangulate_points(self, R, t, pts1, pts2):
        # Projection matrices for frame1 and frame2
        P1 = self.K @ np.eye(3, 4)            # first camera as origin
        P2 = self.K @ np.hstack((R, t))       # second camera relative to first
        
        pts1_h = pts1.T  # shape 2xN
        pts2_h = pts2.T

        points_4d_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # 4xN
        points_3d = points_4d_h[:3] / points_4d_h[3]                 # convert from homogeneous
        return points_3d.T

    def estimate_scale(self):
        if len(self.prev_points_3d) == 0 or len(self.points_3d) == 0:
            return 1.0
        
        min_idx = min([self.prev_points_3d.shape[0], self.points_3d.shape[0]])

        prev_points_3d = self.prev_points_3d[:min_idx]
        points_3d = self.points_3d[:min_idx]

        # Compute pairwise distances
        prev_dist = np.linalg.norm(prev_points_3d[1:] - prev_points_3d[:-1], axis=1)
        cur_dist = np.linalg.norm(points_3d[1:] - points_3d[:-1], axis=1)

        # Average ratio
        scale = np.median(prev_dist / (cur_dist + 1e-6))
        scale = np.clip(scale, 0.1, 5.0)
        return scale

    def get_scale(self, R, t, pts1, pts2):
        self.points_3d = self.triangulate_points(R, t, pts1, pts2)
        
        # self.visualize_3d_points()       
        
        scale = self.estimate_scale()
        return scale
    
    def create_camera_frame(self, R, t):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        # Convert rotation and translation to transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        frame.transform(T)
        return frame
    
    def visualize_3d_points(self):
        if len(self.prev_points_3d) == 0 or len(self.points_3d) == 0:
            return 
        
        # Convert to (N, 3) numpy arrays if needed
        prev_points_3d = np.asarray(self.prev_points_3d).T[:10]
        points_3d = np.asarray(self.points_3d).T[:10]

        # Create Open3D PointCloud objects
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(prev_points_3d)
        pcd1.paint_uniform_color([1, 0, 0])  # Red for frame 12

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points_3d)
        pcd2.paint_uniform_color([0, 1, 0])  # Green for frame 23
        
        # Create camera frames
        cam1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Visualization", width=960, height=720)
        for g in [pcd1, pcd2, cam1]:
            vis.add_geometry(g)

        # Access render option
        render_opt = vis.get_render_option()
        render_opt.point_size = 3.0
        # render_opt.background_color = np.asarray([0, 0, 0])  # black background

        # View control to remove near clipping
        view_ctl = vis.get_view_control()
        new_near_plane_distance = 0.001 
        view_ctl.set_constant_z_near(new_near_plane_distance)
        view_ctl.set_zoom(2)

        params = view_ctl.convert_to_pinhole_camera_parameters()

        extrinsic = np.eye(4)
        extrinsic[:3, 3] = [0, 0, 0]

        # Assign back the modified extrinsic
        params.extrinsic = extrinsic
        view_ctl.convert_from_pinhole_camera_parameters(params)

        cv2.waitKey(0)
        vis.run()
        vis.destroy_window()

    def run(self):
        gt_path = []
        estimated_path = []
        gt_scale = []
        estimated_scale = []

        start_time = time.perf_counter()

        for i, gt_pose in enumerate(tqdm(range(1000))):  
            gt_pose = self.gt_poses[i]
            
            if i == 0:
                img1 = cv2.imread(self.images[0], cv2.IMREAD_GRAYSCALE)
                self.kp1, self.des1 = self.feature.detectAndCompute(img1, None)
                cur_pose = gt_pose
            else:
                img2 = cv2.imread(self.images[i], cv2.IMREAD_GRAYSCALE)
                pts1, pts2 = self.get_matches(img2)

                # pts1, pts2, pts3 = pts1[:10], pts2[:10], pts3[:10]
                # self.draw_matches(i, pts1, pts2, pts3)
                
                R, t = self.get_pose(pts1, pts2)
                
                scale = self.get_scale(R, t, pts1, pts2)
                estimated_scale.append(scale)
                
                true_scale = np.linalg.norm(self.gt_poses[i][:3, 3] - self.gt_poses[i-1][:3, 3])
                gt_scale.append(true_scale)
                
                # construct unscaled relative transform betwen frames
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = t.squeeze() * scale

                cur_pose = cur_pose @ np.linalg.inv(T)
                
                # Shift the cache: current becomes previous
                self.kp1, self.des1 = self.kp2, self.des2
                self.prev_points_3d = self.points_3d

            gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
            estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
  
        self.save_paths("./gt_path.txt", "./est_path.txt", "./scale.txt", gt_path, estimated_path, gt_scale, estimated_scale)
        return gt_path, estimated_path

    def save_paths(self, gt_file, est_file, scale_file, gt_path, est_path, gt_scale, est_scale):
        # Save ground truth path
        with open(gt_file, "w") as f_gt:
            for p in gt_path:
                f_gt.write(f"{p[0]} {p[1]}\n")

        # Save estimated path
        with open(est_file, "w") as f_est:
            for p in est_path:
                f_est.write(f"{p[0]} {p[1]}\n")

        # Save scale values
        with open(scale_file, "w") as f_scale:
            for g, e in zip(gt_scale, est_scale):
                f_scale.write(f"{g} {e}\n")

if __name__ == '__main__':
    KITTI_DIR = "/home/d300/VO/data/kitti"
    # KITTI_DIR = "D:\Visual-Odometry-GPU\data\kitti"
    seq = "05"
    vo = VisualOdom(KITTI_DIR, seq, "orb")
    gt_path, orb_path = vo.run()

    # vo = VisualOdom(KITTI_DIR, seq, "sift")
    # _, sift_path = vo.run()

    # fig, ax = plt.subplots()
    # ax.set_xlabel("X")
    # ax.set_ylabel("Z")
    # ax.set_title("Path Visualization")
    # ax.plot([p[0] for p in gt_path], [p[1] for p in gt_path], 'g-', label='Ground Truth')
    # ax.plot([p[0] for p in orb_path], [p[1] for p in orb_path], 'r-', label='ORB')
    # ax.plot([p[0] for p in sift_path], [p[1] for p in sift_path], 'b-', label='SIFT')
    # ax.legend()
    # ax.relim()            # recompute limits
    # ax.autoscale_view()   # rescale axes
    # plt.show()
