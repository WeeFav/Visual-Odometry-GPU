import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
    
class VisualOdom:
    def __init__(self, KITTI_DIR, seq):
        images_dir = f"{KITTI_DIR}/data_odometry_gray/dataset/sequences/{seq}/image_0"
        self.images = [os.path.join(images_dir, p) for p in sorted(os.listdir(images_dir))]
        
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
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
    
    def get_three_frame_matches(self, img3):
        # Find the keypoints and descriptors with ORB
        self.kp3, self.des3 = self.orb.detectAndCompute(img3, None) # train

        # Match frame 1-2
        matches_12 = self.flann.knnMatch(self.des1, self.des2, k=2)
        good_12 = []
        for m, n in matches_12:
            if m.distance < 0.8 * n.distance:
                good_12.append(m)
        
        # Match frame 2-3
        matches_23 = self.flann.knnMatch(self.des2, self.des3, k=2)
        good_23 = []
        for m, n in matches_23:
            if m.distance < 0.8 * n.distance:
                good_23.append(m)
        
        # Find common features in frame 2
        map_2_to_1 = {m.trainIdx: m.queryIdx for m in good_12}
        map_2_to_3 = {m.queryIdx: m.trainIdx for m in good_23}
        
        common_idx_frame2 = set(map_2_to_1.keys()) & set(map_2_to_3.keys())    
      
        # Build correspondence triplets
        triplets = []
        for idx2 in common_idx_frame2:
            idx1 = map_2_to_1[idx2]
            idx3 = map_2_to_3[idx2]
            triplets.append((idx1, idx2, idx3))
        
        # Extract corresponding keypoints
        pts1 = np.float32([self.kp1[idx1].pt for idx1, _, _ in triplets])
        pts2 = np.float32([self.kp2[idx2].pt for _, idx2, _ in triplets])
        pts3 = np.float32([self.kp3[idx3].pt for _, _, idx3 in triplets])
        
        return pts1, pts2, pts3

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

        # construct unscaled relative transform betwen frames
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        
        return R, t, T
    
    def triangulate_points(self, R, t, pts1, pts2):
        # Projection matrices for frame1 and frame2
        P1 = self.K @ np.eye(3, 4)            # first camera as origin
        P2 = self.K @ np.hstack((R, t))       # second camera relative to first
        
        # Convert points to homogeneous coordinates
        pts1_h = pts1.T  # shape 2xN
        pts2_h = pts2.T

        points_4d_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # 4xN
        points_3d = points_4d_h[:3] / points_4d_h[3]                 # convert from homogeneous
        return points_3d

    def estimate_scale(self, points_3d_12, points_3d_23):
        if len(points_3d_12) == 0 or len(points_3d_23) == 0:
            return 1.0  # fallback

        # Compute pairwise distances
        prev_dist = np.linalg.norm(points_3d_12[1:] - points_3d_12[:-1], axis=1)
        cur_dist = np.linalg.norm(points_3d_23[1:] - points_3d_23[:-1], axis=1)

        # Average ratio
        scale = np.median(prev_dist / (cur_dist + 1e-6))
        scale = np.clip(scale, 0.1, 5.0)
        return scale

    def get_scale(self, pts1, pts2, pts3):
        points_3d_12 = self.triangulate_points(self.R_12, self.t_12, pts1, pts2)
        points_3d_23 = self.triangulate_points(self.R_23, self.t_23, pts2, pts3)
        
        # Transform 3D points from frame1 to frame2 coordinate system
        points_3d_12_in_2 = (self.R_12 @ points_3d_12) + self.t_12 

        self.visualize_3d_points(points_3d_12_in_2, points_3d_23)       
        
        scale = self.estimate_scale(points_3d_12_in_2, points_3d_23)
        return scale
    
    def create_camera_frame(self, R, t):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        # Convert rotation and translation to transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        frame.transform(T)
        return frame
    
    def visualize_3d_points(self, points_3d_12_in_2, points_3d_23):
        # Convert to (N, 3) numpy arrays if needed
        points_3d_12_in_2 = np.asarray(points_3d_12_in_2).T
        points_3d_23 = np.asarray(points_3d_23).T

        # Create Open3D PointCloud objects
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points_3d_12_in_2)
        pcd1.paint_uniform_color([1, 0, 0])  # Red for frame 12

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points_3d_23)
        pcd2.paint_uniform_color([0, 1, 0])  # Green for frame 23
        
        # Create camera frames
        cam1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        cam2 = self.create_camera_frame(self.R_12, self.t_12)
        cam3 = self.create_camera_frame(self.R_23 @ self.R_12, self.R_23 @ self.t_12 + self.t_23)
        
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

        for i, gt_pose in enumerate(tqdm(range(225))):  
            gt_pose = self.gt_poses[i]
            
            if i == 0:
                img1 = cv2.imread(self.images[0], cv2.IMREAD_GRAYSCALE)
                self.kp1, self.des1 = self.orb.detectAndCompute(img1, None)
                cur_pose = gt_pose
            elif i == 1:
                img2 = cv2.imread(self.images[1], cv2.IMREAD_GRAYSCALE)
                self.kp2, self.des2 = self.orb.detectAndCompute(img2, None)
                self.R_12 = np.eye(3)
                self.t_12 = np.zeros((3,1))
                cur_pose = gt_pose
            else:
                img3 = cv2.imread(self.images[i], cv2.IMREAD_GRAYSCALE)
                pts1, pts2, pts3 = self.get_three_frame_matches(img3)

                pts1, pts2, pts3 = pts1[:10], pts2[:10], pts3[:10]
                
                self.draw_matches(i, pts1, pts2, pts3)
                
                self.R_23, self.t_23, T = self.get_pose(pts2, pts3)
                
                scale = self.get_scale(pts1, pts2, pts3)
                
                true_scale = np.linalg.norm(self.gt_poses[i][:3, 3] - self.gt_poses[i-1][:3, 3])
                
                # print("")
                # print(scale, true_scale)
                
                T[:3,3] *= scale
                    
                cur_pose = cur_pose @ np.linalg.inv(T)
                
                # Shift the cache: current becomes previous
                self.kp1, self.des1 = self.kp2, self.des2
                self.kp2, self.des2 = self.kp3, self.des3
                self.R_12, self.t_12 = self.R_23, self.t_23

            gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
            estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
            
        fig, ax = plt.subplots()
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title("Path Visualization")
        gt_line, = ax.plot([], [], 'g-', label='Ground Truth')
        est_line, = ax.plot([], [], 'r-', label='Estimated')
        ax.legend()
        gt_line.set_data([p[0] for p in gt_path], [p[1] for p in gt_path])
        est_line.set_data([p[0] for p in estimated_path], [p[1] for p in estimated_path])
        ax.relim()            # recompute limits
        ax.autoscale_view()   # rescale axes
        plt.show()
        

if __name__ == '__main__':
    KITTI_DIR = "/home/d300/VO/data/kitti"
    # KITTI_DIR = "D:\Visual-Odometry-GPU\data\kitti"
    seq = "08"
    vo = VisualOdom(KITTI_DIR, seq)
    vo.run()