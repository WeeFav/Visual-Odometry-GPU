import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
    
class VisualOdom:
    def __init__(self):
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
    
    def triangulate_points(R, t, q1, q2):
        # Projection matrices for frame1 and frame2
        P1 = K @ np.eye(3, 4)            # first camera at origin
        P2 = K @ np.hstack((R, t))       # second camera relative to first
        
        # Convert points to homogeneous coordinates
        q1_h = q1.T  # shape 2xN
        q2_h = q2.T

        points_4d_h = cv2.triangulatePoints(P1, P2, q1_h, q2_h)  # 4xN
        points_3d = points_4d_h[:3] / points_4d_h[3]             # convert from homogeneous
        return points_3d

    def get_scale(self, pts1, pts2, pts3):
        points_3d_12 = self.triangulate_points()
        points_3d_23 = self.triangulate_points()
        
  
    def run(self):
        gt_path = []
        estimated_path = []
        prev_points_3d = None

        for i, gt_pose in enumerate(tqdm(range(100))):  
            gt_pose = self.gt_poses[i]
            
            if i == 0:
                img1 = cv2.imread(self.images[0], cv2.IMREAD_GRAYSCALE)
                self.kp1, self.des1 = self.orb.detectAndCompute(img1, None)
                cur_pose = gt_pose
            elif i == 1:
                img2 = cv2.imread(self.images[1], cv2.IMREAD_GRAYSCALE)
                self.kp2, self.des2 = self.orb.detectAndCompute(img2, None)
                self.prev_R = np.eye(3)
                self.prev_t = np.zeros((3,1))
                cur_pose = gt_pose
            else:
                img3 = cv2.imread(self.images[i], cv2.IMREAD_GRAYSCALE)
                pts1, pts2, pts3 = self.get_three_frame_matches(img3)
                
                self.R, self.t, T = self.get_pose(pts2, pts3)
                
                scale = self.get_scale(pts1, pts2, pts3)
                
                # Triangulate points for scale
                points_3d = triangulate_points(T[:3,:3], T[:3,3].reshape(3,1), q1, q2, K)
                if prev_points_3d is not None:
                    scale = estimate_scale(prev_points_3d, points_3d)
                    # print(scale)
                    T[:3,3] *= scale
                    
                cur_pose = cur_pose @ np.linalg.inv(T)
                prev_points_3d = points_3d
                
            # Shift the cache: current becomes previous
            
            gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
            estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
            
        gt_line.set_data([p[0] for p in gt_path], [p[1] for p in gt_path])
        est_line.set_data([p[0] for p in estimated_path], [p[1] for p in estimated_path])
        ax.relim()            # recompute limits
        ax.autoscale_view()   # rescale axes
        plt.show()
        

if __name__ == '__main__':
    # KITTI_DIR = "/home/d300/VO/data/kitti"
    KITTI_DIR = "D:\Visual-Odometry-GPU\data\kitti"
    seq = "02"
    vo = VisualOdom(KITTI_DIR, seq)
    vo.run()