import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
    
class VisualOdom:
    def __init__(self, KITTI_DIR, seq):
        images_dir = f"{KITTI_DIR}/data_odometry_gray/dataset/sequences/{seq}/image_0"
        self.images = [os.path.join(images_dir, p) for p in sorted(os.listdir(images_dir))]
        
        self.orb = cv2.ORB_create(3000)
        self.sift = cv2.SIFT_create(nfeatures=1000, 
                                    nOctaveLayers=3,
                                    contrastThreshold=0.04,
                                    edgeThreshold=10,
                                    sigma=1.6)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
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
    
    def get_matches(self, img2):
        # Find the keypoints and descriptors with ORB
        self.kp2, self.des2 = self.orb.detectAndCompute(img2, None) # train

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

    def get_pose(self, pts1, pts2):
        # find essential matrix using RANSAC 5-point algorithm
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, prob=0.999, threshold=1.0, mask=None)
        # keep inliers from RANSAC
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]
           
        # decompose essential matrix to get best R and best unscaled t
        retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        # construct unscaled relative transform betwen frames
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        
        return T

    def run(self):
        gt_path = []
        estimated_path = []

        for i, gt_pose in enumerate(tqdm(range(1000))):  
            gt_pose = self.gt_poses[i]
            
            if i == 0:
                img1 = cv2.imread(self.images[0], cv2.IMREAD_GRAYSCALE)
                self.kp1, self.des1 = self.orb.detectAndCompute(img1, None)
                cur_pose = gt_pose
            else:
                img2 = cv2.imread(self.images[i], cv2.IMREAD_GRAYSCALE)
                pts1, pts2 = self.get_matches(img2)
                
                T = self.get_pose(pts1, pts2)
                                
                true_scale = np.linalg.norm(self.gt_poses[i][:3, 3] - self.gt_poses[i-1][:3, 3])
                                
                T[:3,3] *= true_scale

                cur_pose = cur_pose @ np.linalg.inv(T)
                
                # Shift the cache: current becomes previous
                self.kp1, self.des1 = self.kp2, self.des2

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
    seq = "05"
    vo = VisualOdom(KITTI_DIR, seq)
    vo.run()