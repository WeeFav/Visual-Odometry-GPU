import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class VisualOdometry:
    def __init__(self, image_dir, gt_file, K, max_frames=300):
        self.images = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])[:max_frames]
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.gt_poses = self.load_ground_truth(gt_file, max_frames)

        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.estimated_positions = []

    def load_ground_truth(self, gt_file, max_frames):
        poses = []
        with open(gt_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_frames:
                    break
                data = np.fromstring(line, sep=' ')
                pose = data.reshape(3, 4)
                t = pose[:, 3]
                poses.append(t)
        return np.array(poses)

    def get_keypoints_descriptors(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    def match_features(self, des1, des2):
        matches = self.flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        return good

    def triangulate_points(self, R, t, pts1, pts2):
        """Triangulate points between two frames"""
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t))
        pts4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T
        return pts3d

    def process(self):
        prev_img = cv2.imread(self.images[0], cv2.IMREAD_GRAYSCALE)
        kp1, des1 = self.get_keypoints_descriptors(prev_img)
        self.estimated_positions.append(self.t.copy())

        for idx in tqdm(range(1, len(self.images))):
            curr_img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
            kp2, des2 = self.get_keypoints_descriptors(curr_img)

            matches = self.match_features(des1, des2)
            if len(matches) < 8:
                self.estimated_positions.append(self.t.copy())
                kp1, des1 = kp2, des2
                continue

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, self.K)

            # Triangulate 3D points for scale
            pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), self.K, None).reshape(-1, 2)
            pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), self.K, None).reshape(-1, 2)
            pts3d_prev = self.triangulate_points(np.eye(3), np.zeros((3,1)), pts1_norm, pts2_norm)

            # Compute scale (median distance of triangulated points)
            if len(pts3d_prev) > 1:
                scale = np.median(np.linalg.norm(pts3d_prev[1:] - pts3d_prev[:-1], axis=1))
                print(scale)
            else:
                scale = 1.0

            # Apply scale to translation
            self.t += scale * self.R.dot(t)
            self.R = R.dot(self.R)

            self.estimated_positions.append(self.t.copy())
            kp1, des1 = kp2, des2

        self.estimated_positions = np.array(self.estimated_positions)

    def plot_trajectory(self):
        plt.figure(figsize=(10,6))
        plt.plot(self.gt_poses[:,0], self.gt_poses[:,2], label='Ground Truth', color='blue')
        plt.plot(self.estimated_positions[:,0], self.estimated_positions[:,2], label='VO with Scale', color='red')
        plt.legend()
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters)')
        plt.title('Visual Odometry Trajectory vs Ground Truth')
        plt.grid()
        plt.show()


if __name__ == "__main__":
    image_dir = "/home/d300/VO/data/kitti/data_odometry_gray/dataset/sequences/00/image_0"
    gt_file = "/home/d300/VO/data/kitti/data_odometry_poses/dataset/poses/00.txt"

    # KITTI sequence 00 intrinsic parameters
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])

    vo = VisualOdometry(image_dir, gt_file, K, max_frames=225)
    vo.process()
    vo.plot_trajectory()
