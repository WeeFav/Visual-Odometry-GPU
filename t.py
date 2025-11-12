import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

images_dir = "/home/d300/VO/data/kitti/data_odometry_gray/dataset/sequences/07/image_0"
images = [os.path.join(images_dir, p) for p in sorted(os.listdir(images_dir))]
orb = cv2.ORB_create(3000)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

gt_poses = []
with open("/home/d300/VO/data/kitti/data_odometry_poses/dataset/poses/07.txt", 'r') as f:
    for line in f.readlines():
        T = np.fromstring(line, dtype=np.float64, sep=' ')
        T = T.reshape(3, 4)
        T = np.vstack((T, [0, 0, 0, 1]))
        gt_poses.append(T)
        
with open("/home/d300/VO/data/kitti/data_odometry_gray/dataset/sequences/07/calib.txt", 'r') as f:
    params = np.fromstring(f.readline()[4:], dtype=np.float64, sep=' ')
    P = np.reshape(params, (3, 4))
    K = P[0:3, 0:3]
    
def get_matches(i):
    img1 = cv2.imread(images[i - 1], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # Find matches
    matches = flann.knnMatch(des1, des2, k=2)

    # Find the matches there do not have a to high distance
    good = []
    try:
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
    except ValueError:
        pass

    # draw_params = dict(matchColor = -1, # draw matches in green color
    #             singlePointColor = None,
    #             matchesMask = None, # draw only inliers
    #             flags = 2)

    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good ,None,**draw_params)
    # cv2.imshow("image", img3)
    # cv2.waitKey(1)

    # Get the image points form the good matches
    q1 = np.float32([kp1[m.queryIdx].pt for m in good])
    q2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return q1, q2    
  
def get_pose(q1, q2):
    E, mask = cv2.findEssentialMat(q1, q2, K, cv2.RANSAC, prob=0.999, threshold = 0.4, mask=None)
    # q1 = q1[mask.ravel()==1]
    # q2 = q2[mask.ravel()==1]    
    # retval, R, t, mask = cv2.recoverPose(E, q1, q2, K)
    R, t = decomp_essential_mat(E, q1, q2)
    
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()
    return T

def triangulate_points(R, t, q1, q2, K):
    # Projection matrices for frame1 and frame2
    P1 = K @ np.eye(3, 4)            # first camera at origin
    P2 = K @ np.hstack((R, t))       # second camera relative to first
    
    # Convert points to homogeneous coordinates
    q1_h = q1.T  # shape 2xN
    q2_h = q2.T

    points_4d_h = cv2.triangulatePoints(P, P2, q1_h, q2_h)  # 4xN
    points_3d = points_4d_h[:3] / points_4d_h[3]             # convert from homogeneous
    return points_3d

def estimate_scale(prev_points_3d, cur_points_3d):
    if len(prev_points_3d) == 0 or len(cur_points_3d) == 0:
        return 1.0  # fallback

    # Compute pairwise distances
    prev_dist = np.linalg.norm(prev_points_3d[1:] - prev_points_3d[:-1], axis=1)
    cur_dist = np.linalg.norm(cur_points_3d[1:] - cur_points_3d[:-1], axis=1)

    # Average ratio
    scale = np.median(prev_dist / (cur_dist + 1e-6))
    scale = np.clip(scale, 0.1, 5.0)
    return scale

def sum_z_cal_relative_scale(R, t):
    # Get the transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()

    # Make the projection matrix
    P2 = np.matmul(np.concatenate((K, np.zeros((3, 1))), axis=1), T)

    # Triangulate the 3D points
    hom_Q1 = cv2.triangulatePoints(P, P2, q1.T, q2.T)
    # Also seen from cam 2
    hom_Q2 = np.matmul(T, hom_Q1)

    # Un-homogenize
    uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
    uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

    # Find the number of points there has positive z coordinate in both cameras
    sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
    sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

    # Form point pairs and calculate the relative scale
    relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
    return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

def decomp_essential_mat(E, q1, q2):
    # Decompose the essential matrix
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = np.squeeze(t)

    # Make a list of the different possible pairs
    pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

    # Check which solution there is the right one
    z_sums = []
    relative_scales = []
    for R, t in pairs:
        z_sum, scale = sum_z_cal_relative_scale(R, t)
        z_sums.append(z_sum)
        relative_scales.append(scale)

    # Select the pair there has the most points with positive z coordinate
    right_pair_idx = np.argmax(z_sums)
    right_pair = pairs[right_pair_idx]
    relative_scale = relative_scales[right_pair_idx]
    R1, t = right_pair
    t = t * relative_scale

    # print("")
    # print(relative_scale)

    return [R1, t]
    
fig, ax = plt.subplots()
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_title("Real-time Path Visualization")
gt_line, = ax.plot([], [], 'g-', label='Ground Truth')
est_line, = ax.plot([], [], 'r-', label='Estimated')
ax.legend()

gt_path = []
estimated_path = []
prev_points_3d = None

for i, gt_pose in enumerate(tqdm(range(225))):  
    gt_pose = gt_poses[i]
    
    if i == 0:
        cur_pose = gt_pose
    else:
        q1, q2 = get_matches(i)
        T = get_pose(q1, q2)
        
        # # Triangulate points for scale
        # points_3d = triangulate_points(T[:3,:3], T[:3,3].reshape(3,1), q1, q2, K)
        # if prev_points_3d is not None:
        #     scale = estimate_scale(prev_points_3d, points_3d)
        #     # print(scale)
        #     T[:3,3] *= scale
            
        cur_pose = cur_pose @ np.linalg.inv(T)
        # prev_points_3d = points_3d
        
    gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
    estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    
gt_line.set_data([p[0] for p in gt_path], [p[1] for p in gt_path])
est_line.set_data([p[0] for p in estimated_path], [p[1] for p in estimated_path])
ax.relim()            # recompute limits
ax.autoscale_view()   # rescale axes
plt.show()