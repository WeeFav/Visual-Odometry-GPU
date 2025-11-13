

# https://github.com/sushlokshah/visual-odometry/blob/main/2d-2d/with_scale.ipynb



import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm

def point3D(k,R,t,pts1,pts2):
    P1 = k @ np.eye(3, 4)            
    P2 = k @ np.hstack((R, t))

    pts1_h = pts1.T
    pts2_h = pts2.T

    points_4d_h = cv.triangulatePoints(P1,P2,pts1_h ,pts2_h)
    points3D = points_4d_h / points_4d_h[3,:]
    return points3D.T

def triangulate_points(k,R,t,pts1,pts2):
    P1 = k @ np.eye(3, 4)       
    P2 = k @ np.hstack((R, t))
    
    pts1_h = pts1.T
    pts2_h = pts2.T

    points_4d_h = cv.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d_h[:3] / points_4d_h[3]             
    return points_3d.T

def RelativeScale1(last_cloud, new_cloud):
    min_idx = min([new_cloud.shape[0],last_cloud.shape[0]])
    p_Xk = new_cloud[:min_idx]
    Xk = np.roll(p_Xk,shift = -10)
    p_Xk_1 = last_cloud[:min_idx]
    Xk_1 = np.roll(p_Xk_1,shift = -10)
    d_ratio = (np.linalg.norm(p_Xk_1 - Xk_1,axis = -1))/(np.linalg.norm(p_Xk - Xk,axis = -1))

    return np.clip(np.median(d_ratio), 0.1, 5.0)

translations1 = []
translations1.append(np.zeros((3,1)))
rotations1 = []
rotations1.append(np.identity(3))
scale1 = []
pointcloud = []
error_set = []

KITTI_DIR = "/home/d300/VO/data/kitti"
seq = "00"
images_dir = f"{KITTI_DIR}/data_odometry_gray/dataset/sequences/{seq}/image_0"
images = [os.path.join(images_dir, p) for p in sorted(os.listdir(images_dir))]
sift = cv.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

with open(f"{KITTI_DIR}/data_odometry_gray/dataset/sequences/{seq}/calib.txt", 'r') as f:
    params = np.fromstring(f.readline()[4:], dtype=np.float64, sep=' ')
    P = np.reshape(params, (3, 4))
    k = P[0:3, 0:3]

L = 150

ground_truth = np.loadtxt(f"{KITTI_DIR}/data_odometry_poses/dataset/poses/{seq}.txt", delimiter = ' ')
gx = ground_truth[:L,3]
gz = ground_truth[:L,11]

for a in tqdm(range(len(images[:L])-2)):
    old_cloud = pointcloud
    img1 = cv.imread(images[a],cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(images[a+1],cv.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    R = []
    t = []
    E, mask = cv.findEssentialMat(pts1,pts2,k,cv.RANSAC, prob = 0.999,threshold = 0.4,mask=None)
    retval, R, t, mask = cv.recoverPose(E, pts1, pts2, k)

    if a == 0:
        pointcloud = triangulate_points(k,R,t,pts1,pts2)
        translations1.append(t)
        rotations1.append(R)
        pointcloud = pointcloud[:,:3]
        p3d = np.ones((4,pointcloud.shape[0]))
        p3d[:3,:] = pointcloud.T
        projection_mat = np.zeros((3,4))
        projection_mat[:3,:3] = rotations1[a]
        projection_mat[:,3] = translations1[a].reshape((3))
        # proj_pt = k@projection_mat@p3d
        # proj_pt = proj_pt.T
        # proj_pt= proj_pt / proj_pt[:,2].reshape((-1,1))
        # proj_pt = proj_pt[:,:2]
        # err = proj_pt - pts2.reshape((-1,2))
        # norm_err = np.linalg.norm(err, axis=1)
        # error_set.append(np.min(norm_err))
        #print(a,np.min(norm_err))
        
    elif a == len(images[:L])-1:
        pointcloud = triangulate_points(k,R,t,pts1,pts2)
        #print(t)
        rotations1.append(rotations1[a]@R)
        s1 = RelativeScale1(old_cloud, pointcloud)
        #s1 = RelativeScale2(old_cloud, pointcloud,rotations[a+1],t)
        scale1.append(s1)
        translations1[a] = translations1[a-1] - s1*rotations1[a]@translations1[a]
        translations1.append(t+ translations1[a])
        pointcloud = pointcloud[:,:3]
        p3d = np.ones((4,pointcloud.shape[0]))
        p3d[:3,:] = pointcloud.T
        projection_mat = np.zeros((3,4))
        projection_mat[:3,:3] = rotations1[a]
        projection_mat[:,3] = translations1[a].reshape((3))
        # proj_pt = k@projection_mat@p3d
        # proj_pt = proj_pt.T
        # proj_pt= proj_pt / proj_pt[:,2].reshape((-1,1))
        # proj_pt = proj_pt[:,:2]
        # err = proj_pt - pts2.reshape((-1,2))
        # norm_err = np.linalg.norm(err, axis=1)
        # error_set.append(np.min(norm_err))
        #print(a,np.min(norm_err))
        
    else:
        pointcloud = triangulate_points(k,R,t,pts1,pts2)
        #print(t)
        rotations1.append(rotations1[a]@R)
        s1 = RelativeScale1(old_cloud, pointcloud)
        #s1 = RelativeScale2(old_cloud, pointcloud,rotations1[a+1],t)
        scale1.append(s1)
        translations1[a] = translations1[a-1] - s1*rotations1[a]@translations1[a]
        translations1.append(t)
        pointcloud = pointcloud[:,:3]
        p3d = np.ones((4,pointcloud.shape[0]))
        p3d[:3,:] = pointcloud.T
        projection_mat = np.zeros((3,4))
        projection_mat[:3,:3] = rotations1[a]
        projection_mat[:,3] = translations1[a].reshape((3))
        # proj_pt = k@projection_mat@p3d
        # proj_pt = proj_pt.T
        # proj_pt= proj_pt / proj_pt[:,2].reshape((-1,1))
        # proj_pt = proj_pt[:,:2]
        # err = proj_pt - pts2.reshape((-1,2))
        # norm_err = np.linalg.norm(err, axis=1)
        # error_set.append(np.min(norm_err))
        #print(a,np.min(norm_err))

x1 = []
y1 = []

for i in range(len(translations1)-1):
    y1.append(translations1[i][2])
    x1.append(-1*translations1[i][0])

fig,ax = plt.subplots()
ax.plot(gx, gz,color = 'red',label = 'ground_truth')
ax.plot(x1, y1,label = 'scale using dist ratio')

ax.legend()

plt.show()