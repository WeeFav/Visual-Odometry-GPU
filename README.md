# Monocular Visual Odometry

This project serves as a simple implementation of monocular visual odometry algorithm, where multiple methods will be compared, including feature detection, matching, scale estimation, optimization, and hardware implementation.

Overall Pipeline:
1. Detect features and descriptors in first image and second image
2. Match features
3. Compute essential matrix using 5-point algorithm from matched features
4. Decompose essential matrix to get rotation R and unit translation t
5. Use triangulation on matched features to estimate 3D points
6. Estimate relative scale between 2 transform by comparing 2 sets of 3D points. This requires at least 3 frames to form 2 sets of 3D points
7. Get current camera pose by concatenating previous pose with recovered relative transformation with scale

### SIFT vs ORB
Comparison between different feature detection methods, using scale-invariant feature transform (SIFT) vs Oriented FAST and Rotated BRIEF (ORB)


### Matching vs Tracking
Comparison between feature association methods, using flann knn matching vs optical flow tracking

### Scale estimation using matched vs unmatched 3D points
Comparison between 


### Local Windowed Bundle Adjustment


### CPU vs GPU