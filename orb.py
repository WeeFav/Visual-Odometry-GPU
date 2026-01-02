import cv2
import numpy as np

class OrientedFAST():
    def __init__(self, threshold=50, n=9, nms_window=1, patch_size=9):
        self.threshold = threshold
        self.n = n
        self.nms_window = nms_window
        self.patch_size = patch_size
    
        # Precompute circle offsets (16 pixels on Bresenham circle of radius 3)
        self.circle_offsets = np.array([
            (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
            (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)
        ])

    def detect(self, image):
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    

        keypoints = []
        score_img = np.zeros(image.shape)

        for y in range(3, image.shape[0] - 3):
            for x in range(3, image.shape[1] - 3):
                Ip = int(image[y, x])

                # FAST check: examine pixels 1, 5, 9, 13 first (indices 0, 4, 8, 12)
                check_pixels = [0, 4, 8, 12]
                brighter = 0
                darker = 0
                for idx in check_pixels:
                    cp = int(image[y + self.circle_offsets[idx][0], x + self.circle_offsets[idx][1]])
                    if cp >= Ip + self.threshold:
                        brighter += 1
                    elif cp <= Ip - self.threshold:
                        darker += 1

                # If none are sufficiently bright/dark, skip
                if max(brighter, darker) < 3:
                    continue

                # Full test: check for n contiguous pixels
                circle_values = [int(image[y + dy, x + dx]) for dy, dx in self.circle_offsets]
                
                # Concatenate circle to itself to handle wrap-around [a0, a1, ..., a15, a0, a1, ..., a15]
                circle_values = circle_values + circle_values 

                for i in range(16):
                    segment = circle_values[i:i+self.n]
                    if all(v >= Ip + self.threshold for v in segment) or all(v <= Ip - self.threshold for v in segment):
                        keypoints.append((x, y))
                        score_img[y, x] = sum(abs(Ip - int(image[y + dy, x + dx])) for dy, dx in self.circle_offsets) # for nms
                        break

        # NMS - Non Maximal Suppression
        if self.nms_window != 0:
            nms_keypoints = []
            for x, y in keypoints:
                window = score_img[y-self.nms_window:y+self.nms_window+1, x-self.nms_window:x+self.nms_window+1]
                # Check if the center pixel is the maximum in the window
                if score_img[y, x] == np.max(window):
                    nms_keypoints.append([x, y])
        else:
            nms_keypoints = keypoints

        return np.array(nms_keypoints)
    
    def keypoint_orientations(self, image, keypoints):
        patch_radius = self.patch_size // 2
        orientations = []

        for x, y in keypoints:
            # Skip keypoints too close to border
            if x - patch_radius < 0 or x + patch_radius >= image.shape[1] or y - patch_radius < 0 or y + patch_radius >= image.shape[0]:
                orientations.append(0)
                continue

            patch = image[y-patch_radius:y+patch_radius+1, x-patch_radius:x+patch_radius+1].astype(np.float32)
            m_10, m_01, m_00 = 0, 0, 0

            for r in range(patch.shape[0]):
                for c in range(patch.shape[1]):
                    dy = r - patch_radius
                    dx = c - patch_radius
                    m_10 += dx * patch[r, c]
                    m_01 += dy * patch[r, c]
                    m_00 += patch[r, c]

            angle = np.arctan2(m_01, m_10)
            orientations.append(angle)

        return np.array(orientations)

class RotatedBRIEF:
    def __init__(self, patch_size=31, n_bits=256):
        self.patch_size = patch_size
        self.patch_radius = patch_size // 2
        self.n_bits = n_bits

        # Generate test pattern (random pairs)
        # Each pair = (x1, y1, x2, y2)
        self.pattern = np.random.randint(
            -self.patch_radius, self.patch_radius + 1,
            size=(n_bits, 4)
        ).astype(np.float32)
    
    def rotate_pattern(self, angle):
        # rotational matrix
        R = np.array([
            [ np.cos(angle), -np.sin(angle)],
            [ np.sin(angle),  np.cos(angle)]
        ], dtype=np.float32)

        rotated = []
        for x1, y1, x2, y2 in self.pattern:
            p1 = R @ np.array([x1, y1])
            p2 = R @ np.array([x2, y2])
            rotated.append([p1[0], p1[1], p2[0], p2[1]])
        return np.array(rotated, dtype=np.float32)        

    def compute(self, image, keypoints, orientations):
        """
        image:      grayscale image (H,W)
        keypoints:  list of (x, y)
        orientations: list of angles for each keypoint

        returns: descriptors, shape = (N, desc_size)
        """ 

        descriptors = []

        for (x, y), angle in zip(keypoints, orientations):
            # Skip keypoints too close to the border
            if (x - self.patch_radius < 0 or
                x + self.patch_radius >= image.shape[1] or
                y - self.patch_radius < 0 or
                y + self.patch_radius >= image.shape[0]
            ):
                descriptors.append(np.zeros(self.n_bits, dtype=np.uint8))
                continue              

            # rotate sampling pairs
            rot_pattern = self.rotate_pattern(angle)

            bits = np.zeros(self.n_bits, dtype=np.uint8)

            for i, (dx1, dy1, dx2, dy2) in enumerate(rot_pattern):
                x1 = int(round(x + dx1))
                y1 = int(round(y + dy1))

                x2 = int(round(x + dx2))
                y2 = int(round(y + dy2))

                bits[i] = 1 if image[y1, x1] < image[y2, x2] else 0
            
            descriptors.append(bits)
        
        return np.array(descriptors)

if __name__ == '__main__':
    image = cv2.imread("/home/marvin/Visual-Odometry-GPU/000000.png", cv2.IMREAD_GRAYSCALE)

    fast = OrientedFAST()
    keypoints = fast.detect(image)
    print(len(keypoints))
    orientations = fast.keypoint_orientations(image, keypoints)

    # brief = RotatedBRIEF()
    # descriptors = brief.compute(image, keypoints, orientations)
    
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for [x, y], angle in zip(keypoints, orientations):
        dx = 10 * np.cos(angle)
        dy = 10 * np.sin(angle)
        cv2.circle(img_color, (int(x), int(y)), 3, (0, 0, 255), 1)
        cv2.arrowedLine(img_color, (int(x), int(y)), (int(x + dx), int(y + dy)), (0, 0, 255), 1, tipLength=0.3)

    # fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
    # keypoints = fast.detect(image, None)
    # output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    
    cv2.imshow('FAST Corners', img_color)
    # cv2.imshow("FAST Keypoints", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()