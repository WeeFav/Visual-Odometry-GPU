import numpy as np
import cv2

def conv2d_numpy(image, kernel):
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Calculate output dimensions
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    output = np.zeros((output_h, output_w))

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            # Extract the region of interest (receptive field)
            region = image[i:i + kernel_h, j:j + kernel_w]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
    return output

if __name__ == '__main__':
    # input_data = np.random.randint(0, 10, (6, 6))
    # kernel_data = np.random.randint(0, 10, (3, 3))
    input_data = cv2.imread("/home/d300/VO/data/kitti/data_odometry_gray/dataset/sequences/00/image_0/000000.png", cv2.IMREAD_GRAYSCALE)

    kernel = np.array([[1, 4,  7,  4,  1],
                        [4, 16, 26, 16, 4],
                        [7, 26, 41, 26, 7],
                        [4, 16, 26, 16, 4],
                        [1, 4,  7,  4,  1]])/273      # 5x5 Gaussian Window

    output_data = conv2d_numpy(input_data, kernel)
    output_data = output_data.astype('uint8')

    print(input_data.shape)
    print(output_data.shape)

    cv2.imshow("raw", input_data)
    cv2.imshow("blur", output_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


