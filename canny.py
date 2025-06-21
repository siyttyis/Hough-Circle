import cv2
import numpy as np
import time
from scipy import signal

class Canny:
    def __init__(self, gaussian_kernel_size :int =3, sigma : float =0.8, low_threshold : int =50, high_threshold : int =100):
        """
        initialize Canny edge detector

        :param gaussian_kernel_size: 高斯核大小
        :param sigma: 高斯分布标准差
        :param low_threshold: 双阈值中的低阈值
        :param high_threshold: 双阈值中的高阈值
        """
        self.gaussian_kernel_size = gaussian_kernel_size
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self.gaussian_kernel = self.build_gaussian_kernel()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Do Canny edge detection on image
        """
        start = time.time()
        # Step 1: Padding image
        padded_img = self.padding(image)
        # Step 2: Gaussian blur
        blurred_img = self.gaussian_blur_sripy(padded_img)
        # Step 3: Sobel operator
        sobel_x = self.sobel_scipy(is_horizontal=True, image=blurred_img)
        sobel_y = self.sobel_scipy(is_horizontal=False, image=blurred_img)
        # Step 4: Caculate gradient magnitude and angle
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        gradient_angle[gradient_angle > 180] -= 180
        # print(gradient_magnitude.shape, gradient_angle.shape)
        # Step 5: Non-maximum suppression
        gradient_magnitude_nms = self.non_maximum_suppression(gradient_magnitude, gradient_angle)
        # Step 6: Double thresholding
        threshold_img = self.double_threshold_filter(gradient_magnitude_nms)
        # Step 7: Edge tracking by BFS
        output_img = self.edge_tracking(threshold_img)
        end = time.time()
        print(f"Canny use time {end-start:.2f}s")

        return output_img.clip(0, 255).astype(np.uint8), gradient_angle

    def padding(self, image: np.ndarray) -> np.ndarray:
        """
        Padding image with zeros
        """
        h = image.shape[0]
        w = image.shape[1]
        # Calculate the padding size
        pad_size = self.gaussian_kernel_size // 2

        output_img = np.zeros((h + pad_size * 2, w + pad_size * 2), dtype=np.uint8)
        output_img[pad_size:pad_size + h, pad_size:pad_size + w] = image.copy().astype(np.uint8)

        return output_img
    
    def build_gaussian_kernel(self) -> np.ndarray:
        """
        Build a Gaussian kernel
        """
        gaussian_kernel = np.zeros((self.gaussian_kernel_size, self.gaussian_kernel_size))

        half_size = self.gaussian_kernel_size // 2
        for x in range(self.gaussian_kernel_size):
            for y in range(self.gaussian_kernel_size):
                gaussian_kernel[x, y] = np.exp(-((x-half_size)**2 + (y-half_size)**2) / (2*(self.sigma**2)))
        gaussian_kernel /= (2 * np.pi * self.sigma**2)
        # Normalize the kernel
        gaussian_kernel /= np.sum(gaussian_kernel)

        return gaussian_kernel
    
    def gaussian_blur(self, padded_img) -> np.ndarray:
        """
        Apply Gaussian blur to image after padding
        """
        s = time.time()
        h, w = padded_img.shape
        h = h - (self.gaussian_kernel_size // 2) * 2
        w = w - (self.gaussian_kernel_size // 2) * 2
        output_img = np.zeros((h, w), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                output_img[i, j] = np.sum(padded_img[i:i+self.gaussian_kernel_size, j:j+self.gaussian_kernel_size]*self.gaussian_kernel)
        e = time.time()
        print(e-s)
        return output_img
    
    def gaussian_blur_sripy(self, padded_img) -> np.ndarray:
        """
        Apply Gaussian blur to image after padding by scipy.signal.convolved2d.
        
        :param param: [description]
        :return: [description]
        """
        output_img = signal.convolve2d(padded_img, self.gaussian_kernel, mode='valid')

        return output_img

    def sobel(self, is_horizontal: bool, image: np.ndarray) -> np.ndarray:
        """
        Apply Sobel operator to image

        :params is_horizontal: Operate on horizontal or vertical, true on horizontal
        """
        if is_horizontal:
            sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        else:
            sobel_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        h, w = image.shape
        padded_img = self.padding(image)
        output_img = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                output_img[i, j] = abs(np.sum(padded_img[i:i+self.gaussian_kernel_size, j:j+self.gaussian_kernel_size]*sobel_kernel))
        
        return output_img
    
    def sobel_scipy(self, is_horizontal: bool, image: np.ndarray) -> np.ndarray:
        """
        Apply Sobel operator to image

        :params is_horizontal: Operate on horizontal or vertical, true on horizontal
        """
        if is_horizontal:
            sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        else:
            sobel_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        padded_img = self.padding(image)

        output_img = abs(signal.convolve2d(padded_img, sobel_kernel, mode='valid'))

        return output_img

    def non_maximum_suppression(self, gradient_magnitude: np.ndarray, gradient_angle: np.ndarray) -> np.ndarray:
        """
        Do non-maximum suppression on gradient magnitude

        :param gradient_magnitude: 梯度矩阵
        :param gradient_angle: 角度矩阵
        :return: 非极大值抑制后的梯度矩阵
        """
        h, w = gradient_magnitude.shape

        for i in range(1, h - 1):
            for j in range(1, w -1):
                angle = gradient_angle[i, j]
                if 22.5 > angle or angle > 157.5:
                    near_gradident = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
                elif 22.5 <= angle < 67.5:
                    near_gradident = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
                elif 67.5 <= angle < 112.5:
                    near_gradident = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
                else:
                    # 112.5 <= angle < 157.5
                    near_gradident = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]

                if gradient_magnitude[i, j] <= max(near_gradident):
                    gradient_magnitude[i, j] = 0
        
        return gradient_magnitude

    def double_threshold_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Do double threshold filter on image
        """

        strong = 255
        weak = 128

        h, w = image.shape
        output_img = np.zeros((h, w), dtype=np.uint8)

        strong_edge = (image >= self.high_threshold)
        weak_edge = (image >= self.low_threshold) & (image < self.high_threshold)
        # Set strong edge to 255    
        output_img[strong_edge] = strong
        # Set weak edge to 128
        output_img[weak_edge] = weak

        return output_img

    def edge_tracking(self, image: np.ndarray) -> np.ndarray:
        """
        Do edge tracking by BFS.
        """

        """
        实现思路：从强边缘点开始，向8个方向扩展，找到弱边缘点，将其标记为强边缘点。
        1. 找到所有强边缘点
        2. 对于每个强边缘点，向8个方向扩展，找到所有弱边缘点
        3. 将所有弱边缘点标记为强边缘点，入栈
        4. 栈空，算法结束
        """
        h, w = image.shape
        output_img = np.zeros((h, w), dtype=np.uint8)
        is_visited = np.zeros_like(image, dtype=bool)
        next_offset = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
        strong_edge = np.argwhere(image == 255).tolist()

        while strong_edge:
            x, y = strong_edge.pop()
            if is_visited[x, y]:
                continue
            is_visited[x, y] = True
            output_img[x, y] = 255

            for dx, dy in next_offset:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    if image[nx, ny] == 128:
                        output_img[nx, ny] = 255
                        strong_edge.append((nx, ny))

        return output_img

   
if __name__ == '__main__':
    canny = Canny()
    img = cv2.imread('image/test.jpg', cv2.IMREAD_GRAYSCALE)
    a, _ = canny(img)
    cv2.imshow("a", a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



