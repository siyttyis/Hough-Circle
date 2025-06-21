import cv2
import numpy as np
import copy
from canny import Canny

import math
import time

class HoughCircle:
    def __init__(self, step_length: int =5, vote_threshold: int =20, distance_threshold: int =100):
        """    
        Init hough circle algorithm arguments.

        :param step_length: 步长
        :param vote_threshold: 投票阈值
        :param distance_threshold: 距离阈值，与圆心间距比较判断是否重叠
        """
        self.step_length = step_length
        self.vote_threshold = vote_threshold
        self.distance_threshold = distance_threshold
        
    def __call__(self, image: np.ndarray, angle_matrix: np.ndarray, step_length: int =None, vote_threshold: int =None, distance_threshold: int =None) -> np.ndarray:
        """
        Do hough cricle algorithm on image.
        
        :param image: 传入的边缘图像
        :param angle_matrix: 传入的方向矩阵，内容为角度
        :param step_length: 步长
        :param vote_threshold: 投票阈值
        :param distance_threshold: 距离阈值，与圆心间距比较判断是否重叠
        
        :return circles: 检测的圆的矩阵，形状为[[y, x, r], ...]
        """

        """
        算法思路：
        Step1: 将所有的边缘点投影到霍夫空间，采用霍夫梯度方法，给梯度方向的可能圆心投票
        Step2: 根据投票阈值，筛选出所有符合条件的圆
        Step3: 进行NMS，去重
        """
        if step_length is not None:
           self.step_length = step_length
        if vote_threshold is not None:
            self.vote_threshold = vote_threshold
        if distance_threshold is not None:
            self.distance_threshold = distance_threshold
        start = time.time()
        # vote
        circles = self.vote(image, angle_matrix)
        # nms
        final_circles = self.non_maximum_suppression(circles)
        end = time.time()
        print(f"Hough circle use time {end-start:.2f}s")

        # print results
        for y, x, r in final_circles:
            print(f"圆心坐标：{x}, {y}, 半径：{r}")

        return final_circles
        
    def vote(self, image: np.ndarray, angle_matrix: np.ndarray) -> list:
        """
        Vote on hough space.
        
        :param image: 传入的边缘图像
        :param angle_matrix: 传入的方向矩阵，内容为角度
        :return circles: 投票得出的结果列表，形状为[[y, x, r], ...]
        """
        h, w = image.shape[:2]
        # 圆的最大长度，设定为当前图片的对角线大小
        max_redius = math.ceil(math.sqrt(h**2+w**2))
        # 转换角度为梯度
        angle_matrix = np.tan(np.deg2rad(angle_matrix))
        # 初始化霍夫空间投票矩阵，通过步长离散空间
        self.vote_matrix = np.zeros([math.ceil(h/self.step_length),
                                     math.ceil(w/self.step_length),
                                     math.ceil(max_redius/self.step_length)])
        # 对所有的边缘点做投票
        edge_pixels = np.argwhere(image > 0).tolist()
        for (i, j) in edge_pixels:
            x = j
            y = i
            r = 0
            # 在正梯度方向投票
            while 0 < x < w and 0 < y < h:
                self.vote_matrix[math.floor(y/self.step_length),
                                    math.floor(x/self.step_length),
                                    math.floor(r/self.step_length)] += 1
                # step
                x += self.step_length
                y += self.step_length * angle_matrix[i][j]
                r += math.sqrt(self.step_length**2 + (self.step_length*angle_matrix[i][j])**2)
            # 在负梯度方向投票
            x = j - self.step_length
            y = i - self.step_length * angle_matrix[i][j]
            r = math.sqrt(self.step_length**2 + (self.step_length*angle_matrix[i][j])**2)

            while 0 < x < w and 0 < y < h:
                    self.vote_matrix[math.floor(y/self.step_length),
                                        math.floor(x/self.step_length),
                                        math.floor(r/self.step_length)] += 1
                    # step
                    x -= self.step_length
                    y -= self.step_length * angle_matrix[i][j]
                    r += math.sqrt(self.step_length**2 + (self.step_length*angle_matrix[i][j])**2)
        # 检测高于投票阈值的圆
        circles = np.argwhere(self.vote_matrix > self.vote_threshold).tolist()

        return  circles

    def non_maximum_suppression(self, circles: list) -> np.ndarray:
        """
        Do nms on circles.
        
        :param circles: 检测的圆的矩阵，形状为[[y, x, r], ...]
        :return circles: 最终检测的圆的矩阵，形状为[[y, x, r], ...]
        """
        final_circles = copy.deepcopy(circles)
        # 对每个圆，与附近的圆相比，丢弃投票数更小的圆
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                y1, x1, r1 = circles[i]
                y2, x2, r2 = circles[j]
                
                distance = math.sqrt((x1-x2)**2+(y1-y2)**2) * self.step_length
                if distance <= self.distance_threshold:
                    if self.vote_matrix[y1][x1][r1] <= self.vote_matrix[y2][x2][r2]:
                        if circles[i] in final_circles:
                            final_circles.remove(circles[i])
                            break
                        else:
                            continue
                    else: 
                        if circles[j] in final_circles:
                            final_circles.remove(circles[j])
        # 将离散的数值转换为原空间数值
        final_circles = np.ceil(np.array(final_circles)*self.step_length+self.step_length/2).astype(np.int16)

        return final_circles


if __name__ == '__main__':
    canny = Canny()
    houghcircle = HoughCircle()

    img_path = 'image/test.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edge, angle_matrix = canny(img_gray)
    circles = houghcircle(img_edge, angle_matrix, 5, 40, 60)

    # draw the circles
    for y, x, r in circles:
        cv2.circle(img, (x, y), r, (255, 0, 0), 2)

    cv2.imshow('result', img)
    cv2.imshow('edge img', img_edge)
    cv2.imwrite("result.png", img)
    cv2.imwrite("edge.png", img_edge)
    cv2.waitKey(0)