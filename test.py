import cv2
import os
import numpy as np
from canny import Canny
from hough_circle import HoughCircle

def test_self(canny_args, hough_args):
    canny = Canny(*canny_args)
    houghcircle = HoughCircle(*hough_args)
    folder_path = 'image'
    save_folder_path = 'result/selfh75'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    imgs = os.listdir(folder_path)
    for i, img in enumerate(imgs):
        img = cv2.imread(os.path.join(folder_path, img), cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_edge, angle_matrix = canny(img_gray)
        circles = houghcircle(img_edge, angle_matrix, 5, 40, 60)

        # draw the circles
        for y, x, r in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0), -1)

        save_path = os.path.join(save_folder_path, f"{i}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(os.path.join(save_path, "result.png"), img)
        cv2.imwrite(os.path.join(save_path, "edge.png"), img_edge)
        cv2.waitKey(0)

def test_cv2():
    folder_path = 'image'
    save_folder_path = 'result/cv'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    imgs = os.listdir(folder_path)
    for i, img in enumerate(imgs):
        img = cv2.imread(os.path.join(folder_path, img), cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        edges = cv2.Canny(img_blurred, 50, 100)

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=80,
            minRadius=10,
            maxRadius=150
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0), -1)

        save_path = os.path.join(save_folder_path, f"{i}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(os.path.join(save_path, "result.png"), img)
        cv2.imwrite(os.path.join(save_path, "edge.png"), edges)
        cv2.waitKey(0)

if __name__ == '__main__':
    canny_args = [3, 0.8, 50, 75]
    hough_args = [5, 100, 100]
    test_self(canny_args, hough_args)
    test_cv2()