import cv2
import numpy as np
if __name__ == '__main__':
 # Generator an empty image with 480 height and 640 width
    blanks = np.zeros((480, 640, 3), dtype=np.uint8)
 #blanks.fill(255)
    for i in range(120):
        for j in range(160):
            blanks[i, j] = np.array([0, 0, 255])
    for i in range(120, 240):
        for j in range(160, 320):
            blanks[i, j] = np.array([255, 0, 0])
 ###FILL GREEN when x in range [320, 480] and y in range [240, 360]
    for j in range(320,480):
        for i in range(240,360):
            blanks[i, j] = np.array([0, 255, 0])
    for i in range(360, 480):
        for j in range(480, 640):
            blanks[i, j] = np.array([128, 128, 128])
    cv2.imwrite("test.png", blanks)