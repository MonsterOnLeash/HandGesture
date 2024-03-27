import cv2 as cv
import numpy as np
import imutils
import pandas as pd
import pyautogui


camera = cv.VideoCapture(1)
background = None
PREPARATION_FRAMES = 30


def averaging(cur_frame, cur_weight):
    global background
    if background is None:
        background = cur_frame.copy().astype("float")
        return
    background = cv.accumulateWeighted(cur_frame, background, cur_weight)
    return


def locator(cur_frame, threshold=25):
    global background
    difference = cv.absdiff(background.astype("uint8"), cur_frame)
    thresholded = cv.threshold(difference, threshold, 255, cv.THRESH_BINARY)[1]
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        segmented = max(contours, key=cv.contourArea)
        return thresholded, segmented


def get_tg(p1, p2):
    if p1[0] == p2[0]:
        return float('inf')
    else:
        return 1.0*(p1[1]-p2[1])/(p1[0]-p2[0])


def cross_product(p1, p2, p3):
    return ((p2[0] - p1[0])*(p3[1] - p1[1])) - ((p2[1] - p1[1])*(p3[0] - p1[0]))


def graham(points):
    stack = []
    points = sorted(points, key=lambda x: [x[0], x[1]])
    start = points.pop(0)
    points = sorted(points, key=lambda p: (get_tg(p, start), -p[1], p[0]))
    stack.append(start)
    if len(points) < 3:
        return None
    stack.append(points[0])
    stack.append(points[1])
    for i in range(2, len(points)):
        stack.append(points[i])
        while len(stack) > 2 and cross_product(stack[-3], stack[-2], stack[-1]) < 0:
            stack.pop(-2)
    return stack


def is_on_line(p1, p2, p3):
    eps = 0
    cp = cross_product(p1, p2, p3)
    if 2000 > abs(cp) > 200:
        return False
    return True


def defects(cont, hul):
    counter = 0
    for i in range(1, len(hul)):
        st = hul[i - 1]
        end = hul[i]
        p1 = 0
        p2 = 0
        for idx, k in enumerate(cont):
            if (k == st).all():
                p1 = idx
                break
        for idx, k in enumerate(cont):
            if (k == end).all():
                p2 = idx
                break
        for j in range(p1 + 1, p2):
            if not is_on_line(st, end, cont[j]):
                counter += 1
                break
        for j in range(p2 + 1, p1):
            if not is_on_line(st, end, cont[j]):
                counter += 1
                break
    return counter

if __name__ == '__main__':
    camera = cv.VideoCapture(1)
    num_frames = 0
    roi_top, roi_bottom, roi_left, roi_right = 10, 350, 300, 600
    timer = 0
    while True:
        _, frame = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv.flip(frame, 1)
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        gray_scale = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray_blur = cv.GaussianBlur(gray_scale, (7, 7), 0)
        if num_frames < PREPARATION_FRAMES:
            averaging(gray_blur, 0.5)
        else:
            hand = locator(gray_blur)
            if hand is not None:
                thresholded, segmented = hand
                hull = graham(np.squeeze(segmented))
                if hull is None:
                    continue
                sg = np.array(pd.DataFrame(np.squeeze(segmented)).drop_duplicates())
                cv.drawContours(frame, [segmented + (roi_right//2, roi_top)], -1, (0, 0, 255))
                cv.drawContours(frame, [np.array(hull) + (roi_right//2, roi_top)], -1, (0, 255, 255))
                defs = defects(np.squeeze(segmented), hull)
                if defs >= 7 and timer <= 0:
                    pyautogui.press('space')
                    timer = 60
                cv.imshow("Thresholded", thresholded)
        num_frames += 1
        timer -= 1
        cv.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        cv.imshow("Video Feed", frame)
        if cv.waitKey(10) == ord('x'):
            break 
    camera.release()
    cv.destroyAllWindows()