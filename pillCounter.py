# Ref: https://github.com/quinson/tablet-counter/blob/master/tablet-counter/tablet-counter.cpp
# Ref: https://github.com/kineticR/Image-coin-counter

# Ref: RE: Segfault when using createTrackbar sliders on macOS in opencv-python <5
#   https://github.com/opencv/opencv/pull/22255
#     https://github.com/opencv/opencv-python/issues/691
#   https://github.com/opencv/opencv/issues/22561
#     pip install opencv-python-rolling

# TODO: Could using watershed improve this?
#   https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
#   https://pyimagesearch.com/2015/11/02/watershed-opencv/

# TODO: Could using Canny edge detection improve this? https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

import random
import numpy as np
import cv2 as cv

hueShiftSlider: int = 0
hueShiftSliderOld: int = hueShiftSlider

hueSliderMin: int = 13
hueSliderMax: int = 22

saturationSliderMin: int = 68
saturationSliderMax: int = 118

valueSliderMin: int = 119
valueSliderMax: int = 182

srcImg: cv.Mat
tmpImg: cv.Mat
hsv: cv.Mat
mask: cv.Mat


def onTrackbar(foo):
    global hsv, mask
    global hueShiftSlider, hueShiftSliderOld
    global hueSliderMin, saturationSliderMin, valueSliderMin
    global hueSliderMax, saturationSliderMax, valueSliderMax

    hueShiftSlider = cv.getTrackbarPos("Hue Shift", "Ranged HSV")
    hueSliderMin = cv.getTrackbarPos("Hue Min", "Ranged HSV")
    hueSliderMax = cv.getTrackbarPos("Hue Max", "Ranged HSV")
    saturationSliderMin = cv.getTrackbarPos("Saturation Min", "Ranged HSV")
    saturationSliderMax = cv.getTrackbarPos("Saturation Max", "Ranged HSV")
    valueSliderMin = cv.getTrackbarPos("Value Min", "Ranged HSV")
    valueSliderMax = cv.getTrackbarPos("Value Max", "Ranged HSV")

    # print("----------------------")
    # print(f'[onTrackbar] Hue:        ({hueSliderMin}, {hueSliderMax})')
    # print(
    #     f'[onTrackbar] Saturation: ({saturationSliderMin}, {saturationSliderMax})')
    # print(f'[onTrackbar] Value:      ({valueSliderMin}, {valueSliderMax})')
    # print("----------------------\n")

    if (hueShiftSlider != hueShiftSliderOld):
        [maskRows, maskColumns] = mask.shape

        for row in range(maskRows):
            for column in range(maskColumns):
                h = hsv[row, column, 0]

                h_plus_shift = h
                h_plus_shift += hueShiftSlider - hueShiftSliderOld

                if (h_plus_shift < 0):
                    h = 180 + h_plus_shift
                elif (h_plus_shift > 180):
                    h = h_plus_shift - 180
                else:
                    h = h_plus_shift

                hsv[row, column, 0] = h

        hueShiftSliderOld = hueShiftSlider

    COLOR_MIN = (hueSliderMin, saturationSliderMin, valueSliderMin)
    COLOR_MAX = (hueSliderMax, saturationSliderMax, valueSliderMax)

    mask = cv.inRange(hsv, COLOR_MIN, COLOR_MAX)
    cv.imshow("Ranged HSV", mask)


srcImg = cv.imread('test.jpg')

# TODO: Using this pyrDown function breaks something that is causing the contour drawing/counting to only show 1 big contour,
# rather than lots of smaller individual ones. I suspect it's reducing the resolution too much
# tmpImg = cv.pyrDown(srcImg)
# srcImg = cv.pyrDown(tmpImg)

hsv = cv.cvtColor(srcImg, cv.COLOR_BGR2HSV)
# mask: cv.Mat = np.zeros(hsv.shape, dtype=np.uint8)
# imgray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)

(srcImgHeight, srcImgWidth, srcImgChannels) = srcImg.shape

resizedSrcImgWidth = int(srcImgWidth / 4)
resizedSrcImgHeight = int(srcImgHeight / 4)

cv.namedWindow("Ranged HSV")
cv.namedWindow("Original")
cv.namedWindow("Drawing")
cv.resizeWindow("Ranged HSV", resizedSrcImgWidth, resizedSrcImgHeight)
cv.resizeWindow("Original", resizedSrcImgWidth, resizedSrcImgHeight)
cv.resizeWindow("Drawing", resizedSrcImgWidth, resizedSrcImgHeight)

cv.createTrackbar("Hue Shift", "Ranged HSV", hueShiftSlider, 180, onTrackbar)
cv.createTrackbar("Hue Min", "Ranged HSV", hueSliderMin, 180, onTrackbar)
cv.createTrackbar("Hue Max", "Ranged HSV", hueSliderMax, 180, onTrackbar)
cv.createTrackbar("Saturation Min", "Ranged HSV",
                  saturationSliderMin, 255, onTrackbar)
cv.createTrackbar("Saturation Max", "Ranged HSV",
                  saturationSliderMax, 255, onTrackbar)
cv.createTrackbar("Value Min", "Ranged HSV", valueSliderMin, 255, onTrackbar)
cv.createTrackbar("Value Max", "Ranged HSV", valueSliderMax, 255, onTrackbar)

onTrackbar(0)

cv.imshow("Original", srcImg)
# cv.imshow("Drawing", imgray)

# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv.findContours(
#     thresh,
#     cv.RETR_TREE,
#     cv.CHAIN_APPROX_SIMPLE
# )

try:
    while True:
        cv.waitKey()

        print("----------------------")
        print(f'Hue:        ({hueSliderMin}, {hueSliderMax})')
        print(f'Saturation: ({saturationSliderMin}, {saturationSliderMax})')
        print(f'Value:      ({valueSliderMin}, {valueSliderMax})')
        print("----------------------\n")

        contours, hierarchy = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE
        )  # RETR_TREE

        count: int = 0
        drawingImg: cv.Mat = np.zeros(srcImg.shape, dtype=np.uint8)

        m = []

        for idx, contour in enumerate(contours):
            color = (
                random.uniform(0, 255),
                random.uniform(0, 255),
                random.uniform(0, 255)
            )

            length = cv.arcLength(contour, 0)
            area = cv.contourArea(contour, 0)

            if length == 0 or area == 0:
                continue

            # print("Contour", idx, "length=", length, "area=", area)

            if (area / length > 1.5):
                cv.drawContours(
                    drawingImg,
                    contours,
                    idx,
                    color,
                    2,  # Thickness
                    8,  # Line Type
                    hierarchy,
                    0,  # Max Level
                    (0, 0)  # Offset
                )

                cv.drawContours(
                    srcImg,
                    contours,
                    idx,
                    color,
                    2,  # Thickness
                    8,  # Line Type
                    hierarchy,
                    0,  # Max Level
                    (0, 0)  # Offset
                )

                # cv.drawContours(src, contours, (int)i, color, 2, 8, hierarchy, 0, Point())

                m.append((length, area))
                count += 1

        # cv.drawContours(drawingImg, contours, -1, (0, 255, 0), 3)

        if len(m) != 0:
            m.sort()

            if (count % 2 == 0):
                medianLength = (m[int(len(m) / 2) - 1][0] +
                                m[int(len(m) / 2)][0]) / 2

                medianArea = (m[int(len(m) / 2) - 1][1] +
                              m[int(len(m) / 2)][1]) / 2
            else:
                medianLength = m[int(len(m) / 2)][0]
                medianArea = m[int(len(m) / 2)][1]

            print(f'median: ({medianLength}, {medianArea})')

        medLengthCount = 0
        medAreaCount = 0
        for idx, t in enumerate(m):
            l = t[0] / medianLength
            l = 1 if (l < 1) else round(l)

            a = t[1] / medianArea
            a = 1 if (a < 1) else round(a)

            print(f'{idx+1}: ({t}) -> ({l}, {a})')

            medLengthCount += l
            medAreaCount += a

        print("Count:", count)
        print("Median Length Count:", medLengthCount)

        cv.imshow("Drawing", drawingImg)
        cv.imshow("Original", srcImg)


except KeyboardInterrupt:
    pass

# cv.imwrite('test-contours.jpg', drawingImg)

cv.destroyAllWindows()
