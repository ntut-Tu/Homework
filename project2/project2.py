import cv2
import numpy as np
import os


INPUT_SET = ["1.jpg", "2.jpg", "3.jpg"]
INPUT_PATH = "dataset/"
OUTPUT_BASE = "dist/"
ARG_THRESHOLD = [155, 160, 100]
ARG_BLUR = [7, 5, 5]
ARG_CANNY = [[50, 150], [50, 150], [50, 150]]
ARG_HLP_SET = [[50, 60, 5], [90, 30, 25], [90, 30, 25]]

for i, _input in enumerate(INPUT_SET):
    OUTPUT_PATH = OUTPUT_BASE + str(i + 1) + "/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    img = cv2.imread(INPUT_PATH + _input)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(OUTPUT_PATH + "gray_img.jpg", gray_img)
    blur_img =  cv2.blur(gray_img, (ARG_BLUR[i], ARG_BLUR[i]), 0)
    cv2.imwrite(OUTPUT_PATH + "blur_img.jpg", blur_img)
    thres, bin_img = cv2.threshold(blur_img,ARG_THRESHOLD[i],255,cv2.THRESH_BINARY)
    cv2.imwrite(OUTPUT_PATH + "bin_img.jpg", bin_img)

    kernel = np.ones((3, 3), np.uint8)

    open_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    cv2.imwrite(OUTPUT_PATH + "morphology_img.jpg", close_img)

    edge_img = cv2.Canny(close_img, 50, 150)
    cv2.imwrite(OUTPUT_PATH + "edge_img.jpg", edge_img)
    linesP = cv2.HoughLinesP(
        edge_img,
        1,
        np.pi / 180,
        threshold=ARG_HLP_SET[i][0],
        lines=None,
        minLineLength=ARG_HLP_SET[i][1],
        maxLineGap=ARG_HLP_SET[i][2]
    )

    line_img = np.copy(img)
    if linesP is not None:
        print(len(linesP))
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(line_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 10, cv2.LINE_AA)


    #cv2.imshow("Detected Lines", edge_img)
    cv2.imwrite(OUTPUT_PATH + "Detected Lines.jpg", line_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()