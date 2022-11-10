import cv2 as cv
import numpy as np
import time

def current_milli_time():
    return round(time.time() * 1000)

def make_hist_image(src, hist_w, hist_h, hist_size):
    histRange = (0, 256) # the upper boundary is exclusive
    hist = cv.calcHist(src, [0], None, [hist_size], histRange)
    bin_w = int(round( hist_w/hist_size ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(hist, hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    for i in range(1, hist_size):
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(hist[i-1]) ),
            ( bin_w*(i), hist_h - int(hist[i]) ),
            ( 255, 0, 0), thickness=1)
    return histImage

not_terminated = True
while (not_terminated):
    cap = cv.VideoCapture('/home/sailor/itmo_labs/cv/lab_1/samples/1.mp4')
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = 1000 / fps
    equalized = False
    while cap.isOpened():
        startTime = current_milli_time()
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        src = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if equalized:
            src = cv.equalizeHist(src)
        srcHistImage = make_hist_image(src, 1024, 400, 256)
        cv.imshow('Source image', src)
        cv.imshow("Source Hist", srcHistImage)
        execTime = current_milli_time() - startTime
        timeout = max(1, round(delay - execTime))
        k = cv.waitKey(timeout)
        if k == ord('q'):
            not_terminated = False
            break
        if k == ord('e'):
            equalized = not equalized

    cap.release()
cv.destroyAllWindows()