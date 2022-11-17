import cv2 as cv
import numpy as np
import time

def current_milli_time():
    return round(time.time() * 1000)

def calcHist(source):
    hist = [0 for i in range(256)]
    rows, cols = source.shape
    for x in range(rows):
        for y in range(cols):
            value = source[x, y]
            hist[value] += 1
    return hist


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

def makeCDF(hist):
    cdf = [0 for i in range(len(hist))]
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    return cdf

def equalizeHist(source):
    hist = calcHist(source)
    rows, cols = source.shape
    cdf = makeCDF(hist)
    size = rows, cols, 1
    eMap = [0 for i in range(256)]
    cdf_min = 0
    for i in range(256):
        if (cdf[i] != 0):
            cdf_min = cdf[i]
            break
    k = 255.0 / (rows*cols - cdf_min)
    for i in range(256):
        eMap[i] = (k*(cdf[i] - cdf_min))
    equalizedImg = np.zeros(size, dtype=np.uint8)
    for x in range(rows):
        for y in range(cols):
            equalizedImg[x, y] = eMap[source[x, y]]
    return equalizedImg


not_terminated = True
overallTime = 0
frameCounter = 0
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
            startEqualizationTime = current_milli_time()
            src = equalizeHist(src)
            endEqualizationTime = current_milli_time()
            equalizationTime = endEqualizationTime - startEqualizationTime
            overallTime += equalizationTime
            frameCounter += 1
        srcHistImage = make_hist_image(src, 1024, 400, 256)
        cv.imshow('Source image', src)
        cv.imshow("Source Hist", srcHistImage)
        execTime = current_milli_time() - startTime
        print("Processing time of one frame: ", execTime , "ms")
        timeout = max(1, round(delay - execTime))
        k = cv.waitKey(timeout)
        if k == ord('q'):
            not_terminated = False
            print("Mean equalization time: ", overallTime / frameCounter, "ms")
            break
        if k == ord('e'):
            equalized = not equalized
            print("Equalized: ", equalized)

    cap.release()
cv.destroyAllWindows()
