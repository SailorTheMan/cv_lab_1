#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include <iostream>
#include <stdio.h>
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std::chrono;

static void calcHist(Mat& src, uint32_t histSize, double* histArray)
{
    for (int i = 0; i < histSize; i++)
        histArray[i] = 0.f;
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
        {
            int value = (int)src.at<uchar>(y, x);
            histArray[value]++;
        }
}

static void normalizeHist(uint32_t histSize, double* histArray, double newMax)
{
    double histMax = 0.f;
    for (int i = 0; i < histSize; i++)
        if (histMax < histArray[i])
            histMax = histArray[i];
    for (int i = 0; i < histSize; i++)
        histArray[i] *= (newMax/histMax);
}

static void makeCDF(uint32_t histSize, double* histArray, uint32_t cdfSize, double* cdfArray)
{
    for (int i = 0; i < cdfSize; i++)
        cdfArray[i] = 0;
    cdfArray[0] = histArray[0];
    for (int i = 1; i < cdfSize; i++)
        cdfArray[i] = cdfArray[i-1] + histArray[i];
}

static void equalizeHist(Mat& source, Mat& dst)
{
    double histArray[256];
    calcHist(source, 256, histArray);
    double cdfArray[256];
    makeCDF(256, histArray, 256, cdfArray);
    double eMap[256];
    double cdf_min = 0.f;
    int rows = source.rows, cols = source.cols;
    for (int i = 0; i < 256; i++)
        if (cdfArray[i] != 0)
        {
            cdf_min = cdfArray[i];
            break;
        }
    double k = 255.0f / (double)(rows*cols - cdf_min);
    for (int i = 0; i < 256; i++)
        eMap[i] = (k * (cdfArray[i] - cdf_min));
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
        {
            int value = (int)eMap[(int)source.at<uchar>(y, x)];
            // uchar value = 255;
            dst.at<uchar>(y, x) = value;
        }
}

static void make_hist_image(Mat& src, Mat& histImage)
{
    double histArray[256];
    calcHist(src, 256, histArray);
    int hist_w = histImage.cols;
    int hist_h = histImage.rows;
    normalizeHist(256, histArray, (double)hist_h);
    int bin_w = hist_w/256;
    for (int i = 1; i < 256; i++)
        line(histImage,
            Point2i(bin_w*(i-1), hist_h - (int)histArray[i-1]),
            Point2i(bin_w*i, hist_h - (int)histArray[i]),
            Scalar(255, 0, 0)
        );
}

int main(int argc, char const *argv[])
{
    long int frame_counter_equalized = 0;
    long int overall_time_equalized = 0;
    bool not_terminated = true;
    bool equalized = false;
    while(not_terminated)
    {
        Mat frame;
        VideoCapture cap;
        cap.open("/home/sailor/itmo_labs/cv/lab_1/samples/1.mp4");
        double fps = cap.get(CAP_PROP_FPS);
        double delay = 1000.f / fps;
        while (cap.isOpened())
        {
            auto timeNow = system_clock::now();
            auto startTime = duration_cast<milliseconds>(timeNow.time_since_epoch()).count();
            cap.read(frame);
            if (frame.empty()) {
                std::cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            Mat histImg(512, 1024, CV_8UC1, Scalar(0, 0, 0));
            Mat histImg1(512, 1024, CV_8UC1, Scalar(0, 0, 0));
            Mat outImg = gray.clone();

            if (equalized)
            {
                timeNow = system_clock::now();
                auto start_equalization_time = duration_cast<milliseconds>(timeNow.time_since_epoch()).count();
                equalizeHist(gray, outImg);
                timeNow = system_clock::now();
                auto end_equalization_time = duration_cast<milliseconds>(timeNow.time_since_epoch()).count();
                int equalization_time = end_equalization_time - start_equalization_time;
                overall_time_equalized += equalization_time;
                frame_counter_equalized ++;
            }

            make_hist_image(outImg, histImg);
            imshow("hist", histImg);
            imshow("frame", outImg);
            timeNow = system_clock::now();
            auto endTime = duration_cast<milliseconds>(timeNow.time_since_epoch()).count();
            int execTime = endTime - startTime;
            std::cout << "Processing time of one frame: " << execTime << "ms" << std::endl;
            int timeout = max(1, (int)(delay - execTime));
            int k = waitKey(timeout);
            if (k == 113) // 'q'
            {
                std::cout << "Mean processing time of one equalized frame: " << (double)overall_time_equalized/(double)frame_counter_equalized << "ms" << std::endl;
                not_terminated = false;
                break;
            }
            if (k == 101)
            {
                equalized = !equalized;
                std::cout << "Equalized: " << equalized << std::endl;
            }
        }
    }
    return 0;
}
