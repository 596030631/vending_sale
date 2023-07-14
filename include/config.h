#pragma once

#define USE_FP16
#include "opencv2/opencv.hpp"

//#define USE_INT8

const static char *kInputTensorName = "images";
const static char *kOutputTensorName = "output";

const static int kNumClass = 1;
const static int kBatchSize = 1;

const static int kInputH = 640;
const static int kInputW = 640;

const static float kNmsThresh = 0.4f;
const static float kConfThresh = 0.5f;

const static int kMaxInputImageSize = 1920 * 1080;
const static int kMaxNumOutputBbox = 1000;

struct alignas(float) Detection {
    cv::Rect_<float> rect;
    float prob;
    float label;
};
