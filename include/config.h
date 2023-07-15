#pragma once

#define USE_FP16

#include "opencv2/opencv.hpp"

//#define USE_INT8
//x1 = 0;
//y1 = -320;
//x2 = 1280;
//y2 = -400;
const static cv::Point_<float> point_begin(0, 320);
const static cv::Point_<float> point_end(1280, 400);

const static char *kInputTensorName = "images";
const static char *kOutputTensorName = "output";

const static int NUM_CLASSES = 1;
const static int BATCH_SIZE = 1;
const static std::string sub_type = "m";

const static std::string model_name = "yolov8" + sub_type + "_n" + std::to_string(NUM_CLASSES);
const static std::string wts_name = "../weights/" + model_name + ".wts";
const static std::string engine_name = "../weights/" + model_name + "_b" + std::to_string(BATCH_SIZE) + ".engine";
const static std::string class_file = "../weights/classes" + std::to_string(NUM_CLASSES) + ".txt";

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
